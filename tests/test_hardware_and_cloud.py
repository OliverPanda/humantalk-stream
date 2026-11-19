"""A7 硬件分档 + 云端客户端 单元测试（全 mock，不打真 API、不碰 GPU）

只跑本文件：.venv\\Scripts\\python.exe -m pytest tests/test_hardware_and_cloud.py -v
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import OpenAI

from krvoiceai.core import hardware as hw
from krvoiceai.modules.omnihuman_client import OmniHumanClient, OmniHumanError
from krvoiceai.modules.seedance_client import SeedanceClient, SeedanceError


# ============ 硬件分档 ============

def test_detect_tier_no_gpu():
    assert hw.detect_tier([]) == "cloud"


def test_detect_tier_lite_2060():
    gpus = [{"name": "NVIDIA GeForce RTX 2060", "total_mb": 6144, "free_mb": 5000}]
    assert hw.detect_tier(gpus) == "lite"


def test_detect_tier_pro_4090():
    gpus = [{"name": "NVIDIA GeForce RTX 4090", "total_mb": 24564, "free_mb": 20000}]
    assert hw.detect_tier(gpus) == "pro"


def test_tier_allows_ordering():
    assert hw.tier_allows("pro", "lite")
    assert hw.tier_allows("lite", "cloud")
    assert not hw.tier_allows("lite", "pro")
    assert not hw.tier_allows("cloud", "lite")


def test_ensure_vram_blocks_small_card():
    """2060(6GB) 上申请 18GB —— 必须拦截且报错含人话"""
    gpus = [{"name": "NVIDIA GeForce RTX 2060", "total_mb": 6144, "free_mb": 5000}]
    with patch.object(hw, "query_gpus", return_value=gpus):
        with pytest.raises(hw.VramGateError) as ei:
            hw.ensure_vram(18, "LatentSync 1.6")
        assert "18" in str(ei.value)
        assert "拦截" in str(ei.value)


def test_ensure_vram_blocks_no_gpu():
    with patch.object(hw, "query_gpus", return_value=[]):
        with pytest.raises(hw.VramGateError):
            hw.ensure_vram(12, "EchoMimic V3")


def test_ensure_vram_passes_big_card():
    gpus = [{"name": "NVIDIA GeForce RTX 4090", "total_mb": 24564, "free_mb": 20000}]
    with patch.object(hw, "query_gpus", return_value=gpus):
        hw.ensure_vram(18, "LatentSync 1.6")  # 不应抛


def test_model_catalog_allowed_flags_on_lite():
    """lite 档：pro 模型 allowed=False，云端/lite 模型 allowed=True"""
    gpus = [{"name": "RTX 2060", "total_mb": 6144, "free_mb": 5000}]
    with patch.object(hw, "query_gpus", return_value=gpus):
        catalog = hw.get_model_catalog(None)
    by_id = {m["id"]: m for m in catalog}
    assert by_id["latentsync"]["allowed"] is False
    assert by_id["echomimic_v3"]["allowed"] is False
    assert by_id["musetalk"]["allowed"] is True
    assert by_id["omnihuman_api"]["allowed"] is True
    assert by_id["seedance_api"]["allowed"] is True


# ============ Seedance 客户端 ============

class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=0):
        yield b"FAKE_MP4_BYTES" * 100

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_seedance_requires_relay_even_with_official_keys(monkeypatch):
    for key in ("VIDEO_API_BASE", "VIDEO_API_KEY", "VIDEO_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ARK_API_KEY", "unused-official-key")
    monkeypatch.setenv("VOLC_ACCESS_KEY", "unused-official-ak")
    client = SeedanceClient()
    ok, reason = client.available()
    assert not ok and "VIDEO_API_BASE" in reason


def test_seedance_delegates_to_selected_relay_model(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_API_BASE", "https://relay.example/v1")
    monkeypatch.setenv("VIDEO_API_KEY", "relay-key")
    monkeypatch.setenv("VIDEO_MODEL", "minimax-h3")
    seen = {}

    def fake_generate(self, prompt, output_path, duration, ratio, reference_image, generate_audio):
        seen.update(model=self.model, prompt=prompt, duration=duration)
        output_path.write_bytes(b"video")
        return output_path

    monkeypatch.setattr("krvoiceai.modules.video_relay_client.VideoRelayClient.generate_clip", fake_generate)
    output = tmp_path / "clip.mp4"
    assert SeedanceClient().generate_clip("城市夜景", output) == output
    assert seen == {"model": "minimax-h3", "prompt": "城市夜景", "duration": 5}


def test_openai_video_relay_seedance_and_h3(tmp_path, monkeypatch):
    """两种模型共用同一 /v1/videos 协议，切换模型不改请求端点。"""
    monkeypatch.setenv("VIDEO_API_BASE", "https://relay.example/v1")
    monkeypatch.setenv("VIDEO_API_KEY", "relay-key")
    monkeypatch.setenv("VIDEO_MODEL", "seedance-2.0-mini")
    from krvoiceai.modules.video_relay_client import VideoRelayClient
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer relay-key"
        if request.method == "POST":
            assert request.url.path == "/v1/videos"
            return httpx.Response(200, json={
                "id": "task-1", "status": "queued", "object": "video",
                "created_at": 1, "progress": 0,
            })
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"FAKE_MP4_BYTES" * 100)
        assert request.url.path == "/v1/videos/task-1"
        return httpx.Response(200, json={
            "id": "task-1", "status": "completed", "object": "video",
            "created_at": 1, "progress": 100,
            "video_url": "https://relay.example/video.mp4",
        })

    monkeypatch.setattr(
        "krvoiceai.modules.video_relay_client.OpenAI",
        lambda **kwargs: OpenAI(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs))
    image = tmp_path / "人物.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    for model in ("seedance-2.0-mini", "minimax-h3"):
        monkeypatch.setenv("VIDEO_MODEL", model)
        client = VideoRelayClient()
        client.poll_interval = 0
        client.generate_clip("人物走动，中文招牌", tmp_path / f"{model}.mp4",
                             reference_image=image)
        submitted = [request for request in calls if request.method == "POST"][-1]
        assert submitted.headers["Idempotency-Key"]
        assert submitted.headers["Content-Type"].startswith("application/json")
        payload = json.loads(submitted.content)
        assert payload["model"] == model
        assert payload["images"][0].startswith("data:image/jpeg;base64,")
        assert "@Image1" in payload["prompt"]
    assert (tmp_path / "minimax-h3.mp4").stat().st_size > 0


def test_video_relay_timeout_resumes_without_reposting(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_API_BASE", "https://relay.example/v1")
    monkeypatch.setenv("VIDEO_API_KEY", "relay-key")
    monkeypatch.setenv("VIDEO_MODEL", "minimax-h3")
    from krvoiceai.modules.video_relay_client import VideoRelayClient, VideoRelayError
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={
                "id": "task-1", "status": "queued", "object": "video",
                "created_at": 1, "progress": 0,
            })
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"FAKE_MP4_BYTES")
        return httpx.Response(200, json={
            "id": "task-1", "status": "completed", "object": "video",
            "created_at": 1, "progress": 100,
        })

    monkeypatch.setattr(
        "krvoiceai.modules.video_relay_client.OpenAI",
        lambda **kwargs: OpenAI(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs))
    output = tmp_path / "clip.mp4"
    first = VideoRelayClient()
    first.poll_timeout = 0
    with pytest.raises(VideoRelayError, match="超时"):
        first.generate_clip("中文空镜", output)
    state = json.loads((tmp_path / "clip.mp4.task.json").read_text(encoding="utf-8"))
    assert state["task_id"] == "task-1"
    with pytest.raises(VideoRelayError, match="其他视频任务"):
        VideoRelayClient().generate_clip("不同提示词", output)

    second = VideoRelayClient()
    second.poll_interval = 0
    assert second.generate_clip("中文空镜", output) == output
    assert output.read_bytes() == b"FAKE_MP4_BYTES"
    assert len([r for r in calls if r.method == "POST"]) == 1


def test_video_relay_download_retry_uses_existing_task(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_API_BASE", "https://relay.example/v1")
    monkeypatch.setenv("VIDEO_API_KEY", "relay-key")
    monkeypatch.setenv("VIDEO_MODEL", "minimax-h3")
    from krvoiceai.modules.video_relay_client import VideoRelayClient, VideoRelayError
    calls = {"post": 0, "download": 0}

    def handler(request):
        if request.method == "POST":
            calls["post"] += 1
            return httpx.Response(200, json={
                "id": "task-1", "status": "queued", "object": "video",
                "created_at": 1, "progress": 0,
            })
        if request.url.path.endswith("/content"):
            calls["download"] += 1
            if calls["download"] == 1:
                return httpx.Response(503, json={"error": {"message": "busy"}})
            return httpx.Response(200, content=b"FAKE_MP4_BYTES")
        return httpx.Response(200, json={
            "id": "task-1", "status": "completed", "object": "video",
            "created_at": 1, "progress": 100,
        })

    monkeypatch.setattr(
        "krvoiceai.modules.video_relay_client.OpenAI",
        lambda **kwargs: OpenAI(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs))
    output = tmp_path / "clip.mp4"
    first = VideoRelayClient()
    first.poll_interval = 0
    with pytest.raises(VideoRelayError, match="下载失败"):
        first.generate_clip("中文空镜", output)
    assert not output.exists()
    assert not (tmp_path / "clip.mp4.part").exists()
    second = VideoRelayClient()
    second.poll_interval = 0
    second.generate_clip("中文空镜", output)
    assert calls == {"post": 1, "download": 2}


def test_video_relay_recovers_ambiguous_create_without_reposting(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_API_BASE", "https://relay.example/v1")
    monkeypatch.setenv("VIDEO_API_KEY", "relay-key")
    monkeypatch.setenv("VIDEO_MODEL", "seedance-2.0-mini")
    from krvoiceai.modules.video_relay_client import VideoRelayClient, VideoRelayError
    keys = []

    def handler(request):
        if request.method == "POST":
            keys.append(request.headers["Idempotency-Key"])
            return httpx.Response(503, json={"error": {"message": "retry"}})
        if "/task-requests/" in request.url.path:
            assert request.url.path.endswith(keys[0])
            return httpx.Response(200, json={"id": "task-2", "status": "queued"})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"FAKE_MP4_BYTES")
        return httpx.Response(200, json={
            "id": "task-2", "status": "completed", "object": "video",
            "created_at": 1, "progress": 100,
        })

    monkeypatch.setattr(
        "krvoiceai.modules.video_relay_client.OpenAI",
        lambda **kwargs: OpenAI(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs))
    output = tmp_path / "clip.mp4"
    with pytest.raises(VideoRelayError, match="创建失败"):
        VideoRelayClient().generate_clip("中文空镜", output)
    second = VideoRelayClient()
    second.poll_interval = 0
    second.generate_clip("中文空镜", output)
    assert len(keys) == 1
    state = json.loads((tmp_path / "clip.mp4.task.json").read_text(encoding="utf-8"))
    assert state["request_id"] == keys[0]
    assert state["task_id"] == "task-2"


def test_video_relay_unknown_create_result_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_API_BASE", "https://relay.example/v1")
    monkeypatch.setenv("VIDEO_API_KEY", "relay-key")
    monkeypatch.setenv("VIDEO_MODEL", "minimax-h3")
    from krvoiceai.modules.video_relay_client import VideoRelayClient, VideoRelayError
    posts = []

    def handler(request):
        if request.method == "POST":
            posts.append(request)
            return httpx.Response(503, json={"error": {"message": "unknown"}})
        return httpx.Response(404, json={"error": {"message": "not found"}})

    monkeypatch.setattr(
        "krvoiceai.modules.video_relay_client.OpenAI",
        lambda **kwargs: OpenAI(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs))
    output = tmp_path / "clip.mp4"
    with pytest.raises(VideoRelayError, match="创建失败"):
        VideoRelayClient().generate_clip("中文空镜", output)
    with pytest.raises(VideoRelayError, match="停止重新提交"):
        VideoRelayClient().generate_clip("中文空镜", output)
    assert len(posts) == 1


def test_broll_retry_reuses_original_paid_plan(tmp_path, monkeypatch):
    from krvoiceai.core.base_module import JobContext
    from krvoiceai.core import llm_client
    from krvoiceai.modules import seedance_client
    from krvoiceai.modules.broll_engine import BRollEngine

    calls = {"llm": 0, "video": 0}

    class FakeLLM:
        is_mock = False

        def chat(self, **_kwargs):
            calls["llm"] += 1
            return '[{"start": 8, "scene_prompt": "中文场景", "reason": "演示"}]'

    class FakeSeedance:
        model = "minimax-h3"

        def __init__(self, *_args):
            pass

        def available(self):
            return True, "ok"

        def generate_clip(self, _prompt, output_path, **_kwargs):
            calls["video"] += 1
            if calls["video"] == 1:
                raise RuntimeError("poll timeout")
            output_path.write_bytes(b"completed video")

    monkeypatch.setattr(llm_client, "LLMClient", FakeLLM)
    monkeypatch.setattr(seedance_client, "SeedanceClient", FakeSeedance)
    engine = BRollEngine.__new__(BRollEngine)
    engine.config = SimpleNamespace(get=lambda _key, default=None: default)
    engine.logger = SimpleNamespace(info=lambda _msg: None, warning=lambda _msg: None)
    engine.output_resolution = (1080, 1920)
    ctx = JobContext(work_dir=tmp_path, audio_duration=30, script_text="一段中文文案")
    assert engine._auto_generate_clips(ctx) == []
    assert (tmp_path / "broll_auto_plan.json").exists()
    clips = engine._auto_generate_clips(ctx)
    assert len(clips) == 1
    assert calls == {"llm": 1, "video": 2}


def test_seedance_client_routes_to_video_relay(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_API_BASE", "https://relay.example/v1")
    monkeypatch.setenv("VIDEO_API_KEY", "key")
    monkeypatch.setenv("VIDEO_MODEL", "minimax-h3")
    client = SeedanceClient()
    assert client.available()[0]
    from krvoiceai.modules.video_relay_client import VideoRelayClient
    called = {}
    monkeypatch.setattr(
        VideoRelayClient, "generate_clip",
        lambda self, *args: called.setdefault("model", self.model) or tmp_path / "x.mp4")
    client.generate_clip("中文提示词", tmp_path / "x.mp4")
    assert called["model"] == "minimax-h3"


def test_video_model_list_uses_openai_sdk(monkeypatch):
    from fastapi.testclient import TestClient
    from krvoiceai.web.server import create_app

    monkeypatch.setenv("VIDEO_API_BASE", "https://relay.example/v1")
    monkeypatch.setenv("VIDEO_API_KEY", "relay-key")
    seen = {}

    class FakeSDK:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[
                SimpleNamespace(id="vendor-video-a", type="video"),
                SimpleNamespace(id="vendor-video-b", category="text-to-video"),
                SimpleNamespace(id="text-only", type="text"),
            ]))

    monkeypatch.setattr("openai.OpenAI", FakeSDK)
    response = TestClient(create_app()).get("/api/settings/video-models")
    assert response.status_code == 200
    assert response.json()["models"] == ["vendor-video-a", "vendor-video-b"]
    assert response.json()["unclassified"] is False
    assert seen["base_url"] == "https://relay.example/v1"
    assert seen["api_key"] == "relay-key"
    assert seen["max_retries"] == 0


def test_video_model_preview_uses_unsaved_credentials_without_leaking_saved_key(monkeypatch):
    from fastapi.testclient import TestClient
    from krvoiceai.web.server import create_app

    monkeypatch.setenv("VIDEO_API_BASE", "https://saved.example/v1")
    monkeypatch.setenv("VIDEO_API_KEY", "saved-secret")
    seen = {}

    class FakeSDK:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[
                SimpleNamespace(id="new-provider-video"),
            ]))

    monkeypatch.setattr("openai.OpenAI", FakeSDK)
    client = TestClient(create_app())
    missing_key = client.post("/api/settings/video-models", json={
        "base_url": "https://other.example/v1",
    })
    assert missing_key.json()["models"] == []
    assert seen == {}
    preview = client.post("/api/settings/video-models", json={
        "base_url": "https://other.example/v1", "api_key": "new-secret",
    })
    assert preview.json()["models"] == ["new-provider-video"]
    assert preview.json()["unclassified"] is True
    assert seen["base_url"] == "https://other.example/v1"
    assert seen["api_key"] == "new-secret"
    assert client.get("/api/settings/cloud-keys").json()["VIDEO_API_BASE"] == "https://saved.example/v1"


def test_omnihuman_settings_live_only_in_avatar_page():
    html = (Path(__file__).resolve().parents[1] / "krvoiceai" / "web" / "static" /
            "index.html").read_text(encoding="utf-8")
    cloud = html.split('id="subpage-cloud"', 1)[1].split('id="subpage-llm"', 1)[0]
    avatar = html.split('id="subpage-avatar"', 1)[1].split('id="page-settings-video"', 1)[0]
    assert 'id="cloud-omni-base"' not in cloud
    assert 'id="cloud-omni-base"' in avatar
    assert 'id="cloud-omni-model"' in avatar
    assert "seedance-2.0-mini" not in cloud


def test_ui_script_urls_match_current_content():
    import hashlib
    from fastapi.testclient import TestClient
    from krvoiceai.web.server import create_app

    client = TestClient(create_app())
    page = client.get("/")
    assert page.status_code == 200
    static = Path(__file__).resolve().parents[1] / "krvoiceai" / "web" / "static"
    for name in ("app.js", "settings.js"):
        digest = hashlib.sha256((static / "js" / name).read_bytes()).hexdigest()[:16]
        url = f"/static/js/{name}?v={digest}"
        assert url in page.text
        asset = client.get(url)
        assert asset.status_code == 200
        assert "no-cache" in asset.headers["cache-control"]
    assert "async function refreshVideoModels(" in client.get("/static/js/settings.js").text


def test_cloud_relay_settings_can_be_cleared(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from krvoiceai.core import config as config_module
    from krvoiceai.web.server import create_app

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ARK_API_KEY", "official-key")
    client = TestClient(create_app())
    saved = client.post("/api/settings/cloud-keys", json={
        "video_api_base": "https://relay.example/v1",
        "video_api_key": "relay-key",
        "video_model": "minimax-h3",
    })
    assert saved.json()["success"] is True
    cleared = client.post("/api/settings/cloud-keys", json={
        "clear": ["VIDEO_API_BASE", "VIDEO_API_KEY", "VIDEO_MODEL"],
    })
    assert cleared.json()["cleared"] == ["VIDEO_API_BASE", "VIDEO_API_KEY", "VIDEO_MODEL"]
    assert client.get("/api/settings/cloud-keys").json()["VIDEO_API_BASE"] == ""
    assert "VIDEO_API_BASE=" not in (tmp_path / ".env").read_text(encoding="utf-8")
    assert "ARK_API_KEY" not in client.get("/api/settings/cloud-keys").json()
    rejected = client.post("/api/settings/cloud-keys", json={"ark_api_key": "official-key"})
    assert rejected.json()["success"] is False


def test_avatar_generation_retry_uses_stable_task_path(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from krvoiceai.modules import seedance_client
    from krvoiceai.web import server

    paths = []

    class FakeSeedance:
        def __init__(self, *_args):
            pass

        def generate_clip(self, _prompt, output_path, **_kwargs):
            paths.append(output_path)
            state_path = output_path.with_name(output_path.name + ".task.json")
            if len(paths) == 1:
                state_path.write_text('{"task_id":"paid-task"}', encoding="utf-8")
                raise RuntimeError("poll timeout")
            assert state_path.exists()
            output_path.write_bytes(b"completed video")

    monkeypatch.setattr(seedance_client, "SeedanceClient", FakeSeedance)
    fake_app = SimpleNamespace(
        config=SimpleNamespace(get=lambda _key, default=None: str(tmp_path / "work")),
        register_avatar=lambda _avatar_id, _clip: True,
    )
    monkeypatch.setattr(server, "_get_app", lambda: fake_app)
    client = TestClient(server.create_app())
    for expected in (False, True):
        response = client.post(
            "/api/avatars/generate",
            data={"avatar_id": "person", "prompt": "中文角色"},
            files={"file": ("face.jpg", b"photo", "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.json()["success"] is expected
    assert paths[0] == paths[1]
    assert not paths[1].with_name(paths[1].name + ".task.json").exists()


# ============ OmniHuman 客户端 ============

def test_omnihuman_requires_relay_even_with_official_keys(monkeypatch):
    monkeypatch.delenv("OMNIHUMAN_API_BASE", raising=False)
    monkeypatch.delenv("OMNIHUMAN_API_KEY", raising=False)
    monkeypatch.setenv("VOLC_ACCESS_KEY", "unused-official-ak")
    monkeypatch.setenv("VOLC_SECRET_KEY", "unused-official-sk")
    ok, reason = OmniHumanClient().available()
    assert not ok and "OMNIHUMAN_API_BASE" in reason


def test_omnihuman_model_selects_req_key(monkeypatch):
    monkeypatch.setenv("OMNIHUMAN_API_BASE", "https://omni-relay.example")
    monkeypatch.setenv("OMNIHUMAN_API_KEY", "omni-key")
    monkeypatch.setenv("OMNIHUMAN_MODEL", "v1")
    client = OmniHumanClient()
    assert client.available()[0]
    assert client.version == "v1"
    monkeypatch.setenv("OMNIHUMAN_MODEL", "unknown")
    assert not OmniHumanClient().available()[0]


def test_omnihuman_relay_upload_and_bearer(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIHUMAN_API_BASE", "https://omni-relay.example")
    monkeypatch.setenv("OMNIHUMAN_API_KEY", "omni-key")
    monkeypatch.delenv("VOLC_ACCESS_KEY", raising=False)
    monkeypatch.delenv("VOLC_SECRET_KEY", raising=False)
    client = OmniHumanClient()
    assert client.available()[0]
    calls = []

    def fake_post(url, headers=None, files=None, json=None, timeout=None):
        calls.append((url, headers, files, json))
        if files is not None:
            return _FakeResp(200, {"url": "https://media.example/image.jpg"})
        return _FakeResp(200, {"code": 10000, "data": {"task_id": "t1"}})

    monkeypatch.setattr("krvoiceai.modules.omnihuman_client.requests.post", fake_post)
    image = tmp_path / "人像.jpg"
    image.write_bytes(b"image")
    assert client._ensure_url(image) == "https://media.example/image.jpg"
    result = client._signed_request("CVSubmitTask", {"req_key": "test"})
    assert result["data"]["task_id"] == "t1"
    assert calls[0][0] == "https://omni-relay.example/uploads"
    assert calls[1][0] == "https://omni-relay.example/?Action=CVSubmitTask&Version=2022-08-31"
    assert all(call[1]["Authorization"] == "Bearer omni-key" for call in calls)


def test_omnihuman_api_error_raises(monkeypatch):
    monkeypatch.setenv("OMNIHUMAN_API_BASE", "https://omni-relay.example")
    monkeypatch.setenv("OMNIHUMAN_API_KEY", "omni-key")
    c = OmniHumanClient(None, None)
    monkeypatch.setattr(
        "krvoiceai.modules.omnihuman_client.requests.post",
        lambda *a, **k: _FakeResp(200, {"code": 50411, "message": "审核不通过"}))
    with pytest.raises(OmniHumanError, match="50411"):
        c._signed_request("CVSubmitTask", {"req_key": "x"})


def test_omnihuman_generate_flow(tmp_path, monkeypatch):
    """主体识别 → 视频生成 → 下载 全流程（mock 网络层）"""
    monkeypatch.setenv("OMNIHUMAN_API_BASE", "https://omni-relay.example")
    monkeypatch.setenv("OMNIHUMAN_API_KEY", "omni-key")
    c = OmniHumanClient(None, None)
    c.poll_interval = 0.01
    c.poll_timeout = 5

    calls = []

    def fake_signed(action, body):
        calls.append((action, body.get("req_key")))
        if action == "CVSubmitTask":
            return {"code": 10000, "data": {"task_id": f"task{len(calls)}"}}
        # CVGetResult
        if "create_role" in body["req_key"]:
            return {"code": 10000, "data": {
                "status": "done",
                "resp_data": json.dumps({"status": 1}),
            }}
        return {"code": 10000, "data": {
            "status": "done",
            "video_url": "https://fake/v.mp4",
        }}

    monkeypatch.setattr(c, "_signed_request", fake_signed)
    monkeypatch.setattr(c, "_ensure_url", lambda p: f"https://fake.tos/{Path(str(p)).name}")
    monkeypatch.setattr(
        "krvoiceai.modules.omnihuman_client.requests.get",
        lambda *a, **k: _FakeResp(200))

    out = tmp_path / "omni.mp4"
    started = []
    c.generate(Path("face.jpg"), Path("voice.wav"), out,
               on_video_submit_start=lambda: started.append(True))
    assert out.exists() and out.stat().st_size > 0
    assert started == [True]
    # 默认 v15：先 create_role 再 video
    req_keys = [rk for _, rk in calls if rk]
    assert any("create_role_omni_v15" in rk for rk in req_keys)
    assert any(rk.endswith("omni_v15") and "create_role" not in rk for rk in req_keys)


def test_omnihuman_resume_task_skips_submit(tmp_path, monkeypatch):
    """续轮询模式：不做主体识别、不重新提交（防 50430 撞并发 + 双倍计费），
    直接轮询给定 task_id 并下载；on_video_task 回调收到复用的 id。"""
    monkeypatch.setenv("OMNIHUMAN_API_BASE", "https://omni-relay.example")
    monkeypatch.setenv("OMNIHUMAN_API_KEY", "omni-key")
    c = OmniHumanClient(None, None)
    c.poll_interval = 0.01
    c.poll_timeout = 5

    calls = []

    def fake_signed(action, body):
        calls.append((action, body.get("task_id")))
        assert action == "CVGetResult", "续轮询模式不允许出现提交调用"
        assert body["task_id"] == "orphan-task-1"
        return {"code": 10000, "data": {
            "status": "done",
            "video_url": "https://fake/v.mp4",
        }}

    monkeypatch.setattr(c, "_signed_request", fake_signed)
    monkeypatch.setattr(
        c, "_ensure_url",
        lambda p: (_ for _ in ()).throw(AssertionError("续轮询不应上传素材")))
    monkeypatch.setattr(
        "krvoiceai.modules.omnihuman_client.requests.get",
        lambda *a, **k: _FakeResp(200))

    seen = []
    out = tmp_path / "omni_resume.mp4"
    c.generate(Path("face.jpg"), Path("voice.wav"), out,
               resume_task_id="orphan-task-1", on_video_task=seen.append)
    assert out.exists() and out.stat().st_size > 0
    assert seen == ["orphan-task-1"]
    assert all(a == "CVGetResult" for a, _ in calls)


def test_omnihuman_task_survives_new_context(tmp_path, monkeypatch):
    from krvoiceai.modules.avatar_engine import AvatarEngine
    from krvoiceai.core.base_module import JobContext
    from krvoiceai.modules import omnihuman_client

    avatar_dir = tmp_path / "avatars" / "person"
    avatar_dir.mkdir(parents=True)
    (avatar_dir / "reference.jpg").write_bytes(b"photo")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"audio")
    work_dir = tmp_path / "job"
    work_dir.mkdir()
    output = work_dir / "avatar_output.mp4"
    engine = AvatarEngine.__new__(AvatarEngine)
    engine.avatars_dir = tmp_path / "avatars"
    engine.config = SimpleNamespace(get=lambda _key, default=None: default)
    engine.logger = SimpleNamespace(info=lambda _msg: None)
    engine._to_portrait_or_original = lambda path, _ctx: path
    resume_ids = []
    versions = []

    class FakeClient:
        default_version = "v15"

        def __init__(self, *_args):
            self.version = self.default_version

        def generate(self, _image, _audio, output_path, resume_task_id=None,
                     on_video_task=None, on_video_submit_start=None):
            resume_ids.append(resume_task_id)
            versions.append(self.version)
            if resume_task_id is None:
                on_video_submit_start()
                on_video_task("paid-task-1")
                raise RuntimeError("poll timeout")
            output_path.write_bytes(b"completed video")
            return output_path

    monkeypatch.setattr(omnihuman_client, "OmniHumanClient", FakeClient)
    first_ctx = JobContext(work_dir=work_dir, audio_path=audio)
    with pytest.raises(RuntimeError, match="poll timeout"):
        engine._generate_omnihuman(first_ctx, "person", output)
    assert (work_dir / "omnihuman_task_id.txt").read_text() == "paid-task-1"

    FakeClient.default_version = "v1"
    second_ctx = JobContext(work_dir=work_dir, audio_path=audio)
    assert engine._generate_omnihuman(second_ctx, "person", output) == output
    assert resume_ids == [None, "paid-task-1"]
    assert versions == ["v15", "v15"]
    assert not (work_dir / "omnihuman_task_id.txt").exists()
    assert not (work_dir / "omnihuman_submitting.txt").exists()
    assert not (work_dir / "omnihuman_task_model.txt").exists()


def test_omnihuman_unknown_submit_result_never_resubmits(tmp_path, monkeypatch):
    from krvoiceai.modules.avatar_engine import AvatarEngine
    from krvoiceai.core.base_module import JobContext
    from krvoiceai.modules import omnihuman_client

    avatar_dir = tmp_path / "avatars" / "person"
    avatar_dir.mkdir(parents=True)
    (avatar_dir / "reference.jpg").write_bytes(b"photo")
    work_dir = tmp_path / "job"
    work_dir.mkdir()
    (work_dir / "omnihuman_submitting.txt").write_text("submitting", encoding="utf-8")
    engine = AvatarEngine.__new__(AvatarEngine)
    engine.avatars_dir = tmp_path / "avatars"
    engine.config = SimpleNamespace(get=lambda _key, default=None: default)
    engine.logger = SimpleNamespace(info=lambda _msg: None)

    class FakeClient:
        def __init__(self, *_args):
            pass

        def generate(self, *_args, **_kwargs):
            raise AssertionError("结果不明时不允许重新提交")

    monkeypatch.setattr(omnihuman_client, "OmniHumanClient", FakeClient)
    ctx = JobContext(work_dir=work_dir, audio_path=tmp_path / "voice.wav")
    with pytest.raises(RuntimeError, match="提交结果不明"):
        engine._generate_omnihuman(ctx, "person", work_dir / "avatar_output.mp4")


def test_omnihuman_rejects_photo_without_person(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIHUMAN_API_BASE", "https://omni-relay.example")
    monkeypatch.setenv("OMNIHUMAN_API_KEY", "omni-key")
    c = OmniHumanClient(None, None)
    c.poll_interval = 0.01
    c.poll_timeout = 5

    def fake_signed(action, body):
        if action == "CVSubmitTask":
            return {"code": 10000, "data": {"task_id": "t"}}
        return {"code": 10000, "data": {
            "status": "done", "resp_data": json.dumps({"status": 0}),
        }}

    monkeypatch.setattr(c, "_signed_request", fake_signed)
    monkeypatch.setattr(c, "_ensure_url", lambda p: "https://fake/x")
    with pytest.raises(OmniHumanError, match="人物主体"):
        c.generate(Path("cat.jpg"), Path("a.wav"), tmp_path / "o.mp4")


def test_omnihuman_missing_local_file(monkeypatch):
    monkeypatch.setenv("OMNIHUMAN_API_BASE", "https://omni-relay.example")
    monkeypatch.setenv("OMNIHUMAN_API_KEY", "omni-key")
    client = OmniHumanClient()
    with pytest.raises(OmniHumanError, match="素材不存在"):
        client._ensure_url(Path("local_face.jpg"))
    assert client._ensure_url("https://example.com/a.jpg") == "https://example.com/a.jpg"
