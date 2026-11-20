from types import SimpleNamespace

import httpx
import pytest

from krvoiceai.modules.musetalk_api_client import MuseTalkAPIClient


def make_client(tmp_path, monkeypatch, handler):
    real_client = httpx.Client
    monkeypatch.setattr(
        "krvoiceai.modules.musetalk_api_client.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    values = {
        "avatar.api_base": "https://gpu.example/musetalk",
        "avatar.api_key": "proxy-key",
        "avatar.timeout": 1800,
    }
    source = tmp_path / "人物.jpg"
    source.write_bytes(b"source-image")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"spoken-audio")
    return MuseTalkAPIClient(SimpleNamespace(get=values.get)), source, audio


def test_upload_and_download_with_proxy_prefix(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer proxy-key"
        if request.method == "POST":
            assert request.url.path == "/musetalk/generate"
            assert b'name="source"' in request.content
            assert b'name="audio"' in request.content
            assert b"source-image" in request.content
            assert b"spoken-audio" in request.content
            assert b"freevideoai_" in request.content
            return httpx.Response(200, json={
                "status": "success", "download_url": "/download/result.mp4",
            })
        assert request.url.path == "/musetalk/download/result.mp4"
        return httpx.Response(200, content=b"generated-video")

    client, source, audio = make_client(tmp_path, monkeypatch, handler)
    output = tmp_path / "avatar.mp4"
    assert client.generate(source, audio, output) == output
    assert output.read_bytes() == b"generated-video"
    client.generate(source, audio, output)
    assert len(calls) == 2


def test_unknown_submit_result_does_not_repeat_inference(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("generation response lost", request=request)

    client, source, audio = make_client(tmp_path, monkeypatch, handler)
    output = tmp_path / "avatar.mp4"
    with pytest.raises(httpx.ReadTimeout):
        client.generate(source, audio, output)
    with pytest.raises(RuntimeError, match="停止自动重提"):
        client.generate(source, audio, output)
    assert len(calls) == 1


def test_failed_download_resumes_without_upload(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(200, json={
                "status": "success", "download_url": "/download/result.mp4",
            })
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, content=b"downloaded")

    client, source, audio = make_client(tmp_path, monkeypatch, handler)
    output = tmp_path / "avatar.mp4"
    with pytest.raises(httpx.HTTPStatusError):
        client.generate(source, audio, output)
    client.generate(source, audio, output)
    assert calls == ["POST", "GET", "GET"]
    assert output.read_bytes() == b"downloaded"


def test_health_requires_ready_models_and_gpu(monkeypatch):
    monkeypatch.setattr("krvoiceai.modules.musetalk_api_client.httpx.get", lambda *a, **kw:
        httpx.Response(200, json={"status": "healthy", "models_loaded": False},
                       request=httpx.Request("GET", a[0])))
    client = MuseTalkAPIClient(SimpleNamespace(get=lambda k, d=None:
        "https://gpu.example" if k == "avatar.api_base" else d))
    with pytest.raises(RuntimeError, match="尚未加载"):
        client.health()
