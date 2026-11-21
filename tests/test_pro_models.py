"""Pro downloads must be complete, repairable and offline at inference time."""
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def models(tmp_path, monkeypatch):
    from krvoiceai.core import pro_models

    content = b"verified model"
    item = {
        "repo": "owner/model", "revision": "fixed-revision",
        "source": "weights.bin", "path": "model.bin",
        "size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr(pro_models, "MANIFEST", {"whisper": [item]})
    calls = []

    def download(repo, source, *, revision, local_dir):
        calls.append((repo, source, revision))
        path = Path(local_dir) / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    monkeypatch.setitem(sys.modules, "modelscope.hub.file_download",
                        SimpleNamespace(model_file_download=download))
    return pro_models, tmp_path / "weights", calls, content


def test_modelscope_reuses_complete_files_and_repairs_same_size_corruption(models):
    module, dest, calls, content = models
    module.download_model("whisper", dest)
    assert (dest / "model.bin").read_bytes() == content
    module.download_model("whisper", dest)
    assert len(calls) == 1
    (dest / "model.bin").write_bytes(b"x" * len(content))
    module.download_model("whisper", dest)
    assert len(calls) == 2
    assert (dest / "model.bin").read_bytes() == content


def test_failed_download_keeps_existing_target_and_does_not_report_success(models, monkeypatch):
    module, dest, _, _ = models
    dest.mkdir(parents=True)
    (dest / "model.bin").write_bytes(b"old")

    def corrupt(*args, local_dir, **kwargs):
        path = Path(local_dir) / "weights.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"bad")
        return str(path)

    monkeypatch.setitem(sys.modules, "modelscope.hub.file_download",
                        SimpleNamespace(model_file_download=corrupt))
    with pytest.raises(RuntimeError, match="校验"):
        module.download_model("whisper", dest)
    assert (dest / "model.bin").read_bytes() == b"old"


def test_missing_local_weights_fail_without_starting_download(models):
    module, dest, calls, _ = models
    with pytest.raises(RuntimeError, match="本地"):
        module.require_model("whisper", dest)
    assert calls == []


def test_pro_plan_selects_only_supported_local_models(tmp_path):
    from krvoiceai.core import installation

    plan = installation.make_plan(tmp_path, "pro", [{"total_mb": 24576}],
                                  system="Linux", machine="x86_64")
    assert plan["models"] == ["voxcpm", "musetalk", "whisper"]
    assert plan["optional_models"] == []
    config = plan["config"]
    assert config["llm"]["model"] == "deepseek-chat"
    assert config["tts"]["provider"] == "voxcpm"
    assert config["tts"]["fallback_chain"] == ["voxcpm"]
    assert config["avatar"]["provider"] == "musetalk_local"
    assert config["asr"]["whisper"]["model_dir"].endswith("/faster-whisper-small")
    assert config["asr"]["whisper"]["vad_filter"] is False


def test_whisper_uses_only_local_files(tmp_path, monkeypatch):
    from krvoiceai.core import asr_utils

    seen = []
    monkeypatch.setattr(asr_utils, "_MODEL_CACHE", {})
    monkeypatch.setattr("krvoiceai.core.pro_models.require_model", lambda *args: None)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        WhisperModel=lambda model, **kwargs: seen.append((model, kwargs)) or object()))
    folder = tmp_path / "中文模型"
    asr_utils.get_whisper_model_from_cfg({
        "model_size": "small", "model_dir": str(folder), "device": "cpu",
        "compute_type": "int8",
    })
    assert seen[0][0] == str(folder)
    assert seen[0][1]["local_files_only"] is True


def test_musetalk_failure_preserves_root_cause_without_wav2lip(
        isolated_config, tmp_path, monkeypatch):
    from krvoiceai.core.base_module import JobContext
    from krvoiceai.modules.avatar_engine import AvatarEngine

    isolated_config.set("avatar.provider", "musetalk_local")
    engine = AvatarEngine(config=isolated_config)
    monkeypatch.setattr(engine, "_check_musetalk_env", lambda: (False, "CUDA dependency missing"))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    result = engine.run(JobContext(work_dir=tmp_path, audio_path=audio))
    assert not result.success
    assert "CUDA dependency missing" in result.error
    assert "Wav2Lip" not in result.error


def test_musetalk_subprocess_keeps_both_streams_and_original_error(
        isolated_config, tmp_path, monkeypatch):
    from krvoiceai.core.base_module import JobContext
    from krvoiceai.modules.avatar_engine import AvatarEngine

    engine = AvatarEngine(config=isolated_config)
    monkeypatch.setattr(engine, "_get_avatar_reference", lambda _: tmp_path / "face.jpg")
    engine.musetalk_config = {
        "env_python": str(tmp_path / "python"), "repo_dir": str(tmp_path),
        "worker_script": str(tmp_path / "worker.py"),
    }

    def fail(cmd, *, stdout, stderr, env, **kwargs):
        assert env["HF_HUB_OFFLINE"] == "1"
        assert stderr == -2  # subprocess.STDOUT
        stdout.write("Traceback:\n原始原因：CUDA kernel unavailable\n")
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr("krvoiceai.modules.avatar_engine.subprocess.run", fail)
    context = JobContext(work_dir=tmp_path, audio_path=tmp_path / "audio.wav")
    with pytest.raises(RuntimeError, match="CUDA kernel unavailable"):
        engine._generate_musetalk_local(context, "person", tmp_path / "out.mp4")
    assert "原始原因" in (tmp_path / "musetalk.log").read_text(encoding="utf-8")


def test_whisper_subtitle_does_not_switch_to_mock_when_package_is_missing(
        isolated_config, tmp_path, monkeypatch):
    from krvoiceai.core.base_module import JobContext
    from krvoiceai.modules.subtitle_engine import SubtitleEngine

    isolated_config.set("asr.provider", "whisper_local")
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    engine = SubtitleEngine(config=isolated_config)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    result = engine.execute(JobContext(work_dir=tmp_path, audio_path=audio))
    assert engine.provider == "whisper_local"
    assert not result.success
    assert "faster-whisper" in result.error


def test_reference_extraction_uses_same_local_whisper_runtime(
        isolated_config, tmp_path, monkeypatch):
    from krvoiceai.modules.script_extractor import ScriptExtractor

    isolated_config.set("asr.provider", "whisper_local")
    used = []

    def get_model(config):
        used.append(config)
        return SimpleNamespace(transcribe=lambda *a, **kw: (
            iter([SimpleNamespace(text="这是中文参考文案。")]), None))

    monkeypatch.setattr("krvoiceai.core.asr_utils.get_whisper_model_from_cfg", get_model)
    result = ScriptExtractor(config=isolated_config)._transcribe_local(tmp_path / "audio.wav")
    assert "中文参考文案" in result
    assert len(used) == 1


def test_reference_extraction_run_really_transcribes_for_whisper(
        isolated_config, tmp_path, monkeypatch):
    from krvoiceai.core.base_module import JobContext
    from krvoiceai.modules.script_extractor import ScriptExtractor

    isolated_config.set("asr.provider", "whisper_local")
    engine = ScriptExtractor(config=isolated_config)
    engine._ytdlp_available = True
    monkeypatch.setattr(engine, "_extract_real", lambda *a: "真实参考视频的口播稿。")
    result = engine.run(JobContext(work_dir=tmp_path, reference_video_url="https://video.example"))
    assert result.success
    assert result.data["mock"] is False
    assert "真实参考" in result.data["script_text"]


def test_reference_article_failure_keeps_original_error(isolated_config, monkeypatch):
    from krvoiceai.modules.script_extractor import ScriptExtractor

    isolated_config.set("asr.provider", "whisper_local")
    engine = ScriptExtractor(config=isolated_config)
    monkeypatch.setattr(engine, "_extract_url_from_text", lambda _: "https://article.example")
    monkeypatch.setattr(engine, "_is_video_url", lambda _: False)

    def fail(_):
        raise RuntimeError("文章抓取原始错误")

    monkeypatch.setattr(engine, "_extract_article", fail)
    with pytest.raises(RuntimeError, match="文章抓取原始错误"):
        engine.extract("https://article.example")


def test_missing_deepseek_key_is_an_error_instead_of_template(isolated_config):
    from krvoiceai.core.llm_client import LLMClient, LLMServiceError

    isolated_config.set("llm.provider", "deepseek")
    isolated_config.set("llm.api_key", "")
    with pytest.raises(LLMServiceError, match="Key"):
        LLMClient().chat([{"role": "user", "content": "写口播稿"}])


def test_pro_rejects_environment_that_switches_back_to_cloud(tmp_path, monkeypatch):
    from krvoiceai.core import installation

    plan = installation.make_plan(tmp_path, "pro", [{"total_mb": 24576}],
                                  system="Linux", machine="x86_64")
    monkeypatch.setenv("KRVOICEAI_AVATAR_PROVIDER", "latentsync")
    with pytest.raises(RuntimeError, match="KRVOICEAI_AVATAR_PROVIDER"):
        installation.validate_pro_environment(plan["config"])


def test_pro_activation_preserves_deepseek_endpoint_key_and_user_assets(tmp_path):
    import yaml

    from krvoiceai.core import installation

    path = tmp_path / "user_config.yaml"
    existing = {
        "llm": {"api_key": "private-key", "base_url": "https://proxy.example/v1"},
        "avatar": {"provider": "latentsync", "avatars_dir": "/my/avatars"},
    }
    path.write_text(yaml.safe_dump(existing), encoding="utf-8")
    installation.save_config(path, {
        "llm": {"provider": "deepseek", "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com/v1"},
        "avatar": {"provider": "musetalk_local"},
    }, activate=True)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["llm"]["api_key"] == "private-key"
    assert saved["llm"]["base_url"] == "https://proxy.example/v1"
    assert saved["llm"]["model"] == "deepseek-chat"
    assert saved["avatar"] == {"provider": "musetalk_local", "avatars_dir": "/my/avatars"}
    assert yaml.safe_load(path.with_suffix(".yaml.bak").read_text(encoding="utf-8")) == existing


def test_musetalk_checks_detector_source_path_to_prevent_implicit_download(tmp_path, monkeypatch):
    from krvoiceai.core import pro_models

    monkeypatch.setattr(pro_models, "MANIFEST", {"musetalk": [{
        "path": "face-detection/s3fd.pth", "size": 5,
    }]})
    repo = tmp_path / "MuseTalk"
    weight = repo / "models/face-detection/s3fd.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"model")
    with pytest.raises(RuntimeError, match="S3FD"):
        pro_models.require_musetalk_assets(repo)
