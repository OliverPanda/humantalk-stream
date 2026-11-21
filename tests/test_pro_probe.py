"""The probe must exercise real subprocesses and never turn missing checks into passes."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from krvoiceai.core.config import Config
from krvoiceai.core.installation import pro_config
from scripts import probe_pro


@pytest.fixture
def probe_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_pro, "ROOT", tmp_path)
    cfg = Config(pro_config(tmp_path))
    cfg.set("llm.api_key", "private-probe-key")
    for key in (
        "tts.voxcpm.venv_python", "tts.voxcpm.worker_script",
        "avatar.musetalk_local.env_python", "avatar.musetalk_local.worker_script",
    ):
        path = Path(cfg.get(key))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    repo = Path(cfg.get("avatar.musetalk_local.repo_dir"))
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "scripts/inference.py").touch()
    monkeypatch.setattr(probe_pro, "model_status",
                        lambda *a, **k: {"ready": True, "missing": []})
    monkeypatch.setattr(probe_pro, "require_musetalk_assets", lambda *a, **k: None)
    monkeypatch.setattr(probe_pro.shutil, "which", lambda name: name)
    monkeypatch.setattr(probe_pro, "query_gpus", lambda: [{"name": "test GPU", "total_mb": 24576}])
    return cfg, tmp_path / "probe-output"


def fake_process(argv, **kwargs):
    """Only simulate GPU/model stages; filesystem checks and orchestration remain real."""
    if "-show_entries" in argv:
        stdout = json.dumps({"streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                             "format": {"duration": "1.0"}})
    else:
        stdout = "loaded and computed"
    return subprocess.CompletedProcess(argv, 0, stdout=stdout)


def test_probe_runs_offline_model_subprocesses_and_labels_unverified_scope(probe_setup, monkeypatch):
    cfg, output = probe_setup
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if str(argv[0]) == "ffmpeg":
            Path(argv[-1]).write_bytes(b"video")
        if Path(argv[0]) == Path(cfg.get("tts.voxcpm.venv_python")):
            task = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
            Path(task["out_wav"]).write_bytes(b"audio")
        return fake_process(argv, **kwargs)

    monkeypatch.setattr(probe_pro.subprocess, "run", run)
    report = probe_pro.run_probe(cfg, output, timeout=17)
    assert report["ok"] is True
    checks = {item["id"]: item for item in report["checks"]}
    assert checks["voxcpm"]["status"] == "pass"
    assert checks["musetalk"]["status"] == "pass"
    assert checks["whisper"]["status"] == "pass"
    assert checks["ffmpeg"]["status"] == "pass"
    assert checks["deepseek_api"]["status"] == "skip"
    assert checks["avatar_video"]["status"] == "skip"
    assert report["api_verified"] is False
    assert report["avatar_verified"] is False
    commands = [argv for argv, _ in calls]
    assert any("verify_musetalk.py" in " ".join(map(str, argv)) for argv in commands)
    assert any(argv[0] == sys.executable and "get_whisper_model_from_cfg" in " ".join(argv)
               for argv in commands)
    for _, kwargs in calls:
        assert kwargs["timeout"] == 17
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    serialized = (output / "report.json").read_text(encoding="utf-8")
    assert "private-probe-key" not in serialized
    assert json.loads(serialized)["ok"] is True


def test_probe_keeps_collecting_after_timeout_and_redacts_child_logs(probe_setup, monkeypatch):
    cfg, output = probe_setup

    def run(argv, **kwargs):
        if "verify_musetalk.py" in " ".join(map(str, argv)):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"],
                                            output=b"private-probe-key: CUDA load stalled")
        if str(argv[0]) == "ffmpeg":
            Path(argv[-1]).write_bytes(b"video")
        if Path(argv[0]) == Path(cfg.get("tts.voxcpm.venv_python")):
            task = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
            Path(task["out_wav"]).write_bytes(b"audio")
        result = fake_process(argv, **kwargs)
        if "-show_entries" not in argv:
            result.stdout += "\nprivate-probe-key"
        return result

    monkeypatch.setattr(probe_pro.subprocess, "run", run)
    report = probe_pro.run_probe(cfg, output, timeout=1)
    checks = {item["id"]: item for item in report["checks"]}
    assert report["ok"] is False
    assert checks["musetalk"]["status"] == "fail"
    assert checks["whisper"]["status"] == "pass"
    assert checks["ffmpeg"]["status"] == "pass"
    text = "\n".join(path.read_text(encoding="utf-8") for path in output.glob("*.log"))
    assert "CUDA load stalled" in text
    assert "private-probe-key" not in text
    assert "private-probe-key" not in (output / "report.json").read_text(encoding="utf-8")


def test_missing_weights_fail_and_skip_load_instead_of_downloading(probe_setup, monkeypatch):
    cfg, output = probe_setup
    monkeypatch.setattr(probe_pro, "model_status",
                        lambda *a, **k: {"ready": False, "missing": ["weights.bin"]})
    def missing_assets(*args, **kwargs):
        raise RuntimeError("MuseTalk weights missing")
    monkeypatch.setattr(probe_pro, "require_musetalk_assets", missing_assets)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, stdout="tool failed")

    monkeypatch.setattr(probe_pro.subprocess, "run", run)
    report = probe_pro.run_probe(cfg, output)
    checks = {item["id"]: item for item in report["checks"]}
    assert report["ok"] is False
    assert checks["voxcpm_weights"]["status"] == "fail"
    assert checks["voxcpm"]["status"] == "skip"
    assert checks["musetalk"]["status"] == "skip"
    assert checks["whisper"]["status"] == "skip"
    assert not any("modelscope" in " ".join(map(str, argv)).lower() for argv in calls)
    assert not any(Path(argv[0]) == Path(cfg.get("tts.voxcpm.venv_python")) for argv in calls)


def test_configuration_and_missing_key_are_failures(probe_setup, monkeypatch):
    cfg, output = probe_setup
    cfg.set("llm.api_key", "")
    cfg.set("avatar.provider", "latentsync")
    monkeypatch.setattr(probe_pro.subprocess, "run",
                        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, stdout="no GPU"))
    report = probe_pro.run_probe(cfg, output)
    checks = {item["id"]: item for item in report["checks"]}
    assert checks["configuration"]["status"] == "fail"
    assert "avatar.provider" in checks["configuration"]["detail"]
    assert checks["deepseek_key"]["status"] == "fail"
    assert report["ok"] is False


def test_successful_exit_without_voxcpm_audio_is_failure(probe_setup, monkeypatch):
    cfg, output = probe_setup
    monkeypatch.setattr(probe_pro.subprocess, "run", fake_process)
    report = probe_pro.run_probe(cfg, output)
    checks = {item["id"]: item for item in report["checks"]}
    assert checks["voxcpm"]["status"] == "fail"
    assert report["ok"] is False


def test_probe_rejects_old_artifacts_in_output_directory(probe_setup):
    cfg, output = probe_setup
    output.mkdir()
    (output / "voxcpm.wav").write_bytes(b"old audio")
    with pytest.raises(ValueError, match="非空"):
        probe_pro.run_probe(cfg, output)
    assert (output / "voxcpm.wav").read_bytes() == b"old audio"


def test_ffmpeg_output_must_have_both_video_and_audio(probe_setup, monkeypatch):
    cfg, output = probe_setup

    def run(argv, **kwargs):
        if str(argv[0]) == "ffmpeg":
            Path(argv[-1]).write_bytes(b"video")
        result = fake_process(argv, **kwargs)
        if "-show_entries" in argv:
            result.stdout = json.dumps({"streams": [{"codec_type": "video"},
                                                   {"codec_type": "subtitle"}],
                                        "format": {"duration": "1"}})
        return result

    monkeypatch.setattr(probe_pro.subprocess, "run", run)
    report = probe_pro.run_probe(cfg, output)
    checks = {item["id"]: item for item in report["checks"]}
    assert checks["ffmpeg"]["status"] == "fail"


def test_insufficient_first_gpu_is_reported(probe_setup, monkeypatch):
    cfg, output = probe_setup
    monkeypatch.setattr(probe_pro, "query_gpus", lambda: [
        {"name": "small GPU", "total_mb": 2048}, {"name": "large GPU", "total_mb": 24576}])
    monkeypatch.setattr(probe_pro.subprocess, "run", fake_process)
    report = probe_pro.run_probe(cfg, output)
    checks = {item["id"]: item for item in report["checks"]}
    assert checks["gpu"]["status"] == "fail"
    assert checks["voxcpm"]["status"] == "skip"
    assert checks["musetalk"]["status"] == "skip"


def test_s3fd_alias_checksum_is_checked(tmp_path, monkeypatch):
    import hashlib

    from krvoiceai.core import pro_models

    repo = tmp_path / "MuseTalk"
    item = {"path": "face-detection/s3fd.pth", "size": 5,
            "sha256": hashlib.sha256(b"model").hexdigest()}
    monkeypatch.setattr(pro_models, "MANIFEST", {"musetalk": [item]})
    target = repo / "models" / item["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"model")
    alias = repo / "musetalk/utils/face_detection/detection/sfd/s3fd.pth"
    alias.parent.mkdir(parents=True)
    alias.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="S3FD"):
        pro_models.require_musetalk_assets(repo, checksum=True)


def test_cli_returns_failure_and_writes_report_without_mutating_config(
        probe_setup, monkeypatch):
    cfg, output = probe_setup
    monkeypatch.setattr(probe_pro.Config, "load", lambda **kwargs: cfg)
    monkeypatch.setattr(probe_pro.subprocess, "run",
                        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, stdout="no GPU"))
    before = cfg.as_dict()
    assert probe_pro.main(["--output", str(output)]) == 1
    assert cfg.as_dict() == before
    assert (output / "report.json").exists()


def test_optional_api_and_avatar_use_real_request_and_worker_contracts(probe_setup, monkeypatch):
    import httpx

    cfg, output = probe_setup
    face = output.parent / "reference.mp4"
    face.write_bytes(b"reference")
    requests = []

    def post(url, **kwargs):
        requests.append((url, kwargs))
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    def run(argv, **kwargs):
        if str(argv[0]) == "ffmpeg":
            Path(argv[-1]).write_bytes(b"video")
        if Path(argv[0]) == Path(cfg.get("tts.voxcpm.venv_python")):
            task = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
            Path(task["out_wav"]).write_bytes(b"audio")
        if "--outfile" in argv:
            assert argv[argv.index("--face") + 1] == str(face)
            Path(argv[argv.index("--outfile") + 1]).write_bytes(b"avatar video")
        return fake_process(argv, **kwargs)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(probe_pro.subprocess, "run", run)
    report = probe_pro.run_probe(cfg, output, check_api=True, face=face)
    assert report["ok"] is True
    assert report["api_verified"] is True
    assert report["avatar_verified"] is True
    assert requests[0][0].endswith("/chat/completions")
    assert requests[0][1]["json"]["model"] == "deepseek-chat"
    assert requests[0][1]["json"]["max_tokens"] == 8


def test_offline_guard_blocks_socket_connections():
    result = subprocess.run(
        [sys.executable, "-c",
         probe_pro.OFFLINE_CODE + "\nsocket.create_connection(('127.0.0.1', 1))"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, check=False,
    )
    assert result.returncode != 0
    assert "已禁止网络连接" in result.stderr
