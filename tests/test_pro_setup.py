"""Model downloads configure their actual paths; startup reports effective settings."""
import pytest
import yaml

from krvoiceai.core import config as config_module
from scripts import download_pro_models as downloader


@pytest.fixture
def setup_root(tmp_path, monkeypatch):
    from krvoiceai.core import startup

    monkeypatch.setattr(downloader, "ROOT", tmp_path)
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(startup, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_module, "USER_CONFIG_PATH", tmp_path / "config/user_config.yaml")
    monkeypatch.setattr(config_module.Config, "load", lambda: config_module.Config({}))
    calls = []
    monkeypatch.setattr(downloader, "download_model", lambda name, path: calls.append((name, path)))
    return tmp_path, calls


def test_download_configures_all_models_and_preserves_credentials_and_assets(setup_root):
    root, calls = setup_root
    destination = root / "config/user_config.yaml"
    destination.parent.mkdir()
    existing = {
        "llm": {"api_key": "private-key", "base_url": "https://proxy.example/v1"},
        "tts": {"provider": "mimo", "voices_dir": "/my/voices"},
        "avatar": {"provider": "latentsync", "avatars_dir": "/my/avatars"},
    }
    destination.write_text(yaml.safe_dump(existing), encoding="utf-8")
    downloader.main(["--runtime", "custom-runtime"])
    saved = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert len(calls) == 3
    assert all(path.is_absolute() for _, path in calls)
    assert saved["llm"] == {**existing["llm"], "provider": "deepseek", "model": "deepseek-chat"}
    assert saved["tts"]["provider"] == "voxcpm"
    assert saved["tts"]["voices_dir"] == "/my/voices"
    assert saved["tts"]["voxcpm"]["model_dir"] == (
        root / "custom-runtime/models/VoxCPM-0.5B").as_posix()
    assert saved["avatar"]["musetalk_local"]["repo_dir"] == (
        root / "custom-runtime/MuseTalk").as_posix()
    assert saved["avatar"]["musetalk_local"]["version"] == "v15"
    assert saved["avatar"]["avatars_dir"] == "/my/avatars"
    assert saved["asr"]["whisper"]["model_dir"] == (
        root / "custom-runtime/models/faster-whisper-small").as_posix()
    assert saved["asr"]["whisper"]["vad_filter"] is False
    assert destination.with_suffix(".yaml.bak").exists()


def test_single_model_download_configures_custom_destination_only(setup_root):
    root, _ = setup_root
    destination = root / "config/user_config.yaml"
    destination.parent.mkdir()
    existing = {"avatar": {"provider": "keep-avatar"}, "tts": {"provider": "keep-tts"}}
    destination.write_text(yaml.safe_dump(existing), encoding="utf-8")
    downloader.main(["--models", "whisper", "--destination", "my-models/whisper"])
    saved = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert saved["avatar"] == existing["avatar"]
    assert saved["tts"] == existing["tts"]
    assert saved["asr"]["provider"] == "whisper_local"
    assert saved["asr"]["whisper"]["model_dir"] == (root / "my-models/whisper").as_posix()


def test_musetalk_destination_configures_its_actual_repo(setup_root):
    root, _ = setup_root
    downloader.main(["--models", "musetalk", "--destination", "custom/MuseTalk/models"])
    saved = yaml.safe_load((root / "config/user_config.yaml").read_text(encoding="utf-8"))
    assert saved["avatar"]["musetalk_local"]["repo_dir"] == (root / "custom/MuseTalk").as_posix()


def test_download_failure_keeps_config_unchanged(setup_root, monkeypatch):
    root, _ = setup_root
    destination = root / "config/user_config.yaml"
    destination.parent.mkdir()
    original = b"tts:\n  provider: keep\n"
    destination.write_bytes(original)

    def fail(name, path):
        if name == "musetalk":
            raise RuntimeError("download failed")

    monkeypatch.setattr(downloader, "download_model", fail)
    with pytest.raises(RuntimeError, match="download failed"):
        downloader.main([])
    assert destination.read_bytes() == original
    assert not destination.with_suffix(".yaml.bak").exists()


def test_installer_can_download_without_activating_unverified_environment(setup_root):
    root, calls = setup_root
    downloader.main(["--models", "whisper", "--no-config"])
    assert len(calls) == 1
    assert not (root / "config/user_config.yaml").exists()


@pytest.mark.parametrize(("name", "value"), [
    ("KRVOICEAI_ASR_PROVIDER", "mimo"),
    ("KRVOICEAI_AVATAR_MUSETALK_LOCAL_VERSION", "v1"),
])
def test_conflicting_env_does_not_save_misleading_config(setup_root, monkeypatch, name, value):
    root, _ = setup_root
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        downloader.main([])
    assert not (root / "config/user_config.yaml").exists()


def test_musetalk_rejects_layout_that_worker_cannot_load(setup_root):
    root, calls = setup_root
    with pytest.raises(SystemExit):
        downloader.main(["--models", "musetalk", "--destination", "wrong-layout"])
    assert calls == []
    assert not (root / "config/user_config.yaml").exists()


def test_startup_reports_effective_paths_missing_assets_and_never_secrets(
        setup_root, monkeypatch, capsys):
    from krvoiceai.core import pro_models
    from krvoiceai.core.startup import print_startup_summary

    root, _ = setup_root
    monkeypatch.setattr(pro_models, "MANIFEST", {
        "voxcpm": [{"path": "model.bin", "size": 5}],
        "whisper": [{"path": "model.bin", "size": 5}],
        "musetalk": [{"path": "face-detection/s3fd.pth", "size": 5}],
    })
    for name in ("voxcpm", "whisper"):
        directory = root / name
        directory.mkdir()
        (directory / "model.bin").write_bytes(b"model")
    repo = root / "MuseTalk"
    (repo / "models/face-detection").mkdir(parents=True)
    (repo / "models/face-detection/s3fd.pth").write_bytes(b"model")
    cfg = config_module.Config({
        "llm": {"provider": "deepseek", "model": "deepseek-chat",
                "api_key": "do-not-print-key", "base_url": "https://secret-token@proxy.example"},
        "tts": {"provider": "voxcpm", "voxcpm": {
            "model_dir": "./voxcpm", "venv_python": "./missing-voxcpm-python"}},
        "avatar": {"provider": "musetalk_local", "musetalk_local": {
            "repo_dir": "./MuseTalk", "env_python": "./missing-musetalk-python", "version": "v15"}},
        "asr": {"provider": "whisper_local", "whisper": {
            "model_dir": "./whisper", "model_size": "small", "device": "cpu",
            "compute_type": "int8", "vad_filter": False}},
    })
    print_startup_summary(cfg)
    output = capsys.readouterr().out
    assert "DeepSeek Chat API" in output
    assert "ModelScope" in output
    assert "已设置" in output
    assert "do-not-print-key" not in output
    assert "secret-token" not in output
    assert str(root / "whisper") in output
    assert "S3FD" in output
    assert "missing-voxcpm-python" in output
    assert "missing-musetalk-python" in output
    assert "download_pro_models.py" in output
    assert "install.sh" in output
    assert "校验" in output


def test_startup_warns_about_provider_overrides_and_missing_key(setup_root, capsys):
    from krvoiceai.core.startup import print_startup_summary

    print_startup_summary(config_module.Config({
        "llm": {"provider": "deepseek", "api_key": ""},
        "avatar": {"provider": "latentsync"},
    }))
    output = capsys.readouterr().out
    assert "未设置" in output
    assert "KRVOICEAI_" in output
    assert "latentsync" in output
    assert "musetalk_local" in output


def test_startup_summary_is_wired_into_web_command(isolated_config, monkeypatch):
    from krvoiceai.ui import cli
    from krvoiceai.web import server

    events = []
    monkeypatch.setattr(cli, "FreeVideoAI", lambda: type("App", (), {"config": isolated_config})())
    monkeypatch.setattr(cli, "print_startup_summary", lambda cfg: events.append(("summary", cfg)))
    monkeypatch.setattr(server, "launch", lambda **kwargs: events.append(("launch", kwargs)))
    monkeypatch.setattr("sys.argv", ["freevideoai", "web", "--port", "9001"])
    assert cli.main() == 0
    assert events == [("summary", isolated_config), ("launch", {"host": "127.0.0.1", "port": 9001})]
