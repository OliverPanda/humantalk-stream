"""Installation policy and failure recovery; no model downloads or GPU required."""
import json
import os
import sys
from types import SimpleNamespace

import pytest
import yaml

from krvoiceai.core import installation as ins
from krvoiceai.core.config import Config
from krvoiceai.core import hardware


@pytest.mark.parametrize(
    "mb,expected", [(0, "cloud"), (2048, "cloud"), (4096, "voice"),
                    (5120, "voice"), (6144, "standard"), (16384, "standard"),
                    (18432, "pro"), (24576, "pro")],
)
def test_recommendation_respects_model_memory(mb, expected):
    assert ins.recommend_profile(mb) == expected


def test_manual_profile_cannot_bypass_real_hardware(tmp_path):
    with pytest.raises(ValueError, match="显存"):
        ins.make_plan(tmp_path, "standard", [], system="Linux", machine="x86_64")


def test_cpu_plan_has_no_cuda_dependencies(tmp_path):
    plan = ins.make_plan(tmp_path, "auto", [], system="Linux", machine="x86_64")
    assert plan["models"] == []
    assert plan["profile"] == "cloud"
    assert plan["config"]["tts"]["provider"] == "edge_tts"
    assert plan["config"]["avatar"]["provider"] == "musetalk_api"
    assert "Scripts" not in json.dumps(plan)


def test_linux_pro_has_three_fixed_local_models(tmp_path):
    gpus = [{"name": "RTX 4090", "total_mb": 24576, "free_mb": 20000}]
    plan = ins.make_plan(tmp_path, "auto", gpus, system="Linux", machine="x86_64")
    assert plan["profile"] == "pro"
    assert plan["models"] == ["voxcpm", "musetalk", "whisper"]
    assert plan["optional_models"] == []
    assert plan["config"]["avatar"]["musetalk_local"]["env_python"].endswith("/bin/python")
    assert plan["config"]["tts"]["voxcpm"]["venv_python"].endswith("/bin/python")


def test_windows_plan_uses_windows_interpreters(tmp_path):
    plan = ins.make_plan(tmp_path, "voice", [{"total_mb": 4096}],
                         system="Windows", machine="AMD64")
    assert plan["config"]["tts"]["voxcpm"]["venv_python"].endswith("/Scripts/python.exe")


def test_cuda_plan_rejects_unsupported_architecture(tmp_path):
    with pytest.raises(ValueError, match="x86_64"):
        ins.make_plan(tmp_path, "auto", [{"total_mb": 24576}],
                      system="Linux", machine="aarch64")


def test_write_config_preserves_existing_choices_and_secrets(tmp_path):
    dest = tmp_path / "user_config.yaml"
    existing = {"llm": {"api_key": "keep-secret"}, "tts": {"provider": "mimo"},
                "avatar": {"provider": "omnihuman_api"}}
    dest.write_text(yaml.safe_dump(existing), encoding="utf-8")
    ins.save_config(dest, {"tts": {"provider": "voxcpm", "voxcpm": {"model_dir": "/models"}}})
    saved = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert saved["llm"]["api_key"] == "keep-secret"
    assert saved["tts"]["provider"] == "mimo"
    assert saved["tts"]["voxcpm"]["model_dir"] == "/models"
    assert saved["avatar"] == existing["avatar"]


def test_failed_install_does_not_activate_partial_environment(tmp_path, monkeypatch):
    plan = ins.make_plan(tmp_path, "auto", [], system="Linux", machine="x86_64")
    monkeypatch.setattr(ins, "preflight", lambda *a: None)
    monkeypatch.setattr(ins, "install_base", lambda *a: (_ for _ in ()).throw(
        RuntimeError("download failed")))
    with pytest.raises(RuntimeError, match="download failed"):
        ins.execute_plan(plan)
    assert not (tmp_path / "config/user_config.yaml").exists()


def test_catalog_uses_configured_linux_paths_and_requires_weights(tmp_path, monkeypatch):
    repo = tmp_path / "MuseTalk"
    env = tmp_path / "musetalk-env/bin/python"
    env.parent.mkdir(parents=True)
    env.touch()
    repo.mkdir()
    cfg = Config({"avatar": {"musetalk_local": {
        "repo_dir": str(repo), "env_python": str(env),
    }}})
    profile = {"tier": "pro", "gpus": [{"total_mb": 16384}]}
    first = {m["id"]: m for m in hardware.get_model_catalog(cfg, profile)}
    assert not first["musetalk"]["installed"]
    from krvoiceai.core import pro_models
    # 小文件替身仅用于验证安装状态组合，不伪造真实完整模型。
    monkeypatch.setitem(pro_models.MANIFEST, "musetalk",
                        [{"path": name, "size": 6} for name in ins.MUSETALK_FILES])
    for name in ins.MUSETALK_FILES:
        path = repo / "models" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weight")
    second = {m["id"]: m for m in hardware.get_model_catalog(cfg, profile)}
    assert second["musetalk"]["installed"]
    assert set(second) == {"voxcpm", "musetalk", "whisper"}


def test_small_gpu_does_not_unlock_musetalk_by_tier_alone():
    cfg = Config({})
    profile = {"tier": "lite", "gpus": [{"total_mb": 4096}]}
    catalog = {m["id"]: m for m in hardware.get_model_catalog(cfg, profile)}
    assert catalog["voxcpm"]["allowed"]
    assert not catalog["musetalk"]["allowed"]


def test_default_inference_gpu_is_used_for_recommendation(tmp_path):
    # Workers use cuda:0; a larger second GPU must not unlock models on the first one.
    plan = ins.make_plan(tmp_path, "auto", [{"total_mb": 2048}, {"total_mb": 24576}],
                         system="Linux", machine="x86_64")
    assert plan["profile"] == "cloud"


def test_activate_only_changes_recommended_fields(tmp_path):
    dest = tmp_path / "user_config.yaml"
    dest.write_text("llm:\n  api_key: keep\ntts:\n  provider: mimo\n", encoding="utf-8")
    ins.save_config(dest, {"tts": {"provider": "edge_tts"}}, activate=True)
    assert yaml.safe_load(dest.read_text()) == {
        "llm": {"api_key": "keep"}, "tts": {"provider": "edge_tts"}}
    assert dest.with_suffix(".yaml.bak").is_file()


def test_web_installer_rejects_invalid_profile_before_starting():
    from fastapi.testclient import TestClient
    from krvoiceai.web.server import create_app
    client = TestClient(create_app())
    assert client.post("/api/installation", json={"profile": ";touch /tmp/oops"}).status_code == 422


def test_web_installation_plan_disables_native_install_in_containers(monkeypatch):
    from fastapi.testclient import TestClient
    from krvoiceai.web.server import create_app
    from krvoiceai.core import installation_jobs as jobs
    monkeypatch.setattr(jobs, "native_available", lambda: False)
    client = TestClient(create_app())
    payload = client.get("/api/installation").json()
    assert payload["available"] is False
    assert client.post("/api/installation", json={"profile": "auto"}).status_code == 409


def test_voxcpm_official_bin_weights_are_detected(tmp_path, monkeypatch):
    executable = tmp_path / "env/bin/python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    weights = tmp_path / "models"
    weights.mkdir()
    cfg = Config({"tts": {"voxcpm": {
        "venv_python": str(executable), "model_dir": str(weights)}}})
    from krvoiceai.core import pro_models
    names = ("config.json", "pytorch_model.bin", "audiovae.pth", "tokenizer.json")
    monkeypatch.setitem(pro_models.MANIFEST, "voxcpm",
                        [{"path": name, "size": 5} for name in names])
    for name in names:
        (weights / name).write_bytes(b"model")
    catalog = {m["id"]: m for m in hardware.get_model_catalog(cfg, {
        "tier": "pro", "gpus": [{"total_mb": 24576}]})}
    assert catalog["voxcpm"]["installed"]


def test_vram_guard_matches_default_worker_device(monkeypatch):
    monkeypatch.setattr(hardware, "query_gpus", lambda: [
        {"name": "small", "total_mb": 2048}, {"name": "large", "total_mb": 24576}])
    with pytest.raises(hardware.VramGateError):
        hardware.ensure_vram(5, "MuseTalk")


def test_voxcpm_worker_matches_pinned_package_api(tmp_path, monkeypatch, capsys):
    """VoxCPM 1.0.5 accepts no seed keyword; seed must be set through torch."""
    import numpy as np
    from scripts import voxcpm_worker

    calls = []

    class VoxCPM:
        tts_model = SimpleNamespace(sample_rate=16000)

        @classmethod
        def from_pretrained(cls, path, load_denoiser=False, optimize=False):
            return cls()

        def generate(self, *, text, prompt_wav_path, prompt_text, cfg_value,
                     inference_timesteps, normalize, denoise):
            calls.append(text)
            return np.zeros(1600, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "voxcpm", SimpleNamespace(VoxCPM=VoxCPM))
    monkeypatch.setattr("krvoiceai.core.pro_models.require_model", lambda *args: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(write=lambda *args: None))
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"text": "你好，世界。", "model_dir": str(tmp_path),
                               "out_wav": str(tmp_path / "out.wav")}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["voxcpm_worker.py", str(task)])
    voxcpm_worker.main()
    assert calls == ["你好，世界。"]
    assert '"success": true' in capsys.readouterr().out


def test_default_config_adapts_interpreters_for_linux(tmp_path, monkeypatch):
    from krvoiceai.core import config as config_module
    monkeypatch.setattr(config_module, "os", SimpleNamespace(name="posix", environ=os.environ))
    cfg = Config.load(config_path=tmp_path / "absent.yaml", env_file=tmp_path / "absent.env")
    assert cfg.get("tts.voxcpm.venv_python").endswith("/bin/python")
    assert cfg.get("avatar.musetalk_local.env_python").endswith("/bin/python")
    assert cfg.get("asr.whisper.model_size") == "small"


def test_uv_bootstrap_path_survives_switching_python_environments(tmp_path, monkeypatch):
    binary = tmp_path / "uv.exe"
    binary.touch()
    monkeypatch.setenv("FREEVIDEOAI_UV", str(binary))
    assert ins.uv_command() == [str(binary)]
