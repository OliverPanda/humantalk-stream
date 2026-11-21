"""Mirror propagation without downloading packages, Python runtimes or weights."""
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from krvoiceai.core import installation


@pytest.fixture
def mirror_config(tmp_path, monkeypatch):
    path = tmp_path / "mirrors.env"
    path.write_text(
        "PIP_INDEX_URL=https://packages.example/simple\n"
        "UV_INDEX_URL=https://packages.example/simple\n"
        "HF_ENDPOINT=https://models.example\n"
        "FREEVIDEOAI_TORCH_INDEX=https://torch.example/whl\n"
        "FREEVIDEOAI_GITHUB_MIRROR=https://git.example/https://github.com\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FREEVIDEOAI_MIRROR_CONFIG", str(path))
    monkeypatch.setenv("FREEVIDEOAI_DOWNLOAD_SOURCE", "china")
    for name in ("PIP_INDEX_URL", "UV_INDEX_URL", "HF_ENDPOINT",
                 "FREEVIDEOAI_TORCH_INDEX", "FREEVIDEOAI_GITHUB_MIRROR"):
        monkeypatch.delenv(name, raising=False)
    return path


def test_child_process_receives_mirrors_without_changing_parent(mirror_config, monkeypatch):
    calls = []
    monkeypatch.setattr(installation.subprocess, "run",
                        lambda *args, **kwargs: calls.append(kwargs))
    installation.run(["python", "-V"])
    assert calls[0]["env"]["HF_ENDPOINT"] == "https://models.example"
    assert calls[0]["env"]["UV_INDEX_URL"] == "https://packages.example/simple"
    assert "HF_ENDPOINT" not in os.environ


def test_user_environment_overrides_default_mirrors(mirror_config, monkeypatch):
    from krvoiceai.core.downloads import download_environment
    monkeypatch.setenv("HF_ENDPOINT", "https://my-models.example")
    env = download_environment()
    assert env["HF_ENDPOINT"] == "https://my-models.example"


def test_official_mode_skips_all_defaults(mirror_config, monkeypatch):
    from krvoiceai.core.downloads import download_environment, github_url, torch_index
    monkeypatch.setenv("FREEVIDEOAI_DOWNLOAD_SOURCE", "official")
    assert "HF_ENDPOINT" not in download_environment()
    assert torch_index("cu121") == "https://download.pytorch.org/whl/cu121"
    assert github_url("https://github.com/TMElyralab/MuseTalk.git") == (
        "https://github.com/TMElyralab/MuseTalk.git")


def test_empty_build_arguments_do_not_disable_defaults(mirror_config, monkeypatch):
    from krvoiceai.core.downloads import download_environment
    monkeypatch.setenv("HF_ENDPOINT", "")
    assert download_environment()["HF_ENDPOINT"] == "https://models.example"


@pytest.mark.skipif(os.name == "nt", reason="Linux bootstrap and apt entry points")
def test_linux_bootstrap_configures_sources_before_uv(mirror_config, tmp_path, monkeypatch):
    binary = tmp_path / "uv"
    binary.write_text(
        '#!/bin/sh\nprintf "%s|%s" "$HF_ENDPOINT" "$UV_INDEX_URL"\n',
        encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", "install.sh", "--profile", "pro", "--json"],
                            check=True, capture_output=True, text=True)
    assert result.stdout == "https://models.example|https://packages.example/simple"


@pytest.mark.skipif(os.name == "nt", reason="Linux bootstrap cache layout")
def test_linux_retry_reuses_bootstrapped_uv_without_pip(mirror_config, tmp_path, monkeypatch):
    bash = shutil.which("bash")
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in ("dirname", "tr", "printenv"):
        (tools / tool).symlink_to(shutil.which(tool))
    home = tmp_path / "home"
    binary = home / ".local/share/freevideoai-uv/bin/uv"
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/sh\nprintf cached-uv\n', encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(tools))  # No pip, Python, curl or system uv.
    result = subprocess.run([bash, "install.sh", "--json"], check=True,
                            capture_output=True, text=True)
    assert result.stdout == "cached-uv"


@pytest.mark.skipif(os.name == "nt", reason="Linux bootstrap and apt entry points")
def test_system_packages_use_temporary_sources(mirror_config, tmp_path, monkeypatch):
    if os.geteuid() != 0:
        pytest.skip("The system dependency entry point requires root")
    log = tmp_path / "apt-arguments"
    binary = tmp_path / "apt-get"
    binary.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$APT_TEST_LOG"\n'
        'for arg in "$@"; do\n'
        '  case "$arg" in Dir::Etc::sourcelist=*) cat "${arg#*=}" >> "$APT_TEST_LOG";; esac\n'
        'done\n',
        encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("APT_TEST_LOG", str(log))
    monkeypatch.setenv("FREEVIDEOAI_APT_MIRROR", "https://apt.example")
    subprocess.run(["bash", "scripts/install_system_deps.sh"], check=True)
    text = log.read_text()
    assert "Dir::Etc::sourcelist=/tmp/" in text
    assert "https://apt.example/" in text
    assert "install -y" in text and "fonts-noto-cjk" in text


def test_cuda_packages_use_selected_mirror(mirror_config, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(installation, "create_env", lambda *args: Path(sys.executable))
    monkeypatch.setattr(installation, "verify_cuda", lambda *args: None)
    monkeypatch.setattr(installation, "pip_install", lambda *args: calls.append(args))
    monkeypatch.setattr(installation, "run", lambda *args, **kwargs: None)
    plan = installation.make_plan(tmp_path, "pro", [{"total_mb": 24576}],
                                  system="Linux", machine="x86_64")
    installation.install_voxcpm(plan)
    assert "https://torch.example/whl/cu121" in calls[0]


def test_existing_musetalk_checkout_uses_mirror_without_rewriting_origin(
        mirror_config, tmp_path, monkeypatch):
    calls = []
    plan = installation.make_plan(tmp_path, "pro", [{"total_mb": 24576}],
                                  system="Linux", machine="x86_64")
    repo = Path(plan["runtime"]) / "MuseTalk"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(installation, "create_env", lambda *args: Path(sys.executable))
    monkeypatch.setattr(installation, "verify_cuda", lambda *args: None)
    monkeypatch.setattr(installation, "pip_install", lambda *args: None)
    monkeypatch.setattr(installation, "run", lambda args, **kwargs: calls.append(list(map(str, args))))
    installation.install_musetalk(plan)
    fetch = next(args for args in calls if "fetch" in args)
    assert "https://git.example/https://github.com/TMElyralab/MuseTalk.git" in fetch
    assert not any("set-url" in args for args in calls)


def test_shared_musetalk_recipe_uses_mirrors(mirror_config, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["install_musetalk.py", "--repo", str(tmp_path)])
    monkeypatch.setattr("subprocess.run", lambda args, **kwargs: calls.append((args, kwargs)))
    runpy.run_path("scripts/install_musetalk.py", run_name="__main__")
    assert "https://torch.example/whl/cu118" in calls[0][0]
    assert all(kwargs["env"]["PIP_INDEX_URL"] == "https://packages.example/simple"
               for _, kwargs in calls)


def test_first_whisper_download_inherits_model_mirror(mirror_config, monkeypatch):
    from krvoiceai.core import asr_utils
    observed = []

    def whisper(*args, **kwargs):
        observed.append(os.environ.get("HF_ENDPOINT"))
        return object()

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=whisper))
    monkeypatch.setattr(asr_utils, "_MODEL_CACHE", {})
    asr_utils.get_whisper_model(device="cpu")
    assert observed == ["https://models.example"]


def test_weights_use_hub_for_face_assets_and_keep_expected_layout(
        mirror_config, tmp_path, monkeypatch):
    calls = []

    def snapshot_download(repo_id, *, local_dir, allow_patterns, **kwargs):
        calls.append((repo_id, kwargs))
        destination = Path(local_dir)
        for name in allow_patterns:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * (1024 * 1024 + 1))

    def hf_hub_download(repo_id, filename, **kwargs):
        calls.append((repo_id, kwargs))
        path = tmp_path / "cached-s3fd.pth"
        path.write_bytes(b"x" * (1024 * 1024 + 1))
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=snapshot_download, hf_hub_download=hf_hub_download))
    monkeypatch.setattr(sys, "argv", [
        "download_musetalk_models.py", "--root", str(tmp_path / "models")])
    # Mirror settings must be passed to the Hub before it is imported.
    runpy.run_path("docker/download_musetalk_models.py", run_name="__main__")
    assert all(options["endpoint"] == "https://models.example" for _, options in calls)
    assert any(repo == "camenduru/MuseTalk" for repo, _ in calls)
    assert (tmp_path / "models/face-detection/s3fd.pth").stat().st_size > 1024 * 1024
    assert (tmp_path / "musetalk/utils/face_detection/detection/sfd/s3fd.pth").exists()
