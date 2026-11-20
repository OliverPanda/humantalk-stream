"""Real process/thread contention between the installer and Web configuration writer."""
from __future__ import annotations

import multiprocessing
import threading
import traceback

import pytest
import yaml

from krvoiceai.core.config import Config


def _installer_writer(path, attempted, read, release, finished, errors, pause):
    from krvoiceai.core.installation import save_config

    original_load = yaml.safe_load
    if pause:
        def paused_load(*args, **kwargs):
            data = original_load(*args, **kwargs)
            read.set()
            if not release.wait(15):
                raise TimeoutError("Installer transaction was not released")
            return data
        yaml.safe_load = paused_load
    attempted.set()
    try:
        save_config(path, {"tts": {"provider": "edge_tts"}})
    except BaseException:
        errors.put(traceback.format_exc())
    finally:
        finished.set()


@pytest.fixture
def settings_writer(tmp_path, monkeypatch):
    from krvoiceai.core import settings_manager as settings

    path = tmp_path / "user_config.yaml"
    path.write_text("llm:\n  api_key: old-key\n", encoding="utf-8")
    monkeypatch.setattr(settings, "USER_CONFIG_PATH", path)
    monkeypatch.setattr(
        settings, "get_config",
        lambda **kwargs: Config(yaml.safe_load(path.read_text(encoding="utf-8"))
                                if path.exists() else {}),
    )
    # Avoid changing the application's process-wide singleton.
    manager = object.__new__(settings.SettingsManager)
    manager._listeners = []
    manager._user_data = {}
    return path, manager


@pytest.mark.parametrize("paused_writer", ["installer", "web"])
def test_installer_and_web_preserve_each_others_updates(settings_writer, paused_writer):
    path, manager = settings_writer
    ctx = multiprocessing.get_context("spawn")
    attempted, read, release, finished = (ctx.Event() for _ in range(4))
    errors = ctx.Queue()
    web_done, web_attempted = threading.Event(), threading.Event()
    web_errors = []
    original_load = manager._load_user_config

    if paused_writer == "web":
        def paused_load():
            original_load()
            read.set()
            if not release.wait(15):
                raise TimeoutError("Web transaction was not released")
        manager._load_user_config = paused_load

    def web_writer():
        web_attempted.set()
        try:
            manager.update_section("llm", {"api_key": "new-key"})
        except BaseException:
            web_errors.append(traceback.format_exc())
        finally:
            web_done.set()

    process = ctx.Process(
        target=_installer_writer,
        args=(path, attempted, read, release, finished, errors,
              paused_writer == "installer"),
    )
    thread = threading.Thread(target=web_writer, daemon=True)
    try:
        if paused_writer == "installer":
            process.start()
            assert read.wait(15), "Installer did not reach the read checkpoint"
            thread.start()
            assert web_attempted.wait(5)
            blocked = not web_done.wait(1)
        else:
            thread.start()
            assert read.wait(15), "Web writer did not reach the read checkpoint"
            process.start()
            assert attempted.wait(15)
            blocked = not finished.wait(1)
    finally:
        release.set()
        if process.pid is not None:
            process.join(15)
            if process.is_alive():
                process.terminate()
                process.join(5)
        if thread.ident is not None:
            thread.join(15)

    assert blocked, "The second writer entered a transaction before the first released its lock"
    assert process.exitcode == 0
    assert errors.empty() and not web_errors, web_errors
    assert web_done.is_set() and finished.is_set()
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["llm"]["api_key"] == "new-key"
    assert saved["tts"]["provider"] == "edge_tts"


@pytest.mark.parametrize("operation", ["reset_section", "reset_all"])
def test_web_reset_waits_for_installer_transaction(settings_writer, operation):
    path, manager = settings_writer
    ctx = multiprocessing.get_context("spawn")
    attempted, read, release, finished = (ctx.Event() for _ in range(4))
    errors = ctx.Queue()
    web_attempted, web_done = threading.Event(), threading.Event()
    web_errors = []

    def reset():
        web_attempted.set()
        try:
            if operation == "reset_section":
                manager.reset_section("llm")
            else:
                manager.reset_all()
        except BaseException:
            web_errors.append(traceback.format_exc())
        finally:
            web_done.set()

    process = ctx.Process(
        target=_installer_writer,
        args=(path, attempted, read, release, finished, errors, True),
    )
    thread = threading.Thread(target=reset, daemon=True)
    try:
        process.start()
        assert read.wait(15)
        thread.start()
        assert web_attempted.wait(5)
        blocked = not web_done.wait(1)
    finally:
        release.set()
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join(5)
        if thread.ident is not None:
            thread.join(15)
    assert blocked, "Reset bypassed the installer's configuration lock"
    assert process.exitcode == 0 and errors.empty() and not web_errors
    assert web_done.is_set() and finished.is_set()
    if operation == "reset_all":
        assert not path.exists()
    else:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
            "tts": {"provider": "edge_tts"}}


@pytest.mark.parametrize("writer", ["installer", "web"])
def test_serialization_failure_preserves_current_configuration(settings_writer, monkeypatch, writer):
    from krvoiceai.core.installation import save_config

    path, manager = settings_writer
    original = path.read_bytes()

    def failed_dump(data, stream, **kwargs):
        stream.write("llm: [incomplete")
        raise OSError("Simulated write failure")

    monkeypatch.setattr(yaml, "safe_dump", failed_dump)
    with pytest.raises(OSError, match="Simulated write failure"):
        if writer == "installer":
            save_config(path, {"tts": {"provider": "edge_tts"}})
        else:
            manager.update_section("llm", {"api_key": "new-key"})
    assert path.read_bytes() == original
