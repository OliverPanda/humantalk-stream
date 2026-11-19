from krvoiceai.core.config import _apply_env_overrides


def test_empty_environment_value_does_not_erase_saved_api_key(monkeypatch):
    monkeypatch.setenv("KRVOICEAI_LLM_API_KEY", "")

    result = _apply_env_overrides({"llm": {"api_key": "saved-key"}})

    assert result["llm"]["api_key"] == "saved-key"


def test_nonempty_environment_value_still_overrides_saved_api_key(monkeypatch):
    monkeypatch.setenv("KRVOICEAI_LLM_API_KEY", "env-key")

    result = _apply_env_overrides({"llm": {"api_key": "saved-key"}})

    assert result["llm"]["api_key"] == "env-key"
