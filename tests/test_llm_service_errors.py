import httpx
import pytest

from krvoiceai.core.llm_client import LLMClient, LLMServiceError


@pytest.mark.parametrize("status", [503, 429])
def test_upstream_failure_retains_diagnostics_and_stops_retrying(
    isolated_config, monkeypatch, status
):
    isolated_config.set("llm.provider", "deepseek")
    isolated_config.set("llm.api_key", "secret-test-key")
    isolated_config.set("llm.base_url", "https://relay.example/v1")
    isolated_config.set("llm.model", "test-model")
    client = LLMClient()
    attempts = []
    sleeps = []

    def post(url, **kwargs):
        attempts.append(url)
        return httpx.Response(
            status,
            json={"error": {"message": "Upstream unavailable secret-test-key"}},
            headers={"x-request-id": "request-test-123"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("krvoiceai.core.llm_client.httpx.post", post)
    monkeypatch.setattr("krvoiceai.core.llm_client.time.sleep", sleeps.append)
    with pytest.raises(LLMServiceError) as error:
        client.chat([{"role": "user", "content": "test"}])
    assert len(attempts) == client.MAX_RETRIES + 1
    assert len(sleeps) == client.MAX_RETRIES
    message = str(error.value)
    assert f"HTTP {status}" in message
    assert "relay.example" in message
    assert "test-model" in message
    assert "request-test-123" in message
    assert "secret-test-key" not in message
