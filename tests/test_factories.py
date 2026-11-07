import pytest

from virtus.providers import OpenAI, omlx, openrouter


def describe_provider_factories():
    def it_points_openrouter_at_its_url_with_the_key_from_the_environment(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        provider = openrouter("openai/gpt-audio-mini", voice="echo", temperature=0)

        assert isinstance(provider, OpenAI)
        assert (provider.model, provider.voice, provider.temperature) == ("openai/gpt-audio-mini", "echo", 0)
        assert str(provider._client.base_url) == "https://openrouter.ai/api/v1/"
        assert provider._client.api_key == "or-key"

    def it_requires_the_openrouter_key(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(KeyError, match="OPENROUTER_API_KEY"):
            openrouter("openai/gpt-audio-mini")
        assert openrouter("openai/gpt-audio-mini", api_key="given")._client.api_key == "given"

    def it_points_omlx_at_the_local_server_with_a_placeholder_key(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OMLX_API_KEY", raising=False)

        provider = omlx("LFM3-Omni-2B-MLX-8bit")

        assert str(provider._client.base_url) == "http://localhost:8123/v1/"
        assert provider._client.api_key == "omlx"

        monkeypatch.setenv("OMLX_API_KEY", "local")
        elsewhere = omlx("LFM3-Omni-2B-MLX-8bit", base_url="http://studio.local:8123/v1")
        assert (str(elsewhere._client.base_url), elsewhere._client.api_key) == ("http://studio.local:8123/v1/", "local")
