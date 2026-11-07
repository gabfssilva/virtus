from tests.scripted import Scripted, calls
from virtus import AssistantMessage, ToolResult, agent, secrets, tool


def describe_secrets():
    seen: list[str] = []

    @tool
    def probe(token: str) -> str:
        """Call the API with a token."""
        seen.append(token)
        return f"server saw {token}"

    def _chat(provider: Scripted, **values: str | tuple[str, str]):
        @agent(provider=provider, capabilities=[probe, secrets(**values)])
        async def chat(message: str) -> str: ...

        return chat

    async def it_fills_placeholders_in_and_redacts_values_out():
        seen.clear()
        provider = Scripted(calls(("probe", {"token": "Bearer {{secrets.api}}"})), AssistantMessage("done"))

        await _chat(provider, api="s3cret")("go")

        assert seen == ["Bearer s3cret"]
        assert provider.requests[1].messages[-1] == ToolResult("call-0", "probe", "server saw Bearer {{secrets.api}}")

    async def it_lists_the_names_and_descriptions_but_never_the_values():
        provider = Scripted(AssistantMessage("done"))

        await _chat(provider, api=("s3cret", "API token"), other="x")("go")

        assert provider.requests[0].system == (
            "Secrets you can use as placeholders in tool arguments; their values are never shown to you:\n"
            "- {{secrets.api}}: API token\n"
            "- {{secrets.other}}"
        )
        assert "s3cret" not in provider.requests[0].system

    async def it_keeps_the_arguments_valid_json_whatever_the_value():
        seen.clear()
        provider = Scripted(calls(("probe", {"token": "{{ secrets.api }}"})), AssistantMessage("done"))

        await _chat(provider, api='a"b\\c\n')("go")

        assert seen == ['a"b\\c\n']

    async def it_rejects_unknown_secrets_without_running_the_tool():
        seen.clear()
        provider = Scripted(calls(("probe", {"token": "{{secrets.nope}}"})), AssistantMessage("done"))

        await _chat(provider, api="s3cret")("go")

        assert seen == []
        result = provider.requests[1].messages[-1]
        assert isinstance(result, ToolResult)
        assert result.is_error
        assert result.content == "unknown secret 'nope'; known: api"

    async def it_keeps_the_tool_schema():
        provider = Scripted(AssistantMessage("done"))

        await _chat(provider, api="s3cret")("go")

        [guarded] = provider.requests[0].tools
        assert guarded.schema == probe.schema
