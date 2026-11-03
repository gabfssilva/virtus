from liteagent import AssistantMessage, ToolResult, UserMessage, agent, compaction, tool
from liteagent.compaction import SUMMARY_PROMPT
from tests.scripted import Scripted, calls

SUMMARY = UserMessage("Summary of the earlier conversation:\n\nAna likes tea")


def describe_compaction():
    async def it_leaves_short_conversations_untouched():
        provider = Scripted(AssistantMessage("a"), AssistantMessage("b"))

        @agent(provider=provider, capabilities=[compaction(max_tokens=1_000, keep=1)])
        async def chat(message: str) -> str: ...

        thread = chat.thread()
        await thread("hello")
        await thread("again")

        assert len(provider.requests) == 2
        assert provider.requests[1].messages == (UserMessage("hello"), AssistantMessage("a"), UserMessage("again"))

    async def it_summarizes_older_messages_and_keeps_the_recent_ones():
        provider = Scripted(
            AssistantMessage("noted"),
            AssistantMessage("Ana likes tea"),  # the summary
            AssistantMessage("Tea it is"),
        )

        @agent(provider=provider, capabilities=[compaction(max_tokens=5, keep=1)])
        async def chat(message: str) -> str: ...

        thread = chat.thread()
        await thread("I'm Ana and I like tea")
        await thread("What should I drink?")

        summary_request, answer_request = provider.requests[1:]
        assert summary_request.system == SUMMARY_PROMPT
        assert summary_request.messages == (UserMessage("user: I'm Ana and I like tea\nassistant: noted"),)
        assert answer_request.messages == (SUMMARY, UserMessage("What should I drink?"))
        assert thread.history == (SUMMARY, UserMessage("What should I drink?"), AssistantMessage("Tea it is"))

    async def it_never_separates_a_tool_result_from_its_call():
        @tool
        def lookup() -> str:
            return "tea"

        provider = Scripted(
            calls(("lookup", {})),
            AssistantMessage("Ana likes tea"),  # the summary, before the second step
            AssistantMessage("Tea"),
        )

        @agent(provider=provider, capabilities=[compaction(max_tokens=5, keep=1), lookup])
        async def chat(message: str) -> str: ...

        await chat("What does Ana like? Look it up please")

        assert provider.requests[2].messages == (
            SUMMARY,
            calls(("lookup", {})),
            ToolResult("call-0", "lookup", "tea"),
        )
