import json

from tests.scripted import Scripted, calls
from virtus import AssistantMessage, Capabilities, ToolResult, agent, context, lazy, tool
from virtus.lazy import HEADER, Query, Ranking, find


@tool
def clock() -> str:
    """The time now.

    Always UTC.
    """
    return "12:00"


@tool
def weather(city: str) -> str:
    """The weather in a city."""
    return f"sunny in {city}"


def describe_lazy():
    async def it_defers_the_tools_and_loads_what_a_search_finds():
        provider = Scripted(
            calls(("tool_search", {"query": "weather in a city"})),
            calls(("weather", {"city": "Lisbon"})),
            AssistantMessage("Sunny."),
        )

        @agent(provider=provider, system="Be brief.", capabilities=[lazy(clock, weather)])
        async def assistant(message: str) -> str: ...

        assert await assistant.thread("u1")("Weather in Lisbon?") == "Sunny."

        first, second, third = provider.requests
        assert [t.name for t in first.tools] == ["tool_search"]
        assert [t.name for t in first.deferred] == ["clock", "weather"]
        assert first.loaded == ()
        assert first.system == f"Be brief.\n\n{HEADER} clock, weather"
        assert first.tools[0].schema == Query.model_json_schema()
        loaded = "Loaded, usable from your next step on:\n- weather: The weather in a city."
        assert second.messages[-1] == ToolResult("call-0", "tool_search", loaded, tools=("weather",))
        assert [t.name for t in second.tools] == ["tool_search"]
        assert [t.name for t in second.loaded] == ["weather"]
        assert second.system == first.system
        assert third.messages[-1] == ToolResult("call-0", "weather", "sunny in Lisbon")

    async def it_takes_the_header_description_and_default_limit():
        provider = Scripted(AssistantMessage("ok"))
        deferring = lazy(clock, weather, header="Also:", description="Find tools.", limit=1)

        @agent(provider=provider, capabilities=[deferring])
        async def assistant(message: str) -> str: ...

        await assistant("?")

        [search] = provider.requests[0].tools
        assert provider.requests[0].system == "Also: clock, weather"
        assert (search.description, search.parameters.model_fields["limit"].default) == ("Find tools.", 1)

    async def it_keeps_what_was_loaded_for_the_rest_of_the_conversation_only():
        replies = calls(("tool_search", {"query": "time"})), AssistantMessage("ok"), AssistantMessage("ok")
        provider = Scripted(*replies, AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[lazy(clock)])
        async def assistant(message: str) -> str: ...

        await assistant.thread("u1")("load the clock")
        await assistant.thread("u1")("again")
        await assistant.thread("u2")("fresh")

        assert [[t.name for t in r.loaded] for r in provider.requests[1:]] == [["clock"], ["clock"], []]
        assert all([t.name for t in r.tools] == ["tool_search"] for r in provider.requests)

    async def it_says_so_when_nothing_matches_and_rejects_bad_arguments():
        provider = Scripted(
            calls(("tool_search", {"query": "oracle"})), calls(("tool_search", {"limit": 0})), AssistantMessage("ok")
        )

        @agent(provider=provider, capabilities=[lazy(clock, weather)])
        async def assistant(message: str) -> str: ...

        await assistant("go")

        nothing, invalid = provider.requests[1].messages[-1], provider.requests[2].messages[-1]
        assert nothing == ToolResult("call-0", "tool_search", "No tools match 'oracle'. The tools are: clock, weather")
        assert isinstance(invalid, ToolResult)
        assert invalid.is_error and invalid.content.startswith("ValidationError: 2 validation errors")
        assert provider.requests[2].loaded == ()

    async def it_takes_any_capability_and_keeps_what_it_adds_to_the_prompt():
        class Notes(Capabilities):
            @context
            def count(self) -> str:
                return "You have 2 notes."

            @tool
            def add(self, text: str) -> str:
                """Add a note."""
                return f"added {text!r}"

        search, add = calls(("tool_search", {"query": "add a note"})), calls(("notes__add", {"text": "milk"}))
        provider = Scripted(search, add, AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[clock, lazy(Notes())])
        async def assistant(message: str) -> str: ...

        await assistant("go")

        first, second, third = provider.requests
        assert [t.name for t in first.tools] == ["clock", "tool_search"]
        assert first.system == f"You have 2 notes.\n\n{HEADER} notes__add"
        assert [t.name for t in second.loaded] == ["notes__add"]
        assert json.loads(json.dumps(third.messages[-1].content)) == "added 'milk'"


def describe_find():
    @tool
    def wikipedia(query: str) -> str:
        """Look an article up on Wikipedia."""
        return ""

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression."""
        return ""

    @tool
    def http(url: str, method: str = "GET") -> str:
        """Make an HTTP request.

        Arguments:
            url: where to send it
            method: GET or POST
        """
        return ""

    catalog = (wikipedia, calculator, http, weather)

    def it_ranks_by_the_words_in_names_descriptions_and_parameters():
        assert find(catalog, "arithmetic") == [calculator]
        assert find(catalog, "look up an encyclopedia article") == [wikipedia]
        assert find(catalog, "weather in Lisbon") == [weather]
        assert find(catalog, "send a POST to a url") == [http]
        assert find(catalog, "http request for the weather", limit=1) == [http]
        assert find(catalog, "the weather") == [weather]

    def it_finds_nothing_for_unknown_or_short_words_and_honours_the_limit():
        assert find(catalog, "oracle") == []
        assert find(catalog, "an a to up") == []
        assert find((), "weather") == []
        assert len(find(catalog, "an http request or a wikipedia article", limit=2)) == 2

    def it_takes_the_ranking():
        assert len(find(catalog, "an http request or a wikipedia article", ranking=Ranking(cutoff=1))) == 1
        assert find(catalog, "url", ranking=Ranking(stopwords=frozenset({"url"}))) == []
        assert find(catalog, "up", ranking=Ranking(shortest=2)) != []
