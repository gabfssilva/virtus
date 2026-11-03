import os

import pytest
from pydantic import BaseModel

from liteagent import AssistantMessage, ToolCall, ToolResult, UserMessage, agent, tool
from liteagent.providers import OpenAI
from liteagent.providers.openai import messages


def describe_messages():
    def it_converts_the_conversation_to_chat_completions_format():
        history = (
            UserMessage("Weather?"),
            AssistantMessage(tool_calls=(ToolCall("c1", "weather", '{"city": "Paris"}'),)),
            ToolResult("c1", "weather", "Sunny"),
            AssistantMessage("It's sunny"),
        )

        assert messages("Be brief.", history) == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
            {"role": "assistant", "content": "It's sunny"},
        ]


class Forecast(BaseModel):
    city: str
    sunny: bool


@pytest.mark.integration
def describe_openai():
    async def it_calls_tools_and_returns_structured_output():
        assert os.environ.get("OPENAI_API_KEY"), "integration tests need OPENAI_API_KEY"

        @tool
        def weather(city: str) -> str:
            """Current weather for a city."""
            return f"Sunny and 25°C in {city}"

        @agent(provider=OpenAI(temperature=0), capabilities=[weather])
        async def forecast(city: str) -> Forecast:
            """Use the weather tool to tell whether it is sunny in {city}."""
            ...

        assert await forecast("Paris") == Forecast(city="Paris", sunny=True)
