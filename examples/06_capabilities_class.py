"""A `Capabilities` class groups tools and context that share state. Tools get the class as a prefix."""

import asyncio

from virtus import Capabilities, agent, context, tool
from virtus.providers import ChatCompletions


class Todo(Capabilities):
    def __init__(self) -> None:
        self.items: dict[str, bool] = {}

    @context
    def summary(self) -> str:
        pending = sum(not done for done in self.items.values())
        return f"The user has {len(self.items)} tasks, {pending} pending."

    @tool
    def add(self, task: str) -> str:
        """Add a task to the list."""
        self.items[task] = False
        return f"added {task!r}"

    @tool
    def complete(self, task: str) -> str:
        """Mark a task as done."""
        self.items[task] = True
        return f"completed {task!r}"


async def main() -> None:
    todo = Todo()

    @agent(provider=ChatCompletions(), capabilities=[todo])
    async def assistant(message: str) -> str: ...

    thread = assistant.thread()
    print(await thread("I need to buy milk and call the dentist."))
    print(await thread("Done with the milk."))
    print(await thread("What's left?"))
    print(todo.items)


asyncio.run(main())
