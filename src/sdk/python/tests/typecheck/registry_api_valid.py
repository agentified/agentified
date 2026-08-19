"""Consumer calls that the public registry typing must accept."""

from ratel_ai import Skill, SkillRegistry, Tool, ToolRegistry
from ratel_ai.experimental import Fact, FactRegistry

tools = ToolRegistry()
ToolRegistry(embedding={"local": "/models/local"})
ToolRegistry(spec="/models/local", pooling="mean")
ToolRegistry(huggingface="org/model", revision="main", download=False)
ToolRegistry(local="/models/local", pooling="cls")
ToolRegistry(ollama="nomic-embed-text", query_prefix="query: ")
ToolRegistry(url="https://example.test/embeddings", model="embed-v1", api_key_env="API_KEY")

skills = SkillRegistry(embedding={"ollama": "nomic-embed-text"})
facts = FactRegistry()


async def _register() -> None:
    await tools.register(
        Tool(
            id="read",
            name="read",
            description="Read a file",
            experimental_searchable_description="filesystem content retrieval",
        )
    )
    await tools.register([Tool(id="write", name="write", description="Write a file")])
    await tools.register("send", "send", "Send a message", {}, {})

    await skills.register(
        Skill(
            id="deploy",
            name="deploy",
            description="Deploy an app",
            experimental_searchable_description="production release",
        )
    )
    await skills.register([Skill(id="lint", name="lint", description="Lint the code")])
    await skills.register("auth", "auth", "Set up auth", [], [], {}, "# Auth")
    await facts.register(
        Fact(
            id="hours",
            name="hours",
            description="Opening hours",
            experimental_searchable_description="business schedule",
        )
    )
