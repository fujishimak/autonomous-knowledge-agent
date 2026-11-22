# STARTER (from EcoHome): the `Agent` class below is a reusable wrapper around create_react_agent.
#   Keep the idea, adapt the details (see the "TODO (adapt)" notes inline). Rubric 2 needs >= 4 agents,
#   each with a single, documented responsibility that matches agentic/design/architecture.md.
#   Suggested: reuse this class once per agent with different `name`, `instructions` and `tools`.
#
# pseudocode (one entry per agent):
#   classifier_agent
#       input : ticket text + metadata (channel, created_at, user history)
#       output: category, urgency, complexity   (structured output via a Pydantic schema)
#       tools : none (optionally long-term memory lookup)
#   resolver_agent  (knowledge / RAG)
#       input : ticket text + category
#       does  : call the knowledge search tool, answer ONLY from returned articles, cite article titles
#       output: draft answer + confidence score
#   account_ops_agent  (account operations)
#       input : ticket needing account data
#       tools : account lookup, subscription status, reservation lookup/cancel (see tools.py)
#       output: structured result the resolver / final answer can use
#   escalation_agent
#       input : ticket + everything gathered so far
#       output: hand-off summary for a human, ticket status set to "escalated"
#   (optional) supervisor / router
#       decides which agent runs next; or implement the routing as conditional edges in workflow.py
#
#   for each agent:
#       model = ChatOpenAI(...)   # read base_url / api key from .env (OPENAI_BASE_URL, VOCAREUM_API_KEY), no hard-coding
#       system prompt = role, allowed tools, output format, "never invent policy not in the KB"
#       wrap in create_react_agent(...) or a plain node function that returns a state update
#
import os
import time
from typing import Sequence

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent

from agentic.observability import log_event, log_tool_results

load_dotenv()

# Appended to every agent's instructions: agents must not claim work the tools did not do.
TOOL_USE_RULES = """
Rules for using tools:
- Only state facts that a tool returned or that the task gave you. Never guess ids, names or statuses.
- Only say you created, updated, found or notified something if the tool result had "ok": true.
- If a tool returns "ok": false, say so plainly and include its "error" code. Do not retry with invented values.
- Copy ids exactly as returned (ticket_id, user_id, article titles).
"""


def build_llm(model: str = "gpt-4o-mini") -> ChatOpenAI:
    """The chat model used by the agents and the orchestrator."""
    return ChatOpenAI(
        model=model,
        temperature=0.0,
        # TODO (adapt): .env defines OPENAI_BASE_URL and VOCAREUM_API_KEY. Read the base URL from
        #   os.getenv("OPENAI_BASE_URL") instead of hard-coding it, and check the key name matches .env.
        base_url="https://openai.vocareum.com/v1",
        api_key=os.getenv("VOCAREUM_API_KEY"),
    )


class Agent:
    """A ReAct agent with its own prompt and its own tools.

    Subclasses set the name and instructions. The tools are passed in by whoever builds the
    agent (see registry.py), so each agent only sees the tools it is bound to.
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = "gpt-4o-mini",
        tools: Sequence[BaseTool] | None = None,
        response_format: type[BaseModel] | None = None,
    ):
        self.name = name
        self.tools: list[BaseTool] = list(tools or [])

        llm = build_llm(model)

        self.graph = create_react_agent(
            name=name,
            prompt=SystemMessage(content=instructions),
            model=llm,
            tools=self.tools,
            # With a schema, the result also has `structured_response` (an instance of it).
            response_format=response_format,
        )

    async def ainvoke(
        self, question: str, context: str | None = None, config: dict | None = None
    ) -> dict:
        """
        Args:
            question (str): The customer's question about Cultpass
            context (str): Any useful context related to the user's Cultpass activity
            config (dict): Optional LangGraph config, e.g. {"configurable": {"thread_id": ticket_id}}

        Returns:
            dict: The graph result; the reply is `result["messages"][-1].content`, and
                `result["structured_response"]` holds the parsed schema if one was set.

        Async only: the MCP tools cannot be called from the synchronous `invoke`.
        """
        messages = []
        if context:
            # Add some context to the question as a system message
            messages.append(("system", context))

        messages.append(("user", question))

        started = time.perf_counter()
        result = await self.graph.ainvoke(input={"messages": messages}, config=config)
        tool_calls = log_tool_results(self.name, result["messages"])
        log_event(
            "agent_run",
            agent=self.name,
            tool_calls=tool_calls,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
        return result

    def get_agent_tools(self) -> list[str]:
        """Names of the tools bound to this agent."""
        return [t.name for t in self.tools]
