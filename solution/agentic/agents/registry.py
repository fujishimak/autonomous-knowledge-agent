"""Which tools each agent may use, and a factory that builds the agents with those tools.

Tool names are the names the MCP servers expose (see agentic/mcp_servers/*/tools.py).
The keys are the agents' snake_case names.
"""

from agentic.agents.account_ops_agent import AccountOperationsAgent
from agentic.agents.agent import Agent
from agentic.agents.classifier_agent import ClassifierAgent
from agentic.agents.escalation_agent import EscalationAgent
from agentic.agents.resolver_agent import ResolverAgent
from agentic.agents.service_desk_agent import ServiceDeskAgent
from agentic.mcp_servers.toolbox import McpToolbox

AGENT_CLASSES: dict[str, type[Agent]] = {
    "classifier_agent": ClassifierAgent,
    "service_desk_agent": ServiceDeskAgent,
    "resolver_agent": ResolverAgent,
    "account_ops_agent": AccountOperationsAgent,
    "escalation_agent": EscalationAgent,
}

AGENT_TOOLS: dict[str, list[str]] = {
    "classifier_agent": [],  # no tools bound
    "service_desk_agent": [
        "get_support_user",
        "get_support_user_by_external_id",
        "get_ticket_history",
        "get_open_tickets",
        "create_ticket",
        "update_ticket",
        "add_ticket_message",
    ],
    "escalation_agent": ["escalate_to_human"],
    "resolver_agent": ["get_knowledge_articles"],
    "account_ops_agent": [
        "lookup_user_account",
        "lookup_user_by_name",
        "lookup_user_by_email",
        "get_reservations",
        "get_subscriptions",
        "get_experience",
        "set_blocked",
        "set_unblocked",
    ],
}

assert AGENT_CLASSES.keys() == AGENT_TOOLS.keys(), "every agent needs an entry in both maps"


def build_agents(toolbox: McpToolbox, model: str = "gpt-4o-mini") -> dict[str, Agent]:
    """Build every agent with only the tools it is bound to. The toolbox must be connected."""
    return {
        key: agent_class(model=model, tools=toolbox.for_agent(AGENT_TOOLS[key]))
        for key, agent_class in AGENT_CLASSES.items()
    }
