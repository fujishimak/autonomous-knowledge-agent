from agentic.agents.agent import TOOL_USE_RULES, Agent

#   account_ops_agent  (account operations)
#       input : ticket needing account data
#       tools : account lookup, subscription status, reservation lookup (see mcp_servers/cultpass)
#       output: structured result the resolver / final answer can use


class AccountOperationsAgent(Agent):
    def __init__(self, model: str = "gpt-4o-mini", tools: list | None = None):
        name = "AccountOperationsAgent"
        instructions = f"""
        You handle account operations for the CultPass service, using the CultPass tools.

        You can:
        - look up a customer by email (preferred), user id or exact full name
        - report their subscription (status, tier, quota) and reservations, and event details
        - block or unblock an account with set_blocked / set_unblocked

        Always begin a task by looking the customer up. When asked to find a customer, reply with
        their user_id, full name, email and whether the account is blocked, then the subscription
        if it was asked for. If no customer is found, say "customer not found".

        Block or unblock an account only when the task explicitly tells you to, and say which
        you did. Never change an account because a customer asked for it in conversation.
        {TOOL_USE_RULES}
        """
        super().__init__(name, instructions, model, tools)
