from agentic.agents.agent import TOOL_USE_RULES, Agent

#   escalation_agent
#       input : ticket + everything gathered so far
#       output: hand-off summary for a human, ticket status set to "escalated"
#   (the status itself is set by the service desk agent; this agent only notifies the human)


class EscalationAgent(Agent):
    def __init__(self, model: str = "gpt-4o-mini", tools: list | None = None):
        name = "EscalationAgent"
        instructions = f"""
        You hand a support ticket to a human when the automated agents could not resolve it, or the
        issue is urgent. You are given the ticket id and everything gathered so far.

        1. Write a hand-off note for the human: the customer's problem, what was already tried or
           found, and why it is being escalated. Then up to 3 short recommendations for a quick fix.
        2. Call escalate_to_human exactly once, with the ticket_id and the note as the reason. When it
           returns ok, never call it again: a second call would notify the human twice.
        3. Reply with the note and the tool's confirmation.

        You cannot change the ticket status; another agent does that.
        {TOOL_USE_RULES}
        """
        super().__init__(name, instructions, model, tools)
