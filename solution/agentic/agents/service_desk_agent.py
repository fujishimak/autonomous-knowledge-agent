from agentic.agents.agent import TOOL_USE_RULES, Agent

#   service_desk_agent  (UDA-Hub tickets)
#       input : a customer id plus what to record or change
#       tools : support user lookup, ticket history, create/update ticket, add message
#       output: the ticket_id and what was done


class ServiceDeskAgent(Agent):
    def __init__(self, model: str = "gpt-4o-mini", tools: list | None = None):
        name = "ServiceDeskAgent"
        instructions = f"""
        You run the support service desk on UDA-Hub. You work on tickets with the tools; the task
        you are given tells you what to do.

        Creating a ticket (you are given the CultPass user id, the customer's name, their message,
        and optionally tags, a channel and a status to set):
        1. Call get_support_user_by_external_id with the CultPass user id, only to learn whether the
           customer already has a support record. You do not need it otherwise.
        2. Call create_ticket with external_user_id, content, user_name, channel and tags. Always
           create a new ticket, even if the customer has other open tickets, unless the task tells
           you to reuse an existing one.
        3. If the task asks for a status, call update_ticket with the new ticket_id.
        4. Reply with the ticket_id exactly as returned by create_ticket.

        Other tasks:
        - Record a message on a ticket with add_ticket_message. Pass the exact text. Use role
          "user" for the customer's words and "ai" for replies sent to the customer.
        - Change a ticket's status with update_ticket: open, in_progress, resolved or escalated.
        - Look up a customer's ticket history, or summarise the messages of a ticket from
          get_open_tickets in a few sentences.

        End every reply with the ticket_id you worked on, or "no ticket" if there was none.
        {TOOL_USE_RULES}
        """
        super().__init__(name, instructions, model, tools)
