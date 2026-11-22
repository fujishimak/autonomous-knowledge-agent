"""UDA-Hub MCP server: tools that work on the support platform database (udahub.db)."""

import logging

from agentic.mcp_servers.common import (
    NOT_FOUND,
    ToolInputError,
    fail,
    jsonable,
    ok,
    pick,
    require_int_range,
    require_text,
    safe_tool,
)
from agentic.mcp_servers.udahub.knowledge_index import KnowledgeIndex
from agentic.mcp_servers.udahub.repository import (
    MemoryType,
    MessageRole,
    TicketStatus,
    UdaHubRepository,
)
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

MAX_ARTICLES = 10
MAX_TICKETS = 20


def build_udahub_server(
    repository: UdaHubRepository, knowledge_index: KnowledgeIndex
) -> FastMCP:
    """Create the UDA-Hub server. Dependencies are passed in so tests can fake them."""
    server = FastMCP("udahub")

    # --- users -------------------------------------------------------------

    @server.tool
    @safe_tool
    def get_support_user(user_id: str) -> dict:
        """Look up a support-platform user by their UDA-Hub user id.

        This is the support-side record of a customer (the one tickets belong to), not
        the CultPass account. Its `external_user_id` is the CultPass user id.

        Args:
            user_id: The UDA-Hub user id (`user_id` on a ticket).

        Returns:
            {"ok": True, "user": {"user_id", "account_id", "external_user_id", "user_name",
             "created_at"}}, or {"ok": False, "error": "not_found" | "invalid_input", ...}.
        """
        user_id = require_text(user_id, "user_id")
        user = repository.get_user_by_id(user_id)
        if not user:
            return fail(NOT_FOUND, "No support user with that id.")
        return ok(
            user=pick(
                user,
                "user_id",
                "account_id",
                "external_user_id",
                "user_name",
                "created_at",
            )
        )

    @server.tool
    @safe_tool
    def get_support_user_by_external_id(external_user_id: str) -> dict:
        """Find the support-platform user for a CultPass customer id.

        Use this to get the UDA-Hub `user_id` (needed for ticket history) from a CultPass
        user id.

        Args:
            external_user_id: The CultPass user id.

        Returns:
            Same shape as `get_support_user`.
        """
        external_user_id = require_text(external_user_id, "external_user_id")
        user = repository.get_user_by_external_id(external_user_id)
        if not user:
            return fail(NOT_FOUND, "No support user for that CultPass user id.")
        return ok(
            user=pick(
                user,
                "user_id",
                "account_id",
                "external_user_id",
                "user_name",
                "created_at",
            )
        )

    # --- tickets -----------------------------------------------------------

    @server.tool
    @safe_tool
    def get_ticket_history(user_id: str, max_results: int = 5) -> dict:
        """List a customer's previous support tickets, newest first.

        Use it to recognise returning customers and repeated issues.

        Args:
            user_id: The UDA-Hub user id (not the CultPass id).
            max_results: How many tickets to return, 1 to 20 (default 5).

        Returns:
            {"ok": True, "count": int, "tickets": [{"ticket_id", "channel", "created_at",
             "status", "main_issue_type", "tags"}]}. Empty list if there are none.
        """
        user_id = require_text(user_id, "user_id")
        max_results = require_int_range(max_results, "max_results", 1, MAX_TICKETS)
        tickets = jsonable(repository.get_ticket_history(user_id, max_results))
        return ok(count=len(tickets), tickets=tickets)

    @server.tool
    @safe_tool
    def get_open_tickets(user_id: str, max_results: int = 10) -> dict:
        """List a customer's open tickets, each with its full message thread.

        Args:
            user_id: The UDA-Hub user id (not the CultPass id).
            max_results: How many tickets to return, 1 to 20 (default 10).

        Returns:
            {"ok": True, "count": int, "tickets": [{"ticket_id", "channel", "created_at",
             "status", "main_issue_type", "tags", "messages": [{"message_id", "role",
             "content", "created_at"}]}]}. Messages are oldest first; `role` is one of
            "user", "agent", "ai", "system". Empty list if there are no open tickets.
        """
        user_id = require_text(user_id, "user_id")
        max_results = require_int_range(max_results, "max_results", 1, MAX_TICKETS)
        tickets = jsonable(repository.get_open_tickets_for_user(user_id, max_results))
        return ok(count=len(tickets), tickets=tickets)

    # --- ticket writes -----------------------------------------------------

    @server.tool
    @safe_tool
    def create_ticket(
        external_user_id: str,
        content: str,
        channel: str = "chat",
        user_name: str | None = None,
        tags: str | None = None,
    ) -> dict:
        """Open a new support ticket for a customer, with their message as the first message.

        The ticket starts with status "open". Check `get_open_tickets` first so you do not
        open a duplicate for an issue that already has an open ticket.

        Args:
            external_user_id: The customer's CultPass user id.
            content: The customer's message or a faithful summary of the problem.
            channel: Where the request came from, e.g. "chat" (default), "email", "phone".
            user_name: The customer's full name. Required only the first time a customer
                is seen (when `get_support_user_by_external_id` returns not_found).
            tags: Optional comma-separated keywords, e.g. "login, access".

        Returns:
            {"ok": True, "ticket": {"ticket_id", "user_id", "account_id", "channel",
             "status", "tags", "created_at"}}, or {"ok": False, "error": "invalid_input", ...}.
        """
        external_user_id = require_text(external_user_id, "external_user_id")
        content = require_text(content, "content", max_length=5000)
        channel = require_text(channel, "channel", max_length=50)
        tags = require_text(tags, "tags", max_length=500) if tags else None
        if not repository.get_user_by_external_id(external_user_id):
            # A first-time customer needs a name for the new support-side user record.
            if not (user_name and user_name.strip()):
                raise ToolInputError(
                    "user_name is required because this customer has no support record yet"
                )
            user_name = require_text(user_name, "user_name", max_length=200)
        ticket = repository.create_ticket(
            external_user_id, content, channel=channel, user_name=user_name, tags=tags
        )
        return ok(ticket=jsonable(ticket))

    @server.tool
    @safe_tool
    def update_ticket(ticket_id: str, status: TicketStatus) -> dict:
        """Set a ticket's status. Safe to repeat: setting the current status changes nothing.

        Statuses: "open" (new or waiting for work), "in_progress" (being worked on),
        "resolved" (the customer's issue is solved), "escalated" (handed to a human; use
        `escalate_to_human` to notify them).

        Args:
            ticket_id: The ticket to update.
            status: One of "open", "in_progress", "resolved", "escalated".

        Returns:
            {"ok": True, "ticket": {"ticket_id", "status", "previous_status",
             "updated_at"}}, or {"ok": False, "error": "not_found" | "invalid_input", ...}.
        """
        ticket_id = require_text(ticket_id, "ticket_id")
        ticket = repository.update_ticket_status(ticket_id, status)
        if not ticket:
            return fail(NOT_FOUND, "No ticket with that id.")
        return ok(ticket=jsonable(ticket))

    @server.tool
    @safe_tool
    def add_ticket_message(
        ticket_id: str, content: str, role: MessageRole = "user"
    ) -> dict:
        """Append a message to a ticket's conversation.

        Use it to record what the customer says after the ticket was opened, and what was
        replied to them. Always pass the exact text, not a paraphrase.

        Args:
            ticket_id: The ticket to add the message to.
            content: The message text.
            role: Who wrote it: "user" (the customer, default), "ai" (the assistant's reply
                to the customer), "agent" (a human support agent) or "system" (an internal note).

        Returns:
            {"ok": True, "message": {"message_id", "ticket_id", "role", "content",
             "created_at"}}, or {"ok": False, "error": "not_found" | "invalid_input", ...}.
        """
        ticket_id = require_text(ticket_id, "ticket_id")
        content = require_text(content, "content", max_length=5000)
        message = repository.add_ticket_message(ticket_id, role, content)
        if not message:
            return fail(NOT_FOUND, "No ticket with that id.")
        return ok(message=jsonable(message))

    # --- long-term memory --------------------------------------------------

    @server.tool
    @safe_tool
    def get_customer_memory(
        user_id: str, memory_type: MemoryType | None = None, max_results: int = 20
    ) -> dict:
        """Recall what is remembered about a customer from earlier tickets.

        Args:
            user_id: The UDA-Hub user id (not the CultPass id).
            memory_type: Optional filter: "resolved_issue" or "preference". Omit for both.
            max_results: How many memories to return, 1 to 50 (default 20).

        Returns:
            {"ok": True, "count": int, "memories": [{"memory_id", "memory_type", "key",
             "value", "source_ticket_id", "created_at"}]}, newest first. For a
            "resolved_issue", `key` is the issue category and `value` says how it was fixed.
            For a "preference", `key` is the setting (e.g. "language") and `value` the choice.
        """
        user_id = require_text(user_id, "user_id")
        max_results = require_int_range(max_results, "max_results", 1, 50)
        rows = repository.get_customer_memories(user_id, memory_type, max_results)
        memories = [
            pick(
                r,
                "memory_id",
                "memory_type",
                "key",
                "value",
                "source_ticket_id",
                "created_at",
            )
            for r in rows
        ]
        return ok(count=len(memories), memories=memories)

    @server.tool
    @safe_tool
    def save_customer_memory(
        user_id: str,
        memory_type: MemoryType,
        key: str,
        value: str,
        ticket_id: str | None = None,
    ) -> dict:
        """Remember something about a customer for future tickets.

        Use "resolved_issue" after an issue was fixed (key = category, value = problem and fix).
        Use "preference" for a stated choice such as language or preferred contact; saving the
        same key again replaces the earlier value.

        Args:
            user_id: The UDA-Hub user id.
            memory_type: "resolved_issue" or "preference".
            key: Issue category, or preference name.
            value: What to remember, in a short sentence.
            ticket_id: The ticket this came from, if any.

        Returns:
            {"ok": True, "memory": {...}}, or {"ok": False, "error": "not_found" |
            "invalid_input", ...}.
        """
        user_id = require_text(user_id, "user_id")
        key = require_text(key, "key", max_length=100)
        value = require_text(value, "value", max_length=1000)
        ticket_id = require_text(ticket_id, "ticket_id") if ticket_id else None
        memory = repository.save_customer_memory(user_id, memory_type, key, value, ticket_id)
        if not memory:
            return fail(NOT_FOUND, "No support user with that id.")
        return ok(
            memory=pick(
                memory,
                "memory_id",
                "memory_type",
                "key",
                "value",
                "source_ticket_id",
                "created_at",
            )
        )

    # --- knowledge base ----------------------------------------------------

    @server.tool
    @safe_tool
    def get_knowledge_articles(query: str, k: int = 3) -> dict:
        """Search the support knowledge base for articles relevant to a customer question.

        Args:
            query: The customer's question or issue, in natural language.
            k: How many articles to return, from 1 to 10 (default 3).

        Returns:
            {"ok": True, "results": [{"article_id", "title", "content", "tags", "score"}],
             "top_score": float}. Results are best first. `score` is a 0..1 similarity;
            low scores mean the article is probably not relevant. `results` is empty and
            `top_score` is 0.0 when nothing matches.
            On failure: {"ok": False, "error": <code>, "message": <text>}.
        """
        query = require_text(query, "query", max_length=1000)
        k = require_int_range(k, "k", 1, MAX_ARTICLES)
        result = knowledge_index.search(query, k=k)
        return ok(results=result["results"], top_score=result["top_score"])

    # --- escalation --------------------------------------------------------

    @server.tool
    @safe_tool
    def escalate_to_human(ticket_id: str, reason: str) -> dict:
        """Hand a ticket over to a human support agent.

        Use it when the knowledge base has no relevant answer, the customer asks for a
        person, or the issue is urgent or outside what the tools can do. Placeholder
        for now: it only records that a human was notified.

        Args:
            ticket_id: The ticket being escalated.
            reason: Short explanation for the human, including what was already tried.

        Returns:
            {"ok": True, "message": "Human notified"}, or
            {"ok": False, "error": "invalid_input", ...}.
        """
        ticket_id = require_text(ticket_id, "ticket_id")
        reason = require_text(reason, "reason", max_length=2000)
        # The reason can contain personal data, so only its length is logged.
        logger.info("escalate_to_human ticket_id=%s reason_chars=%d", ticket_id, len(reason))
        return ok(message="Human notified")

    return server
