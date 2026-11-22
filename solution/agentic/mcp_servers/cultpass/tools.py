"""CultPass MCP server: tools on the CultPass customer database (cultpass.db)."""

from agentic.mcp_servers.common import (
    NOT_FOUND,
    ToolInputError,
    fail,
    ok,
    pick,
    require_text,
    safe_tool,
)
from agentic.mcp_servers.cultpass.repository import CultpassRepository
from fastmcp import FastMCP


def _public_user(user: dict) -> dict:
    """The user fields an agent may see."""
    return pick(user, "user_id", "full_name", "email", "is_blocked", "created_at")


def build_cultpass_server(repository: CultpassRepository) -> FastMCP:
    """Create the CultPass server. The repository is passed in so tests can use a test database."""
    server = FastMCP("cultpass")

    @server.tool
    @safe_tool
    def lookup_user_account(user_id: str) -> dict:
        """Look up a CultPass customer by their user id.

        Args:
            user_id: The CultPass user id (the `external_user_id` of a support ticket's user).

        Returns:
            {"ok": True, "user": {"user_id", "full_name", "email", "is_blocked",
             "created_at"}}, or {"ok": False, "error": "not_found" | "invalid_input", ...}.
            `is_blocked` true means the account is suspended.
        """
        user_id = require_text(user_id, "user_id")
        user = repository.get_user_by_id(user_id)
        if not user:
            return fail(NOT_FOUND, "No CultPass user with that id.")
        return ok(user=_public_user(user))

    @server.tool
    @safe_tool
    def lookup_user_by_name(name: str) -> dict:
        """Look up a CultPass customer by exact full name (case-sensitive).

        Names are not unique, so only the first match is returned. Prefer
        `lookup_user_account` when a user id is available.

        Args:
            name: The customer's full name exactly as registered, e.g. "Bob Stone".

        Returns:
            {"ok": True, "user": {...same fields as lookup_user_account}}, or
            {"ok": False, "error": "not_found" | "invalid_input", ...}.
        """
        name = require_text(name, "name")
        user = repository.get_user_by_name(name)
        if not user:
            return fail(NOT_FOUND, "No CultPass user with that name.")
        return ok(user=_public_user(user))

    @server.tool
    @safe_tool
    def lookup_user_by_email(email: str) -> dict:
        """Look up a CultPass customer by email address (case-insensitive).

        Emails are unique, so this finds at most one customer. Use it when the customer
        gives an email and no user id.

        Args:
            email: The customer's email address, e.g. "bob.stone@granite.com".

        Returns:
            {"ok": True, "user": {...same fields as lookup_user_account}}, or
            {"ok": False, "error": "not_found" | "invalid_input", ...}.
        """
        email = require_text(email, "email")
        if "@" not in email:
            raise ToolInputError("email must be a valid email address")
        user = repository.get_user_by_email(email)
        if not user:
            return fail(NOT_FOUND, "No CultPass user with that email.")
        return ok(user=_public_user(user))

    @server.tool
    @safe_tool
    def get_reservations(user_id: str) -> dict:
        """List a customer's experience reservations, newest first.

        Args:
            user_id: The CultPass user id.

        Returns:
            {"ok": True, "count": int, "reservations": [{"reservation_id", "experience_id",
             "status", "created_at"}]}. The list is empty if the user has no reservations
            (or does not exist: check with `lookup_user_account`). Use `get_experience`
            with an `experience_id` to get the event's title, place and date.
        """
        user_id = require_text(user_id, "user_id")
        rows = repository.get_reservations(user_id)
        reservations = [
            pick(r, "reservation_id", "experience_id", "status", "created_at")
            for r in rows
        ]
        return ok(count=len(reservations), reservations=reservations)

    @server.tool
    @safe_tool
    def get_subscriptions(user_id: str) -> dict:
        """Get a customer's CultPass subscription (a user has at most one).

        Args:
            user_id: The CultPass user id.

        Returns:
            {"ok": True, "count": int, "subscriptions": [{"subscription_id", "status",
             "tier", "monthly_quota", "started_at", "ended_at"}]}. `status` is "active" or
            "cancelled". The list is empty if the user has no subscription.
        """
        user_id = require_text(user_id, "user_id")
        rows = repository.get_subscriptions(user_id)
        subscriptions = [
            pick(
                s,
                "subscription_id",
                "status",
                "tier",
                "monthly_quota",
                "started_at",
                "ended_at",
            )
            for s in rows
        ]
        return ok(count=len(subscriptions), subscriptions=subscriptions)

    @server.tool
    @safe_tool
    def get_experience(experience_id: str) -> dict:
        """Get the details of a CultPass experience (event), e.g. from a reservation.

        Args:
            experience_id: The experience id, as found on a reservation.

        Returns:
            {"ok": True, "experience": {"experience_id", "title", "description", "location",
             "when", "slots_available", "is_premium"}}, or
            {"ok": False, "error": "not_found" | "invalid_input", ...}.
        """
        experience_id = require_text(experience_id, "experience_id")
        experience = repository.get_experience(experience_id)
        if not experience:
            return fail(NOT_FOUND, "No experience with that id.")
        return ok(
            experience=pick(
                experience,
                "experience_id",
                "title",
                "description",
                "location",
                "when",
                "slots_available",
                "is_premium",
            )
        )

    @server.tool
    @safe_tool
    def set_blocked(user_id: str) -> dict:
        """Block a customer's account. Safe to repeat: an already blocked account stays blocked.

        This takes effect immediately. Only block when the situation calls for it
        (for example fraud or abuse, or a human instruction).

        Args:
            user_id: The CultPass user id.

        Returns:
            {"ok": True, "user": {"user_id", "full_name", "email", "is_blocked",
             "created_at"}} with `is_blocked` true, or
            {"ok": False, "error": "not_found" | "invalid_input", ...}.
        """
        user_id = require_text(user_id, "user_id")
        user = repository.set_user_blocked(user_id, True)
        if not user:
            return fail(NOT_FOUND, "No CultPass user with that id.")
        return ok(user=_public_user(user))

    @server.tool
    @safe_tool
    def set_unblocked(user_id: str) -> dict:
        """Unblock a customer's account. Safe to repeat: an already unblocked account stays so.

        This takes effect immediately and restores the customer's access.

        Args:
            user_id: The CultPass user id.

        Returns:
            {"ok": True, "user": {"user_id", "full_name", "email", "is_blocked",
             "created_at"}} with `is_blocked` false, or
            {"ok": False, "error": "not_found" | "invalid_input", ...}.
        """
        user_id = require_text(user_id, "user_id")
        user = repository.set_user_blocked(user_id, False)
        if not user:
            return fail(NOT_FOUND, "No CultPass user with that id.")
        return ok(user=_public_user(user))

    return server
