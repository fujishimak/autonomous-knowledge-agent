import uuid
from typing import Literal, get_args

import utils as u
from data.models import udahub
from sqlalchemy import select
from sqlalchemy.orm import contains_eager, selectinload

DEFAULT_ACCOUNT_ID = "cultpass"

TicketStatus = Literal["open", "in_progress", "resolved", "escalated"]
TICKET_STATUSES: tuple[str, ...] = get_args(TicketStatus)

MessageRole = Literal["user", "agent", "ai", "system"]
MESSAGE_ROLES: tuple[str, ...] = get_args(MessageRole)


class UdaHubRepository:
    def __init__(self, engine):
        self.engine = engine

    # USER

    def get_user_by_id(self, user_id: str) -> dict:
        with u.get_session(self.engine) as session:
            user = session.get(udahub.User, user_id)
            if user is None:
                return {}
            return u.model_to_dict(user)

    def get_user_by_external_id(
        self, external_user_id: str, account_id: str = DEFAULT_ACCOUNT_ID
    ) -> dict:
        """Find the support-side user for a customer id from the client system (e.g. CultPass).

        external_user_id is only unique within an account, so the account is part of the lookup.
        """
        with u.get_session(self.engine) as session:
            stmt = select(udahub.User).where(
                udahub.User.account_id == account_id,
                udahub.User.external_user_id == external_user_id,
            )
            user = session.execute(stmt).scalars().first()
            if user is None:
                return {}
            return u.model_to_dict(user)

    # TICKET HISTORY

    def get_ticket_history(self, user_id: str, max_results: int = 5) -> list:
        """The user's most recent tickets (newest first), with status and issue type."""
        with u.get_session(self.engine) as session:
            stmt = (
                select(
                    udahub.Ticket.ticket_id,
                    udahub.Ticket.channel,
                    udahub.Ticket.created_at,
                    udahub.TicketMetadata.status,
                    udahub.TicketMetadata.main_issue_type,
                    udahub.TicketMetadata.tags,
                )
                # outer join: a ticket without a metadata row still appears in the history
                .outerjoin(
                    udahub.TicketMetadata,
                    udahub.TicketMetadata.ticket_id == udahub.Ticket.ticket_id,
                )
                .where(udahub.Ticket.user_id == user_id)
                .order_by(udahub.Ticket.created_at.desc())
                .limit(max_results)
            )
            return [dict(r) for r in session.execute(stmt).mappings().all()]

    # OPEN TICKETS

    def get_open_tickets_for_user(self, user_id: str, max_results: int = 10) -> list:
        """The user's open tickets (newest first), each with its messages in time order."""
        with u.get_session(self.engine) as session:
            stmt = (
                select(udahub.Ticket)
                .join(udahub.Ticket.ticket_metadata)
                .where(
                    udahub.Ticket.user_id == user_id,
                    udahub.TicketMetadata.status == "open",
                )
                .order_by(udahub.Ticket.created_at.desc())
                .limit(max_results)
                .options(
                    contains_eager(udahub.Ticket.ticket_metadata),
                    selectinload(udahub.Ticket.messages),
                )
            )
            tickets = session.execute(stmt).scalars().all()

            # Build plain dicts inside the session; ORM objects can't be used after it closes.
            return [
                {
                    "ticket_id": t.ticket_id,
                    "channel": t.channel,
                    "created_at": t.created_at,
                    "status": t.ticket_metadata.status,
                    "main_issue_type": t.ticket_metadata.main_issue_type,
                    "tags": t.ticket_metadata.tags,
                    "messages": [
                        {
                            "message_id": m.message_id,
                            "role": m.role.value,
                            "content": m.content,
                            "created_at": m.created_at,
                        }
                        for m in sorted(t.messages, key=lambda m: m.created_at)
                    ],
                }
                for t in tickets
            ]

    # TICKET WRITES

    def create_ticket(
        self,
        external_user_id: str,
        content: str,
        channel: str = "chat",
        user_name: str | None = None,
        tags: str | None = None,
        account_id: str = DEFAULT_ACCOUNT_ID,
    ) -> dict:
        """Open a ticket with its first customer message, in one transaction.

        The support-side user is found by (account_id, external_user_id) and created when
        missing, which needs `user_name`. Returns the new ticket as stored.
        """
        with u.get_session(self.engine) as session:
            user = session.execute(
                select(udahub.User).where(
                    udahub.User.account_id == account_id,
                    udahub.User.external_user_id == external_user_id,
                )
            ).scalars().first()
            if user is None:
                if not user_name:
                    raise ValueError("user_name is required to create a new support user")
                user = udahub.User(
                    user_id=str(uuid.uuid4()),
                    account_id=account_id,
                    external_user_id=external_user_id,
                    user_name=user_name,
                )
                session.add(user)

            ticket = udahub.Ticket(
                ticket_id=str(uuid.uuid4()),
                account_id=account_id,
                user_id=user.user_id,
                channel=channel,
            )
            metadata = udahub.TicketMetadata(
                ticket_id=ticket.ticket_id, status="open", tags=tags
            )
            first_message = udahub.TicketMessage(
                message_id=str(uuid.uuid4()),
                ticket_id=ticket.ticket_id,
                role=udahub.RoleEnum.user,
                content=content,
            )
            session.add_all([ticket, metadata, first_message])
            session.flush()  # insert rows so database defaults (created_at) are available
            session.refresh(ticket)
            return {
                "ticket_id": ticket.ticket_id,
                "user_id": ticket.user_id,
                "account_id": ticket.account_id,
                "channel": ticket.channel,
                "status": metadata.status,
                "tags": metadata.tags,
                "created_at": ticket.created_at,
            }

    def update_ticket_status(self, ticket_id: str, status: str) -> dict:
        """Set a ticket's status. Idempotent. Returns {} if the ticket does not exist."""
        if status not in TICKET_STATUSES:
            raise ValueError(f"status must be one of {TICKET_STATUSES}, got {status!r}")
        with u.get_session(self.engine) as session:
            if session.get(udahub.Ticket, ticket_id) is None:
                return {}
            metadata = session.get(udahub.TicketMetadata, ticket_id)
            if metadata is None:  # ticket without a metadata row: create it
                metadata = udahub.TicketMetadata(ticket_id=ticket_id, status=status)
                session.add(metadata)
                previous = None
            else:
                previous = metadata.status
                if previous != status:
                    metadata.status = status
            session.flush()
            session.refresh(metadata)
            return {
                "ticket_id": ticket_id,
                "status": metadata.status,
                "previous_status": previous,
                "updated_at": metadata.updated_at,
            }

    def add_ticket_message(self, ticket_id: str, role: str, content: str) -> dict:
        """Append a message to a ticket's thread. Returns {} if the ticket does not exist."""
        if role not in MESSAGE_ROLES:
            raise ValueError(f"role must be one of {MESSAGE_ROLES}, got {role!r}")
        with u.get_session(self.engine) as session:
            if session.get(udahub.Ticket, ticket_id) is None:
                return {}
            message = udahub.TicketMessage(
                message_id=str(uuid.uuid4()),
                ticket_id=ticket_id,
                role=udahub.RoleEnum(role),
                content=content,
            )
            session.add(message)
            session.flush()  # insert so the database default (created_at) is available
            session.refresh(message)
            return {
                "message_id": message.message_id,
                "ticket_id": message.ticket_id,
                "role": message.role.value,
                "content": message.content,
                "created_at": message.created_at,
            }
