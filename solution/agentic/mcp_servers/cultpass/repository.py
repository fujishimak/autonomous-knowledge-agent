from pathlib import Path

import utils as u
from data.models import cultpass
from sqlalchemy import Engine, create_engine, func, select

# solution/agentic/mcp_servers/cultpass/repository.py -> parents[3] == solution/
CULTPASS_DB_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "external" / "cultpass.db"
)


def create_cultpass_engine(
    db_path: str | Path | None = None, echo: bool = False
) -> Engine:
    """Engine for cultpass.db. Defaults to solution/data/external/cultpass.db."""
    path = Path(db_path) if db_path else CULTPASS_DB_PATH
    return create_engine(f"sqlite:///{path}", echo=echo)


class CultpassRepository:
    def __init__(self, engine):
        self.engine = engine

    # USER

    def get_user_by_id(self, user_id: str) -> dict:
        with u.get_session(self.engine) as session:
            user = session.get(cultpass.User, user_id)
            if user is None:
                return {}
            return u.model_to_dict(user)

    def get_user_by_name(self, name: str) -> dict:
        with u.get_session(self.engine) as session:
            stmt = select(cultpass.User).where(cultpass.User.full_name == name)
            # Names are not unique, so take the first match instead of requiring exactly one.
            user = session.execute(stmt).scalars().first()
            if user is None:
                return {}
            return u.model_to_dict(user)

    def get_user_by_email(self, email: str) -> dict:
        """Find a user by email, ignoring case ({} if none). Emails are unique."""
        with u.get_session(self.engine) as session:
            stmt = select(cultpass.User).where(
                func.lower(cultpass.User.email) == email.strip().lower()
            )
            user = session.execute(stmt).scalars().first()
            if user is None:
                return {}
            return u.model_to_dict(user)

    def set_user_blocked(self, user_id: str, blocked: bool) -> dict:
        """Set `is_blocked` to `blocked` and return the user as stored ({} if no such user).

        Idempotent: if the user is already in that state nothing is written.
        """
        with u.get_session(self.engine) as session:
            user = session.get(cultpass.User, user_id)
            if user is None:
                return {}
            if bool(user.is_blocked) != blocked:  # None (unset) counts as not blocked
                user.is_blocked = blocked
                session.flush()  # apply the UPDATE (and onupdate timestamp) before reading back
                session.refresh(user)
            return u.model_to_dict(user)

    # RESERVATIONS

    def get_reservations(self, user_id: str) -> list:
        with u.get_session(self.engine) as session:
            stmt = (
                select(cultpass.Reservation)
                .where(cultpass.Reservation.user_id == user_id)
                .order_by(cultpass.Reservation.created_at.desc())
            )
            reservations = session.execute(stmt).scalars().all()
            return [u.model_to_dict(r) for r in reservations]

    # SUBSCRIPTIONS

    def get_subscriptions(self, user_id: str) -> list:
        with u.get_session(self.engine) as session:
            stmt = select(cultpass.Subscription).where(
                cultpass.Subscription.user_id == user_id
            )
            subscriptions = session.execute(stmt).scalars().all()
            return [u.model_to_dict(s) for s in subscriptions]

    # EXPERIENCES

    def get_experience(self, experience_id: str) -> dict:
        with u.get_session(self.engine) as session:
            experience = session.get(cultpass.Experience, experience_id)
            if experience is None:
                return {}
            return u.model_to_dict(experience)
