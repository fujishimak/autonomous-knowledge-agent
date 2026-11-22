# reset_udahub.py
import os
from contextlib import contextmanager

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


def reset_db(db_path: str, echo: bool = True):
    """Drops the existing udahub.db file and recreates all tables."""

    # Remove the file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")

    # Create a new engine and recreate tables
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    print(f"✅ Recreated {db_path} with fresh schema")


def get_connection(db_path: str) -> Engine:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    return engine


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance) -> dict:
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }


async def chat_interface(agent: CompiledStateGraph, ticket_id: str):
    """Talk to the orchestrator in the terminal or a notebook. Use: `await chat_interface(...)`.

    `ticket_id` is the conversation's thread id: the same id continues the same conversation.
    Type quit, exit or q to stop.
    """
    config = {"configurable": {"thread_id": ticket_id}}
    while True:
        user_input = input("User: ")
        print("User:", user_input)
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Assistant: Goodbye!")
            break
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        print("Assistant:", result["messages"][-1].content)


# TODO (Rubric 5): Add a structured logging helper used by workflow.py, agents and tools.
# pseudocode:
#   def get_logger(name) -> logger writing JSON lines to logs/agent_events.jsonl (and optionally stdout)
#   def log_event(logger, ticket_id, agent, event, **fields):
#       drop or mask sensitive keys (email, api keys, raw message text)
#       emit {"ts", "ticket_id", "agent", "event", ...fields}
#   def summarize_logs(path) -> dict with counts of
#       kb_resolved vs escalated, tool usage by name, routes by category
