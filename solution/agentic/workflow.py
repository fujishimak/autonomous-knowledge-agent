"""The support orchestrator: a hub-and-spoke LangGraph over the specialist agents.

    START -> orchestrator (hub) -> one specialist -> orchestrator -> ... -> deliver -> END

The hub is a router. It looks at the saved state and picks the next piece of work, so the
order of steps is decided by code, and each specialist gets an explicit task built from state.
Ids (user_id, ticket_id) always come from tool results, never from an agent's wording.

Conversation phases (state["phase"]) across the customer's turns:
    intake                 collecting the four intake answers
    processing             classify -> find account -> create ticket -> resolve / escalate -> reply
    awaiting_confirmation  the fix was sent, waiting for "did that work?"
    escalated              a human has the ticket; later messages are added to it
    closed                 resolved; a new message starts a new ticket

Use:
    orchestrator = await build_orchestrator(toolbox)            # toolbox must be connected
    config = {"configurable": {"thread_id": "1"}}               # one thread per conversation
    result = await orchestrator.ainvoke({"messages": [HumanMessage("hi")]}, config)
    print(result["messages"][-1].content)
"""

import logging
import re
from typing import Annotated, Literal, TypedDict

from agentic.agents.agent import build_llm
from agentic.agents.registry import build_agents
from agentic.agents.resolver_agent import KB_CONFIDENCE_THRESHOLD
from agentic.mcp_servers.toolbox import McpToolbox
from agentic.tool_results import ok_results
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

logger = logging.getLogger(__name__)

APOLOGY = (
    "Sorry, something went wrong on our side while handling your request. "
    "Please send your message again in a moment."
)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# One graph run takes about two supersteps per step (hub + specialist).
RECURSION_LIMIT = 60


# --------------------------------------------------------------------------- #
# State and structured outputs
# --------------------------------------------------------------------------- #
class TicketState(TypedDict, total=False):
    messages: Annotated[
        list[AnyMessage], add_messages
    ]  # the customer-facing conversation
    history_start: (
        int  # index in `messages` where the current ticket's conversation begins
    )
    phase: Literal[
        "intake", "processing", "awaiting_confirmation", "escalated", "closed"
    ]
    intake: dict  # issue, when, activity, name, email
    classification: dict | None  # category, urgency, complexity
    user_id: str | None  # CultPass user id, from the account lookup
    customer_name: str | None
    account_blocked: bool
    ticket_id: str | None  # UDA-Hub ticket id, from the create_ticket result
    kb: dict | None  # found_answer, answer, article_titles, top_score
    escalation_reason: str | None
    escalated: bool
    reply: str | None  # text waiting to be sent to the customer


class IntakeExtraction(BaseModel):
    issue: str | None
    when: str | None
    activity: str | None
    name: str | None
    email: str | None


class Confirmation(BaseModel):
    satisfied: Literal["yes", "no", "unclear"]


NEW_TICKET_STATE: dict = {
    "phase": "intake",
    "classification": None,
    "user_id": None,
    "customer_name": None,
    "account_blocked": False,
    "ticket_id": None,
    "kb": None,
    "escalation_reason": None,
    "escalated": False,
    "reply": None,
}

INTAKE_QUESTIONS = {
    "issue": "What is the issue you are experiencing?",
    "when": "When did it happen?",
    "activity": "What were you doing when it happened?",
}


# --------------------------------------------------------------------------- #
# Pure helpers (no LLM, no I/O)
# --------------------------------------------------------------------------- #
def missing_intake(intake: dict) -> list[str]:
    missing = [
        key for key in ("issue", "when", "activity", "name") if not intake.get(key)
    ]
    email = intake.get("email")
    if not email or not EMAIL_PATTERN.match(email):
        missing.append("email")
    return missing


def intake_reply(missing: list[str], first_turn: bool) -> str:
    lines, number = [], 1
    for key in ("issue", "when", "activity"):
        if key in missing:
            lines.append(f"{number}. {INTAKE_QUESTIONS[key]}")
            number += 1
    contact = [
        label
        for key, label in (("name", "your name"), ("email", "your email address"))
        if key in missing
    ]
    if contact:
        lines.append(f"{number}. What is {' and '.join(contact)}?")
    intro = (
        "I'm sorry you're having trouble, and I'd like to help. To get started, could you tell me:"
        if first_turn
        else "Thanks! I just need a little more information:"
    )
    return intro + "\n" + "\n".join(lines)


def kb_is_confident(kb: dict | None) -> bool:
    return bool(
        kb
        and kb.get("found_answer")
        and kb.get("top_score", 0.0) >= KB_CONFIDENCE_THRESHOLD
    )


def escalation_reason(state: TicketState) -> str | None:
    """Why a human must take this ticket, or None. Checked after each step that adds facts."""
    if state.get("escalation_reason"):
        return state["escalation_reason"]
    if state.get("account_blocked"):
        return "The customer's account is blocked."
    classification = state.get("classification") or {}
    if (
        classification.get("urgency") == "high"
        and classification.get("complexity") == "complex"
    ):
        return "Classified as high urgency and complex."
    kb = state.get("kb")
    if kb is not None and not kb_is_confident(kb):
        return (
            "The knowledge base has no confident answer "
            f"(best score {kb.get('top_score', 0.0):.2f}, threshold {KB_CONFIDENCE_THRESHOLD})."
        )
    return None


def route(state: TicketState) -> str:
    """The hub's decision: which node runs next."""
    if state.get("reply"):
        return "deliver"
    phase = state.get("phase") or "intake"
    if phase in ("intake", "closed"):
        return "intake"
    if phase == "awaiting_confirmation":
        return "confirm"
    if phase == "escalated":
        return "followup"

    # phase == "processing": the first missing piece of state decides the step
    if not state.get("classification"):
        return "classify"
    if not state.get("user_id"):
        return "account_lookup"
    if not state.get("ticket_id"):
        return "create_ticket"
    reason = escalation_reason(state)
    # Some reasons are known before the knowledge base is searched (blocked account, customer
    # said the fix failed). Those escalate straight away and skip the search.
    skip_search = bool(state.get("account_blocked") or state.get("escalation_reason"))
    if skip_search and not state.get("escalated"):
        return "escalate"
    if state.get("kb") is None and not skip_search:
        return "resolve"
    # After the search: urgent + complex tickets and weak search results go to a human too.
    if reason and not state.get("escalated"):
        return "escalate"
    return "respond"


def _transcript(messages: list[AnyMessage]) -> str:
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"Customer: {message.content}")
        elif isinstance(message, AIMessage) and message.content:
            lines.append(f"Assistant: {message.content}")
    return "\n".join(lines)


def _pick_ticket_id(messages: list[AnyMessage]) -> str | None:
    """A ticket id the service desk agent really created or found, taken from tool results."""
    created = ok_results(messages, "create_ticket")
    if created:
        return created[0]["ticket"]["ticket_id"]
    final_text = str(messages[-1].content) if messages else ""
    for result in ok_results(messages, "get_open_tickets"):
        for ticket in result.get("tickets", []):
            if ticket["ticket_id"] in final_text:  # the agent chose to reuse this one
                return ticket["ticket_id"]
    return None


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #
def _guard(name: str, node):
    """If a node fails, log it and tell the customer to retry instead of crashing the turn."""

    async def guarded(state: TicketState) -> dict:
        try:
            return await node(state)
        except Exception:
            logger.exception("node %s failed", name)
            return {"reply": APOLOGY}

    return guarded


async def build_orchestrator(
    toolbox: McpToolbox, model: str = "gpt-4o-mini", checkpointer=None
):
    """Compile the orchestrator graph. `toolbox` must already be connected."""
    agents = build_agents(toolbox, model=model)
    llm = build_llm(model)
    extract_intake = llm.with_structured_output(IntakeExtraction)
    judge_confirmation = llm.with_structured_output(Confirmation)

    async def run_service_desk(task: str) -> list[AnyMessage]:
        result = await agents["service_desk_agent"].ainvoke(task)
        return result["messages"]

    async def record_message(
        ticket_id: str, role: str, text: str, also: str = ""
    ) -> list:
        """Ask the service desk to add a message to the ticket (and optionally do more)."""
        return await run_service_desk(
            f'On ticket {ticket_id}: record this message with role "{role}", using exactly this '
            f"text:\n'''{text}'''\n{also}"
        )

    # -- hub ---------------------------------------------------------------- #
    async def orchestrator(state: TicketState) -> dict:
        return {}  # routing happens on the conditional edges, using `route`

    # -- intake (diagram step 1) --------------------------------------------- #
    async def intake(state: TicketState) -> dict:
        updates: dict = {}
        messages = state.get("messages", [])
        intake_data = dict(state.get("intake") or {})
        if state.get("phase") == "closed":
            # a new message after a resolved ticket: new ticket, same customer
            updates.update(NEW_TICKET_STATE)
            updates["history_start"] = max(len(messages) - 1, 0)
            intake_data = {k: intake_data.get(k) for k in ("name", "email")}
        start = updates.get("history_start", state.get("history_start", 0))

        extraction = await extract_intake.ainvoke(
            "Extract details about the customer's support request from their messages. "
            "Use only what the customer wrote. Leave a field null when it was not stated. "
            "If the customer corrected a value, use the newest one.\n"
            "issue: what is wrong, in a short phrase (null if they only said they need help). when: when it happened, as they said. "
            "activity: what they were doing when it happened. name: their name. email: their email.\n\n"
            + _transcript(messages[start:])
        )
        first_turn = not any(intake_data.values())
        for key, value in extraction.model_dump().items():
            if value and value.strip():
                intake_data[key] = value.strip()

        missing = missing_intake(intake_data)
        if missing:
            return {
                **updates,
                "phase": "intake",
                "intake": intake_data,
                "reply": intake_reply(missing, first_turn),
            }
        return {**updates, "phase": "processing", "intake": intake_data}

    # -- classifier (step 2) ------------------------------------------------- #
    async def classify(state: TicketState) -> dict:
        intake_data = state["intake"]
        result = await agents["classifier_agent"].ainvoke(
            "Classify this support request.\n"
            f"Issue: {intake_data['issue']}\n"
            f"When it happened: {intake_data['when']}\n"
            f"What the customer was doing: {intake_data['activity']}"
        )
        return {"classification": result["structured_response"].model_dump()}

    # -- account ops (step 3) ------------------------------------------------ #
    async def account_lookup(state: TicketState) -> dict:
        email = state["intake"]["email"]
        result = await agents["account_ops_agent"].ainvoke(
            f"Find the customer with email {email}. Report their user_id, full name and "
            "whether the account is blocked."
        )
        found = ok_results(result["messages"], "lookup_user_by_email")
        if not found:
            # Unknown email: ask again (the intake step will ask only for the email)
            intake_data = {**state["intake"], "email": None}
            return {
                "phase": "intake",
                "intake": intake_data,
                "reply": (
                    "I couldn't find a CultPass account with that email address. "
                    "Could you double-check it and send it again?"
                ),
            }
        user = found[0]["user"]
        return {
            "user_id": user["user_id"],
            "customer_name": user["full_name"],
            "account_blocked": bool(user["is_blocked"]),
        }

    # -- service desk: create the ticket (step 4) ---------------------------- #
    async def create_ticket(state: TicketState) -> dict:
        intake_data, classification = state["intake"], state["classification"]
        content = (
            f"{intake_data['issue']}\n"
            f"When it happened: {intake_data['when']}\n"
            f"What the customer was doing: {intake_data['activity']}"
        )
        tags = f"{classification['category']}, {classification['urgency']}"
        messages = await run_service_desk(
            "Create a support ticket and then set its status to in_progress.\n"
            f"CultPass user id: {state['user_id']}\n"
            f"Customer name: {state.get('customer_name') or intake_data['name']}\n"
            f"Channel: chat\nTags: {tags}\n"
            f"Customer message:\n'''{content}'''\n"
            "Reply with the ticket_id."
        )
        ticket_id = _pick_ticket_id(messages)
        if not ticket_id:
            logger.error("service desk returned no ticket id")
            return {"reply": APOLOGY}
        return {"ticket_id": ticket_id}

    # -- resolver (step 5) --------------------------------------------------- #
    async def resolve(state: TicketState) -> dict:
        intake_data, classification = state["intake"], state["classification"]
        result = await agents["resolver_agent"].ainvoke(
            "Resolve this customer issue from the knowledge base.\n"
            f"Issue: {intake_data['issue']}\n"
            f"What the customer was doing: {intake_data['activity']}\n"
            f"Category: {classification['category']}"
        )
        resolution = result["structured_response"]
        searches = ok_results(result["messages"], "get_knowledge_articles")
        top_score = max((s.get("top_score", 0.0) for s in searches), default=0.0)
        return {
            "kb": {
                "found_answer": resolution.found_answer,
                "answer": resolution.answer,
                "article_titles": resolution.article_titles,
                "top_score": top_score,
            }
        }

    # -- escalation ----------------------------------------------------------- #
    async def escalate(state: TicketState) -> dict:
        intake_data, classification = state["intake"], state["classification"]
        reason = escalation_reason(state)
        kb = state.get("kb")
        kb_summary = (
            "not searched"
            if kb is None
            else f"found_answer={kb['found_answer']}, top score {kb['top_score']:.2f}, "
            f"articles={kb['article_titles']}"
        )
        result = await agents["escalation_agent"].ainvoke(
            f"Escalate ticket {state['ticket_id']} to a human.\n"
            f"Issue: {intake_data['issue']}\n"
            f"When it happened: {intake_data['when']}\n"
            f"What the customer was doing: {intake_data['activity']}\n"
            f"Classification: {classification}\n"
            f"Knowledge base: {kb_summary}\n"
            f"Reason for escalation: {reason}"
        )
        if not ok_results(result["messages"], "escalate_to_human"):
            logger.error("escalate_to_human did not succeed")
            return {"escalation_reason": reason, "reply": APOLOGY}

        status = await run_service_desk(
            f"Set the status of ticket {state['ticket_id']} to escalated."
        )
        if not any(
            r["ticket"]["status"] == "escalated"
            for r in ok_results(status, "update_ticket")
        ):
            logger.warning("ticket status was not set to escalated")
        return {"escalated": True, "escalation_reason": reason}

    # -- compose the answer for the customer ---------------------------------- #
    async def respond(state: TicketState) -> dict:
        kb = state.get("kb") or {}
        name = (state.get("customer_name") or state["intake"].get("name") or "").split(
            " "
        )[0]
        has_fix = bool(kb.get("found_answer"))
        ticket_id = state["ticket_id"]

        if state.get("escalated") and has_fix:
            situation = (
                "A human colleague has also been notified because the issue is urgent and "
                f"complex. Ticket reference: {ticket_id}.\n"
                f"Knowledge base answer:\n<<<{kb['answer']}>>>\n"
                "Briefly give the steps from that text, then say a colleague will follow up and "
                "mention the ticket reference."
            )
            phase = "escalated"
        elif state.get("escalated"):
            situation = (
                "You could not resolve this automatically, so a human colleague has been "
                f"notified and will follow up. Ticket reference: {ticket_id}.\n"
                "Apologise briefly, reassure them, and mention the ticket reference. "
                "Do not give troubleshooting steps."
            )
            phase = "escalated"
        else:
            situation = (
                f"The knowledge base has this answer for their problem:\n<<<{kb['answer']}>>>\n"
                "Tell them the fix as short numbered steps using only that text, then ask them "
                "to try it and say whether it solved the problem. Do not say goodbye yet."
            )
            phase = "awaiting_confirmation"

        message = await llm.ainvoke(
            "You write the next message from the CultPass support assistant to a customer. "
            "Be warm and concise. Plain text only, no markdown headings or bold. "
            "Never invent steps, policies or promises.\n\n"
            f"Customer first name: {name or 'unknown'}\n{situation}\n\nWrite the message."
        )
        return {"reply": str(message.content).strip(), "phase": phase}

    # -- later turns ---------------------------------------------------------- #
    def _last_customer_text(state: TicketState) -> str:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    async def confirm(state: TicketState) -> dict:
        customer_text = _last_customer_text(state)
        last_reply = next(
            (
                str(m.content)
                for m in reversed(state["messages"])
                if isinstance(m, AIMessage)
            ),
            "",
        )
        verdict = await judge_confirmation.ainvoke(
            "The assistant sent a fix and asked the customer whether it solved their problem.\n"
            f"Assistant: {last_reply}\nCustomer: {customer_text}\n"
            "Did the customer say it is fixed (yes), say it is not fixed (no), or neither "
            "(unclear, for example a new question)?"
        )
        ticket_id = state["ticket_id"]
        if verdict.satisfied == "yes":
            messages = await record_message(
                ticket_id,
                "user",
                customer_text,
                "Then set the ticket status to resolved.",
            )
            if not any(
                r["ticket"]["status"] == "resolved"
                for r in ok_results(messages, "update_ticket")
            ):
                logger.warning("ticket status was not set to resolved")
            return {
                "phase": "closed",
                "reply": "I'm glad that fixed it! Your ticket is now closed. Have a wonderful day!",
            }
        if verdict.satisfied == "no":
            await record_message(ticket_id, "user", customer_text)
            return {
                "phase": "processing",
                "escalation_reason": "The customer says the suggested fix did not work.",
                "kb": {**(state.get("kb") or {}), "found_answer": False},
            }
        await record_message(ticket_id, "user", customer_text)
        answer = await llm.ainvoke(
            "You are the CultPass support assistant. You sent the customer the message below, and "
            "they replied with something other than yes or no.\n"
            f"Your message: {last_reply}\nCustomer: {customer_text}\n"
            "Answer their question briefly, using only what your message says. If it cannot be "
            "answered from your message, say a colleague can help with that. Then ask whether the "
            "steps solved their problem. Plain text, short."
        )
        return {"reply": str(answer.content).strip()}

    async def followup(state: TicketState) -> dict:
        await record_message(state["ticket_id"], "user", _last_customer_text(state))
        return {
            "reply": (
                f"Thank you. I've added that to your ticket ({state['ticket_id']}). "
                "A member of our team will follow up with you."
            )
        }

    # -- send the reply ------------------------------------------------------- #
    async def deliver(state: TicketState) -> dict:
        reply = state["reply"]
        if state.get("ticket_id"):
            try:
                messages = await record_message(state["ticket_id"], "ai", reply)
                if not ok_results(messages, "add_ticket_message"):
                    logger.warning("reply was not recorded on the ticket")
            except Exception:
                logger.exception(
                    "recording the reply failed"
                )  # the customer still gets it
        return {"messages": [AIMessage(content=reply)], "reply": None}

    # -- wire it up ----------------------------------------------------------- #
    specialists = {
        "intake": intake,
        "classify": classify,
        "account_lookup": account_lookup,
        "create_ticket": create_ticket,
        "resolve": resolve,
        "escalate": escalate,
        "respond": respond,
        "confirm": confirm,
        "followup": followup,
    }
    graph = StateGraph(TicketState)
    graph.add_node("orchestrator", orchestrator)
    for name, node in specialists.items():
        graph.add_node(name, _guard(name, node))
        graph.add_edge(name, "orchestrator")  # every spoke reports back to the hub
    graph.add_node("deliver", deliver)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator", route, {name: name for name in [*specialists, "deliver"]}
    )
    graph.add_edge("deliver", END)

    compiled = graph.compile(checkpointer=checkpointer or MemorySaver())
    return compiled.with_config(recursion_limit=RECURSION_LIMIT)
