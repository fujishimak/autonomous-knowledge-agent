"""The support orchestrator: a hub-and-spoke LangGraph over the specialist agents.

    START -> orchestrator (hub) -> one specialist -> orchestrator -> ... -> deliver -> END

The hub is a router. It looks at the saved state and picks the next piece of work, so the
order of steps is decided by code, and each specialist gets an explicit task built from state.
Ids (user_id, ticket_id) always come from tool results, never from an agent's wording.

Conversation phases (state["phase"]) across the customer's turns:
    intake                 collecting the four intake answers
    processing             classify -> find account -> create ticket -> recall memory ->
                           resolve / escalate -> reply
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
import time
from typing import Annotated, Literal, TypedDict

from agentic.agents.agent import build_llm
from agentic.agents.registry import build_agents
from agentic.agents.resolver_agent import KB_CONFIDENCE_THRESHOLD
from agentic.mcp_servers.toolbox import McpToolbox
from agentic.observability import bind_context, log_event, reset_context, tool_outcome
from agentic.tool_results import ok_results, parse_tool_content
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
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

# A customer whose problem in the same category was resolved this many times before and has
# come back again is sent to a human: the standard fix evidently does not last.
REPEAT_ISSUE_THRESHOLD = 2


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
    support_user_id: str | None  # UDA-Hub user id, from the create_ticket result
    memory: dict | None  # recalled: resolved_issues, preferences, previous_tickets
    kb: dict | None  # found_answer, answer, article_titles, top_score
    escalation_reason: str | None
    escalation_code: str | None  # why, as a short code for the logs and metrics
    escalated: bool
    reply: str | None  # text waiting to be sent to the customer


class IntakeExtraction(BaseModel):
    issue: str | None
    when: str | None
    activity: str | None
    name: str | None
    email: str | None


class PreferenceExtraction(BaseModel):
    preferred_contact: str | None
    language: str | None


class Confirmation(BaseModel):
    satisfied: Literal["yes", "no", "unclear"]


NEW_TICKET_STATE: dict = {
    "phase": "intake",
    "classification": None,
    "user_id": None,
    "customer_name": None,
    "account_blocked": False,
    "ticket_id": None,
    "support_user_id": None,
    "memory": None,
    "kb": None,
    "escalation_reason": None,
    "escalation_code": None,
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


def escalation_code(state: TicketState) -> str | None:
    """The short code for `escalation_reason`, used in logs and metrics."""
    if state.get("escalation_code"):
        return state["escalation_code"]
    if state.get("escalation_reason"):
        return "other"
    if state.get("account_blocked"):
        return "account_blocked"
    classification = state.get("classification") or {}
    if classification.get("urgency") == "high" and classification.get("complexity") == "complex":
        return "urgent_complex"
    kb = state.get("kb")
    if kb is not None and not kb_is_confident(kb):
        return "low_confidence"
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
    if state.get("memory") is None:
        return "recall"
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
def _pick_support_user_id(messages: list[AnyMessage]) -> str | None:
    """The UDA-Hub user id of the customer, from the create_ticket result."""
    created = ok_results(messages, "create_ticket")
    return created[0]["ticket"].get("user_id") if created else None


def _memory_prompt(memory: dict | None) -> str:
    """What is known about a returning customer, as text for the reply writer."""
    if not memory:
        return ""
    lines = []
    if memory.get("preferences"):
        prefs = ", ".join(f"{k}: {v}" for k, v in memory["preferences"].items())
        lines.append(f"Customer preferences: {prefs}. Follow them (for example reply in their language).")
    if memory.get("resolved_issues"):
        past = "; ".join(f"{m['key']}: {m['value']}" for m in memory["resolved_issues"][:3])
        lines.append(
            f"Earlier resolved issues: {past}. If one is clearly the same problem, say briefly "
            "that you recognise it. Otherwise do not mention them."
        )
    if memory.get("previous_tickets"):
        lines.append(f"They have contacted support {memory['previous_tickets']} time(s) before.")
    return "\n".join(lines)


def _traced(name: str, node, recover: bool = True, quiet: bool = False):
    """Wrap a node: bind the log context, time it, and log how it ended.

    With `recover`, a failing node is logged and the customer is told to retry instead of the
    whole turn crashing. `quiet` skips the per-run timing event (used for the hub).
    """

    async def traced(state: TicketState, config: RunnableConfig) -> dict:
        thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
        token = bind_context(thread_id=thread_id, ticket_id=state.get("ticket_id"), node=name)
        started = time.perf_counter()
        try:
            result = await node(state)
        except Exception as exc:
            logger.exception("node %s failed", name)
            log_event("node_failed", level="error", error_type=type(exc).__name__)
            if not recover:
                raise
            return {"reply": APOLOGY}
        else:
            if not quiet:
                log_event(
                    "node_completed", duration_ms=round((time.perf_counter() - started) * 1000)
                )
            return result
        finally:
            reset_context(token)

    return traced


async def build_orchestrator(
    toolbox: McpToolbox, model: str = "gpt-4o-mini", checkpointer=None
):
    """Compile the orchestrator graph. `toolbox` must already be connected."""
    agents = build_agents(toolbox, model=model)
    llm = build_llm(model)
    extract_intake = llm.with_structured_output(IntakeExtraction)
    judge_confirmation = llm.with_structured_output(Confirmation)
    extract_preferences = llm.with_structured_output(PreferenceExtraction)

    async def call_tool(name: str, args: dict) -> dict:
        """Call an MCP tool directly (bookkeeping the graph does itself) and log the outcome."""
        payload = parse_tool_content(await toolbox.tool(name).ainvoke(args)) or {}
        outcome = tool_outcome(payload)
        log_event(
            "tool_call",
            agent="orchestrator",
            tool=name,
            outcome=outcome,
            error_code=None if outcome == "ok" else payload.get("error"),
        )
        return payload

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
        # Routing happens on the conditional edges, using `route`. The hub records the choice.
        log_event(
            "routing_decision",
            phase=state.get("phase") or "intake",
            next=route(state),
            escalated=bool(state.get("escalated")),
        )
        return {}

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
        log_event("intake_checked", missing=missing, complete=not missing, first_turn=first_turn)
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
        classification = result["structured_response"].model_dump()
        log_event("classified", **classification)
        return {"classification": classification}

    # -- account ops (step 3) ------------------------------------------------ #
    async def account_lookup(state: TicketState) -> dict:
        email = state["intake"]["email"]
        result = await agents["account_ops_agent"].ainvoke(
            f"Find the customer with email {email}. Report their user_id, full name and "
            "whether the account is blocked."
        )
        found = ok_results(result["messages"], "lookup_user_by_email")
        log_event(
            "account_lookup",
            found=bool(found),
            blocked=bool(found and found[0]["user"]["is_blocked"]),
        )
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
        log_event("ticket_created", ticket_id=ticket_id)
        return {
            "ticket_id": ticket_id,
            "support_user_id": _pick_support_user_id(messages),
        }

    # -- long-term memory: recall (before the answer is worked out) ----------- #
    async def recall(state: TicketState) -> dict:
        memory: dict = {"resolved_issues": [], "preferences": {}, "previous_tickets": 0}
        support_user_id = state.get("support_user_id")
        updates: dict = {}
        if support_user_id:
            try:
                recalled = await call_tool(
                    "get_customer_memory", {"user_id": support_user_id, "max_results": 50}
                )
                history = await call_tool(
                    "get_ticket_history", {"user_id": support_user_id, "max_results": 10}
                )
                for item in recalled.get("memories", []):
                    if item["memory_type"] == "resolved_issue":
                        memory["resolved_issues"].append(item)
                    else:
                        memory["preferences"][item["key"]] = item["value"]
                memory["previous_tickets"] = sum(
                    1
                    for t in history.get("tickets", [])
                    if t["ticket_id"] != state["ticket_id"]
                )
            except Exception:
                logger.exception("recalling customer memory failed")  # the ticket carries on

        category = (state.get("classification") or {}).get("category")
        repeats = [m for m in memory["resolved_issues"] if m["key"] == category]
        if len(repeats) >= REPEAT_ISSUE_THRESHOLD:
            updates["escalation_reason"] = (
                f"Recurring issue: this customer had {len(repeats)} earlier {category} "
                "issues resolved and it came back."
            )
            updates["escalation_code"] = "repeat_issue"
        log_event(
            "memory_recalled",
            resolved_issues=len(memory["resolved_issues"]),
            preferences=len(memory["preferences"]),
            previous_tickets=memory["previous_tickets"],
            repeat_issue=bool(updates),
        )
        return {"memory": memory, **updates}

    # -- long-term memory: remember what was learned once the issue is resolved -- #
    async def remember(state: TicketState, customer_text: str) -> None:
        support_user_id, ticket_id = state.get("support_user_id"), state["ticket_id"]
        if not support_user_id:
            return
        category = (state.get("classification") or {}).get("category", "other")
        kb = state.get("kb") or {}
        fix = ", ".join(kb.get("article_titles") or []) or "standard steps"
        issue = (state.get("intake") or {}).get("issue", "")
        saved = 0
        try:
            result = await call_tool(
                "save_customer_memory",
                {
                    "user_id": support_user_id,
                    "memory_type": "resolved_issue",
                    "key": category,
                    "value": f"{issue} (fixed with: {fix})"[:900],
                    "ticket_id": ticket_id,
                },
            )
            saved += result.get("ok") is True
            start = state.get("history_start", 0)
            customer_lines = "\n".join(
                str(m.content)
                for m in state.get("messages", [])[start:]
                if isinstance(m, HumanMessage)
            )
            found = await extract_preferences.ainvoke(
                "Find stated preferences in what the customer wrote: how they want to be "
                "contacted (preferred_contact, e.g. email, phone, chat) and the language they "
                "want replies in (language). Use null when not stated. Do not guess.\n\n"
                f"{customer_lines}\n{customer_text}"
            )
            for key, value in found.model_dump().items():
                if value and value.strip():
                    result = await call_tool(
                        "save_customer_memory",
                        {
                            "user_id": support_user_id,
                            "memory_type": "preference",
                            "key": key,
                            "value": value.strip(),
                            "ticket_id": ticket_id,
                        },
                    )
                    saved += result.get("ok") is True
        except Exception:
            logger.exception("saving customer memory failed")  # never blocks the reply
        log_event("memory_saved", saved=saved)

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
        kb = {
            "found_answer": resolution.found_answer,
            "answer": resolution.answer,
            "article_titles": resolution.article_titles,
            "top_score": top_score,
        }
        log_event(
            "kb_retrieval",
            top_score=round(top_score, 3),
            found_answer=kb["found_answer"],
            confident=kb_is_confident(kb),
            articles_used=len(kb["article_titles"]),
        )
        return {"kb": kb}

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
            log_event("escalation_failed", level="error", reason_code=escalation_code(state))
            return {"escalation_reason": reason, "reply": APOLOGY}

        status = await run_service_desk(
            f"Set the status of ticket {state['ticket_id']} to escalated."
        )
        if not any(
            r["ticket"]["status"] == "escalated"
            for r in ok_results(status, "update_ticket")
        ):
            logger.warning("ticket status was not set to escalated")
        code = escalation_code(state)
        log_event("escalated", ticket_id=state["ticket_id"], reason_code=code)
        return {"escalated": True, "escalation_reason": reason, "escalation_code": code}

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
            f"Customer first name: {name or 'unknown'}\n"
            f"{_memory_prompt(state.get('memory'))}\n{situation}\n\nWrite the message."
        )
        memory = state.get("memory") or {}
        log_event(
            "reply_composed",
            outcome="escalated" if state.get("escalated") else "fix_offered",
            has_fix=has_fix,
            personalised=bool(memory.get("preferences") or memory.get("resolved_issues")),
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
        log_event("confirmation", verdict=verdict.satisfied)
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
            log_event(
                "ticket_resolved",
                ticket_id=ticket_id,
                category=(state.get("classification") or {}).get("category"),
                resolved_via="knowledge_base",
            )
            await remember(state, customer_text)
            return {
                "phase": "closed",
                "reply": "I'm glad that fixed it! Your ticket is now closed. Have a wonderful day!",
            }
        if verdict.satisfied == "no":
            await record_message(ticket_id, "user", customer_text)
            return {
                "phase": "processing",
                "escalation_reason": "The customer says the suggested fix did not work.",
                "escalation_code": "fix_failed",
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
        log_event("followup_recorded", ticket_id=state["ticket_id"])
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
        log_event("turn_completed", reply_length=len(reply), phase=state.get("phase"))
        return {"messages": [AIMessage(content=reply)], "reply": None}

    # -- wire it up ----------------------------------------------------------- #
    specialists = {
        "intake": intake,
        "classify": classify,
        "account_lookup": account_lookup,
        "create_ticket": create_ticket,
        "recall": recall,
        "resolve": resolve,
        "escalate": escalate,
        "respond": respond,
        "confirm": confirm,
        "followup": followup,
    }
    graph = StateGraph(TicketState)
    graph.add_node("orchestrator", _traced("orchestrator", orchestrator, recover=False, quiet=True))
    for name, node in specialists.items():
        graph.add_node(name, _traced(name, node))
        graph.add_edge(name, "orchestrator")  # every spoke reports back to the hub
    graph.add_node("deliver", _traced("deliver", deliver, recover=False))
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator", route, {name: name for name in [*specialists, "deliver"]}
    )
    graph.add_edge("deliver", END)

    compiled = graph.compile(checkpointer=checkpointer or MemorySaver())
    return compiled.with_config(recursion_limit=RECURSION_LIMIT)
