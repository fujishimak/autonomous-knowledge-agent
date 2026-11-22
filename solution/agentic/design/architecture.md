# CultPass Support Agent: Architecture Design

> TODO (Rubric 2): Complete this document BEFORE implementing the graph.
> Replace every TODO block below with your own content. The implementation in
> `agentic/workflow.py` and `agentic/agents/` must match what you write here.

## 1. Pattern and rationale

TODO: Name the pattern (Supervisor, Hierarchical, Network, ...) and say why it suits
support-ticket handling. Mention at least one alternative you rejected and why.

## 2. Diagram

TODO: Add a Mermaid (or ASCII) diagram that matches the code exactly.

```mermaid
flowchart TD
    %% pseudocode: one node per agent, one edge per possible hand-off
    %% START --> classifier
    %% classifier -->|needs account data| tool_agent
    %% classifier -->|FAQ / technical| resolver
    %% classifier -->|urgent / sensitive| escalation
    %% resolver -->|confidence >= threshold| memory
    %% resolver -->|confidence < threshold or no article| escalation
    %% tool_agent --> resolver
    %% escalation --> memory
    %% memory --> END
```

## 3. Agents (at least 4)

TODO: One subsection per agent with: role, inputs, outputs, tools it may call, and what it must NOT do.

| Agent | Responsibility | Inputs | Outputs | Tools |
|-------|----------------|--------|---------|-------|
| TODO  | TODO           | TODO   | TODO    | TODO  |

## 4. Shared state

TODO: List the fields of the graph state (messages, category, urgency, complexity,
kb_results, confidence, escalated, ...) and which agent writes each one.

## 5. Information flow and routing

TODO: Describe, step by step, how a ticket moves through the graph.
- Which signals drive routing (ticket content, urgency, complexity, date/channel, customer history)?
- What is the confidence threshold for escalation and how is the confidence computed?
- What happens on tool errors, empty retrieval, or an unclassifiable ticket?

## 6. Inputs and expected outputs by ticket type

TODO: A table with at least four ticket types (simple FAQ, technical issue, urgent problem,
no-KB-match) showing the route taken and the expected final output.

| Ticket type | Example | Route | Expected output |
|-------------|---------|-------|-----------------|
| TODO | TODO | TODO | TODO |

## 7. Memory

TODO: Explain short-term memory (thread_id checkpointing), long-term memory (what is stored,
where, when it is read), and how each influences agent decisions.

## 8. Logging and metrics

TODO: List the events logged (classification, routing, retrieval, tool call, escalation, outcome),
the log format, how sensitive data is kept out, and the metrics tracked.
