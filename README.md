# UDA-Hub — Autonomous Knowledge Agent

A LangGraph multi-agent system that resolves customer-support tickets end to end: it asks for missing
details, classifies the issue, opens a ticket, answers from a knowledge base, escalates to a human when it
should, and remembers the outcome for the next conversation.

This is the capstone project for the Udacity *Building Agents* course. It is a learning project, not a
production service: the customer system (**CultPass**, a cultural-experiences subscription) is fictional and
the human escalation is a placeholder.

## Why this exists

Most support automation stops at retrieval-augmented chat. This project explores what it takes to act on a
ticket, not just describe it:

- **Decide, then act.** The system chooses between answering, asking a follow-up question, changing an
  account, or handing the case to a human.
- **Keep the record straight.** Ticket ids and user ids come from tool results, never from model prose, so
  the agent cannot invent a ticket that was never created.
- **Stay grounded.** Answers must cite a knowledge-base article. Unsupported text is discarded before it
  reaches the customer.
- **Remember.** Short-term state, restarts, and cross-conversation customer memory are handled separately.

## How it works

The graph is a **supervisor hub-and-spoke** design, but the supervisor is a router written in code. Every
spoke returns to the hub (`orchestrator`), and the hub decides the next step from the saved state. Code owns
the order of steps; the model owns classification, retrieval, understanding, and writing.

```
customer message
      |
  intake ──► classify ──► account lookup ──► recall memory ──► create/find ticket
                                                                     |
                                              ┌──────────────────────┴──────────────┐
                                        knowledge base answer                 escalate to human
                                              └──────────────────────┬──────────────┘
                                                                     ▼
                                                       reply ──► confirm ──► remember outcome
```

Five specialist agents sit behind the spokes, each with its own prompt and only the tools it needs:

| Agent | Job | Tools |
|---|---|---|
| **Classifier** | Category, urgency, complexity | none |
| **Account Operations** | Find the customer, read subscriptions/reservations, block or unblock | CultPass lookups |
| **Service Desk** | Create/update the ticket, add messages | UDA-Hub ticket tools |
| **Resolver** | Answer from the knowledge base (RAG) | `get_knowledge_articles` |
| **Escalation** | Hand the ticket to a human, exactly once | `escalate_to_human` |

Data is reached only through two in-memory **FastMCP** servers (`cultpass` and `udahub`, 19 tools total).
Every tool returns `{"ok": true, ...}` or `{"ok": false, "error": ..., "message": ...}` and never raises.

For the full design — diagrams, state fields, alternatives that were rejected, and what each agent must not
do — see [`solution/agentic/design/architecture.md`](solution/agentic/design/architecture.md).

## Repository layout

```
solution/
├── 01_external_db_setup.ipynb   # builds CultPass: users, experiences, subscriptions, reservations
├── 02_core_db_setup.ipynb       # builds UDA-Hub: account, knowledge base, ticket, memory + vector index
├── 03_agentic_app.ipynb         # builds the agents, runs scenarios, memory demos, logs
├── agentic/
│   ├── workflow.py              # the LangGraph hub-and-spoke orchestrator
│   ├── agents/                  # specialist agents + tool bindings (registry.py)
│   ├── mcp_servers/             # CultPass and UDA-Hub MCP servers + repositories
│   ├── checkpointing.py         # SQLite conversation checkpointer
│   ├── observability.py         # JSON event log and metrics
│   └── design/                  # architecture doc and diagrams
├── data/
│   ├── external/                # CultPass seed data (JSONL) and cultpass.db
│   └── core/                    # udahub.db and checkpoints.db
└── logs/agent_events.jsonl      # generated event log
```

`*.db` files and `logs/` are generated and git-ignored; the notebooks rebuild them from the JSONL seeds.

## Getting started

### Prerequisites

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/) (recommended, `uv.lock` is included) or plain `pip`
- An OpenAI-compatible API key. The default `.env.example` points at the Vocareum sandbox endpoint used by
  the course; any OpenAI-compatible endpoint with chat and embedding models works.

### 1. Install

```bash
git clone <this-repo>
cd autonomous-knowledge-agent
uv sync
# without uv:
# python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Then set your key in `.env`:

```dotenv
OPENAI_BASE_URL=https://openai.vocareum.com/v1
VOCAREUM_API_KEY=your-key-here
```

`.env` is git-ignored. Do not commit real keys.

### 3. Build the databases

Start Jupyter **from `solution/`** — the notebooks use paths relative to that directory.

```bash
cd solution
uv run jupyter lab
```

Run the notebooks in order:

1. **`01_external_db_setup.ipynb`** — creates `data/external/cultpass.db` and seeds users, experiences,
   subscriptions and reservations.
2. **`02_core_db_setup.ipynb`** — creates `data/core/udahub.db` (account, knowledge articles, ticket tables,
   long-term customer memory) and embeds the articles into `sqlite-vec`. The embedding cell calls the API, so
   `.env` must be configured first. Re-run that cell whenever the knowledge base changes.
3. **`03_agentic_app.ipynb`** — mounts the MCP servers, builds the agents and orchestrator, then runs the
   end-to-end scenarios.

Both setup notebooks are idempotent: they delete and recreate their database at the start.

### 4. Talk to it

In notebook 03, run the cells through `await chat_interface(orchestrator, "1")` and type as a customer.
The second argument is the conversation's `thread_id`: reuse it to continue a conversation, use a new one to
start fresh.

Try something like:

> Hi, my app crashes when I open my tickets. It started this morning. I'm Bob Stone, bob.stone@granite.com.

The agent will ask for anything missing, open a ticket, search the knowledge base, and either resolve or
escalate.

### 5. Verify

Notebook 03 is also the test suite. It checks:

- six scripted ticket scenarios against expected outcomes (phase, escalation, ticket status in the database,
  escalation reason);
- short-term, persistent, and long-term memory behaviour, including a returning customer and a recurring
  issue that should escalate;
- that the event log contains no customer names or emails.

There is no standalone `pytest` suite; `pytest` is available if you want to extract the notebook checks.

## Memory

| Kind | Where it lives | What it does |
|---|---|---|
| Short-term | LangGraph state per `thread_id` | Keeps context across messages in one conversation |
| Persistent | `data/core/checkpoints.db` | Survives kernel restarts, so a conversation can continue |
| Long-term | `customer_memory` table in `udahub.db` | Stores preferences and resolved issues, recalled when the customer returns |

Long-term memory also feeds decisions: a customer whose same issue was resolved repeatedly is escalated
instead of answered again.

## Logs and metrics

Every routing decision, classification, knowledge-base search, tool call and outcome is written as one JSON
line to `solution/logs/agent_events.jsonl`. `agentic.observability.summarize_logs()` aggregates resolution
rate, escalation reasons, tool usage (including errors and not-found results), and per-step latency. The log
is designed to hold no customer PII.

## Known limitations

- Human escalation is a stub that reports "Human notified"; no real ticketing or notification integration.
- The MCP servers run in-process with SQLite; there is no authentication, retry policy, or concurrency control.
- The model endpoint and model name (`gpt-4o-mini`) are fixed in `agentic/agents/agent.py` and
  `agentic/workflow.py`; override them through the `model` argument if you switch providers.
- Conversation state and the support database are separate SQLite files; resetting one does not reset the other.

## Built with

- [LangGraph](https://langchain-ai.github.io/langgraph/) — stateful graph orchestration and checkpointing
- [LangChain](https://python.langchain.com/) — ReAct agents, structured output, OpenAI adapters
- [FastMCP](https://gofastmcp.com/) + `langchain-mcp-adapters` — in-memory tool servers
- [SQLAlchemy](https://www.sqlalchemy.org/) + [sqlite-vec](https://github.com/asg017/sqlite-vec) — data access
  and vector search
- [Jupyter](https://jupyter.org/) — setup, demo and verification notebooks
