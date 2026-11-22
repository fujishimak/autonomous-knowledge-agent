# Building Agents Project: Autonomous Knowledge Agent — Rubric

# Project Overview

You’ve joined a fast-growing AI startup building the next frontier in customer support automation.

Your team is responsible for building **UDA-Hub**, a **Universal Decision Agent** designed to plug into existing customer support systems such as **Zendesk, Intercom, Freshdesk, and internal CRMs** and intelligently resolve tickets. But this isn’t just another FAQ bot.

The goal is to build an agentic system that **reads, reasons, routes, and resolves**, acting as the operational brain behind support teams.

Your agent system should be able to:

- Understand customer tickets across channels
- Decide which agent or tool should handle each case
- Retrieve or infer answers when possible
- Escalate or summarize issues when necessary
- Learn from interactions by updating long-term memory

Your agent should not only automate — it should decide how to automate!

## Inputs

- Incoming support ticket (text + metadata)
- Internal knowledge base (FAQ, previous tickets)
- Optional internal tool (for example, refund handling)
- Memory store (for prior conversations and resolutions)

## Deliverables

A **LangGraph-powered multi-agent system** that:

- Understands tickets
- Routes to the correct agent with tools
- Resolves or escalates based on decision logic
- Uses memory appropriately

## Key Capabilities

1. **Multi-Agent Architecture with LangGraph**
   - Design and orchestrate specialized agents such as Supervisor, Classifier, Resolver, and Escalation agents

2. **Input Handling**
   - Accept incoming support tickets in natural language with metadata such as platform, urgency, and history

3. **Decision Routing and Resolution**
   - Route tickets to the right agent based on classification
   - Retrieve relevant knowledge via RAG if needed
   - Resolve or escalate based on confidence and context

4. **Memory Integration**
   - Maintain state during execution steps
   - Use short-term memory to preserve context during the same session
   - Store and recall long-term memory, such as customer preferences


## 1. Data Setup and Knowledge Base Preparation

### Criteria
Set up the database and knowledge base infrastructure.

### To pass
- Successfully set up the database infrastructure and populate the knowledge base with comprehensive support articles
- Successfully run the database management notebook to initialize the databases
- Database contains the required tables:
  - Account
  - User
  - Ticket
  - TicketMetadata
  - TicketMessage
  - Knowledge
- Knowledge base includes at least 10 additional support articles beyond the provided 4
- New articles cover different categories:
  - technical issues
  - billing
  - account management
  - etc.
- All database operations complete without errors
- Can demonstrate successful data retrieval from the database

### Reviewer tips
- Check that the student has executed all cells in the database management notebook and that the database files are properly created
- Verify that the knowledge base contains at least 14 total articles (4 original + 10 additional)
- The additional articles should be relevant to the support domain and demonstrate variety in topics

---

## 2. Multi-Agent Architecture with LangGraph

### Criteria
Design and document multi-agent architecture.

### To pass
- Design and document a comprehensive multi-agent architecture before implementation
- Submit a detailed architecture design document in Markdown format
- Include a visual diagram showing the multi-agent architecture
  - ASCII art, Mermaid, or similar is acceptable
- Document the role and responsibilities of each agent in the system
- Explain the flow of information and decision-making between agents
- Describe how the system handles different types of inputs and expected outputs
- Architecture should be based on one of the standard patterns:
  - Supervisor
  - Hierarchical
  - Network
  - etc.

### Reviewer tips
- The design document should be comprehensive and show clear understanding of multi-agent patterns
- Look for logical agent separation, clear communication flows, and appropriate handling of different ticket types
- The diagram should be clear and match the described architecture
- Students should explain their architectural choices and how they support the project goals

### Criteria
Implement the designed multi-agent architecture using LangGraph.

### To pass
- Implement the designed multi-agent architecture using LangGraph with specialized agents for different tasks
- Implementation matches the documented architecture design
- Project includes at least 4 specialized agents
- Each agent has a clearly defined role and responsibility as documented
- Agents are properly connected using LangGraph's graph structure
- Code demonstrates proper agent state management and message passing

### Reviewer tips
- Compare the implementation to the documented design
- Look for clear separation of concerns between agents
- Each agent should have a specific purpose:
  - classification
  - resolution
  - escalation
  - etc.
- The graph structure should show logical flow between agents
- Check that agents can communicate and pass state appropriately

### Criteria
Implement task routing and role assignment across agents.

### To pass
- Implement intelligent task routing and role assignment across agents based on ticket characteristics
- System can classify incoming tickets and route them to appropriate agents
- Routing logic considers ticket content and metadata:
  - date
  - urgency
  - complexity
  - etc.
- At least one routing decision is made based on ticket classification
- Code includes routing logic that can be demonstrated with sample tickets
- Routing follows the architecture design principles

### Reviewer tips
- The routing should be more sophisticated than simple if/else statements
- Look for classification logic that analyzes ticket content and metadata to make routing decisions
- Students should demonstrate routing with different types of tickets
- The routing should align with the documented architecture

---

## 3. Knowledge Retrieval and Tool Usage

### Criteria
Implement knowledge-based response system with escalation logic.

### To pass
- Implement a knowledge retrieval system that provides responses based on articles and escalates when no relevant knowledge is found
- System retrieves relevant knowledge base articles based on ticket content
- All responses are based on the content of knowledge base articles
- System can demonstrate retrieval of appropriate articles for different ticket types
- Implements escalation logic when no relevant knowledge base article is found
- System includes confidence scoring to determine when to escalate
- Can demonstrate both successful knowledge retrieval and escalation scenarios

### Reviewer tips
- The system should use semantic search or similar techniques to find relevant articles
- Look for proper implementation of RAG (Retrieval-Augmented Generation) where responses are grounded in the knowledge base content
- The escalation logic should trigger when confidence is low or no relevant articles are found
- Students should demonstrate both scenarios:
  - successful resolution using knowledge base
  - escalation when appropriate

### Criteria
Implement support operation tools with database abstraction.

### To pass
- Create and implement at least 2 tools that perform support operations with proper database abstraction
- Implement at least 2 functional tools for support operations:
  - account lookup
  - subscription management
  - refund processing
  - etc.
- Tools abstract the interaction with the CultPass database
- Tools can be invoked by agents and return structured responses
- Tools include proper error handling and validation
- Can demonstrate tool usage with sample operations
- Tools are properly integrated into the agent workflow

### Reviewer tips
- Look for tools that encapsulate database operations and provide a clean interface for agents
- Tools should handle common support scenarios like:
  - looking up user accounts
  - checking subscription status
  - processing refunds
- The tools should abstract away the complexity of direct database access and provide meaningful responses to agents
- Check that tools are properly integrated into the agent decision-making process

---

## 4. Memory and State Management

### Criteria
Persist customer interaction history to enable personalized, context-aware support.

### To pass
- Implement persistent memory to store and retrieve customer interaction history
- System stores conversation history in a persistent database
- Can retrieve previous interactions for returning customers
- Uses historical context to provide personalized responses
- Demonstrates memory retrieval with sample customer interactions

### Reviewer tips
- Check that the database schema includes tables for storing conversation history
- Check that the system can query previous interactions
- Look for code that retrieves and uses historical context when processing new tickets

### Criteria
Implement state, session and long-term memory in agent workflows.

### To pass
- Implement different types of memory in agent workflows
- Agents maintain state during multi-step interactions in one execution
- Based on the appropriate scope (like `thread_id` or `session_id`), it's possible to inspect the workflow:
  - messages
  - tool usage
- Short-term memory is used as context to keep conversation running during the same session
- Long-term memory is used to store resolved issues and customer preferences across different sessions
- Memory is properly integrated into agent decision-making

### Reviewer tips
- Look for LangGraph checkpointing and/or memory features being used
- The system should maintain context within a conversation (session / short-term memory)
- The system should also store information for future reference (long-term memory / across sessions)
- Check that memory is actually used in agent decisions

---

## 5. Integration and Testing

### Criteria
Demonstrate end-to-end ticket processing workflow with proper logging.

### To pass
- Demonstrate a complete end-to-end workflow for processing customer support tickets
- System can process a ticket from initial submission to resolution/escalation
- Workflow includes:
  - classification
  - routing
  - knowledge retrieval
  - tool usage
  - resolution attempt
  - final action
- Demonstrates the complete flow with sample tickets
- Includes proper error handling and edge cases
- System logs agent decisions, routing choices, tool usage, and outcomes
- All logs are structured and searchable
- Shows both successful resolution and escalation scenarios
- Demonstrates tool integration in the workflow

### Reviewer tips
- Ensure that the learner included tests for the system with at least four different types of tickets:
  - simple FAQ questions
  - complex technical issues
  - urgent problems
  - etc.
- The workflow should be smooth and logical, with appropriate handling of different scenarios
- Verify that the system can handle both:
  - cases where knowledge is available
  - cases where escalation is needed
- Check that tools are properly invoked when needed
- Also look for structured logging that captures:
  - agent decisions
  - routing choices
  - outcomes
- The logs should be useful for debugging and improving the system
- Check that logging doesn't expose sensitive information
- The system should track:
  - how often knowledge retrieval succeeds vs. when escalation is needed
  - tool usage patterns
