from typing import Literal

from agentic.agents.agent import Agent
from pydantic import BaseModel

#   classifier_agent
#       input : ticket text + metadata (channel, created_at, user history)
#       output: category, urgency, complexity   (structured output via a Pydantic schema)
#       tools : none (optionally long-term memory lookup)


class Classification(BaseModel):
    category: Literal["technical", "billing", "account", "reservation", "subscription", "other"]
    urgency: Literal["low", "medium", "high"]
    complexity: Literal["simple", "complex"]


class ClassifierAgent(Agent):
    def __init__(self, model: str = "gpt-4o-mini", tools: list | None = None):
        name = "ClassifierAgent"
        instructions = """
        You classify customer support tickets for CultPass, a cultural experiences subscription.
        You are given the customer's description of the problem and return exactly three labels.

        category:
        - technical: the app crashes, will not open, is slow, or notifications misbehave
        - billing: charges, payments, refunds, invoices
        - account: login, password, profile, privacy, a blocked account
        - reservation: booking, cancelling or attending an experience
        - subscription: plan, tier, pausing or cancelling a subscription, what is included
        - other: anything else, including feedback and questions unrelated to CultPass

        urgency:
        - high: the customer cannot log in or use the service at all, money was taken wrongly,
          a security problem, or an event they booked is imminent
        - medium: something is broken but there is a workaround, or it blocks a paid feature
        - low: a question, feedback, or a cosmetic problem

        complexity (independent of urgency: an urgent problem can still be simple):
        - simple: a common problem with a standard fix or a plain answer, such as the app
          crashing or not opening, login or password trouble, how-to and policy questions,
          what a subscription includes, how to book or cancel
        - complex: needs a person to investigate or decide, such as a charge or refund dispute,
          a double charge, a suspected security problem, a blocked account, several unrelated
          problems at once, or a problem that persists after the standard fixes

        Base the labels only on what the customer wrote. Pick the single best fit.
        """
        super().__init__(name, instructions, model, tools, response_format=Classification)
