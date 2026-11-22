from agentic.agents.agent import TOOL_USE_RULES, Agent
from pydantic import BaseModel, model_validator

#   resolver_agent  (knowledge / RAG)
#       input : ticket text + category
#       does  : call the knowledge search tool, answer ONLY from returned articles, cite article titles
#       output: draft answer + confidence score

# Minimum article `score` (cosine similarity, 0..1) for an article to count as relevant.
# Measured on the 15 current articles: relevant questions scored about 0.34 to 0.55,
# an off-topic one about 0.10. Calibrate again with the real test tickets.
KB_CONFIDENCE_THRESHOLD = 0.32


NO_ANSWER_TEXT = "No relevant knowledge base article was found."


class Resolution(BaseModel):
    found_answer: bool
    answer: str
    article_titles: list[str]

    @model_validator(mode="after")
    def _enforce_grounding(self) -> "Resolution":
        # An answer must cite at least one article. Without one, or when nothing was found,
        # drop whatever text the model wrote so ungrounded content can never be sent on.
        if not self.found_answer or not self.article_titles:
            self.found_answer = False
            self.answer = NO_ANSWER_TEXT
            self.article_titles = []
        return self


class ResolverAgent(Agent):
    def __init__(self, model: str = "gpt-4o-mini", tools: list | None = None):
        name = "ResolverAgent"
        instructions = f"""
        You resolve CultPass customer issues using only the support knowledge base.

        Steps:
        1. Call get_knowledge_articles once, with the customer's issue described in plain words.
        2. Consider only articles whose score is at least {KB_CONFIDENCE_THRESHOLD}.
        3. If at least one such article answers the issue, write a short, friendly reply for the
           customer that states the fix step by step, using only what those articles say.
           Set found_answer to true and list the titles of the articles you used.
        4. Otherwise set found_answer to false, set answer to a one-sentence explanation that the
           knowledge base has no answer, and leave article_titles empty.

        Never use outside knowledge, and never promise refunds, credits or exceptions.
        {TOOL_USE_RULES}
        """
        super().__init__(name, instructions, model, tools, response_format=Resolution)
