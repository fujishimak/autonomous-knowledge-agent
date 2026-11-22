"""Vector search over the `knowledge` table of udahub.db, using sqlite-vec.

The `knowledge` table stays the source of truth for article text. This module adds a
`vec_knowledge` virtual table (one embedding per article) next to it and exposes:

    KnowledgeIndex.rebuild()   -> (re)embed every article and store the vectors
    KnowledgeIndex.search()    -> top-k articles for a question, with a 0..1 score

Run from the `solution/` directory so `import utils` and `data.models` resolve.
"""

import logging
import os
from pathlib import Path
from typing import Any, Protocol

import sqlite_vec
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, event, select, text

import utils as u
from data.models import udahub

load_dotenv()

logger = logging.getLogger(__name__)

# The vector column size must equal the embedding model's output size.
# text-embedding-3-small -> 1536. If you change the model, change this and rebuild.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

VEC_TABLE = "vec_knowledge"
DEFAULT_ACCOUNT_ID = "cultpass"

# solution/agentic/mcp_servers/udahub/knowledge_index.py -> parents[3] == solution/
UDAHUB_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "core" / "udahub.db"


class Embedder(Protocol):
    """The two methods of LangChain's `Embeddings` that this module uses."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# --------------------------------------------------------------------------- #
# Engine helpers
# --------------------------------------------------------------------------- #
def register_vec_extension(engine: Engine) -> Engine:
    """Load sqlite-vec on every new connection of `engine`.

    Call this before the engine opens its first connection. A connection opened
    earlier will not have the extension and fails with "no such module: vec0".
    """

    @event.listens_for(engine, "connect")
    def _load_vec(dbapi_connection, _record):
        dbapi_connection.enable_load_extension(True)
        sqlite_vec.load(dbapi_connection)
        dbapi_connection.enable_load_extension(False)

    return engine


def create_udahub_engine(db_path: str | Path | None = None, echo: bool = False) -> Engine:
    """Engine for udahub.db with sqlite-vec loaded. Defaults to solution/data/core/udahub.db."""
    path = Path(db_path) if db_path else UDAHUB_DB_PATH
    return register_vec_extension(create_engine(f"sqlite:///{path}", echo=echo))


def default_embedder() -> Embedder:
    """OpenAI embeddings through the endpoint and key configured in .env."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("VOCAREUM_API_KEY"),
    )


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
class KnowledgeIndex:
    def __init__(
        self,
        engine: Engine,
        embedder: Embedder | None = None,
        dim: int = EMBEDDING_DIM,
    ):
        self.engine = engine
        self._embedder = embedder
        self.dim = dim

    @property
    def embedder(self) -> Embedder:
        # Created lazily so importing this module never needs an API key.
        if self._embedder is None:
            self._embedder = default_embedder()
        return self._embedder

    # -- schema ------------------------------------------------------------ #
    def ensure_table(self) -> None:
        """Create the vector table if it does not exist (idempotent)."""
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0("
                    "article_id text primary key, "
                    "account_id text, "
                    f"embedding float[{self.dim}] distance_metric=cosine)"
                )
            )

    # -- indexing ---------------------------------------------------------- #
    @staticmethod
    def _article_text(title: str, tags: str | None, content: str) -> str:
        # Tags are embedded with the article so they help retrieval without acting as a filter.
        return f"{title}\nTags: {tags or ''}\n{content}"

    def rebuild(self) -> int:
        """Embed every Knowledge row and replace the stored vectors. Returns the row count."""
        with u.get_session(self.engine) as session:
            articles = [
                {
                    "article_id": a.article_id,
                    "account_id": a.account_id,
                    "text": self._article_text(a.title, a.tags, a.content),
                }
                for a in session.execute(select(udahub.Knowledge)).scalars().all()
            ]

        self.ensure_table()
        if not articles:
            with self.engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {VEC_TABLE}"))
            return 0

        # Embed first, then write. A failed API call leaves the old index untouched.
        vectors = self.embedder.embed_documents([a["text"] for a in articles])
        if len(vectors) != len(articles):
            raise ValueError(
                f"Embedder returned {len(vectors)} vectors for {len(articles)} articles"
            )
        for vector in vectors:
            self._check_dim(vector)

        rows = [
            {
                "article_id": a["article_id"],
                "account_id": a["account_id"],
                "embedding": sqlite_vec.serialize_float32(vector),
            }
            for a, vector in zip(articles, vectors)
        ]
        with self.engine.begin() as conn:  # one transaction: delete + insert
            conn.execute(text(f"DELETE FROM {VEC_TABLE}"))
            conn.execute(
                text(
                    f"INSERT INTO {VEC_TABLE}(article_id, account_id, embedding) "
                    "VALUES (:article_id, :account_id, :embedding)"
                ),
                rows,
            )
        return len(rows)

    def _check_dim(self, vector: list[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(
                f"Embedding has {len(vector)} dimensions but the index expects {self.dim}. "
                "Update EMBEDDING_DIM, drop the vec table and rebuild."
            )

    # -- search ------------------------------------------------------------ #
    def search(
        self, query: str, k: int = 3, account_id: str = DEFAULT_ACCOUNT_ID
    ) -> dict[str, Any]:
        """Top-k articles for `query`, best first.

        score = cosine similarity clamped to 0..1. `top_score` is the best score, or 0.0
        when nothing matched, so callers can compare it to an escalation threshold.
        """
        self.ensure_table()
        query_vector = self.embedder.embed_query(query)
        self._check_dim(query_vector)

        with self.engine.connect() as conn:
            matches = conn.execute(
                text(
                    f"SELECT article_id, distance FROM {VEC_TABLE} "
                    "WHERE embedding MATCH :query AND k = :k AND account_id = :account_id "
                    "ORDER BY distance"
                ),
                {
                    "query": sqlite_vec.serialize_float32(query_vector),
                    "k": k,
                    "account_id": account_id,
                },
            ).all()

        scores = {article_id: max(0.0, 1.0 - distance) for article_id, distance in matches}
        if not scores:
            return {"query": query, "results": [], "top_score": 0.0}

        with u.get_session(self.engine) as session:
            articles = {
                a.article_id: a
                for a in session.execute(
                    select(udahub.Knowledge).where(
                        udahub.Knowledge.article_id.in_(scores.keys())
                    )
                ).scalars()
            }
            results = [
                {
                    "article_id": article_id,
                    "title": articles[article_id].title,
                    "content": articles[article_id].content,
                    "tags": articles[article_id].tags,
                    "score": round(score, 4),
                }
                for article_id, score in scores.items()
                if article_id in articles  # skip vectors whose article was deleted
            ]

        results.sort(key=lambda r: r["score"], reverse=True)
        return {
            "query": query,
            "results": results,
            "top_score": results[0]["score"] if results else 0.0,
        }
