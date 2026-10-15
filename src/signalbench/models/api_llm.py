"""Disk-cached API classification with an experiment-wide spend ledger."""

import json
import os
import sqlite3
import time
from pathlib import Path

import httpx
import tiktoken

from signalbench import LABELS
from signalbench.core import digest
from signalbench.models.common import prompt


class BudgetExceeded(RuntimeError):
    """An uncached request would exceed the experiment budget."""


class APIModel:
    """Pinned API snapshot with hard pre-request budget reservation.

    Scores are discrete -1/0/+1; returned labels are NOT calibrated probabilities.
    A reservation survives a crash to conservatively account for possibly billed requests.
    Cache is shared by benchmark and filing jobs; one process owns its write lock.
    """

    def __init__(self, cfg: dict, examples):
        self.cfg, self.examples = cfg["api"], examples
        self.seed = cfg["seed"]
        path = Path(cfg["paths"]["cache"]) / "api.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(key TEXT PRIMARY KEY, request TEXT, response TEXT, cost REAL, seconds REAL)"
        )
        self.encoder = tiktoken.encoding_for_model(self.cfg["model"])

    def request(self, text: str) -> dict:
        """Build every cache-affecting parameter into the canonical request."""
        return {
            "model": self.cfg["model"],
            "messages": prompt(text, self.examples.get(text)),
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": self.cfg["max_output_tokens"],
        }

    def upper_cost(self, body: dict) -> float:
        """Conservative token bound including chat framing and reserved completion tokens."""
        tokens = len(self.encoder.encode(json.dumps(body))) + 100
        return (
            tokens * self.cfg["input_per_million"]
            + body["max_tokens"] * self.cfg["output_per_million"]
        ) / 1e6

    def estimate(self, texts: list[str]) -> float:
        """Estimate only requests missing from the local cache."""
        requests = {digest(self.request(t)): self.request(t) for t in texts}
        return sum(
            self.upper_cost(body)
            for key, body in requests.items()
            if not self.db.execute("SELECT 1 FROM responses WHERE key=?", (key,)).fetchone()
        )

    def predict(self, texts: list[str]) -> list[dict]:
        """Classify with durable raw responses and token usage."""
        rows = []
        for text in texts:
            body = self.request(text)
            key = digest(body)
            self.db.execute("BEGIN IMMEDIATE")
            cached = self.db.execute(
                "SELECT response,cost,seconds FROM responses WHERE key=?", (key,)
            ).fetchone()
            if cached:
                self.db.commit()
                if cached[0] is None:
                    raise RuntimeError(
                        "Unresolved billed reservation; reconcile ledger before retry"
                    )
                raw, cost, seconds = json.loads(cached[0]), cached[1], cached[2]
            else:
                reserve = self.upper_cost(body)
                spent = self.db.execute("SELECT COALESCE(SUM(cost),0) FROM responses").fetchone()[0]
                if spent + reserve > self.cfg["budget"]:
                    self.db.rollback()
                    raise BudgetExceeded("API budget exceeded; check in with Nish")
                self.db.execute(
                    "INSERT INTO responses VALUES (?,?,NULL,?,NULL)",
                    (key, json.dumps(body), reserve),
                )
                self.db.commit()
                start = time.perf_counter()
                response = httpx.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                    timeout=90,
                )
                response.raise_for_status()
                raw, seconds = response.json(), time.perf_counter() - start
                usage = raw["usage"]
                cost = (
                    usage["prompt_tokens"] * self.cfg["input_per_million"]
                    + usage["completion_tokens"] * self.cfg["output_per_million"]
                ) / 1e6
                self.db.execute(
                    "UPDATE responses SET response=?,cost=?,seconds=? WHERE key=?",
                    (json.dumps(raw), cost, seconds, key),
                )
                self.db.commit()
            label = raw["choices"][0]["message"]["content"].strip().lower()
            if label not in LABELS:
                raise ValueError(
                    f"Invalid cached API output for {key}; do not silently drop failures"
                )
            rows.append(
                {
                    "label": label,
                    "score": {"negative": -1.0, "neutral": 0.0, "positive": 1.0}[label],
                    "cost": cost,
                    "seconds": seconds,
                    "usage": raw["usage"],
                    "cache_key": key,
                }
            )
        return rows
