"""Classic and FinBERT baselines, imported lazily for CPU-only CI."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from signalbench import LABELS
from signalbench.models.common import output


class Classic:
    """Training-only TF-IDF vocabulary with multinomial logistic regression."""

    def __init__(self, train, seed: int, majority: bool = False):
        estimator = (
            DummyClassifier(strategy="prior")
            if majority
            else LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
        )
        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), estimator)
        self.model.fit(train.text, train.label)

    def predict(self, texts: list[str]) -> list[dict]:
        """Return probabilities in the shared label order."""
        p = self.model.predict_proba(texts)
        order = [list(self.model.classes_).index(label) for label in LABELS]
        return output(p[:, order])


class FinBERT:
    """ProsusAI FinBERT, whose PhraseBank contamination must be disclosed."""

    def __init__(self, revision: str):
        from transformers import pipeline

        self.pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            revision=revision,
            device=-1,
            top_k=None,
        )

    def predict(self, texts: list[str]) -> list[dict]:
        """Map model labels by name, never by assumed index."""
        rows = self.pipe(texts, truncation=True, max_length=512, batch_size=8)
        p = [{entry["label"].lower(): entry["score"] for entry in row} for row in rows]
        return output(np.array([[row[label] for label in LABELS] for row in p]))
