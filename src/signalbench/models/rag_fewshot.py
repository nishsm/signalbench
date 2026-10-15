"""Training-only semantic retrieval; validation/test are never indexed."""

import numpy as np

from signalbench import LABELS


class Examples:
    """Fixed stratified shots or semantic nearest neighbors from the training partition."""

    def __init__(self, train, mode: str, shots: int, seed: int, revision: str | None = None):
        if not (train.split == "train").all():
            raise ValueError("Demonstrations must be training-only")
        self.train, self.mode, self.shots = train.reset_index(drop=True), mode, shots
        self.fixed = [
            row
            for label in LABELS
            for row in train[train.label == label]
            .sample(n=shots, random_state=seed)
            .to_dict("records")
        ]
        if mode == "rag":
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", revision=revision
            )
            self.embeddings = self.encoder.encode(
                self.train.text.tolist(), normalize_embeddings=True
            )

    def get(self, text: str) -> list[dict]:
        """Retrieve only from the frozen training corpus."""
        if self.mode == "zero":
            return []
        if self.mode == "few":
            return self.fixed
        query = self.encoder.encode([text], normalize_embeddings=True)[0]
        ids = np.argsort(-(self.embeddings @ query), kind="stable")[: self.shots * 3]
        return self.train.iloc[ids].to_dict("records")
