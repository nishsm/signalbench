"""Dataset acquisition, conservative deduplication and fixed split manifests."""

from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from signalbench import LABELS
from signalbench.core import artifact, digest
from signalbench.db import upsert


def normalize(text: str) -> str:
    """Normalize case, Unicode, punctuation and whitespace without erasing numbers."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def deduplicate(frame: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Keep a deterministic representative per connected near-duplicate component.

    MinHash candidates use exact shingle Jaccard verification; fixed character cosine
    adds a conservative second pass without fitting a vocabulary on test data.
    Conflicting labels in a component are excluded rather than silently relabeled.
    """
    frame = frame.sort_values("id").reset_index(drop=True).copy()
    if frame.empty:
        return frame
    vector = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**18)
    x = vector.transform(frame.text.map(normalize))
    neighbors = NearestNeighbors(metric="cosine", radius=1 - threshold + 1e-10).fit(x)
    parent = list(range(len(frame)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, indices in enumerate(neighbors.radius_neighbors(x, return_distance=False)):
        for j in indices:
            parent[root(int(j))] = root(i)
    # MinHash retrieval is approximate; the cosine pass above also catches close variants.
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    shingles = []
    for i, text in enumerate(frame.text.map(normalize)):
        grams = {text[j : j + 5] for j in range(max(1, len(text) - 4))}
        sketch = MinHash(num_perm=128, seed=42)
        sketch.update_batch([g.encode() for g in sorted(grams)])
        for candidate in lsh.query(sketch):
            previous = shingles[candidate]
            if len(grams & previous) / max(1, len(grams | previous)) >= threshold:
                parent[root(candidate)] = root(i)
        lsh.insert(i, sketch)
        shingles.append(grams)
    groups: dict[int, list[int]] = {}
    for i in range(len(frame)):
        groups.setdefault(root(i), []).append(i)
    keep = [min(ids) for ids in groups.values() if frame.iloc[ids].label.nunique() == 1]
    return frame.iloc[keep].reset_index(drop=True)


def split(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Stratify 70/10/20 after deduplication."""
    train, test = train_test_split(frame, test_size=0.2, stratify=frame.label, random_state=seed)
    train, val = train_test_split(train, test_size=0.125, stratify=train.label, random_state=seed)
    return pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    ).sort_values("id")


def _revision(repo: str, kind: str, manifest: dict) -> str:
    if repo not in manifest:
        manifest[repo] = HfApi().repo_info(repo, repo_type=kind).sha
    return manifest[repo]


def repository_files(repo: str, revision: str, manifest: dict) -> list[str]:
    """Reuse saved file listings; resolve an upstream tree only on first acquisition."""
    key = repo + ":files"
    if key not in manifest:
        from huggingface_hub.constants import HF_HUB_CACHE
        snapshot = Path(HF_HUB_CACHE) / ("datasets--" + repo.replace("/", "--")) / "snapshots" / revision
        cached = [str(path.relative_to(snapshot)) for path in snapshot.rglob("*") if path.is_file()]
        manifest[key] = cached or HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
    return manifest[key]


def cached_download(repo: str, filename: str, revision: str) -> str:
    """Use the immutable local snapshot before attempting network access."""
    try:
        return hf_hub_download(
            repo, filename, repo_type="dataset", revision=revision, local_files_only=True
        )
    except LocalEntryNotFoundError:
        return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def download(cfg: dict, con) -> None:
    """Fetch pinned upstream files without executing deprecated HF dataset scripts."""
    out = Path(cfg["paths"]["results"])
    manifest_path = out / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    repo = cfg["data"]["phrasebank"]
    revision = _revision(repo, "dataset", manifest)
    # Original dataset repository ships the archive; inspect tree rather than guessing URL.
    files = repository_files(repo, revision, manifest)
    archives = [f for f in files if f.lower().endswith(".zip")]
    if len(archives) != 1:
        raise RuntimeError(f"Expected one PhraseBank archive, found {archives}")
    archive = zipfile.ZipFile(cached_download(repo, archives[0], revision))
    phrase_frames = []
    for name in cfg["data"]["configs"]:
        key = name.replace("sentences_", "").lower()
        member = next(n for n in archive.namelist() if key in n.lower() and n.endswith(".txt"))
        rows = []
        for line in archive.read(member).decode("latin-1").splitlines():
            text, label = line.rsplit("@", 1)
            rows.append(
                {
                    "source": name,
                    "id": digest([name, normalize(text)]),
                    "text": text,
                    "label": label.strip(),
                }
            )
        phrase_frames.append(pd.DataFrame(rows))
    # Split primary first. Secondary is a separate experiment; never train on its overlapping test.
    primary = deduplicate(phrase_frames[0], cfg["data"]["near_similarity"])
    frames = [split(primary, cfg["seed"])]
    for secondary in phrase_frames[1:]:
        # Exclude ALL primary neighbors from the secondary-only supplement.
        combined = pd.concat(
            [primary.assign(id=lambda x: "0" + x.id), secondary.assign(id=lambda x: "1" + x.id)]
        )
        clean = deduplicate(combined, cfg["data"]["near_similarity"])
        clean = clean[clean.source != primary.source.iloc[0]].copy()
        if not clean.empty:
            clean["id"] = clean.id.str[1:]
            frames.append(clean.assign(split="secondary_only"))
    twitter = cfg["data"]["twitter"]
    rev = _revision(twitter, "dataset", manifest)
    files = repository_files(twitter, rev, manifest)
    rows = []
    for file in files:
        if file.endswith(".csv") and "valid" in file.lower():
            raw = pd.read_csv(cached_download(twitter, file, rev))
            for row in raw.itertuples():
                text = row.text
                rows.append(
                    {
                        "source": "twitter",
                        "id": digest([twitter, normalize(text)]),
                        "text": text,
                        "label": {0: "negative", 1: "positive", 2: "neutral"}[int(row.label)],
                    }
                )
    if not rows:
        raise RuntimeError("No Twitter validation CSV found; inspect upstream manifest")
    # Remove exact/near overlap with every PhraseBank partition, retaining PhraseBank first.
    combined = pd.concat(
        [
            pd.concat(frames).assign(id=lambda x: "0" + x.id),
            pd.DataFrame(rows).assign(id=lambda x: "1" + x.id),
        ]
    )
    clean = deduplicate(combined, cfg["data"]["near_similarity"])
    ood = clean[clean.source == "twitter"].copy()
    ood["id"] = ood.id.str[1:]
    frames.append(ood.assign(split="ood"))
    all_data = pd.concat(frames)
    assert all_data.text.map(normalize).nunique() == len(all_data)
    assert set(all_data.label) <= set(LABELS)
    con.execute("DELETE FROM labeled_text")
    upsert(con, "labeled_text", all_data)
    artifact(manifest_path, cfg, json.dumps(manifest, indent=2))
    artifact(
        out / "splits/ids.csv",
        cfg,
        all_data[["source", "id", "split"]]
        .assign(text_hash=all_data.text.map(lambda text: digest(normalize(text))))
        .to_csv(index=False),
    )
    eda(all_data, cfg)


def eda(frame: pd.DataFrame, cfg: dict) -> None:
    """Write class counts, length distributions and private example sentences."""
    import matplotlib.pyplot as plt

    out = Path(cfg["paths"]["results"])
    counts = pd.crosstab([frame.source, frame.split], frame.label)
    artifact(out / "eda.csv", cfg, counts.to_csv())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    frame.label.value_counts().reindex(LABELS).plot.bar(ax=axes[0], title="Class balance")
    frame.text.str.split().str.len().hist(ax=axes[1], bins=40)
    axes[1].set(title="Length distribution", xlabel="Words")
    path = out / "figures/eda.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    artifact(str(path) + ".provenance.json", cfg, "{}")
    private = Path("data/eda_examples.csv")
    private.parent.mkdir(exist_ok=True)
    frame.groupby("label").head(3).to_csv(private, index=False)
