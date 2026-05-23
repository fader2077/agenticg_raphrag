"""HotpotQA loading, sampling, normalization, and artifact writing."""

from __future__ import annotations

import json
import random
import time
import urllib.request
from pathlib import Path
from typing import Any

from src.data.schema import ChunkSpec, make_chunks, normalize_hotpot_row
from src.io_utils import write_json, write_jsonl


OFFICIAL_DEV_DISTRACTOR_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HF_DATASET_CANDIDATES = ["hotpotqa/hotpot_qa", "hotpot_qa"]


def _load_hf_hotpotqa(split: str, subset: str) -> tuple[list[dict[str, Any]], str]:
    """Load HotpotQA examples from Hugging Face datasets."""
    from datasets import load_dataset

    errors: list[str] = []
    for dataset_name in HF_DATASET_CANDIDATES:
        try:
            ds = load_dataset(dataset_name, subset, split=split)
            return [dict(row) for row in ds], dataset_name
        except Exception as exc:
            errors.append(f"{dataset_name}:{type(exc).__name__}:{exc}")
    raise RuntimeError("; ".join(errors))


def _download_official_dev(cache_path: Path) -> list[dict[str, Any]]:
    """Download the official HotpotQA distractor dev file as fallback."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        with urllib.request.urlopen(OFFICIAL_DEV_DISTRACTOR_URL, timeout=120) as resp:
            cache_path.write_bytes(resp.read())
    return json.loads(cache_path.read_text(encoding="utf-8"))


def load_hotpotqa_rows(setting: str = "distractor", split: str = "validation") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load raw HotpotQA rows with retry and official-file fallback."""
    errors: list[str] = []
    subset = setting if setting in {"distractor", "fullwiki"} else "distractor"
    if subset != setting:
        errors.append(f"unsupported_setting:{setting}; using distractor")
    for attempt in range(3):
        try:
            rows, dataset_name = _load_hf_hotpotqa(split, subset)
            return rows, {"source": f"huggingface:{dataset_name}/{subset}", "errors": errors}
        except Exception as exc:
            errors.append(f"hf_attempt_{attempt + 1}:{type(exc).__name__}:{exc}")
            time.sleep(1 + attempt)
    if subset == "distractor":
        try:
            rows = _download_official_dev(Path("data/raw/hotpotqa/hotpot_dev_distractor_v1.json"))
            return rows, {"source": OFFICIAL_DEV_DISTRACTOR_URL, "errors": errors}
        except Exception as exc:
            errors.append(f"official_download:{type(exc).__name__}:{exc}")
            cache_path = Path("data/raw/hotpotqa/hotpot_dev_distractor_v1.json")
            if cache_path.exists():
                return json.loads(cache_path.read_text(encoding="utf-8")), {"source": str(cache_path), "errors": errors}
    return _synthetic_hotpotqa_fallback(), {
        "source": "synthetic_hotpotqa_schema_fallback_not_for_scientific_reporting",
        "errors": errors + ["using synthetic schema fallback because no HotpotQA source was reachable"],
    }


def sample_rows(
    rows: list[dict[str, Any]],
    seed: int,
    sample_size: int | str | None = None,
    sample_fraction: float | None = None,
    strategy: str = "random",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose a reproducible sample and return sample metadata."""
    if strategy != "random":
        raise ValueError(f"unsupported sample strategy: {strategy}")
    indexed = list(rows)
    rng = random.Random(seed)
    rng.shuffle(indexed)
    resolved_count: int
    if sample_fraction is not None:
        resolved_count = max(1, int(round(len(indexed) * float(sample_fraction))))
    elif sample_size is None or str(sample_size).lower() in {"full", "all", "none"}:
        resolved_count = len(indexed)
    else:
        size_map = {"smoke": 20, "debug": 50, "main": 500}
        key = str(sample_size).lower()
        resolved_count = size_map[key] if key in size_map else int(sample_size)
    sampled = indexed[: min(resolved_count, len(indexed))]
    sample_ids = [str(row.get("id") or row.get("_id") or "") for row in sampled]
    return sampled, {
        "sample_strategy": strategy,
        "sample_fraction": sample_fraction,
        "sample_size": sample_size if sample_fraction is None else resolved_count,
        "resolved_count": len(sampled),
        "sample_ids": sample_ids,
    }


def choose_sample(rows: list[dict[str, Any]], sample_size: int | str, seed: int) -> list[dict[str, Any]]:
    """Choose a stable sample; kept for Round 1/2 compatibility."""
    sampled, _ = sample_rows(rows, seed, sample_size=sample_size, strategy="random")
    return sampled


def _synthetic_hotpotqa_fallback() -> list[dict[str, Any]]:
    """Return a tiny schema-compatible fallback when HotpotQA cannot be fetched."""
    return [
        {
            "_id": "synthetic-hotpotqa-001",
            "question": "What city is the university that created Python Tutor located in?",
            "answer": "San Diego",
            "type": "bridge",
            "level": "easy",
            "context": [
                [
                    "Python Tutor",
                    [
                        "Python Tutor is an educational tool created by Philip Guo.",
                        "Philip Guo was a professor at the University of California, San Diego.",
                    ],
                ],
                [
                    "University of California, San Diego",
                    [
                        "The University of California, San Diego is a public research university.",
                        "It is located in San Diego, California.",
                    ],
                ],
            ],
            "supporting_facts": [["Python Tutor", 1], ["University of California, San Diego", 1]],
        },
        {
            "_id": "synthetic-hotpotqa-002",
            "question": "Are The Leftovers and Lost both television series?",
            "answer": "yes",
            "type": "comparison",
            "level": "easy",
            "context": [
                ["The Leftovers", ["The Leftovers is an American supernatural drama television series."]],
                ["Lost", ["Lost is an American drama television series that aired on ABC."]],
            ],
            "supporting_facts": [["The Leftovers", 0], ["Lost", 0]],
        },
        {
            "_id": "synthetic-hotpotqa-003",
            "question": "Who wrote the novel that inspired the film Blade Runner?",
            "answer": "Philip K. Dick",
            "type": "bridge",
            "level": "easy",
            "context": [
                ["Blade Runner", ["Blade Runner is a film loosely based on the novel Do Androids Dream of Electric Sheep?."]],
                ["Do Androids Dream of Electric Sheep?", ["Do Androids Dream of Electric Sheep? is a novel by Philip K. Dick."]],
            ],
            "supporting_facts": [["Blade Runner", 0], ["Do Androids Dream of Electric Sheep?", 0]],
        },
    ]


def normalize_dataset(
    rows: list[dict[str, Any]],
    max_sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """Normalize raw HotpotQA rows into questions, documents, chunks, and facts."""
    questions: list[dict[str, Any]] = []
    documents_by_id: dict[str, dict[str, Any]] = {}
    facts: list[dict[str, Any]] = []
    chunk_spec = ChunkSpec(max_sentences_per_chunk, overlap_sentences)
    for row in rows:
        item = normalize_hotpot_row(row)
        q = {k: item[k] for k in ("qid", "dataset", "question", "answer", "type", "level", "gold_titles")}
        q["supporting_facts"] = item["supporting_facts"]
        questions.append(q)
        for doc in item["contexts"]:
            documents_by_id.setdefault(doc["doc_id"], doc)
        for fact in item["supporting_facts"]:
            facts.append({"qid": item["qid"], **fact})
    chunks: list[dict[str, Any]] = []
    for document in documents_by_id.values():
        chunks.extend(make_chunks(document, chunk_spec))
    return {
        "questions": questions,
        "documents": list(documents_by_id.values()),
        "chunks": chunks,
        "supporting_facts": facts,
    }


def write_processed_hotpotqa(
    output_dir: Path,
    rows: list[dict[str, Any]],
    source_meta: dict[str, Any],
    setting: str,
    sample_size: int | str,
    seed: int,
    split: str = "validation",
    sample_strategy: str = "random",
    sample_fraction: float | None = None,
    sample_ids: list[str] | None = None,
    max_sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
) -> dict[str, Any]:
    """Write normalized HotpotQA artifacts and return a manifest."""
    normalized = normalize_dataset(rows, max_sentences_per_chunk, overlap_sentences)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "questions.jsonl", normalized["questions"])
    write_jsonl(output_dir / "documents.jsonl", normalized["documents"])
    write_jsonl(output_dir / "chunks.jsonl", normalized["chunks"])
    write_jsonl(output_dir / "supporting_facts.jsonl", normalized["supporting_facts"])
    manifest = {
        "dataset": "hotpotqa",
        "setting": setting,
        "split": split,
        "sample_size": sample_size,
        "sample_fraction": sample_fraction,
        "sample_strategy": sample_strategy,
        "random_seed": seed,
        "source": source_meta.get("source"),
        "source_errors": source_meta.get("errors", []),
        "sample_ids": sample_ids or [],
        "num_questions": len(normalized["questions"]),
        "num_documents": len(normalized["documents"]),
        "num_chunks": len(normalized["chunks"]),
        "num_supporting_facts": len(normalized["supporting_facts"]),
        "max_sentences_per_chunk": max_sentences_per_chunk,
        "overlap_sentences": overlap_sentences,
    }
    write_json(output_dir / "sample_manifest.json", manifest)
    return manifest
