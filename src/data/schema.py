"""Canonical schema helpers for HotpotQA examples, documents, chunks, and facts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


def normalize_title(title: str) -> str:
    """Normalize a title for stable document identifiers."""
    return re.sub(r"\s+", " ", str(title or "").strip())


def doc_id_for_title(title: str) -> str:
    """Return the canonical HotpotQA document id for a title."""
    return f"hotpotqa::{normalize_title(title)}"


def stable_qid(value: str, question: str) -> str:
    """Return a stable question id when the source id is absent."""
    if value:
        return str(value)
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class ChunkSpec:
    """Configuration for sentence-aware chunking."""

    max_sentences_per_chunk: int = 3
    overlap_sentences: int = 1


def make_chunks(document: dict[str, Any], spec: ChunkSpec) -> list[dict[str, Any]]:
    """Create sentence-aware overlapping chunks from one normalized document."""
    sentences = [str(s).strip() for s in document.get("sentences", []) if str(s).strip()]
    chunks: list[dict[str, Any]] = []
    if not sentences:
        return chunks
    step = max(1, spec.max_sentences_per_chunk - spec.overlap_sentences)
    title = document["title"]
    for idx, start in enumerate(range(0, len(sentences), step)):
        end = min(len(sentences), start + spec.max_sentences_per_chunk)
        sentence_ids = list(range(start, end))
        chunk_sentences = [sentences[i] for i in sentence_ids]
        chunk_id = f"{document['doc_id']}::chunk_{idx:03d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": document["doc_id"],
                "title": title,
                "text": " ".join(chunk_sentences).strip(),
                "sentences": chunk_sentences,
                "sentence_ids": sentence_ids,
                "source": "hotpotqa",
            }
        )
        if end >= len(sentences):
            break
    return chunks


def support_fact_text(contexts: list[dict[str, Any]], title: str, sent_id: int) -> str:
    """Resolve a supporting fact text from normalized contexts."""
    for doc in contexts:
        if normalize_title(doc.get("title")) != normalize_title(title):
            continue
        sentences = doc.get("sentences", [])
        if 0 <= int(sent_id) < len(sentences):
            return str(sentences[int(sent_id)]).strip()
    return ""


def normalize_hotpot_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one Hugging Face or official HotpotQA row into project schema."""
    question = str(row.get("question") or "")
    qid = stable_qid(str(row.get("id") or row.get("_id") or ""), question)
    context = row.get("context") or []
    contexts: list[dict[str, Any]] = []
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentence_groups = context.get("sentences") or []
        for title, sentences in zip(titles, sentence_groups):
            clean_title = normalize_title(title)
            contexts.append(
                {
                    "title": clean_title,
                    "sentences": [str(s).strip() for s in sentences if str(s).strip()],
                    "doc_id": doc_id_for_title(clean_title),
                }
            )
    else:
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, sentences = item[0], item[1]
            else:
                title, sentences = item.get("title"), item.get("sentences")
            clean_title = normalize_title(title)
            contexts.append(
                {
                    "title": clean_title,
                    "sentences": [str(s).strip() for s in sentences if str(s).strip()],
                    "doc_id": doc_id_for_title(clean_title),
                }
            )

    sf = row.get("supporting_facts") or []
    sf_titles: list[str] = []
    sf_ids: list[int] = []
    if isinstance(sf, dict):
        sf_titles = [normalize_title(x) for x in sf.get("title", [])]
        sf_ids = [int(x) for x in sf.get("sent_id", [])]
    else:
        for item in sf:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                sf_titles.append(normalize_title(item[0]))
                sf_ids.append(int(item[1]))
            elif isinstance(item, dict):
                sf_titles.append(normalize_title(item.get("title")))
                sf_ids.append(int(item.get("sent_id", 0)))

    supporting_facts = [
        {
            "title": title,
            "sent_id": sent_id,
            "text": support_fact_text(contexts, title, sent_id),
        }
        for title, sent_id in zip(sf_titles, sf_ids)
    ]
    return {
        "qid": qid,
        "dataset": "hotpotqa",
        "question": question,
        "answer": str(row.get("answer") or ""),
        "type": str(row.get("type") or "unknown"),
        "level": str(row.get("level") or "unknown"),
        "contexts": contexts,
        "supporting_facts": supporting_facts,
        "gold_titles": sorted({fact["title"] for fact in supporting_facts if fact.get("title")}),
    }
