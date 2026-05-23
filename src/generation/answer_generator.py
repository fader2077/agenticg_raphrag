"""Answer generation with Ollama main-path support and deterministic fallback."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.generation.prompts import (
    LLM_ONLY_JSON_INSTRUCTION,
    LLM_ONLY_PROMPT,
    RAG_ANSWER_JSON_INSTRUCTION,
    RAG_ANSWER_PROMPT,
)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9]+", str(text or "")) if len(t) > 2}


def _best_sentence(question: str, chunks: list[dict[str, Any]]) -> tuple[str, str | None]:
    q_tokens = _tokens(question)
    best = ("", None, 0.0)
    for chunk in chunks:
        sentences = chunk.get("sentences") or [chunk.get("text", "")]
        ids = chunk.get("sentence_ids") or list(range(len(sentences)))
        for sid, sent in zip(ids, sentences):
            s_tokens = _tokens(sent)
            score = len(q_tokens & s_tokens) / max(1, len(q_tokens))
            if score > best[2]:
                best = (str(sent).strip(), f"{chunk.get('chunk_id')}#sent_{sid}", score)
    return best[0], best[1]


def _extract_candidate(question: str, sentence: str) -> str:
    q_lower = question.lower()
    s = sentence.strip()
    if not s:
        return "insufficient evidence"
    if q_lower.startswith(("were ", "was ", "is ", "are ", "did ", "does ", "do ", "has ", "have ")):
        if re.search(r"\b(no|not|never|different)\b", s, flags=re.I):
            return "no"
        return "yes"
    quoted = re.findall(r'"([^"]{2,100})"', s)
    if quoted:
        return quoted[0].strip()
    entities = re.findall(r"\b[A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){0,5}\b", s)
    q_tokens = _tokens(question)
    filtered = [e for e in entities if not (_tokens(e) & q_tokens) and e.lower() not in {"the", "this"}]
    if filtered:
        return filtered[-1].strip(" .,;:")
    numbers = re.findall(r"\b\d+(?:\.\d+)?(?:\s*(?:percent|%|years?|minutes?|miles?|km|kg))?\b", s, flags=re.I)
    if numbers:
        return numbers[0]
    words = s.split()
    return " ".join(words[: min(8, len(words))]).strip(" .,;:") or "insufficient evidence"


def _fallback_generate(question: str, evidence_chunks: list[dict[str, Any]], method: str) -> dict[str, Any]:
    """Return the deterministic evidence-sentence fallback answer."""
    if method == "LLM-only":
        return {"answer": "insufficient evidence", "citations": [], "fallback_mode": "deterministic_llm_only"}
    if not evidence_chunks:
        return {"answer": "insufficient evidence", "citations": [], "fallback_mode": "deterministic_empty_evidence"}
    sentence, citation = _best_sentence(question, evidence_chunks)
    return {
        "answer": _extract_candidate(question, sentence),
        "citations": [citation] if citation else [],
        "fallback_mode": "deterministic_evidence_sentence_v1",
    }


def _safe_json_extract(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from a model response."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _evidence_block(evidence_chunks: list[dict[str, Any]], max_context_chars: int) -> str:
    """Format evidence chunks into a bounded prompt block."""
    lines: list[str] = []
    total = 0
    for chunk in evidence_chunks:
        cid = str(chunk.get("chunk_id") or "")
        text = str(chunk.get("text") or "").strip()
        sent_ids = ",".join(str(x) for x in chunk.get("sentence_ids") or [])
        block = f"[{cid}] title={chunk.get('title')} sentence_ids={sent_ids}\n{text}\n"
        if total + len(block) > max_context_chars:
            break
        lines.append(block)
        total += len(block)
    return "\n".join(lines)


class AnswerGenerator:
    """Common answer generator with Ollama main-path and deterministic fallback."""

    def __init__(
        self,
        temperature: float = 0.0,
        max_context_tokens: int = 6000,
        provider: str = "deterministic",
        qa_model: str | None = None,
        ollama_host: str | None = None,
        deterministic_fallback_enabled: bool = True,
        use_deterministic_for_main_eval: bool = True,
    ):
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens
        self.provider = provider
        self.qa_model = qa_model
        self.ollama_host = ollama_host
        self.deterministic_fallback_enabled = deterministic_fallback_enabled
        self.use_deterministic_for_main_eval = use_deterministic_for_main_eval
        self._client = None

    def _ensure_client(self) -> None:
        """Lazily create the Ollama client only when an Ollama call is needed."""
        if self._client is not None or self.provider != "ollama":
            return
        from ollama import Client

        self._client = Client(host=self.ollama_host)

    def _ollama_generate(self, question: str, evidence_chunks: list[dict[str, Any]], method: str) -> dict[str, Any]:
        """Call Ollama and parse a JSON answer object."""
        self._ensure_client()
        if not self._client or not self.qa_model:
            raise RuntimeError("Ollama provider requested but client/model is unavailable")
        prompt = LLM_ONLY_PROMPT if method == "LLM-only" else RAG_ANSWER_PROMPT
        json_instruction = LLM_ONLY_JSON_INSTRUCTION if method == "LLM-only" else RAG_ANSWER_JSON_INSTRUCTION
        evidence_text = "" if method == "LLM-only" else _evidence_block(evidence_chunks, max_context_chars=max(1200, self.max_context_tokens * 4))
        user_content = f"{prompt}\n{json_instruction}\n\nQuestion:\n{question}\n"
        if evidence_text:
            user_content += f"\nEvidence:\n{evidence_text}\n"
        response = self._client.chat(
            model=self.qa_model,
            messages=[{"role": "user", "content": user_content}],
            options={"temperature": self.temperature},
        )
        text = ""
        if isinstance(response, dict):
            text = str(((response.get("message") or {}).get("content")) or "")
        elif hasattr(response, "message"):
            text = str(getattr(response.message, "content", "") or "")
        else:
            text = str(response)
        parsed = _safe_json_extract(text)
        if not parsed or "answer" not in parsed:
            raise ValueError(f"ollama response was not valid answer JSON: {text[:200]}")
        citations = parsed.get("citations") if isinstance(parsed.get("citations"), list) else []
        return {
            "answer": str(parsed.get("answer") or "insufficient evidence").strip() or "insufficient evidence",
            "citations": [str(x) for x in citations if str(x).strip()],
            "raw_response": text,
        }

    def generate(self, question: str, evidence_chunks: list[dict[str, Any]] | None = None, method: str = "") -> dict[str, Any]:
        """Generate a concise answer and citation ids from evidence."""
        start = time.perf_counter()
        evidence_chunks = evidence_chunks or []
        prompt = LLM_ONLY_PROMPT if method == "LLM-only" else RAG_ANSWER_PROMPT
        fallback_used = False
        generation_provider = self.provider
        output: dict[str, Any]
        if self.provider == "ollama" and not self.use_deterministic_for_main_eval:
            try:
                output = self._ollama_generate(question, evidence_chunks, method)
            except Exception as exc:
                if not self.deterministic_fallback_enabled:
                    raise
                output = _fallback_generate(question, evidence_chunks, method)
                output["generation_error"] = f"{type(exc).__name__}: {exc}"
                fallback_used = True
        else:
            output = _fallback_generate(question, evidence_chunks, method)
            fallback_used = self.provider != "deterministic" or self.use_deterministic_for_main_eval
        elapsed = int((time.perf_counter() - start) * 1000)
        return {
            "answer": output.get("answer", "insufficient evidence"),
            "citations": output.get("citations", []),
            "prompt_style": "llm_only" if method == "LLM-only" else "shared_rag_evidence_only",
            "prompt": prompt,
            "latency_ms": elapsed,
            "input_tokens": None,
            "output_tokens": None,
            "generation_provider": generation_provider,
            "generation_model": self.qa_model if self.provider == "ollama" else output.get("fallback_mode", "deterministic"),
            "fallback_used": fallback_used,
            "generation_error": output.get("generation_error"),
            "raw_generation": output.get("raw_response"),
        }
