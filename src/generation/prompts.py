"""Shared generation prompts for HotpotQA methods."""

RAG_ANSWER_PROMPT = """You are given a question and retrieved evidence.
Answer the question using only the provided evidence.
If the evidence is insufficient, answer exactly: "insufficient evidence".
Return a concise final answer.
Do not use external knowledge.
Also return cited evidence ids if available."""

LLM_ONLY_PROMPT = """Answer the question concisely.
If you cannot answer from the question alone, answer exactly: "insufficient evidence"."""

RAG_ANSWER_JSON_INSTRUCTION = """Return valid JSON with this schema:
{"answer": "...", "citations": ["chunk_id#sent_n", "..."]}"""

LLM_ONLY_JSON_INSTRUCTION = """Return valid JSON with this schema:
{"answer": "...", "citations": []}"""
