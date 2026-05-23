"""Adapters for repository judge_binary_correctness.py and judge_openai.py."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from src.eval.answer_metrics import exact_match, normalized_accuracy, token_f1


DEFAULT_API_KEY_FILE = Path(r"C:\Users\kbllm\Downloads\api.txt")


def load_openai_api_key(api_key_file: str | Path | None = None) -> str:
    """Load OpenAI API key from the requested file, then OPENAI_API_KEY, without printing it."""
    path = Path(api_key_file) if api_key_file else DEFAULT_API_KEY_FILE
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
            return key
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _external_calls_enabled(flag: bool | None = None) -> bool:
    if flag is not None:
        return bool(flag)
    return os.environ.get("HOTPOTQA_RUN_OPENAI_JUDGE_CALLS", "").strip() == "1"


def _judge_record(question: str, gold_answer: str, pred_answer: str) -> dict[str, Any]:
    return {
        "question_id": "adapter_single",
        "method": "HotpotQA",
        "variant": "HotpotQA",
        "question": question,
        "ground_truth": gold_answer,
        "reference_answer": gold_answer,
        "prediction": pred_answer,
        "predicted_answer": pred_answer,
    }


def judge_binary_correctness(
    question: str,
    gold_answer: str,
    pred_answer: str,
    evidence: Any = None,
    external_enabled: bool | None = None,
) -> dict[str, Any]:
    """Run or safely fallback for judge_binary_correctness.py."""
    load_openai_api_key()
    local_score = normalized_accuracy(pred_answer, gold_answer)
    if not _external_calls_enabled(external_enabled):
        return {
            "binary_correct": bool(local_score),
            "judge_score": int(local_score),
            "judge_reason": "local normalized-answer fallback; external judge disabled by HOTPOTQA_RUN_OPENAI_JUDGE_CALLS",
            "judge_model": "judge_binary_correctness.py:local_fallback",
            "judge_error": None,
        }
    try:
        key = load_openai_api_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY unavailable")
        mod = _load_module(Path("judge_binary_correctness.py"), "judge_binary_correctness_adapter")
        import openai

        client = openai.OpenAI(api_key=key)
        if hasattr(mod, "JUDGE_MODEL"):
            mod.JUDGE_MODEL = "gpt-5-mini"
        out = mod.call_one(client, _judge_record(question, gold_answer, pred_answer))
        return {
            "binary_correct": out.get("binary_correct"),
            "judge_score": 1 if out.get("binary_correct") is True else 0 if out.get("binary_correct") is False else None,
            "judge_reason": out.get("binary_correctness_raw") or out.get("binary_correctness_label"),
            "judge_model": out.get("binary_judge_model") or getattr(mod, "JUDGE_MODEL", None),
            "judge_error": None if out.get("binary_parse_ok") else out.get("binary_correctness_raw"),
        }
    except Exception as exc:
        return {
            "binary_correct": bool(local_score),
            "judge_score": int(local_score),
            "judge_reason": "local fallback after binary judge error",
            "judge_model": "judge_binary_correctness.py",
            "judge_error": f"{type(exc).__name__}: {exc}",
        }


def judge_openai_correctness(
    question: str,
    gold_answer: str,
    pred_answer: str,
    evidence: Any = None,
    external_enabled: bool | None = None,
) -> dict[str, Any]:
    """Run or safely fallback for judge_openai.py."""
    load_openai_api_key()
    fallback_score = token_f1(pred_answer, gold_answer)
    fallback_label = "correct" if exact_match(pred_answer, gold_answer) else "partial" if fallback_score > 0 else "incorrect"
    if not _external_calls_enabled(external_enabled):
        return {
            "openai_judge_label": fallback_label,
            "openai_judge_score": float(fallback_score),
            "openai_judge_reason": "local token-F1 fallback; external judge disabled by HOTPOTQA_RUN_OPENAI_JUDGE_CALLS",
            "judge_model": "judge_openai.py:local_fallback",
            "judge_error": None,
        }
    try:
        key = load_openai_api_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY unavailable")
        mod = _load_module(Path("judge_openai.py"), "judge_openai_adapter")
        import openai

        client = openai.OpenAI(api_key=key)
        if hasattr(mod, "JUDGE_MODEL"):
            mod.JUDGE_MODEL = "gpt-5-mini"
        out = mod.call_judge(client, _judge_record(question, gold_answer, pred_answer))
        score = out.get("llm_f1")
        label = "correct" if score is not None and float(score) >= 0.8 else "partial" if score is not None and float(score) > 0 else "incorrect"
        return {
            "openai_judge_label": label,
            "openai_judge_score": float(score) if score is not None else None,
            "openai_judge_reason": out.get("assessment") or out.get("differences"),
            "judge_model": getattr(mod, "JUDGE_MODEL", "judge_openai.py"),
            "judge_error": None if score is not None else out.get("assessment") or out.get("differences"),
        }
    except Exception as exc:
        return {
            "openai_judge_label": fallback_label,
            "openai_judge_score": float(fallback_score),
            "openai_judge_reason": "local fallback after OpenAI judge error",
            "judge_model": "judge_openai.py",
            "judge_error": f"{type(exc).__name__}: {exc}",
        }


def judge_prediction(
    question: str,
    gold_answer: str,
    pred_answer: str,
    evidence: Any = None,
    *,
    external_binary_enabled: bool | None = None,
    external_openai_enabled: bool | None = None,
) -> dict[str, Any]:
    """Run both judge adapters and merge their outputs."""
    binary = judge_binary_correctness(question, gold_answer, pred_answer, evidence, external_enabled=external_binary_enabled)
    openai_j = judge_openai_correctness(question, gold_answer, pred_answer, evidence, external_enabled=external_openai_enabled)
    errors = [x.get("judge_error") for x in (binary, openai_j) if x.get("judge_error")]
    return {
        "binary_correct": binary.get("binary_correct"),
        "judge_score": binary.get("judge_score"),
        "judge_reason": binary.get("judge_reason"),
        "openai_judge_label": openai_j.get("openai_judge_label"),
        "openai_judge_score": openai_j.get("openai_judge_score"),
        "openai_judge_reason": openai_j.get("openai_judge_reason"),
        "judge_model": {"binary": binary.get("judge_model"), "openai": openai_j.get("judge_model")},
        "judge_error": " | ".join(errors) if errors else None,
    }
