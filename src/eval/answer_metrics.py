"""HotpotQA answer normalization, exact match, and token F1."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    text = str(text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> int:
    """Return normalized exact match."""
    return int(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Return token-level F1 under standard HotpotQA/SQuAD normalization."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def normalized_accuracy(prediction: str, gold: str) -> int:
    """Return relaxed normalized answer accuracy."""
    pred = normalize_answer(prediction)
    ans = normalize_answer(gold)
    return int(bool(pred and ans and (pred == ans or pred in ans or ans in pred)))
