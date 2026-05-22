from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_cached_base(path: str | Path = "data/results/cached_base_eval_input_cases.jsonl") -> Dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    out: Dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r.get("question_id"))] = r
    return out
