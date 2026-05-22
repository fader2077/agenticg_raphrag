from __future__ import annotations

import argparse
import json

from vg_graphrag.demo.corpus import DEMO_QUESTIONS, build_demo_stores
from vg_graphrag.models import RunConfig
from vg_graphrag.pipeline.runner import run_diagnostic, run_vg_graphrag


def _print_answer(question: str, config: RunConfig) -> None:
    graph, text = build_demo_stores()
    ans = run_vg_graphrag(question, config, graph=graph, text=text)
    print("Question:", question)
    print("Mode:", ans.vg_mode)
    print("Answer:", ans.answer_text)
    print("Confidence:", ans.confidence)
    print("Independent dynamic retrieval:", ans.independent_dynamic_retrieval)
    print("Dynamic tool calls:", ans.dynamic_tool_call_count)
    print("Hop2 usage:", ans.hop2_usage)
    print("V5 usage:", ans.v5_usage)
    print("Invalid VG reason:", ans.invalid_vg_reason)
    print("Supporting paths:", json.dumps(ans.supporting_paths, indent=2))
    print("Supporting chunks:", json.dumps(ans.supporting_chunks, indent=2))
    print("Verifier summary:", json.dumps(ans.verifier_summary, indent=2))
    print("Tool trace:", json.dumps(ans.tool_trace, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m vg_graphrag")
    ap.add_argument("question", nargs="?", help="Question to run against the demo corpus.")
    ap.add_argument("--demo", action="store_true", help="Run built-in demo questions in vg_native_answer mode.")
    ap.add_argument("--graph-run-id", default=None)
    ap.add_argument("--graph-run-dir", default=None)
    ap.add_argument("--case-scope", default="demo", choices=["demo", "triggered", "indirect56", "full158_diagnostic"])
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--max-tool-calls", type=int, default=5)
    ap.add_argument("--max-hops", type=int, default=3)
    args = ap.parse_args()

    config = RunConfig(
        max_iterations=args.max_iterations,
        max_tool_calls=args.max_tool_calls,
        max_hops=args.max_hops,
        graph_run_id=args.graph_run_id,
        graph_run_dir=args.graph_run_dir,
        case_scope=args.case_scope,
    )
    if args.graph_run_id or args.graph_run_dir or args.case_scope != "demo":
        summary = run_diagnostic(config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.demo:
        for q in DEMO_QUESTIONS:
            _print_answer(q, config)
            print()
        return 0
    _print_answer(args.question or DEMO_QUESTIONS[0], config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
