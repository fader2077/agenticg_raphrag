#!/usr/bin/env python
"""Check and repair the HotpotQA Round 1 Python environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = {
    "datasets": "datasets",
    "yaml": "PyYAML",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "networkx": "networkx",
    "openai": "openai",
    "tqdm": "tqdm",
    "rank_bm25": "rank-bm25",
    "sentence_transformers": "sentence-transformers",
    "faiss": "faiss-cpu",
    "neo4j": "neo4j",
}
REQUIRED_PATHS = [
    "judge_binary_correctness.py",
    "judge_openai.py",
    "config.py",
    "builder.py",
    "src",
    "scripts",
    "eval",
    "data",
    "runs",
    "configs",
    "reports",
]


def has_module(import_name: str) -> bool:
    """Return True when an importable module is present."""
    return importlib.util.find_spec(import_name) is not None


def install_package(package: str) -> dict[str, object]:
    """Install one package using the current Python interpreter."""
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return {"package": package, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}


def update_requirements(packages: list[str]) -> None:
    """Append missing packages to requirements.txt without removing existing pins."""
    if not packages:
        return
    path = ROOT / "requirements.txt"
    existing = set()
    if path.exists():
        existing = {line.strip().split("==")[0].lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    with path.open("a", encoding="utf-8") as f:
        for package in packages:
            name = package.split("==")[0].lower()
            if name not in existing:
                f.write(package + "\n")
                existing.add(name)


def main() -> int:
    """Run environment checks and optional package installation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-install", action="store_true", default=True, help="Install missing Python packages.")
    parser.add_argument("--no-auto-install", action="store_false", dest="auto_install")
    args = parser.parse_args()

    for directory in ["data", "runs", "configs", "reports", "scripts"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    for keep in [ROOT / "runs" / ".gitkeep", ROOT / "reports" / ".gitkeep"]:
        keep.touch(exist_ok=True)

    missing = [pkg for module, pkg in REQUIREMENTS.items() if not has_module(module)]
    installs = []
    installed = []
    if missing and args.auto_install:
        for package in missing:
            result = install_package(package)
            installs.append(result)
            if result["returncode"] == 0:
                installed.append(package)
        update_requirements(installed)

    after_missing = [pkg for module, pkg in REQUIREMENTS.items() if not has_module(module)]
    path_status = {path: (ROOT / path).exists() for path in REQUIRED_PATHS}
    api_key_file_exists = Path(r"C:\Users\kbllm\Downloads\api.txt").exists()
    neo4j_available = has_module("neo4j")
    result = {
        "python": sys.version,
        "root": str(ROOT),
        "missing_before_install": missing,
        "install_results": installs,
        "missing_after_install": after_missing,
        "required_path_exists": path_status,
        "api_key_file_exists": api_key_file_exists,
        "openai_api_key_env_present": bool(__import__("os").environ.get("OPENAI_API_KEY", "").strip()),
        "neo4j_driver_importable": neo4j_available,
        "neo4j_status": "driver_importable_service_not_required" if neo4j_available else "fallback_networkx_json",
    }
    out_dir = ROOT / "runs" / "environment"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "check_env.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not after_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
