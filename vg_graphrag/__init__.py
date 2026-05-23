"""Compatibility shim exposing ``vg_graphragVG`` as ``vg_graphrag``."""

from __future__ import annotations

from pathlib import Path

_VG_ROOT = Path(__file__).resolve().parents[1] / "vg_graphragVG"
__path__ = [str(_VG_ROOT)]

from vg_graphragVG.models import RunConfig  # noqa: E402,F401
from vg_graphragVG.pipeline.runner import run_vg_graphrag  # noqa: E402,F401

__all__ = ["RunConfig", "run_vg_graphrag"]
