"""VG-GraphRAG: verifier-guided dynamic retrieval over text and graph evidence."""

from .models import RunConfig
from .pipeline.runner import run_vg_graphrag

__all__ = ["RunConfig", "run_vg_graphrag"]
