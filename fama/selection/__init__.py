"""Factor selection pipeline (RIC + correlation gates + dedup)."""

from .config import apply_selection_overrides, load_selection_config
from .models import SelectionConfig, SelectionInput, SelectionResult
from .pipeline import run_selection_pipeline

__all__ = [
    "SelectionConfig",
    "SelectionInput",
    "SelectionResult",
    "load_selection_config",
    "apply_selection_overrides",
    "run_selection_pipeline",
]

