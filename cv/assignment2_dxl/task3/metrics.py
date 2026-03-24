#!/usr/bin/env python
from pathlib import Path
import importlib.util


def _load_source_module():
    src_path = Path(__file__).resolve().parents[2] / "assignment2" / "task3" / "metrics.py"
    spec = importlib.util.spec_from_file_location("assignment2_task3_metrics", src_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SOURCE = _load_source_module()
solve_scale_shift = _SOURCE.solve_scale_shift
abs_rel_metric = _SOURCE.abs_rel_metric
to_numpy_metrics = _SOURCE.to_numpy_metrics
__all__ = ["solve_scale_shift", "abs_rel_metric", "to_numpy_metrics"]
