#!/usr/bin/env python
from pathlib import Path
import importlib.util


def _load_source_module():
    src_path = Path(__file__).resolve().parents[2] / "assignment2" / "task3" / "depth_model.py"
    spec = importlib.util.spec_from_file_location("assignment2_task3_depth_model", src_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SOURCE = _load_source_module()
ResNet50DepthModel = _SOURCE.ResNet50DepthModel
__all__ = ["ResNet50DepthModel"]
