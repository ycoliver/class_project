#!/usr/bin/env python
from pathlib import Path
import importlib.util


def _load_source_module():
    src_path = Path(__file__).resolve().parents[2] / "assignment2" / "task3" / "scannet_dataset.py"
    spec = importlib.util.spec_from_file_location("assignment2_task3_scannet_dataset", src_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SOURCE = _load_source_module()
ScanNetDepthDataset = _SOURCE.ScanNetDepthDataset
list_scannet_scenes = _SOURCE.list_scannet_scenes
build_train_val_scenes = _SOURCE.build_train_val_scenes
__all__ = ["ScanNetDepthDataset", "list_scannet_scenes", "build_train_val_scenes"]
