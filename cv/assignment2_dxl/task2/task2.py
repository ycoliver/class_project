import os.path as osp
from pathlib import Path
import importlib.util

import cv2


def _load_source_module():
    src_path = Path(__file__).resolve().parents[2] / "assignment2" / "task2" / "task2.py"
    spec = importlib.util.spec_from_file_location("assignment2_task2", src_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SOURCE = _load_source_module()
histogram_equalization = _SOURCE.histogram_equalization
local_histogram_equalization = _SOURCE.local_histogram_equalization


if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "moon.png"), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res_hist_equalization = histogram_equalization(img)
    res_local_hist_equalization = local_histogram_equalization(img)

    cv2.imwrite(osp.join(root_dir, "HistEqualization.jpg"), res_hist_equalization)
    cv2.imwrite(
        osp.join(root_dir, "LocalHistEqualization.jpg"), res_local_hist_equalization
    )
