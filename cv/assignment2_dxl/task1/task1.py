import os.path as osp
from pathlib import Path
import importlib.util

import cv2


def _load_source_module():
    src_path = Path(__file__).resolve().parents[2] / "assignment2" / "task1" / "task1.py"
    spec = importlib.util.spec_from_file_location("assignment2_task1", src_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SOURCE = _load_source_module()
gaussian_filter = _SOURCE.gaussian_filter


if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "Lena-RGB.jpg"))
    kernel_size = 5
    sigma = 1
    res_img = gaussian_filter(img, kernel_size, sigma)

    cv2.imwrite(osp.join(root_dir, "gaussian_result.jpg"), res_img)
