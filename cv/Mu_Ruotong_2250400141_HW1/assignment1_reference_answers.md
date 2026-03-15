# Assignment 1 Reference Answers (Concise)

## Task 1: Image Affine Transformation
- Use image center as rotation origin.
- Transform equation: `p' = s * R(theta) * (p - c) + c + t`.
- Implementation uses 3-point affine matrix with `cv2.getAffineTransform` and `cv2.warpAffine`.
- Example output saved to `results/affine_result.png` and `results/affine_compare.png`.

## Task 2: 3D to 2D Perspective Projection
- Camera intrinsics:
  - `K = [[alpha, 0, cx], [0, beta, cy], [0, 0, 1]]`.
- Projection:
  - `x_h = K @ x_c` (homogeneous), `x_s = x_h[:2] / x_h[2]`.
- Default output saved to `results/task2_cube_default.png`, variant to `results/task2_cube_variant.png`.

## Task 3: Portrait Segmentation
- Baseline: U-Net + CrossEntropy + Adam, 20 epochs.
- Report metrics: training loss, validation loss, validation pixel accuracy.
- Improvements (suggested): data augmentation, pretrained backbone, LR scheduling, CE + Dice loss.
- Face-aware preprocessing: detect face → crop/expand → resize → segment.
- SAM2: use box/point prompts from face detector; compare qualitatively with trained model.

## Code Files
- Task 1: `task1.py`
- Task 2: `task2_solution.py` and `task2.ipynb`
- Report: `report-latex/report.tex`
