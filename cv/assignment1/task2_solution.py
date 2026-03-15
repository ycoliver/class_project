import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial.transform import Rotation as R

H, W = 128, 128


def get_cube(center=(0, 0, 2), rotation_angles=(0.0, 0.0, 0.0), scale=1.0):
    corners = np.array([
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
    ], dtype=np.float32)
    corners = corners - np.array([0.5, 0.5, 0.5], dtype=np.float32)
    corners = corners * scale
    rot_mat = R.from_euler('xyz', rotation_angles, degrees=True).as_matrix()
    corners = corners @ rot_mat.T
    corners = corners + np.array(center, dtype=np.float32)

    faces = np.array([
        [corners[0], corners[1], corners[3], corners[2]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[0], corners[2], corners[6], corners[4]],
        [corners[-1], corners[-2], corners[-4], corners[-3]],
        [corners[-1], corners[-2], corners[-6], corners[-5]],
        [corners[-1], corners[-3], corners[-7], corners[-5]],
    ])
    return faces


def get_camera_intrinsics(alpha=70, beta=70, cx=W / 2.0, cy=H / 2.0):
    K = np.array([
        [alpha, 0.0, cx],
        [0.0, beta, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return K


def get_perspective_projection(x_c, K):
    x_c = np.asarray(x_c, dtype=np.float32)
    if x_c.ndim == 1:
        x_c = x_c[None, :]
    x_h = (K @ x_c.T).T
    x_s = x_h[:, :2] / x_h[:, 2:3]
    return x_s


def project_cube(cube, K):
    projected_faces = []
    for face in cube:
        projected_faces.append(get_perspective_projection(face, K))
    return np.array(projected_faces)


def plot_projected_cube(projected_cube, out_path=None, title='Projected Cube'):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')

    for face in projected_cube:
        poly = Polygon(face, closed=True, fill=False, edgecolor='tab:blue', linewidth=1.5)
        ax.add_patch(poly)

    ax.set_title(title)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)

    K = get_camera_intrinsics()
    cube = get_cube(center=(0, 0, 2), rotation_angles=(30, 50, 0), scale=1.0)
    projected_cube = project_cube(cube, K)
    plot_projected_cube(projected_cube, os.path.join(out_dir, 'task2_cube_default.png'), 'Projected Cube (default)')

    # Example variations
    cube2 = get_cube(center=(0.2, -0.2, 2.5), rotation_angles=(10, 30, 20), scale=1.0)
    projected_cube2 = project_cube(cube2, K)
    plot_projected_cube(projected_cube2, os.path.join(out_dir, 'task2_cube_variant.png'), 'Projected Cube (variant)')


if __name__ == '__main__':
    main()
