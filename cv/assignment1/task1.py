import numpy as np
import cv2

def image_t(im, scale=1.0, rot=45, trans=(50,-50)):
    # TODO Write "image affine transformation" function based on the illustration in specification.
    # Return transformed result image
    h, w = im.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    theta = np.deg2rad(rot)
    cos_t = np.cos(theta) * scale
    sin_t = np.sin(theta) * scale

    # Define three points around the center to construct the affine transform
    pts1 = np.float32([
        [cx, cy],
        [cx + 1.0, cy],
        [cx, cy + 1.0],
    ])

    pts2 = []
    for p in pts1:
        v = p - np.array([cx, cy], dtype=np.float32)
        v2 = np.array([cos_t * v[0] - sin_t * v[1],
                       sin_t * v[0] + cos_t * v[1]], dtype=np.float32)
        p2 = v2 + np.array([cx + trans[0], cy + trans[1]], dtype=np.float32)
        pts2.append(p2)
    pts2 = np.float32(pts2)

    M = cv2.getAffineTransform(pts1, pts2)
    result = cv2.warpAffine(
        im, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return result


if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    im_path = os.path.join(base_dir, 'misc', 'pearl.jpeg')
    im = cv2.imread(im_path)
    
    scale  = 0.5
    rot    = 45
    trans  = (50, -50)
    result = image_t(im, scale, rot, trans)
    out_dir = os.path.join(base_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, 'affine_result.png'), result)
