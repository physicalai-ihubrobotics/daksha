"""Wrist-camera geometry for the final approach of a grasp (eye-in-hand, aligned depth).

Both the fingertips and the object are measured in the WRIST CAMERA frame from depth, so the
correction (object minus fingertips) needs only the camera's rotation in the world, not the
eye-in-hand translation (which is the poorly calibrated part on Daksha).

  fingertips(depth, K, D)    3-D midpoint of the two fingertips: the two nearest depth blobs in the
                             lower part of the image, their top-most (tip) pixels. Measure it with
                             the hand open and the object still far (the fingers are rigid in the
                             camera image while the opening does not change).
  object_point(depth, K, D, box)
                             3-D point of the object: median depth over the central part of its
                             (tracker) bounding box, on the ray through the box centre.
"""

import cv2
import numpy as np


def _backproject(u, v, z, K, D):
    xn = cv2.undistortPoints(np.array([[[float(u), float(v)]]]), K, D).reshape(2)
    return np.array([xn[0] * z, xn[1] * z, z])


def fingertips(depth, K, D, near=(0.04, 0.22), min_area=300):
    """(point_cam, info) or (None, why)."""
    if depth is None:
        return None, 'no depth'
    h, w = depth.shape
    m = np.isfinite(depth) & (depth > near[0]) & (depth < near[1])
    m[: int(h * 0.30)] = False                         # fingers come in from the bottom of the image
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    comps = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    comps = [i for i in comps if stats[i, cv2.CC_STAT_AREA] >= min_area][:2]
    if len(comps) < 2:
        return None, f'fingers not found in depth ({len(comps)} near blobs)'
    tips, px = [], []
    for i in comps:
        vs, us = np.nonzero(lab == i)
        cut = np.percentile(vs, 8)                     # the top-most 8 % = the tip
        sel = vs <= cut
        u, v, z = float(np.median(us[sel])), float(np.median(vs[sel])), float(np.nanmedian(depth[vs[sel], us[sel]]))
        tips.append(_backproject(u, v, z, K, D))
        px.append((u, v))
    p = (tips[0] + tips[1]) / 2
    return p, {'tips_px': px, 'gap_m': float(np.linalg.norm(tips[0] - tips[1])), 'depth_m': float(p[2])}


def object_point(depth, K, D, box, core=0.4):
    """(point_cam, depth_m) or (None, why) from the central `core` fraction of the box."""
    if depth is None or box is None:
        return None, 'no depth / box'
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * core / 2, (y2 - y1) * core / 2
    h, w = depth.shape
    patch = depth[max(0, int(cy - hh)):min(h, int(cy + hh) + 1), max(0, int(cx - hw)):min(w, int(cx + hw) + 1)]
    vals = patch[np.isfinite(patch) & (patch > 0.03)]
    if vals.size < 10:
        return None, 'no depth on the object'
    z = float(np.median(vals))
    return _backproject(cx, cy, z, K, D), z
