# Camera geometry in code

For a frame with intrinsics `K` (3x3), distortion `D` (OpenCV order k1 k2 p1 p2 k3; zeros in
simulation, NOT zero on the real RealSense cameras) and camera pose `T_base_cam` (4x4, camera to
the robot frame), all taken at capture time. The robot's object-localization facts say where a
frame stores them and in which units its depth is saved.

Every function below accepts plain Python lists (as loaded from `meta.json` with `json.load`)
and returns `None` instead of a wrong point when the geometry is invalid. Check for `None`.

```python
import numpy as np

def _arr(x):
    return np.asarray(x, dtype=float)

def normalized(u, v, K, D=None):
    """Pixel -> normalized camera coordinates [x, y, 1], removing lens distortion when D is given.
    Real cameras need D: skipping it shifts points by up to several cm near the image edges."""
    K = _arr(K)
    if D is not None and np.any(_arr(D) != 0):
        import cv2
        xn = cv2.undistortPoints(np.array([[[float(u), float(v)]]]), K, _arr(D)).reshape(2)
        return np.array([xn[0], xn[1], 1.0])
    return np.linalg.solve(K, [float(u), float(v), 1.0])
```

## Depth back-projection

```python
def back_project(u, v, depth_m, K, T_base_cam, D=None):
    """3-D point (robot frame) of pixel (u, v) at depth_m (metres along the optical axis), or None.
    Depth 0 / NaN / inf is "no measurement" (occlusion, reflection, out of range): without this
    check it would map to the camera's own position and send the arm into the camera."""
    if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0:
        return None
    p_cam = float(depth_m) * normalized(u, v, K, D)
    return (_arr(T_base_cam) @ np.append(p_cam, 1.0))[:3]
```

Use the median of valid depth in a small window around the pixel, and skip pixels whose
window is mostly invalid.

## A pixel ray

```python
def ray(u, v, K, T_base_cam, D=None):
    """(origin, unit direction) of the pixel's viewing ray in the robot frame."""
    T = _arr(T_base_cam)
    d = T[:3, :3] @ normalized(u, v, K, D)
    return T[:3, 3].copy(), d / np.linalg.norm(d)
```

## Ray and a horizontal plane

```python
def on_plane(origin, direction, height, eps=1e-6):
    """Where the ray meets the plane z = height, or None. Works for cameras above (looking down)
    and below (looking up) the plane; None when the ray is parallel to it or meets it behind
    the camera."""
    origin, direction = _arr(origin), _arr(direction)
    if abs(direction[2]) < eps:
        return None                          # parallel to the plane
    s = (height - origin[2]) / direction[2]
    if s <= 0:
        return None                          # the plane is behind the camera along this ray
    return origin + s * direction
```

## Two-view triangulation

```python
def triangulate(o1, d1, o2, d2, min_den=1e-5, min_dist=1e-3):
    """Closest point between two viewing rays: (midpoint, gap_m, angle_deg), or (None, inf, angle)
    when the rays are (nearly) parallel or only meet behind a camera (a ghost point)."""
    o1, d1, o2, d2 = _arr(o1), _arr(d1), _arr(o2), _arr(d2)
    d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
    w = o1 - o2
    a, b, c = float(d1 @ d1), float(d1 @ d2), float(d2 @ d2)
    e, f = float(d1 @ w), float(d2 @ w)
    angle = float(np.degrees(np.arccos(np.clip(b, -1.0, 1.0))))
    den = a * c - b * b
    if abs(den) < min_den:
        return None, np.inf, angle           # parallel / collinear rays
    s, t = (b * f - c * e) / den, (a * f - b * e) / den
    if s <= min_dist or t <= min_dist:
        return None, np.inf, angle           # they only meet behind a camera
    p1, p2 = o1 + s * d1, o2 + t * d2
    return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2)), angle
```

Accept the midpoint only when the ray angle and the gap between the rays meet the
robot's measured thresholds; otherwise move the camera further or pick a sharper
feature.

On the real robot you rarely need these by hand: `daksha_pixel_to_position` does the
back-projection (depth when the frame has it, else a plane) with distortion handled, and
`daksha_footprint` localises a box from its bounding box without depth.
