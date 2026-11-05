"""Find a round through-hole (a spool's centre hole) with SAM3 on one camera.

SAM3's text prompt proposes masks; geometry decides which one is the hole:

* **round**: the mask fills most of its enclosing circle (the spool's slot and
  label edges do not);
* **a hole, not a disc**: depth inside the mask reads farther than depth on a
  ring just outside it — the camera sees through the hole to the table;
* **big enough** to be the thing we want (tiny rivet holes are dropped).

Then the rim ring (pixels just outside the mask) is back-projected to world
points and fitted with a plane: that is the spool's top face and its normal is
the hole axis. The hole centre is where the camera ray through the mask centroid
meets that plane (depth inside a hole is the table, not the hole), and the
radius is the median distance of the mask boundary, intersected with the same
plane, from that centre. The bottom is the deepest valid point inside the hole,
or the table if the hole shows nothing valid.
"""

import logging
from typing import TypedDict

import cv2
import numpy as np
from gap_core.types import pose_to_matrix

from gap import NodeContext

logger = logging.getLogger(__name__)

PROMPTS = ("hole", "round hole", "center hole of the spool")
MIN_FILL = 0.75  # mask area / area of its minimum enclosing circle
MIN_RECESS_M = 0.01  # median depth inside must exceed the rim's by this much
MIN_AREA_PX = 150
RING_PX = 12
SNAP_VERTICAL_DEG = 8.0  # a rim plane this close to level is a part lying on the table: trust the table


class Output(TypedDict):
    hole: dict


def _camera(cameras: list, name: str) -> dict:
    for cam in cameras or []:
        if cam.get("name") == name:
            return cam
    raise RuntimeError(f"camera {name!r} not in the observation ({[c.get('name') for c in cameras or []]})")


def _ray(K: np.ndarray, T: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, dtype=float)], axis=-1)
    return d @ T[:3, :3].T


def run(ctx: NodeContext, cameras: list, camera: str = "left_wrist") -> Output:
    cam = _camera(cameras, camera)
    rgb, depth = cam["rgb"], np.asarray(cam["depth"], dtype=np.float32)
    K = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
    T = pose_to_matrix(cam["pose"])

    candidates = []
    for prompt in PROMPTS:
        seg = ctx.tool("sam3.segment_text", image=rgb, query=prompt)
        for mask, score in zip(seg.get("masks") or [], seg.get("scores") or [], strict=False):
            m = np.asarray(mask) > 0
            if m.shape != depth.shape:
                continue
            area = int(m.sum())
            if area < MIN_AREA_PX:
                continue
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv2.contourArea)
            (_, _), r_px = cv2.minEnclosingCircle(contour)
            fill = area / (np.pi * r_px * r_px + 1e-9)
            ring = cv2.dilate(m.astype(np.uint8), np.ones((2 * RING_PX + 1,) * 2, np.uint8)).astype(bool) & ~m
            d_in, d_rim = depth[m & (depth > 0)], depth[ring & (depth > 0)]
            recess = (float(np.median(d_in)) - float(np.median(d_rim))) if d_in.size and d_rim.size else float("nan")
            ok = fill >= MIN_FILL and recess >= MIN_RECESS_M
            logger.info("[hole] %r score=%.3f area=%d r=%.0fpx fill=%.2f recess=%.3fm -> %s",
                        prompt, float(score), area, r_px, fill, recess, "candidate" if ok else "rejected")
            if ok:
                candidates.append((float(score) * fill, area, prompt, m, ring, contour))
    if not candidates:
        raise RuntimeError(f"no round through-hole found by SAM3 on {camera} (prompts {PROMPTS})")
    score, area, prompt, mask, ring, contour = max(candidates, key=lambda c: (round(c[0], 2), c[1]))

    # The rim ring -> world points -> plane (the spool's top face).
    pts = np.asarray(ctx.tool("geometry.mask_to_world_points", mask=(ring * 255).astype(np.uint8), depth=depth,
                              intrinsics=K, camera_pose=cam["pose"])["points"]["points"], dtype=np.float64)
    if len(pts) < 30:
        raise RuntimeError(f"too few valid depth points on the hole's rim ({len(pts)})")
    for _ in range(3):  # trimmed plane fit
        c0 = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c0)
        n = vt[2] / np.linalg.norm(vt[2])
        res = (pts - c0) @ n
        keep = np.abs(res) < max(2.5 * res.std(), 0.002)
        pts = pts[keep]
    if n[2] < 0:
        n = -n
    tilt_deg = float(np.degrees(np.arccos(min(1.0, abs(n[2])))))
    if tilt_deg < SNAP_VERTICAL_DEG:
        # A narrow rim ring fits its normal to a few degrees at best, and over an
        # 8 cm approach 4 degrees is 5 mm sideways; a part lying on the table has
        # the table's normal. Keep the fitted height.
        n = np.array([0.0, 0.0, 1.0])

    o = T[:3, 3]
    ys, xs = np.nonzero(mask)
    d = _ray(K, T, np.array([xs.mean()]), np.array([ys.mean()]))[0]
    center = o + d * float(n @ (c0 - o)) / float(n @ d)
    cu, cv = contour[:, 0, 0].astype(float), contour[:, 0, 1].astype(float)
    rays = _ray(K, T, cu, cv)
    s = (n @ (c0 - o)) / (rays @ n)
    edge = o + rays * s[:, None]
    radii = np.linalg.norm((edge - center) - np.outer((edge - center) @ n, n), axis=1)
    radius = float(np.median(radii))

    inside = (mask & (depth > 0)).astype(np.uint8) * 255
    hole_depth = None
    if int((inside > 0).sum()) > 30:
        pin = np.asarray(ctx.tool("geometry.mask_to_world_points", mask=inside, depth=depth, intrinsics=K,
                                  camera_pose=cam["pose"])["points"]["points"], dtype=np.float64)
        if len(pin) > 30:
            hole_depth = float(np.percentile((center - pin) @ n, 90))  # rim plane down to the bottom
    if hole_depth is None or hole_depth <= 0.0:
        surface_z = float(ctx.tool("robot.describe_workspace")["surface_z"])
        hole_depth = float(center[2] - surface_z)
    hole = {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "axis": {"x": float(n[0]), "y": float(n[1]), "z": float(n[2])},
        "radius_m": radius,
        "depth_m": hole_depth,
        "tilt_deg": tilt_deg,
        "axis_snapped_vertical": bool(tilt_deg < SNAP_VERTICAL_DEG),
        "score": score,
        "prompt": prompt,
        "pixel": [float(xs.mean()), float(ys.mean())],
        "camera": camera,
    }
    logger.info("[hole] %r at (%.3f, %.3f, %.3f) r=%.1f mm depth=%.1f mm tilt=%.1f deg (pixel %.0f,%.0f)",
                prompt, *center, radius * 1e3, hole_depth * 1e3, tilt_deg, *hole["pixel"])
    return {"hole": hole}
