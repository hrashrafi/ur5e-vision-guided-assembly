"""Drawing helpers for turning a detected shape (a hex polygon, a hole
circle) into an annotated frame with a non-overlapping label. Shared by
scripts/record_vision_pipeline.py and qd_sim/viz/live_dashboard.py so both
draw detections the same way.

Color convention: the shape actually being tracked this step gets its own
bright color, any other detected-but-not-tracked shape is dimmed gray, and
an occupied port gets a third color.
"""

import cv2
import numpy as np

HEX_COLOR = (0, 220, 60)       # bright green - the hex flange actually being tracked
HOLE_COLOR = (0, 200, 255)     # bright amber - the empty hole actually being tracked
OCCUPIED_COLOR = (0, 90, 255)  # orange-red - a port classified occupied (hex detected there)
DIM_COLOR = (210, 210, 210)    # light gray - a detected shape that ISN'T the one being tracked this step
COAST_TEXT_COLOR = (0, 0, 255)


def _label_box_size(lines, font, scale, thick, pad):
    sizes = [cv2.getTextSize(line, font, scale, thick)[0] for line in lines]
    w = max(s[0] for s in sizes) + 2 * pad
    line_h = max(s[1] for s in sizes) + 6
    h = line_h * len(lines) + pad
    return w, h, line_h


def _rects_overlap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def draw_labels(img, requests, obstacles=()):
    """requests: list of (anchor_x, anchor_y, lines, color). Places each
    label near its anchor, offset just enough to avoid overlapping any
    label already placed this frame or any rect in `obstacles` (the shape
    markers themselves).

    Side (left/right of the anchor) is picked by which half of the image
    the anchor falls in, so labels spread outward instead of crowding the
    middle - matters most for the manifold's 4-in-a-row holes. Vertical
    placement is a simple greedy search (grow an offset, alternate up/
    down), good enough for the handful of labels any one frame has. A thin
    leader line connects each label back to its anchor."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, pad = 0.5, 1, 4
    placed = list(obstacles)
    img_w = img.shape[1]
    for x, y, lines, color in requests:
        w, h, line_h = _label_box_size(lines, font, scale, thick, pad)
        on_right = x > img_w / 2
        base_tx = int(x) + 16 if on_right else int(x) - 16 - w
        base_ty = int(y) - h // 2
        rect = None
        for step in range(0, 30):
            for dy in ([0] if step == 0 else [step * 16, -step * 16]):
                tx = max(0, min(base_tx, img_w - w))
                ty = max(0, min(base_ty + dy, img.shape[0] - h))
                candidate = (tx, ty, tx + w, ty + h)
                if not any(_rects_overlap(candidate, p) for p in placed):
                    rect = candidate
                    break
            if rect:
                break
        if rect is None:
            tx = max(0, min(base_tx, img_w - w))
            ty = max(0, min(base_ty, img.shape[0] - h))
            rect = (tx, ty, tx + w, ty + h)
        tx, ty, _, _ = rect
        placed.append(rect)

        anchor = (int(x), int(y))
        edge = (tx, ty + h // 2) if on_right else (tx + w, ty + h // 2)
        cv2.line(img, anchor, edge, color, 1, cv2.LINE_AA)
        overlay = img.copy()
        cv2.rectangle(overlay, (tx, ty), (tx + w, ty + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, dst=img)
        cv2.rectangle(img, (tx, ty), (tx + w, ty + h), color, 1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (tx + pad, ty + pad + (i + 1) * line_h - 6),
                        font, scale, color, thick, cv2.LINE_AA)


def annotate(img_bgr, shapes, labels):
    """shapes: list of ("polygon", points_Nx2, color) or ("circle", (cx,cy,r), color)
    - drawn as the actual detected outline, not a generic marker. labels:
    passed to draw_labels(), which keeps each shape's bounding box
    off-limits for label placement."""
    out = img_bgr.copy()
    obstacles = []
    pad = 6  # a little breathing room around each shape
    for kind, data, color in shapes:
        if kind == "polygon":
            pts = np.asarray(data, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
            cx, cy = np.asarray(data).mean(axis=0)
            cv2.circle(out, (int(cx), int(cy)), 3, color, -1, cv2.LINE_AA)
            arr = np.asarray(data)
            x0, y0 = arr[:, 0].min() - pad, arr[:, 1].min() - pad
            x1, y1 = arr[:, 0].max() + pad, arr[:, 1].max() + pad
            obstacles.append((x0, y0, x1, y1))
        else:
            cx, cy, r = data
            cv2.circle(out, (int(cx), int(cy)), int(r), color, 2, cv2.LINE_AA)
            cv2.circle(out, (int(cx), int(cy)), 3, color, -1, cv2.LINE_AA)
            obstacles.append((cx - r - pad, cy - r - pad, cx + r + pad, cy + r + pad))
    draw_labels(out, labels, obstacles=obstacles)
    return out
