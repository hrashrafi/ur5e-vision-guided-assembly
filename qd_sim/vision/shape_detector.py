"""Shape-based detectors for the wrist camera: find a QD's hexagonal flange
and the manifold's port holes from their silhouette, no markers involved.
The caller flies to a rough hover position above the tray or manifold;
these just refine that into pixel centroids for the real features, which
ray_cast.py then turns into world positions.

Both detectors follow the same recipe: threshold against the background,
find contours, filter by shape/size, return each match's centroid (via
image moments, so it's robust to an asymmetric contour).

Size thresholds are pixel-count based and calibrated against a 960x720
render at the hover heights record_full_session.py uses (TRAY_HOVER_Z=0.50,
MANIFOLD_HOVER_Z=0.30). They don't automatically scale to a different
render resolution or hover height - re-measure if either changes.
"""

import cv2
import numpy as np


def _centroid(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def detect_hexagons(image_bgr, min_area_px=400, max_area_px=6000,
                     min_vertices=5, max_vertices=8, min_circularity=0.6,
                     return_polygons=False):
    """QDs render as a bright, near-white hex flange against the blue-tinted
    floor. Brightness alone isn't enough to isolate it - the floor's grid
    lines are bright too - so this also requires low chroma (max channel -
    min channel), which the tinted floor fails and the white flange passes.

    A contour counts as a QD if its polygon approximation has roughly the
    right vertex count (5-8, not exactly 6 - render noise nudges a real hex
    off exact) and its circularity is above min_circularity, which rejects
    the two false positives seen in practice (a merged floor grid-line
    contour, a QD partly shadowed by the arm). Returns a list of (cx, cy)
    pixel centroids, or with return_polygons=True, (centroid, polygon)
    pairs so a caller can draw the actual detected outline instead of a
    generic marker.
    """
    chroma = np.max(image_bgr.astype(np.int16), axis=2) - np.min(image_bgr.astype(np.int16), axis=2)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = ((chroma < 20) & (gray > 150)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area_px < area < max_area_px):
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri ** 2)
        if circularity < min_circularity:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if not (min_vertices <= len(approx) <= max_vertices):
            continue
        centroid = _centroid(c)
        if centroid is not None:
            results.append((centroid, approx.reshape(-1, 2)))
    if return_polygons:
        return results
    return [centroid for centroid, _ in results]


def detect_holes(image_bgr, min_radius_px=10, max_radius_px=45, min_dist_px=50, accumulator_thresh=20,
                  return_radius=False):
    """Manifold ports render as a shaded circular hole (bright highlight on
    one side, dark crescent on the other) cut into the manifold's flat top.
    A plain intensity threshold only catches the dark half, so this uses
    cv2.HoughCircles instead, which handles a shaded-interior circular edge
    much better.

    accumulator_thresh had to drop from an initial 25 to 20 to reliably
    catch all 4 holes. min_dist_px is 50 rather than a tighter value
    because an occupied port's hex-plus-crescent shape registers as two
    overlapping Hough circles a few dozen pixels apart, which a smaller
    minDist wouldn't merge - 50 merges those while staying well under the
    real spacing between distinct ports.

    return_radius=True also returns each circle's detected radius, so an
    overlay can draw it at the size actually found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist_px,
        param1=80, param2=accumulator_thresh, minRadius=min_radius_px, maxRadius=max_radius_px,
    )
    if circles is None:
        return []
    if return_radius:
        return [(float(x), float(y), float(r)) for x, y, r in circles[0]]
    return [(float(x), float(y)) for x, y, _r in circles[0]]


def is_hole_occupied(image_bgr, cx, cy, patch_radius_px=55, min_line_segments=2):
    """True if the hole at (cx, cy) already has a QD installed in it.

    Reusing detect_hexagons() on the whole manifold frame doesn't work -
    its bright/low-chroma mask can't separate a QD from the manifold's own
    bright top face, so they merge into one blob. Instead this crops a
    small patch around the hole and looks for straight edges (Canny +
    probabilistic Hough lines): an empty hole is just a smooth shaded
    circle with no straight lines, while an installed QD's hex flange has
    real straight sides even partially visible. min_line_segments=2 leaves
    comfortable margin between the two cases.
    """
    h, w = image_bgr.shape[:2]
    x0, y0 = max(0, int(cx - patch_radius_px)), max(0, int(cy - patch_radius_px))
    x1, y1 = min(w, int(cx + patch_radius_px)), min(h, int(cy + patch_radius_px))
    crop = image_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=20, maxLineGap=3)
    n = 0 if lines is None else len(lines)
    return n >= min_line_segments
