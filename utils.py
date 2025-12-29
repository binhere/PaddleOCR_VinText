from shapely.ops import unary_union
from shapely.geometry import Polygon, Point
import numpy as np
import cv2


def parse_vintext_label(label_path):
    """Read VinText gt_X.txt and return valid word boxes with text."""
    words = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue

            try:
                coords = list(map(float, parts[:8]))  # first 8 are coordinates
            except ValueError:
                continue

            # The rest (joined by comma) is the text — to handle commas inside text
            text = ",".join(parts[8:]).strip()

            if text == "###" or not text:
                continue

            pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            words.append({"bbox": pts, "text": text})
    return words


def polygon_to_quad(polygon: Polygon, ref_pt: tuple[float, float] | None = None):
    pts = np.array(polygon.exterior.coords, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    # box must be in clockwise order (default in this cv2 version) before going next step
    # otherwise, we have to turn it clockwise, for example: box = ensure_clockwise(box)

    # If a reference point is given, align box accordingly
    if ref_pt is not None:
        ref_pt = np.array(ref_pt, dtype=np.float32)
        dists = np.linalg.norm(box - ref_pt, axis=1)
        start_idx = np.argmin(dists)
        box = np.roll(box, -start_idx, axis=0)

    return box.tolist()


def safe_polygon(points : np.array) -> Polygon:
    """
    Create a valid shapely Polygon.
    - Remove nearly overlapping consecutive points
    - Keep original geometry location
    """
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)  # fix small self-intersections
    if not poly.is_valid: # or poly.area == 0:
        return None  # still invalid — skip
    return poly


def has_row_conflict(words):
    """
    Detect if any pair of words overlaps horizontally too much, 
    which suggests they might belong to different rows.
    
    This assumes nearly horizontal text lines. 
    Conflict occurs if two words have large horizontal overlap OR one word's center x lies within the other's x-range.
    """
    n = len(words)

    for i in range(n):
        for j in range(i + 1, n):
            w1, w2 = words[i], words[j]
            xs1 = [p[0] for p in w1["bbox"]]
            xs2 = [p[0] for p in w2["bbox"]]
            min_x1, max_x1 = min(xs1), max(xs1)
            min_x2, max_x2 = min(xs2), max(xs2)

            # horizontal overlap width
            overlap_x = min(max_x1, max_x2) - max(min_x1, min_x2)
            if overlap_x <= 0:
                continue  # no overlap at all

            width1 = max_x1 - min_x1
            width2 = max_x2 - min_x2
            overlap_ratio = overlap_x / min(width1, width2)

            # center x positions
            cx1 = (min_x1 + max_x1) / 2
            cx2 = (min_x2 + max_x2) / 2

            # check if centers fall inside other's horizontal span
            center_inside = (
                (min_x1 < cx2 < max_x1) or
                (min_x2 < cx1 < max_x2)
            )

            # conflict: strong horizontal overlap + center alignment
            if center_inside or overlap_ratio > 0.6:
                return True

    return False


def group_to_lines(word_boxes, dt_polys, min_words_per_line=2, min_chars_to_split=26, use_extra_ref=False, mode="manual"):
    """
    Group VinText words into line boxes returned by PaddleOCR detector.
    Optionally split long text lines if their total character length exceeds min_chars_to_split.

    Args:
        word_boxes: list of dicts with "bbox" (4-pt polygon) and "text".
        dt_polys: list of line polygons (from PaddleOCR detector).
        min_words_per_line: minimum number of words required for a valid line (default=2).
                            Set to 1 to disable filtering.
        min_chars_to_split: int or None. If set, split lines longer than this threshold.
        external_ref_pt : np.array or None. If set, use this point as reference for *FULL* line alignment only.

    Returns:
        grouped_lines: list of dicts with keys "line_box" and "text".
    """
    assert min_words_per_line >= 1, "min_words_per_line must be >= 1"
    assert mode in ["manual", "auto"], "mode must be `manual` or `auto`"
    
    line_polys = [safe_polygon(poly) for poly in dt_polys]
    grouped_lines = []

    for lpoly in line_polys:
        # find words whose centers are inside this line polygon
        contained_words = []
        for w in word_boxes:
            wx, wy = np.mean(np.array(w["bbox"]), axis=0)
            if lpoly.contains(Point(wx, wy)):
                contained_words.append(w)

        # skip if too few words
        if len(contained_words) < min_words_per_line:
            continue
        
        # only check row conflict if mode is "auto"
        # in mode auto, we assume that text lines are (nearly) horizontal and PaddleOCR TextDetector predicts inaccurate dt_polys 
        if mode=="auto" and has_row_conflict(contained_words):
            continue

        # sort words left to right
        contained_words = sorted(contained_words, key=lambda w: min(p[0] for p in w["bbox"]))

        # --- get full text line ---
        merged_poly = unary_union([safe_polygon(w["bbox"]) for w in contained_words])
        if not merged_poly.is_valid:
            merged_poly = merged_poly.buffer(0)
        merged_poly = merged_poly.convex_hull

        # only support use_extra_ref for full line
        # if use_extra_ref is False, use first point of first sorted word as reference 
        full_text = " ".join(w["text"] for w in contained_words)
        ref_pt = contained_words[0]["bbox"][0]
        if use_extra_ref:
            ref_pt = lpoly.exterior.coords[0]
        full_box = polygon_to_quad(merged_poly, ref_pt=ref_pt)

        grouped_lines.append({
            "line_box": full_box,
            "text": full_text
        })

        # --- get splited text lines at the middle ---
        if min_chars_to_split and len(full_text) >= min_chars_to_split and len(contained_words) >= 3:
            mid_idx = len(contained_words) // 2
            left_words = contained_words[:mid_idx+1]
            right_words = contained_words[mid_idx:]

            for subset in (left_words, right_words):
                if len(subset) < min_words_per_line:
                    continue

                sub_text = " ".join(w["text"] for w in subset)
                merged_poly = unary_union([safe_polygon(w["bbox"]) for w in subset])
                if not merged_poly.is_valid:
                    merged_poly = merged_poly.buffer(0)
                merged_poly = merged_poly.convex_hull

                ref_pt = subset[0]["bbox"][0]
                box4 = polygon_to_quad(merged_poly, ref_pt=ref_pt)

                grouped_lines.append({
                    "line_box": box4,
                    "text": sub_text
                })

    return grouped_lines


def get_rotate_crop_image(img, points, rotate_90_counter_clockwise=False):
	assert len(points) == 4, "shape of points must be 4*2"

	img_crop_width = int(
		max(
			np.linalg.norm(points[0] - points[1]),
			np.linalg.norm(points[2] - points[3]),
		)
	)
	img_crop_height = int(
		max(
			np.linalg.norm(points[0] - points[3]),
			np.linalg.norm(points[1] - points[2]),
		)
	)
	pts_std = np.float32(
		[
			[0, 0],
			[img_crop_width, 0],
			[img_crop_width, img_crop_height],
			[0, img_crop_height],
		]
	)
	M = cv2.getPerspectiveTransform(points, pts_std)
	dst_img = cv2.warpPerspective(
		img,
		M,
		(img_crop_width, img_crop_height),
		borderMode=cv2.BORDER_REPLICATE,
		flags=cv2.INTER_CUBIC,
	)

    # This parameter should be False when processing single words and True when processing text lines
	if rotate_90_counter_clockwise:
		dst_img_height, dst_img_width = dst_img.shape[0:2]
		if dst_img_height * 1.0 / dst_img_width >= 1.5:
			dst_img = np.rot90(dst_img)
        
	return dst_img