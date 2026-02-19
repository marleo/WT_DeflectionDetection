import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from scipy.interpolate import UnivariateSpline

# =============================================================================
# CONFIGURATION
# =============================================================================
VIDEO_PATH     = r"C:\Users\Mario\Videos\ModelTurbine\TopDown\clockwise_8rpm_long.mp4"
MIN_BLADE_AREA = 5000
SPLINE_SMOOTHING = 5000   # Higher = smoother, more outlier rejection

# Define the frame windows in which spline + angle calculation is active.
# Format: list of (start_frame, end_frame) tuples  (inclusive on both ends).
# Example: two separate measurement windows.
CALC_WINDOWS = [
    (74, 83),
    (111, 121),
    (150, 157),
    (194, 206),
    (237, 247),
    (281, 287),
    (324, 333),
    (366, 376)
]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# TEMPORARY HELPER – auto-build CALC_WINDOWS from clockwise_8rpm_analyzed_data.csv
# DELETE this entire block (and the two assignment lines below) when you automate.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def _build_calc_windows_from_csv(
    csv_path: str = r"C:\Users\Mario\Documents\DORBINE_Projects\IQA\clockwise_8rpm_long_analyzed_data.csv",
    buffer: int = 4,
    angle_tolerance_rad: float = 0.05,
) -> tuple:
    """
    Reads *csv_path* and returns:
        windows  : list of (start_frame, end_frame) tuples
        blade_map: dict {window_index (1-based): "BladeX / BladeY" label}

    A window is centred on every frame where ANY blade angle is within
    *angle_tolerance_rad* of 0 rad (horizontal position).
    Buffer of *buffer* frames is added on each side; overlapping windows
    are merged.  When windows are merged the union of all triggering blade
    names is recorded.

    Parameters
    ----------
    csv_path            : path to the analyzed-data CSV
    buffer              : frames to add before and after each zero-crossing hit
    angle_tolerance_rad : how close to 0 rad counts as "horizontal"
    """
    df = pd.read_csv(csv_path)

    blade_cols = [c for c in df.columns if c.startswith("Blade") and "Angle_Rad" in c]

    # For each row, collect which blades are near zero
    near_zero_mask = df[blade_cols].abs() <= angle_tolerance_rad   # DataFrame of bools

    # Build per-frame hit list: [(frame, ["Blade1", ...]), ...]
    hits = []
    for _, row in df[near_zero_mask.any(axis=1)].iterrows():
        frame_no  = int(row["Frame"])
        blades    = [
            # "Blade1_Angle_Rad" → "Blade1"
            col.replace("_Angle_Rad", "")
            for col in blade_cols
            if abs(row[col]) <= angle_tolerance_rad
        ]
        hits.append((frame_no, blades))

    if not hits:
        print("[_build_calc_windows_from_csv] WARNING: no near-zero frames found – CALC_WINDOWS unchanged.")
        return [], {}

    # Build raw windows: (start, end, {blade_names})
    raw_windows = [(f - 4 * buffer, f - 3*buffer , set(blades)) for f, blades in hits] # Buffer right now is not for horizontal position (hor. would be f-buffer, f+buffer)
    raw_windows.sort(key=lambda x: x[0])

    # Merge overlapping / adjacent windows, union their blade sets
    merged = [list(raw_windows[0])]          # [start, end, {blades}]
    for start, end, blades in raw_windows[1:]:
        if start <= merged[-1][1] + 1:       # overlapping or adjacent
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] |= blades
        else:
            merged.append([start, end, blades])

    windows  = [(m[0], m[1]) for m in merged]
    blade_map = {
        i + 1: " / ".join(sorted(m[2]))
        for i, m in enumerate(merged)
    }

    print(f"[_build_calc_windows_from_csv] Found {len(hits)} zero-angle frame(s), "
          f"merged into {len(windows)} window(s):")
    for win_idx, (w, label) in enumerate(zip(windows, blade_map.values()), start=1):
        print(f"  Window {win_idx}: frames {w[0]}-{w[1]}  |  triggered by: {label}")

    return windows, blade_map


# TEMPORARY – override CALC_WINDOWS and build WINDOW_BLADE_LABELS from CSV.
# Remove these two lines together with the function above when automating.
_csv_windows, WINDOW_BLADE_LABELS = _build_calc_windows_from_csv()
CALC_WINDOWS = _csv_windows or CALC_WINDOWS
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# =============================================================================
# PROCESSING FUNCTIONS  (unchanged from DeflectionTracker_spline.py)
# =============================================================================

def get_cleaned_mask(frame, fgbg):
    mask = fgbg.apply(frame, learningRate=0.005)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def prune_skeleton(skeleton_img):
    """Keep only the longest connected component of the skeleton."""
    contours, _ = cv2.findContours(skeleton_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(skeleton_img)
    longest = max(contours, key=lambda c: cv2.arcLength(c, False))
    pruned = np.zeros_like(skeleton_img)
    cv2.drawContours(pruned, [longest], -1, 255, 1)
    return pruned


def fit_smooth_spline(skeleton_img, smoothing=SPLINE_SMOOTHING):
    """
    PCA-ordered, arc-length parameterised smoothing spline.
    Returns (N,2) array of fitted curve points, or None on failure.
    """
    ys, xs = np.where(skeleton_img > 0)
    if len(xs) < 6:
        return None

    pts = np.column_stack([xs, ys]).astype(float)

    # Order by projection onto principal axis (O(n), no loop-closing)
    mean    = pts.mean(axis=0)
    centered = pts - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    order   = np.argsort(centered @ Vt[0])
    pts     = pts[order]

    # Subsample
    if len(pts) > 300:
        idx = np.round(np.linspace(0, len(pts) - 1, 300)).astype(int)
        pts = pts[idx]

    # Arc-length parameter t
    diffs = np.diff(pts, axis=0)
    t = np.concatenate([[0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    _, unique_idx = np.unique(t, return_index=True)
    t   = t[unique_idx]
    pts = pts[unique_idx]

    if len(t) < 6:
        return None

    try:
        spl_x = UnivariateSpline(t, pts[:, 0], s=smoothing, k=3)
        spl_y = UnivariateSpline(t, pts[:, 1], s=smoothing, k=3)
    except Exception:
        return None

    t_fine = np.linspace(t[0], t[-1], 200)
    return np.column_stack([spl_x(t_fine), spl_y(t_fine)])


def measure_deflection_angle(spline_pts, fraction=0.25):
    """
    Angle between PCA tangent at root (first `fraction`) and tip (last `fraction`).
    Returns (angle_deg, v_root, v_tip, root_anchor, tip_anchor).
    """
    n     = len(spline_pts)
    n_end = max(6, int(n * fraction))

    root_pts = spline_pts[:n_end]
    tip_pts  = spline_pts[-n_end:]

    def pca_dir(pts):
        c = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        return Vt[0]

    v_root = pca_dir(root_pts)
    v_tip  = pca_dir(tip_pts)

    cos_a     = np.clip(np.abs(np.dot(v_root, v_tip)), 0.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_a))

    return angle_deg, v_root, v_tip, root_pts.mean(axis=0), tip_pts.mean(axis=0)


# =============================================================================
# HELPERS: styled HUD drawing
# =============================================================================

_FONT      = cv2.FONT_HERSHEY_DUPLEX
_FONT_SM   = 0.55   # small label scale
_FONT_MD   = 0.68   # medium value scale
_PAD       = 8      # internal padding for badge backgrounds


def _draw_badge(frame, text, x, y, font_scale=_FONT_MD,
                txt_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.55,
                thickness=1):
    """
    Draws *text* at (x, y) with a semi-transparent dark background bar.
    Returns the bottom-y of the drawn element so callers can chain vertically.
    """
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    x1, y1 = x - _PAD, y - th - _PAD
    x2, y2 = x + tw + _PAD, y + baseline + _PAD
    # clamp to frame bounds
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size:
        overlay = roi.copy()
        overlay[:] = bg_color
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
        frame[y1:y2, x1:x2] = roi
    cv2.putText(frame, text, (x, y), _FONT, font_scale, txt_color, thickness, cv2.LINE_AA)
    return y2 + 4   # next available y


def draw_results_overlay(frame, completed_results, blade_labels=None):
    """
    Draws only the most recently completed window result (bottom-left, below frame counter).
    Replaces the previous result instead of stacking.
    """
    if not completed_results:
        return
    blade_labels = blade_labels or {}
    latest_idx = max(completed_results.keys())
    avg_angle  = completed_results[latest_idx]
    blade_tag  = blade_labels.get(latest_idx, "")
    blade_part = f"  {blade_tag}" if blade_tag else ""
    # Two-line badge: dim label row + bright value row
    y = _draw_badge(frame, f"LAST RESULT  •  Win {latest_idx}{blade_part}",
                    20, 108, font_scale=_FONT_SM,
                    txt_color=(180, 180, 180), bg_color=(20, 20, 20), alpha=0.6)
    _draw_badge(frame, f"{avg_angle:.2f} deg  (avg)",
                20, y + 18, font_scale=_FONT_MD,
                txt_color=(0, 220, 255), bg_color=(0, 40, 60), alpha=0.65)


# =============================================================================
# MAIN
# =============================================================================

def main(calc_windows=CALC_WINDOWS):
    """
    calc_windows : list of (start_frame, end_frame) tuples.
                   Within each window the spline + angle calculation runs.
                   Masking is always active.
    """
    cap  = cv2.VideoCapture(VIDEO_PATH)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50,
                                               detectShadows=True)

    # Per-window angle accumulator  {window_index: [list of per-frame angles]}
    window_angles = {i + 1: [] for i in range(len(calc_windows))}

    # Completed window averages shown on screen  {window_index: avg_angle}
    completed_results = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # ------------------------------------------------------------------
        # 1. Always compute mask (background model trains on every frame)
        # ------------------------------------------------------------------
        mask = get_cleaned_mask(frame, fgbg)
        display_frame    = frame.copy()
        deflection_angle = None

        # ------------------------------------------------------------------
        # 2. Check which (if any) calc window is active for this frame
        # ------------------------------------------------------------------
        active_window_idx = None
        for i, (start, end) in enumerate(calc_windows):
            win_idx = i + 1
            if start <= frame_idx <= end:
                active_window_idx = win_idx
            # Window just ended on the previous frame → finalise average
            elif frame_idx == end + 1 and win_idx not in completed_results:
                angles = window_angles[win_idx]
                if angles:
                    completed_results[win_idx] = float(np.mean(angles))

        # ------------------------------------------------------------------
        # 3. Spline + angle only inside an active window
        # ------------------------------------------------------------------
        if active_window_idx is not None and np.sum(mask > 0) > MIN_BLADE_AREA:

            skeleton_bool  = skeletonize(mask > 0)
            skeleton_uint8 = (skeleton_bool * 255).astype(np.uint8)
            clean_skeleton = prune_skeleton(skeleton_uint8)

            # --------------------------------------------------
            spline_pts = fit_smooth_spline(clean_skeleton)
            if spline_pts is not None:
                # Draw spline in cyan
                for i in range(len(spline_pts) - 1):
                    p1 = (int(round(spline_pts[i,     0])), int(round(spline_pts[i,     1])))
                    p2 = (int(round(spline_pts[i + 1, 0])), int(round(spline_pts[i + 1, 1])))
                    cv2.line(display_frame, p1, p2, (255, 255, 0), 2)

                # Measure and accumulate angle
                angle, v_root, v_tip, anch_root, anch_tip = measure_deflection_angle(spline_pts)
                deflection_angle = angle
                window_angles[active_window_idx].append(angle)

                # Draw root (green) and tip (red) tangent arrows
                arrow_len = 60
                r0 = (int(round(anch_root[0])), int(round(anch_root[1])))
                r1 = (int(round(anch_root[0] + v_root[0] * arrow_len)),
                      int(round(anch_root[1] + v_root[1] * arrow_len)))
                cv2.arrowedLine(display_frame, r0, r1, (0, 255, 0), 2, tipLength=0.2)

                t0 = (int(round(anch_tip[0])), int(round(anch_tip[1])))
                t1 = (int(round(anch_tip[0] + v_tip[0] * arrow_len)),
                      int(round(anch_tip[1] + v_tip[1] * arrow_len)))
                cv2.arrowedLine(display_frame, t0, t1, (0, 0, 255), 2, tipLength=0.2)
            # --------------------------------------------------

        # ------------------------------------------------------------------
        # 4. HUD  (left = persistent info | right = live active-window panel)
        # ------------------------------------------------------------------
        h_frame, w_frame = display_frame.shape[:2]

        # --- LEFT: frame counter (always visible) ---
        _draw_badge(display_frame, f"FRAME  {frame_idx}",
                    20, 38, font_scale=_FONT_SM,
                    txt_color=(0, 230, 230), bg_color=(10, 10, 10), alpha=0.55)

        # --- LEFT: last completed result (replaces previous each window) ---
        draw_results_overlay(display_frame, completed_results, WINDOW_BLADE_LABELS)

        # --- RIGHT: live active-window panel ---
        if active_window_idx is not None:
            win_start, win_end = calc_windows[active_window_idx - 1]
            blade_tag  = WINDOW_BLADE_LABELS.get(active_window_idx, "")

            # Anchor right-aligned: compute x from right edge
            right_margin = w_frame - 20

            def _rbadge(text, y, font_scale=_FONT_MD, txt_color=(255,255,255),
                        bg_color=(0,0,0), alpha=0.55):
                """Right-aligned badge helper."""
                (tw, _th), _bl = cv2.getTextSize(text, _FONT, font_scale, 1)
                x = right_margin - tw
                return _draw_badge(display_frame, text, x, y,
                                   font_scale=font_scale, txt_color=txt_color,
                                   bg_color=bg_color, alpha=alpha)

            # Header: window index + blade tag
            header = (f"AngleCalc {active_window_idx}  •  {blade_tag}"
                      if blade_tag else f"AngleCalc {active_window_idx}")
            y_r = _rbadge(header, 38, font_scale=_FONT_SM,
                          txt_color=(200, 200, 200), bg_color=(20, 20, 20), alpha=0.6)

            # Frame range
            y_r = _rbadge(f"Frames  {win_start} – {win_end}", y_r + 16,
                          font_scale=_FONT_SM,
                          txt_color=(140, 180, 200), bg_color=(10, 10, 30), alpha=0.55)

            # Live deflection (large + bright when available)
            if deflection_angle is not None:
                _rbadge(f"{deflection_angle:.2f} deg", y_r + 20,
                        font_scale=0.90,
                        txt_color=(50, 230, 120), bg_color=(0, 30, 15), alpha=0.65)
            else:
                _rbadge("-- deg", y_r + 20, font_scale=0.90,
                        txt_color=(120, 120, 120), bg_color=(20, 20, 20), alpha=0.50)

        cv2.imshow("DeflectionTracker (UDP)", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        if key == ord(' '):
            cv2.waitKey(-1)

    # Finalise any window that was still open when the video ended
    for i, (start, end) in enumerate(calc_windows):
        win_idx = i + 1
        if win_idx not in completed_results and window_angles[win_idx]:
            completed_results[win_idx] = float(np.mean(window_angles[win_idx]))

    # Print final summary to console
    print("\n=== Deflection Angle Summary ===")
    for idx, avg in sorted(completed_results.items()):
        start, end = calc_windows[idx - 1]
        blade_tag  = WINDOW_BLADE_LABELS.get(idx, "unknown blade")
        print(f"  AngleCalc {idx}  (frames {start}-{end})  [{blade_tag}]:  avg. {avg:.2f} deg")

    # --- Per-blade total average ---
    # A window may list multiple blades (e.g. "Blade1 / Blade2"); attribute it to each.
    blade_accum: dict[str, list[float]] = {}
    for idx, avg in completed_results.items():
        label = WINDOW_BLADE_LABELS.get(idx, "unknown blade")
        for blade in [b.strip() for b in label.split("/")]:
            blade_accum.setdefault(blade, []).append(avg)

    print("\n=== Per-Blade Total Average ===")
    for blade in sorted(blade_accum):
        values   = blade_accum[blade]
        total    = float(np.mean(values))
        n_win    = len(values)
        print(f"  {blade}:  avg. {total:.2f} deg  (over {n_win} window{'s' if n_win != 1 else ''})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(CALC_WINDOWS)
