import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.interpolate import UnivariateSpline

# --- CONFIGURATION ---
VIDEO_PATH = r"C:\Users\Mario\Videos\ModelTurbine\TopDown\clockwise_14rpm.mp4"
MIN_BLADE_AREA = 5000
# Controls how aggressively outliers are smoothed out.
# Higher = smoother / more outlier rejection. Lower = closer to raw points.
SPLINE_SMOOTHING = 5000

def get_cleaned_mask(frame, fgbg):
    mask = fgbg.apply(frame, learningRate=0.005)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    
    # Initial cleanup
    # cv2.imshow("Mask Before Cleaning", mask)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    # cv2.imshow("Mask After Cleaning", mask)
    return mask

def prune_skeleton(skeleton_img):
    """
    Finds all connected components in the skeleton and keeps only the longest one.
    """
    # Find all connected components (the main spine and the hairs)
    contours, _ = cv2.findContours(skeleton_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.zeros_like(skeleton_img)
    
    # Find the longest contour based on arc length
    longest_contour = max(contours, key=lambda x: cv2.arcLength(x, False))
    
    # Create a blank mask and draw only the longest line
    pruned = np.zeros_like(skeleton_img)
    cv2.drawContours(pruned, [longest_contour], -1, 255, 1)
    
    return pruned

# --------------------------------------------------
def fit_smooth_spline(skeleton_img, smoothing=SPLINE_SMOOTHING):
    """
    Orders skeleton pixels by projecting onto the blade's principal axis (PCA),
    then fits a smoothing UnivariateSpline for x and y independently.

    - O(n) ordering via PCA projection  →  no loop-closing, much faster
    - Subsamples pixels before fitting  →  lower CPU cost per frame

    Returns:
        pts_fit : (N, 2) float array of fitted curve points, or None on failure.
    """
    ys, xs = np.where(skeleton_img > 0)
    if len(xs) < 6:
        return None

    pts = np.column_stack([xs, ys]).astype(float)

    # --- Fast ordering: project onto the 1st principal component ---
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    principal_axis = Vt[0]                        # dominant direction
    projections = centered @ principal_axis        # scalar per point
    order = np.argsort(projections)               # sort along blade axis
    pts = pts[order]
    # ---------------------------------------------------------------

    # Subsample to at most 300 points to keep fitting cheap
    if len(pts) > 300:
        idx = np.round(np.linspace(0, len(pts) - 1, 300)).astype(int)
        pts = pts[idx]

    # Arc-length parameter t
    diffs = np.diff(pts, axis=0)
    t = np.concatenate([[0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])

    # Drop duplicate t values (skeleton pixel clusters)
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

    t_fine = np.linspace(t[0], t[-1], 200)   # 200 pts is plenty to look smooth
    return np.column_stack([spl_x(t_fine), spl_y(t_fine)])
# --------------------------------------------------

# --------------------------------------------------
def measure_deflection_angle(spline_pts, fraction=0.25):
    """
    Computes the deflection angle of a blade by comparing the tangent direction
    at the root (first `fraction` of spline points) vs the tip (last `fraction`).

    Uses PCA on each subset to get a robust dominant direction vector.

    Returns:
        angle_deg  : deflection angle in degrees (0 = perfectly straight blade)
        root_vec   : unit direction vector at the root
        tip_vec    : unit direction vector at the tip
        root_anchor: midpoint of the root subset (for drawing)
        tip_anchor : midpoint of the tip  subset (for drawing)
    """
    n = len(spline_pts)
    n_end = max(6, int(n * fraction))

    root_pts = spline_pts[:n_end]
    tip_pts  = spline_pts[-n_end:]

    def pca_direction(pts):
        centered = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        return Vt[0]   # unit vector along dominant direction

    v_root = pca_direction(root_pts)
    v_tip  = pca_direction(tip_pts)

    # np.abs handles the fact that PCA eigenvectors have arbitrary sign
    cos_angle = np.clip(np.abs(np.dot(v_root, v_tip)), 0.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))

    return angle_deg, v_root, v_tip, root_pts.mean(axis=0), tip_pts.mean(axis=0)
# --------------------------------------------------

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        mask = get_cleaned_mask(frame, fgbg)
        deflection_angle = None

        if np.sum(mask > 0) > MIN_BLADE_AREA:
            # 1. Generate Raw Skeleton
            skeleton_bool = skeletonize(mask > 0)
            skeleton_uint8 = (skeleton_bool * 255).astype(np.uint8)
            
            # 2. Prune: Keep only the longest line
            clean_skeleton = prune_skeleton(skeleton_uint8)
            
            # 3. Visualization
            display_frame = frame.copy()

            # --------------------------------------------------
            # 4. Fit a smoothing spline and draw only that line
            spline_pts = fit_smooth_spline(clean_skeleton)
            if spline_pts is not None:
                # Draw the fitted spline in cyan
                for i in range(len(spline_pts) - 1):
                    p1 = (int(round(spline_pts[i,   0])), int(round(spline_pts[i,   1])))
                    p2 = (int(round(spline_pts[i+1, 0])), int(round(spline_pts[i+1, 1])))
                    cv2.line(display_frame, p1, p2, (255, 255, 0), 2)

                # 5. Measure deflection angle
                angle, v_root, v_tip, anch_root, anch_tip = measure_deflection_angle(spline_pts)
                deflection_angle = angle
                arrow_len = 60
                # Root tangent in green
                r0 = (int(round(anch_root[0])), int(round(anch_root[1])))
                r1 = (int(round(anch_root[0] + v_root[0] * arrow_len)),
                      int(round(anch_root[1] + v_root[1] * arrow_len)))
                cv2.arrowedLine(display_frame, r0, r1, (0, 255, 0), 2, tipLength=0.2)
                # Tip tangent in red
                t0 = (int(round(anch_tip[0])),  int(round(anch_tip[1])))
                t1 = (int(round(anch_tip[0]  + v_tip[0]  * arrow_len)),
                      int(round(anch_tip[1]  + v_tip[1]  * arrow_len)))
                cv2.arrowedLine(display_frame, t0, t1, (0, 0, 255), 2, tipLength=0.2)
            # --------------------------------------------------
            
        else:
            display_frame = frame.copy()

        # Debug UI
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.putText(display_frame, f"Frame: {frame_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if deflection_angle is not None:
            angle_text = f"Deflection: {deflection_angle:.2f} deg"
        else:
            angle_text = "Deflection: N/A"
        cv2.putText(display_frame, angle_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Pruned Skeleton (Longest Line Only)', display_frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27: break
        if key == ord(' '): cv2.waitKey(-1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()