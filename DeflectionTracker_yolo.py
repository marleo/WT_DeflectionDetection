import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = 'C:\\Users\\Mario\\Videos\\ModelTurbine\\TopDown\\clockwise_8rpm.mp4'
MODEL_PATH = 'best.pt'
MIN_BLADE_AREA = 5000
CONF_THRES = 0.25       # YOLO confidence threshold
IMG_SIZE = 640          # YOLO inference resolution

# ROOT: Fixed pixel range for the rigid shaft
ROOT_X_START = 150
ROOT_X_END = 450

# TIP: Percentage of the detected contour
TIP_REGION_START = 0.75


def get_vector_angle(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    det = v1[0]*v2[1] - v1[1]*v2[0]
    return np.degrees(np.arctan2(det, dot))


def draw_line_from_vector(img, fit_data, color, length=400):
    if fit_data is None:
        return
    vx, vy, x, y = fit_data[0][0], fit_data[1][0], fit_data[2][0], fit_data[3][0]
    p1 = (int(x - vx * length), int(y - vy * length))
    p2 = (int(x + vx * length), int(y + vy * length))
    cv2.line(img, p1, p2, color, 2)


def overlay_mask(frame, mask, color=(255, 255, 0), alpha=0.2):
    """Applies a transparent colored tint where the mask is active."""
    colored_mask = np.zeros_like(frame)
    colored_mask[mask > 0] = color
    return cv2.addWeighted(frame, 1.0, colored_mask, alpha, 0)


def get_yolo_mask(frame, model):
    """
    Runs YOLOv8 segmentation on the frame and returns a binary mask
    (uint8, same H×W as frame) covering the largest detected blade.
    Returns an all-zero mask if nothing is detected.
    """
    h, w = frame.shape[:2]
    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES,
                            verbose=False)[0]

    best_mask = None
    best_area = 0

    if results.masks is not None:
        for seg_mask in results.masks.data:
            # seg_mask is a float32 tensor of shape (H_model, W_model)
            # Resize to original frame dimensions
            m = seg_mask.cpu().numpy().astype(np.uint8) * 255
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            area = int(np.count_nonzero(m))
            if area > best_area:
                best_area = area
                best_mask = m

    if best_mask is None:
        best_mask = np.zeros((h, w), dtype=np.uint8)

    return best_mask


def get_midline_from_range(mask, x_start, x_end):
    spine_points = []
    for x in range(x_start, x_end, 5):
        col = mask[:, x]
        indices = np.where(col > 0)[0]
        if len(indices) > 10:
            spine_points.append([x, (indices.min() + indices.max()) / 2])

    if len(spine_points) > 5:
        pts_array = np.array(spine_points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.fitLine(pts_array, cv2.DIST_L2, 0, 0.01, 0.01), spine_points
    return None, []


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    paused, slow_mo = False, False

    while True:
        delay = 0 if paused else (150 if slow_mo else 30)
        key = cv2.waitKey(delay) & 0xFF
        if key == 27:
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('s'):
            slow_mo = not slow_mo
        if key == ord('d'):
            pass
        elif paused and key != ord('d'):
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # --- SEGMENTATION via YOLOv8 ---
        mask = get_yolo_mask(frame, model)

        # Apply the transparent overlay to the frame
        display_frame = overlay_mask(frame, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > MIN_BLADE_AREA:

                fit_root, debug_pts = get_midline_from_range(mask, ROOT_X_START, ROOT_X_END)

                points = c[:, 0, :]
                points = points[points[:, 0].argsort()]
                tip_pts = points[int(len(points) * TIP_REGION_START):]

                if fit_root is not None and len(tip_pts) > 10:
                    fit_tip = cv2.fitLine(tip_pts, cv2.DIST_L2, 0, 0.01, 0.01)
                    v_root = (fit_root[0][0], fit_root[1][0])
                    v_tip = (fit_tip[0][0], fit_tip[1][0])
                    angle = get_vector_angle(v_root, v_tip)

                    # Draw analysis on the display_frame
                    for pt in debug_pts:
                        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 2, (180, 105, 255), -1)

                    draw_line_from_vector(display_frame, fit_root, (0, 255, 0))
                    draw_line_from_vector(display_frame, fit_tip, (0, 0, 255))

                    cv2.putText(display_frame, f"Bend: {abs(angle):.2f} deg", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('YOLO Segmentation Tracker', display_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()