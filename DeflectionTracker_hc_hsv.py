import cv2
import numpy as np

# --- CONFIGURATION ---
VIDEO_PATH = 'C:\\Users\\Mario\\Videos\\ModelTurbine\\TopDown\\clockwise_8rpm.mp4'
MIN_BLADE_AREA = 5000

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
    if fit_data is None: return
    vx, vy, x, y = fit_data[0][0], fit_data[1][0], fit_data[2][0], fit_data[3][0]
    p1 = (int(x - vx * length), int(y - vy * length))
    p2 = (int(x + vx * length), int(y + vy * length))
    cv2.line(img, p1, p2, color, 2)

def overlay_mask(frame, mask, color=(255, 255, 0), alpha=0.2):
    """Applies a transparent colored tint where the mask is active."""
    colored_mask = np.zeros_like(frame)
    colored_mask[mask > 0] = color
    # Blend the original frame with the colored mask
    return cv2.addWeighted(frame, 1.0, colored_mask, alpha, 0)

def get_combined_mask(frame, fgbg):
    motion_mask = fgbg.apply(frame)
    _, motion_mask = cv2.threshold(motion_mask, 250, 255, cv2.THRESH_BINARY)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 160]) 
    upper_white = np.array([180, 60, 255])
    color_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    combined = cv2.bitwise_or(motion_mask, color_mask)
    kernel = np.ones((5,5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return combined

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
    cap = cv2.VideoCapture(VIDEO_PATH)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
    paused, slow_mo = False, False

    while True:
        delay = 0 if paused else (150 if slow_mo else 30)
        key = cv2.waitKey(delay) & 0xFF
        if key == 27: break
        if key == ord(' '): paused = not paused
        if key == ord('s'): slow_mo = not slow_mo
        if key == ord('d'): pass
        elif paused and key != ord('d'): continue

        ret, frame = cap.read()
        if not ret: break

        mask = get_combined_mask(frame, fgbg)
        
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

        cv2.imshow('Robust Tracker', display_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()