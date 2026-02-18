import cv2
import numpy as np
from collections import deque

def process_turbine_sharp(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Use standard MOG2. It provides the most stable detection.
    # We rely on post-processing to keep it sharp.
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    ret, frame = cap.read()
    if not ret: return
    h, w = frame.shape[:2]

    # --- STATE ---
    accumulator = np.zeros((h, w), dtype=np.float32)
    stable_hub_float = None
    stable_hub_int = None
    angle_history = deque(maxlen=20) 
    root_line_history = deque(maxlen=20)
    current_angle_smooth = 0.0
    max_angle = 0.0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: 
                print("End of video.")
                break 

            # 1. MINIMAL PRE-PROCESSING
            # A tiny blur (3x3) removes sensor grain without killing edge sharpness
            blur = cv2.GaussianBlur(frame, (3, 3), 0)
            
            # 2. RAW MASK EXTRACTION
            mask = back_sub.apply(blur)
            
            # 3. BINARY THRESHOLD
            # Ensure strictly black/white (removes gray shadows)
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

            # 4. COMPONENT-BASED CLEANING (The "High Res" Fix)
            # Instead of eroding/dilating, we just delete small objects.
            # connectivity=8 checks all surrounding pixels
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # Create a clean mask canvas
            clean_mask = np.zeros_like(mask)
            
            # Loop through all found components (0 is background, so skip it)
            # We look for the "Blade" which will be the largest moving object
            largest_label = -1
            max_area = 0
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Threshold: Ignore small noise specks (e.g. < 500 pixels)
                if area > 500:
                    if area > max_area:
                        max_area = area
                        largest_label = i
            
            # If we found a blade, draw ONLY that component
            if largest_label != -1:
                # This copies the EXACT pixels of the blade from the raw mask
                # No blurring, no rounding, no "sausage" effect.
                clean_mask[labels == largest_label] = 255

            # --- GEOMETRY & TRACKING (Standard Logic) ---
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            res_frame = frame.copy()
            avg_angle = 0.0

            if contours:
                c = max(contours, key=cv2.contourArea)
                
                # 1. ISOLATE ROOT
                x, y, bw, bh = cv2.boundingRect(c)
                root_limit = x + int(bw * 0.35) 
                root_points = c[c[:, :, 0] < root_limit]

                if len(root_points) > 10:
                    line_params = cv2.fitLine(root_points, cv2.DIST_L2, 0, 0.01, 0.01)
                    raw_vx, raw_vy, raw_x0, raw_y0 = line_params.flatten()
                    
                    mag = np.sqrt(raw_vx**2 + raw_vy**2)
                    if mag > 0: raw_vx /= mag; raw_vy /= mag
                    if raw_vx < 0: raw_vx, raw_vy = -raw_vx, -raw_vy

                    root_line_history.append((raw_vx, raw_vy, raw_x0, raw_y0))
                    avg_data = np.mean(root_line_history, axis=0)
                    vx, vy, x0, y0 = avg_data
                    mag_avg = np.sqrt(vx**2 + vy**2)
                    if mag_avg > 0: vx /= mag_avg; vy /= mag_avg

                    # 2. FIND TIP & ANGLE
                    tip_idx = np.argmax(c[:, 0, 0])
                    tip_pt = tuple(c[tip_idx][0])

                    vec_to_tip_x = tip_pt[0] - x0
                    vec_to_tip_y = tip_pt[1] - y0
                    mag_tip = np.sqrt(vec_to_tip_x**2 + vec_to_tip_y**2)
                    
                    angle_deg = 0.0
                    if mag_tip > 0:
                        dot_prod = vx * vec_to_tip_x + vy * vec_to_tip_y
                        cos_theta = np.clip(dot_prod / mag_tip, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cos_theta))

                    angle_history.append(angle_deg)
                    avg_angle = sum(angle_history) / len(angle_history)
                    current_angle_smooth = current_angle_smooth * 0.9 + angle_deg * 0.1
                    if current_angle_smooth > max_angle: max_angle = current_angle_smooth

                    # Visualization
                    diag_len = 2000
                    pt1_axis = (int(x0 - vx * diag_len), int(y0 - vy * diag_len))
                    pt2_axis = (int(x0 + vx * diag_len), int(y0 + vy * diag_len))
                    cv2.line(res_frame, pt1_axis, pt2_axis, (0, 255, 255), 2)

                    root_center = (int(x0), int(y0))
                    cv2.line(res_frame, root_center, tip_pt, (255, 0, 255), 2)
                    cv2.circle(res_frame, tip_pt, 6, (255, 0, 255), -1)
                    cv2.line(res_frame, (root_limit, 0), (root_limit, h), (0, 255, 0), 1)

                    cv2.line(accumulator, pt1_axis, pt2_axis, 1.0, 3)

            accumulator *= 0.98
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(accumulator)
            
            if max_val > 5:
                current_winner = max_loc
                if stable_hub_float is None:
                    stable_hub_float = np.array(current_winner, dtype=np.float32)
                    stable_hub_int = current_winner
                else:
                    dist = np.linalg.norm(stable_hub_float - np.array(current_winner))
                    if dist < 100:
                        alpha = 0.05
                        stable_hub_float = stable_hub_float * (1 - alpha) + np.array(current_winner) * alpha
                        stable_hub_int = (int(stable_hub_float[0]), int(stable_hub_float[1]))

            if stable_hub_int:
                cv2.circle(res_frame, stable_hub_int, 10, (255, 0, 0), -1) 
                cv2.putText(res_frame, "HUB", (stable_hub_int[0]+15, stable_hub_int[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Display Stats
            cv2.rectangle(res_frame, (w-250, 0), (w, 110), (0,0,0), -1)
            cv2.putText(res_frame, f"Cur: {current_angle_smooth:.2f} deg", (w-240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(res_frame, f"Avg: {avg_angle:.2f} deg", (w-240, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            cv2.putText(res_frame, f"Max: {max_angle:.2f} deg", (w-240, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- DISPLAY (Side by Side) ---
            def resize_h_local(img, th=500):
                h, w = img.shape[:2]
                return cv2.resize(img, (int(th * (w/h)), th))

            # Show the CLEAN mask (no noise, original sharpness)
            final_mask = resize_h_local(cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR))
            final_res = resize_h_local(res_frame)
            combined_view = np.hstack([final_mask, final_res])

            cv2.imshow("Left: Component Cleaned Mask | Right: Tracking", combined_view)

        wait_time = 0 if paused else 1
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'): break
        elif key == 32: paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"C:\Users\Mario\Videos\ModelTurbine\TopDown\clockwise_8rpm.mp4"
    process_turbine_sharp(video_path)