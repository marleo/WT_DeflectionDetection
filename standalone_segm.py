import math
import numpy as np
import cv2
import time
from ultralytics import YOLO

# -------------------- Configuration --------------------
N_BLADES = 3
IMG_SIZE = 640
CONF_THRES = 0.5
TRIGGER_WINDOW_MS = 200 
BLADE_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

# -------------------- Logic Classes --------------------

class BladeStateFilter:
    def __init__(self, initial_angle, dt):
        self.dt = dt
        self.state = np.array([initial_angle, 0.0])
        self.P = np.diag([0.1, 10.0])
        self.R = np.array([[0.01]]) 
        self.Q_mag = 0.01

    def predict(self, dt):
        self.dt = dt
        self.state[0] = (self.state[0] + self.state[1] * self.dt + math.pi) % (2 * math.pi) - math.pi
        F = np.array([[1.0, self.dt], [0.0, 1.0]])
        Q = np.array([[0.25*(dt**4), 0.5*(dt**3)], [0.5*(dt**3), dt**2]]) * self.Q_mag
        self.P = F @ self.P @ F.T + Q
        return self.state[0]

    def update(self, measured_angle):
        innovation = (measured_angle - self.state[0] + math.pi) % (2 * math.pi) - math.pi
        H = np.array([[1.0, 0.0]])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K.flatten() * innovation
        self.state[0] = (self.state[0] + math.pi) % (2 * math.pi) - math.pi
        self.P = (np.eye(2) - K @ H) @ self.P

class BladeTracker:
    def __init__(self, target_count):
        self.target_count = target_count
        self.blades = {}
        self.next_id = 1

    def match(self, detections, dt):
        for b in self.blades.values(): b.predict(dt)
        
        mapping = {}
        for det in detections:
            best_id, min_dist = None, 0.8
            for bid, filt in self.blades.items():
                dist = abs((det['angle'] - filt.state[0] + math.pi) % (2 * math.pi) - math.pi)
                if dist < min_dist:
                    min_dist, best_id = dist, bid
            
            if best_id:
                self.blades[best_id].update(det['angle'])
                mapping[det['tracker_id']] = best_id
            elif len(self.blades) < self.target_count:
                new_id = self.next_id
                self.blades[new_id] = BladeStateFilter(det['angle'], dt)
                mapping[det['tracker_id']] = new_id
                self.next_id += 1
        return mapping

# -------------------- Main Loop --------------------

def run_live_analysis(model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 is the default webcam
    tracker = BladeTracker(N_BLADES)
    
    last_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        if dt <= 0: dt = 0.033 # Fallback to ~30fps if timing glitches

        results = model.track(frame, imgsz=IMG_SIZE, conf=CONF_THRES, persist=True, verbose=False)[0]
        h, w = frame.shape[:2]
        hub = (w // 2, h // 2)
        trigger_active = False

        if results.boxes.id is not None and results.masks is not None:
            raw_dets = []
            for i, track_id in enumerate(results.boxes.id.int().cpu().tolist()):
                poly = results.masks.xy[i]
                cx, cy = np.mean(poly, axis=0)
                angle = math.atan2(cy - hub[1], cx - hub[0])
                raw_dets.append({'tracker_id': track_id, 'angle': angle, 'poly': poly, 'center': (cx, cy)})

            id_map = tracker.match(raw_dets, dt)

            for det in raw_dets:
                sid = id_map.get(det['tracker_id'])
                if sid:
                    filt = tracker.blades[sid]
                    # Trigger logic using filtered velocity
                    omega = abs(filt.state[1])
                    rad_threshold = omega * (TRIGGER_WINDOW_MS / 1000.0)
                    
                    if abs(filt.state[0]) < rad_threshold:
                        trigger_active = True
                    
                    color = BLADE_COLORS[(sid-1) % 3]
                    cv2.fillPoly(frame, [det['poly'].astype(np.int32)], color)
                    cv2.putText(frame, f"B{sid}", (int(det['center'][0]), int(det['center'][1])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if trigger_active:
            print("Trigger")
            cv2.circle(frame, (w - 50, 50), 20, (0, 0, 255), -1)

        cv2.imshow("Live Blade Trigger", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_analysis("best.pt")