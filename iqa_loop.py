import cv2 as cv
import numpy as np
import os
import socket
import json
import time

# -------------------- Networking Setup --------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)  # Important: prevents script from freezing while waiting for data

# -------------------- Configuration --------------------
SAVE_PATH = "captured_bursts"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

active_window = None

# -------------------- Camera Setup --------------------
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_GAIN, 0)
cap.set(cv.CAP_PROP_EXPOSURE, 0)

def analyze_histogram(frame):
    hist = cv.calcHist([frame], [0], None, [256], [0, 256])
    return hist

def check_histogram(hist):
    if hist[255] > 1000:
        return f"Overexposed by {int(hist[255][0])} pixels"
    elif hist[0] > 1000:
        return f"Underexposed by {int(hist[0][0])} pixels"
    else:
        return "Normal"

print(f"Receiver active. Listening on port {UDP_PORT}...")

# -------------------- Main Loop --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    t_now = time.time()

    # 1. Listen for UDP Trigger Windows (Non-blocking)
    try:
        data, addr = sock.recvfrom(1024)
        active_window = json.loads(data.decode())
        print(f"New window received: {active_window['start']} to {active_window['stop']}")
    except (BlockingIOError, socket.error):
        pass

    # 2. Automated Burst Trigger Logic
    if active_window:
        if t_now >= active_window['start'] and t_now <= active_window['stop']:
            # Capture logic
            filename = os.path.join(SAVE_PATH, f"trigger_{t_now}.jpg")
            cv.imwrite(filename, frame)
            # Visual feedback on screen
            cv.putText(frame, "BURST ACTIVE", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif t_now > active_window['stop']:
            active_window = None # Clear window once expired

    # 3. Manual Controls & Display
    cv.imshow("Receiver Camera - Manual & Auto", frame)
    key = cv.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("h"):
        hist = analyze_histogram(frame)
        status = check_histogram(hist)
        print("Status: ", status)
    elif key == ord("o"):
        cap.set(cv.CAP_PROP_EXPOSURE, cap.get(cv.CAP_PROP_EXPOSURE) - 1)
        print("Exposure: ", cap.get(cv.CAP_PROP_EXPOSURE))
    elif key == ord("p"):
        cap.set(cv.CAP_PROP_EXPOSURE, cap.get(cv.CAP_PROP_EXPOSURE) + 1)
        print("Exposure: ", cap.get(cv.CAP_PROP_EXPOSURE))
    elif key == ord("l"):
        cap.set(cv.CAP_PROP_GAIN, cap.get(cv.CAP_PROP_GAIN) - 1)
        print("Gain: ", cap.get(cv.CAP_PROP_GAIN))
    elif key == ord("k"):
        cap.set(cv.CAP_PROP_GAIN, cap.get(cv.CAP_PROP_GAIN) + 1)
        print("Gain: ", cap.get(cv.CAP_PROP_GAIN))

cap.release()
cv.destroyAllWindows()
exit()