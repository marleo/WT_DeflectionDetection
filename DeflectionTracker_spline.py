import cv2
import numpy as np
from skimage.morphology import skeletonize

# --- CONFIGURATION ---
VIDEO_PATH = r"C:\Users\Mario\Videos\ModelTurbine\TopDown\clockwise_14rpm.mp4"
MIN_BLADE_AREA = 5000

def get_cleaned_mask(frame, fgbg):
    mask = fgbg.apply(frame, learningRate=0.005)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    # Fast cleanup
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def get_fast_line_approx(skeleton_img):
    """
    Fits a line through skeleton pixels using M-estimator (Least Squares).
    Highly efficient and ignores minor skeleton 'hairs'.
    """
    # Find all non-zero pixel coordinates
    pts = np.column_stack(np.where(skeleton_img > 0))
    
    if len(pts) < 10: # Minimum points to define a meaningful line
        return None, None

    # cv2.fitLine expects (N, 1, 2) or (N, 2) float32 array in (x, y) format
    # np.where returns (row, col) which is (y, x), so we flip them
    pts_float = pts[:, [1, 0]].astype(np.float32)
    
    # fitLine returns: [vx, vy, x0, y0] where (vx, vy) is a normalized 
    # vector and (x0, y0) is a point on the line.
    [vx, vy, x, y] = cv2.fitLine(pts_float, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Determine line endpoints for visualization (scaled to blade size)
    # Increase 'length' if the line is too short for your turbine
    length = 200 
    pt1 = (int(x - vx * length), int(y - vy * length))
    pt2 = (int(x + vx * length), int(y + vy * length))
    
    return pt1, pt2

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        mask = get_cleaned_mask(frame, fgbg)
        display_frame = frame.copy()
        
        # Check if we have enough white pixels to process
        if np.sum(mask > 0) > MIN_BLADE_AREA:
            # 1. Generate Skeleton
            skeleton_bool = skeletonize(mask > 0)
            skeleton_uint8 = (skeleton_bool * 255).astype(np.uint8)
            
            # 2. Fit Line (The 'Smoothing' step)
            p1, p2 = get_fast_line_approx(skeleton_uint8)
            
            # 3. Draw result
            if p1 is not None:
                cv2.line(display_frame, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display
        cv2.imshow('Fast Line Fit', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        if key == ord(' '): cv2.waitKey(-1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()