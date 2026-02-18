import cv2
import numpy as np
from skimage.morphology import skeletonize

# --- CONFIGURATION ---
VIDEO_PATH = r"C:\Users\Mario\Videos\ModelTurbine\TopDown\clockwise_14rpm.mp4"
MIN_BLADE_AREA = 5000

def get_cleaned_mask(frame, fgbg):
    mask = fgbg.apply(frame, learningRate=0.005)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    
    # Initial cleanup
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
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

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        mask = get_cleaned_mask(frame, fgbg)
        
        if np.sum(mask > 0) > MIN_BLADE_AREA:
            # 1. Generate Raw Skeleton
            skeleton_bool = skeletonize(mask > 0)
            skeleton_uint8 = (skeleton_bool * 255).astype(np.uint8)
            
            # 2. Prune: Keep only the longest line
            clean_skeleton = prune_skeleton(skeleton_uint8)
            
            # 3. Visualization
            display_frame = frame.copy()
            # Draw the pruned skeleton in Green
            display_frame[clean_skeleton > 0] = (0, 255, 0)
            
            # Draw a thick version of the skeleton so it's easier to see
            kernel_dilate = np.ones((3,3), np.uint8)
            thick_skel = cv2.dilate(clean_skeleton, kernel_dilate)
            display_frame[thick_skel > 0] = (0, 255, 0) 
            
        else:
            display_frame = frame.copy()

        # Debug UI
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.putText(display_frame, f"Frame: {frame_idx}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Pruned Skeleton (Longest Line Only)', display_frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27: break
        if key == ord(' '): cv2.waitKey(-1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()