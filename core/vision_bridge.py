from PIL import Image, ImageStat
import numpy as np
import os
import time
from typing import Optional

class VisionBridge:
    """
    Connects image files in the sensory_input/ folder to ECH0-PRIME Level 0.
    This provides a robust, platform-independent way to feed real-world 
    visual data into the architecture.
    """
    def __init__(self, watch_dir: str = "sensory_input", use_webcam: bool = False):
        self.watch_dir = watch_dir
        self.feature_dim = 1000000 
        self.last_processed_key = None # (path, mtime)
        self.use_webcam = use_webcam
        self.cap = None
        
        if use_webcam:
            try:
                import cv2
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("VISION WARNING: Could not open webcam. Falling back to folder watch.")
                    self.use_webcam = False
                else:
                    print("VISION: Webcam initialized and linked to sensory field.")
            except ImportError:
                print("VISION WARNING: opencv-python not installed. Falling back to folder watch.")
                self.use_webcam = False

        if not os.path.exists(watch_dir):
            os.makedirs(watch_dir)

    def process_image(self, img_path: str) -> np.ndarray:
        """Converts an image file into a 1M dimensional sensory vector."""
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                
                # 1. Feature Extraction
                # Global Statistics
                stats = ImageStat.Stat(img)
                avg_colors = stats.mean # [R, G, B]
                
                # Distribution (Histogram)
                hist = img.histogram() # 3 channels * 256 = 768 entries
                
                # Grayscale for structure
                gray = img.convert('L')
                small = gray.resize((200, 200)) # 40,000 pixels
                pixel_data = np.array(small).flatten()

                # 2. Vector Packaging
                sensory_vector = np.zeros(self.feature_dim)
                
                # Insert histogram at the start
                hist_norm = np.array(hist) / (np.max(hist) + 1e-6)
                sensory_vector[:len(hist_norm)] = hist_norm
                
                # Insert averages
                sensory_vector[1000:1003] = np.array(avg_colors) / 255.0
                
                # Insert pixel data (Visual Structure)
                sensory_vector[10000:10000+len(pixel_data)] = pixel_data / 255.0
                
                return sensory_vector
        except Exception as e:
            print(f"Error processing image: {e}")
            return np.zeros(self.feature_dim)

    def get_latest_sensory_vector(self) -> Optional[tuple[np.ndarray, str]]:
        """Checks for new images in the directory OR captures from webcam and returns a (vector, path) if found."""
        
        if self.use_webcam and self.cap:
            import cv2
            ret, frame = self.cap.read()
            if ret:
                # Save frame temporarily for dashboard visualization
                frame_path = os.path.join(self.watch_dir, "webcam_live.png")
                cv2.imwrite(frame_path, frame)
                return self.process_image(frame_path), frame_path
            
        # Fallback to folder watch
        files = [f for f in os.listdir(self.watch_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            return None
        
        # Get the most recently modified file (ignore the live webcam file if watching folder)
        files = [f for f in files if f != "webcam_live.png"]
        if not files: return None

        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.watch_dir, x)), reverse=True)
        latest_file_name = files[0]
        latest_file_path = os.path.join(self.watch_dir, latest_file_name)
        mtime = os.path.getmtime(latest_file_path)
        
        # Check if already processed
        current_key = (latest_file_path, mtime)
        if current_key == self.last_processed_key:
            return None
            
        self.last_processed_key = current_key
        return self.process_image(latest_file_path), latest_file_path

if __name__ == "__main__":
    bridge = VisionBridge()
    print(f"Vision Bridge active. Watch directory: {os.path.abspath(bridge.watch_dir)}")
    # Wait for a file to appear
    while True:
        vec = bridge.get_latest_sensory_vector()
        if vec is not None:
            print(f"Processed latest image. Vector non-zero count: {np.count_nonzero(vec)}")
            break
        time.sleep(1)
