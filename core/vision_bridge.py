from PIL import Image, ImageStat
import numpy as np
import os
import time
from typing import Optional, Dict, Any

# Apple Vision integration with fallback
try:
    import objc
    import Foundation
    import Vision
    APPLE_VISION_AVAILABLE = True
    print("✓ Apple Vision framework available for enhanced computer vision")
except ImportError:
    APPLE_VISION_AVAILABLE = False
    print("⚠️ Apple Vision framework not available, using basic image processing")


class AppleVisionIntegration:
    """Integration with Apple's Vision framework for advanced computer vision"""

    def __init__(self):
        self.available = APPLE_VISION_AVAILABLE
        self.face_detector = None
        self.object_detector = None
        self.text_recognizer = None
        self.image_classifier = None

        if self.available:
            self._initialize_vision_components()

    def _initialize_vision_components(self):
        """Initialize Vision framework components"""
        try:
            # Face detection
            self.face_detector = Vision.VNFaceObservation.accuracyMode()

            # Object detection (general)
            self.object_detector = Vision.VNDetectRectanglesRequest.alloc().init()

            # Text recognition
            self.text_recognizer = Vision.VNRecognizeTextRequest.alloc().init()

            # Image classification (if available)
            try:
                self.image_classifier = Vision.VNClassifyImageRequest.alloc().init()
            except:
                self.image_classifier = None

        except Exception as e:
            print(f"❌ Apple Vision initialization failed: {e}")
            self.available = False

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using Apple Vision framework"""
        if not self.available:
            return self._simulate_vision_analysis()

        try:
            # Load image
            image_url = Foundation.NSURL.fileURLWithPath_(image_path)
            image_request_handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(image_url, None)

            results = {}

            # Face detection
            if self.face_detector:
                face_request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
                success, error = image_request_handler.performRequests_error_([face_request], None)
                if success and face_request.results():
                    results["faces"] = len(face_request.results())
                    # Get face details
                    face_details = []
                    for face in face_request.results():
                        face_details.append({
                            "confidence": float(face.confidence) if hasattr(face, 'confidence') else 1.0,
                            "bounds": {
                                "x": face.boundingBox.origin.x,
                                "y": face.boundingBox.origin.y,
                                "width": face.boundingBox.size.width,
                                "height": face.boundingBox.size.height
                            }
                        })
                    results["face_details"] = face_details

            # Text recognition
            if self.text_recognizer:
                text_request = Vision.VNRecognizeTextRequest.alloc().init()
                text_request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
                success, error = image_request_handler.performRequests_error_([text_request], None)
                if success and text_request.results():
                    text_results = []
                    for observation in text_request.results():
                        text_results.append({
                            "text": str(observation.text),
                            "confidence": float(observation.confidence)
                        })
                    results["text"] = text_results

            # Image classification
            if self.image_classifier:
                success, error = image_request_handler.performRequests_error_([self.image_classifier], None)
                if success and self.image_classifier.results():
                    classifications = []
                    for classification in self.image_classifier.results()[:5]:  # Top 5
                        classifications.append({
                            "identifier": str(classification.identifier),
                            "confidence": float(classification.confidence)
                        })
                    results["classifications"] = classifications

            return results

        except Exception as e:
            print(f"❌ Apple Vision analysis failed: {e}")
            return self._simulate_vision_analysis()

    def _simulate_vision_analysis(self) -> Dict[str, Any]:
        """Simulate vision analysis when framework unavailable"""
        return {
            "faces": 0,
            "text": [],
            "classifications": [],
            "simulated": True
        }


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

        # Initialize Apple Vision integration
        self.apple_vision = AppleVisionIntegration()
        
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

    def process_image(self, img_path: str) -> tuple[np.ndarray, Dict[str, Any]]:
        """Converts an image file into a 1M dimensional sensory vector with Apple Vision analysis."""
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')

                # 1. Basic Feature Extraction
                # Global Statistics
                stats = ImageStat.Stat(img)
                avg_colors = stats.mean # [R, G, B]

                # Distribution (Histogram)
                hist = img.histogram() # 3 channels * 256 = 768 entries

                # Grayscale for structure
                gray = img.convert('L')
                small = gray.resize((200, 200)) # 40,000 pixels
                pixel_data = np.array(small).flatten()

                # 2. Apple Vision Analysis
                vision_analysis = self.apple_vision.analyze_image(img_path)

                # 3. Vector Packaging
                sensory_vector = np.zeros(self.feature_dim)

                # Insert histogram at the start
                hist_norm = np.array(hist) / (np.max(hist) + 1e-6)
                sensory_vector[:len(hist_norm)] = hist_norm

                # Insert averages
                sensory_vector[1000:1003] = np.array(avg_colors) / 255.0

                # Insert pixel data (Visual Structure)
                sensory_vector[10000:10000+len(pixel_data)] = pixel_data / 255.0

                # Encode Apple Vision features
                vision_features = self._encode_vision_features(vision_analysis)
                vision_start_idx = 50000  # Reserve space for vision features
                if len(vision_features) > 0:
                    sensory_vector[vision_start_idx:vision_start_idx+len(vision_features)] = vision_features

                return sensory_vector, vision_analysis
        except Exception as e:
            print(f"Error processing image: {e}")
            return np.zeros(self.feature_dim), {}

    def _encode_vision_features(self, vision_analysis: Dict[str, Any]) -> np.ndarray:
        """Encode Apple Vision analysis into numerical features"""
        features = []

        # Face detection features
        if "faces" in vision_analysis:
            features.append(min(vision_analysis["faces"] / 10.0, 1.0))  # Normalize face count

        # Text detection features
        if "text" in vision_analysis:
            text_count = len(vision_analysis["text"])
            features.append(min(text_count / 50.0, 1.0))  # Normalize text count

            # Average text confidence
            if text_count > 0:
                avg_confidence = sum(t.get("confidence", 0.5) for t in vision_analysis["text"]) / text_count
                features.append(avg_confidence)

        # Classification features (top 5)
        if "classifications" in vision_analysis:
            for i, classification in enumerate(vision_analysis["classifications"][:5]):
                features.append(classification.get("confidence", 0.0))

        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)

        return np.array(features[:10])

    def get_latest_sensory_vector(self) -> Optional[tuple[np.ndarray, str, Dict[str, Any]]]:
        """Checks for new images in the directory OR captures from webcam and returns a (vector, path) if found."""
        
        if self.use_webcam and self.cap:
            import cv2
            ret, frame = self.cap.read()
            if ret:
                # Save frame temporarily for dashboard visualization
                frame_path = os.path.join(self.watch_dir, "webcam_live.png")
                cv2.imwrite(frame_path, frame)
                vector, analysis = self.process_image(frame_path)
                return vector, frame_path, analysis
            
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
        vector, analysis = self.process_image(latest_file_path)
        return vector, latest_file_path, analysis

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
