"""
Enhanced Deepfake Detection Integration
Combines video and image analysis capabilities
"""

import os
import sys
import cv2
import dlib
import torch
import numpy as np
from PIL import Image
import argparse
import tempfile
import subprocess
from pathlib import Path

# Add the current directory to the path for imports
current_dir = Path(__file__).parent
deepfake_dir = current_dir / "Deepfake-Detection"
sys.path.insert(0, str(deepfake_dir))

try:
    from network.models import model_selection
    from dataset.transform import xception_default_data_transforms
    from dataset.mydataset import MyDataset
except ImportError as e:
    print(f"Warning: Could not import deepfake detection modules: {e}")

class DeepfakeAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize dlib face detector
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Load model
        self.model = None
        self.load_model()
    
    def load_model(self, model_path=None):
        """Load the deepfake detection model"""
        if model_path is None:
            # Try different possible model paths
            model_paths = [
                deepfake_dir / "pretrained_model" / "FF++_c23.pth",
                deepfake_dir / "pretrained_model" / "FF++_c40.pth",
                deepfake_dir / "pretrained_model" / "df_c0_best.pkl"
            ]
            
            for path in model_paths:
                if path.exists():
                    model_path = str(path)
                    break
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
                
                # Load model with proper device handling
                if torch.cuda.is_available():
                    print("CUDA is available - using GPU")
                    self.model.load_state_dict(torch.load(model_path))
                    self.model = self.model.cuda()
                else:
                    print("CUDA not available - using CPU")
                    self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                
                self.model.to(self.device)
                self.model.eval()
                print(f"Model loaded from: {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
                return False
        else:
            print("No pretrained model found. Please download FF++_c23.pth or FF++_c40.pth")
            print("Place the model in: backend/Deepfake-Detection/pretrained_model/")
            return False
    
    def get_boundingbox(self, face, width, height, scale=1.3, minsize=None):
        """Generate bounding box for detected face"""
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        if minsize:
            if size_bb < minsize:
                size_bb = minsize
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Check for out of bounds, x-y top left corner
        x1 = max(int(center_x - size_bb // 2), 0)
        y1 = max(int(center_y - size_bb // 2), 0)
        # Check for too big bb size for given x, y
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)

        return x1, y1, size_bb
    
    def detect_faces(self, image):
        """Detect faces in image using dlib"""
        if isinstance(image, str):
            image = cv2.imread(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        return faces
    
    def analyze_image(self, image_path):
        """Analyze a single image for deepfakes"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {"error": "Could not load image"}
            
            height, width = image.shape[:2]
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if not faces:
                return {
                    "prediction": "no_faces",
                    "confidence": 0.0,
                    "message": "No faces detected in the image",
                    "faces_detected": 0
                }
            
            results = []
            
            for i, face in enumerate(faces):
                # Get bounding box
                x, y, size = self.get_boundingbox(face, width, height)
                
                # Extract face region
                cropped_face = image[y:y+size, x:x+size]
                
                if self.model is not None:
                    # Preprocess for model
                    face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                    face_tensor = xception_default_data_transforms['test'](face_pil).unsqueeze(0)
                    
                    # Move to device (CUDA or CPU)
                    face_tensor = face_tensor.to(self.device)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = self.model(face_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        fake_prob = probabilities[0][1].item()
                        
                    prediction = "fake" if fake_prob > 0.5 else "real"
                    confidence = fake_prob if prediction == "fake" else (1 - fake_prob)
                else:
                    prediction = "unknown"
                    confidence = 0.0
                
                results.append({
                    "face_id": i + 1,
                    "prediction": prediction,
                    "confidence": confidence,
                    "bbox": {"x": x, "y": y, "size": size}
                })
            
            # Overall prediction (worst case)
            overall_prediction = "real"
            max_fake_confidence = 0.0
            
            for result in results:
                if result["prediction"] == "fake" and result["confidence"] > max_fake_confidence:
                    max_fake_confidence = result["confidence"]
                    overall_prediction = "fake"
            
            return {
                "overall_prediction": overall_prediction,
                "overall_confidence": max_fake_confidence if overall_prediction == "fake" else (1 - max_fake_confidence),
                "faces_detected": len(faces),
                "face_results": results
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_video(self, video_path, output_path=None):
        """Analyze video for deepfakes"""
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}
        
        # Use the detect_from_video.py script
        cmd = [
            sys.executable, 
            str(deepfake_dir / "detect_from_video.py"),
            "-i", str(video_path),
            "-m", str(deepfake_dir / "pretrained_model" / "FF++_c23.pth")
        ]
        
        if output_path:
            cmd.extend(["-o", str(output_path)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(deepfake_dir))
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": "Video analysis completed",
                    "output": result.stdout
                }
            else:
                return {
                    "error": "Video analysis failed",
                    "details": result.stderr
                }
        except Exception as e:
            return {"error": str(e)}

def test_setup():
    """Test the deepfake detection setup"""
    print("Testing Deepfake Detection Setup")
    print("=" * 40)
    
    analyzer = DeepfakeAnalyzer()
    
    # Test face detection
    print(f"dlib face detector: {'✓' if analyzer.face_detector else '✗'}")
    print(f"Model loaded: {'✓' if analyzer.model else '✗'}")
    print(f"Device: {analyzer.device}")
    
    # Test with sample image if available
    sample_video = deepfake_dir / "videos" / "994.mp4"
    if sample_video.exists():
        print(f"Sample video found: {sample_video}")
        
        # Extract a frame for testing
        cap = cv2.VideoCapture(str(sample_video))
        ret, frame = cap.read()
        if ret:
            test_image_path = "test_frame.jpg"
            cv2.imwrite(test_image_path, frame)
            
            result = analyzer.analyze_image(test_image_path)
            print(f"Test analysis result: {result}")
            
            # Cleanup
            os.remove(test_image_path)
        cap.release()
    
    print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Deepfake Detection')
    parser.add_argument('--test', action='store_true', help='Test the setup')
    parser.add_argument('--image', type=str, help='Path to image file to analyze')
    parser.add_argument('--video', type=str, help='Path to video file to analyze')
    parser.add_argument('--output', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.test:
        test_setup()
    elif args.image:
        analyzer = DeepfakeAnalyzer()
        result = analyzer.analyze_image(args.image)
        print(f"Analysis result: {result}")
    elif args.video:
        analyzer = DeepfakeAnalyzer()
        result = analyzer.analyze_video(args.video, args.output)
        print(f"Video analysis result: {result}")
    else:
        print("Use --test to test setup, --image <path> to analyze image, or --video <path> to analyze video")
