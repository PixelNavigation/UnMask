"""
UnMask Backend API - Cleaned Version
Using only Xception models and OpenCV face detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import os
import tempfile
import logging

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our cleaned modules
import sys
import os

# Add the kaggle-dfdc directory to Python path
kaggle_dfdc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kaggle-dfdc')
sys.path.insert(0, kaggle_dfdc_path)

try:
    from face_utils import FaceDetector, norm_crop
    from model_def import WSDAN, xception
    logger.info("Successfully imported cleaned modules")
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    # For development/testing, create dummy classes
    class FaceDetector:
        def __init__(self, **kwargs): pass
        def detect(self, img): return [], []
    
    class WSDAN:
        def __init__(self, **kwargs): pass
        def __call__(self, x): return torch.zeros(1, 2), None, None
    
    def xception(**kwargs):
        return torch.nn.Linear(1, 2)
    
    def norm_crop(img, landmarks, **kwargs):
        return img

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class DeepfakeDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize face detector (OpenCV)
        self.face_detector = FaceDetector(device="cpu")  # OpenCV runs on CPU
        logger.info("OpenCV face detector initialized")
        
        # Load models
        self.load_models()
        
        # Image transforms
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        
        # Normalization parameters for WSDAN
        self.zhq_nm_avg = torch.Tensor([.4479, .3744, .3473]).view(1, 3, 1, 1)
        self.zhq_nm_std = torch.Tensor([.2537, .2502, .2424]).view(1, 3, 1, 1)
        
        if self.device.type == 'cuda':
            self.zhq_nm_avg = self.zhq_nm_avg.cuda()
            self.zhq_nm_std = self.zhq_nm_std.cuda()
    
    def load_models(self):
        """Load the cleaned Xception models"""
        try:
            # Model 1: Standalone Xception
            self.model1 = xception(num_classes=2, pretrained=False)
            
            # Try to load weights if available
            xception_weights_path = "kaggle-dfdc/model_def/xception-hg-2.pth"  # Standalone Xception weights
            if os.path.exists(xception_weights_path):
                logger.info("Loading Xception weights...")
                ckpt = torch.load(xception_weights_path, map_location=self.device, weights_only=False)
                self.model1.load_state_dict(ckpt.get("state_dict", ckpt))
            else:
                logger.warning("Xception weights not found, using random initialization")
            
            self.model1 = self.model1.to(self.device)
            self.model1.eval()
            
            # Model 2: WSDAN + Xception
            self.model2 = WSDAN(num_classes=2, M=8, net="xception", pretrained=False)
            
            # Try to load WSDAN weights if available
            wsdan_weights_path = "kaggle-dfdc/model_def/ckpt_x.pth"  # Update path as needed
            if os.path.exists(wsdan_weights_path):
                logger.info("Loading WSDAN+Xception weights...")
                ckpt = torch.load(wsdan_weights_path, map_location=self.device, weights_only=False)
                self.model2.load_state_dict(ckpt.get("state_dict", ckpt))
            else:
                logger.warning("WSDAN weights not found, using random initialization")
                
            self.model2 = self.model2.to(self.device)
            self.model2.eval()
            
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Initialize with random weights for testing
            self.model1 = xception(num_classes=2, pretrained=False).to(self.device)
            self.model2 = WSDAN(num_classes=2, M=8, net="xception", pretrained=False).to(self.device)
            self.model1.eval()
            self.model2.eval()
            logger.info("Using models with random weights for testing")
    
    def detect_faces(self, image):
        """Detect faces in image using OpenCV"""
        boxes, landms = self.face_detector.detect(image)
        return boxes, landms
    
    def predict(self, image_path):
        """
        Predict if image/video contains deepfake
        Returns: dict with prediction results
        """
        try:
            # Load image
            if image_path.lower().endswith(('.mp4', '.avi', '.mov')):
                # Handle video - extract first frame for now
                cap = cv2.VideoCapture(image_path)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return {"error": "Could not read video file"}
                image = frame
            else:
                # Handle image
                image = cv2.imread(image_path)
                if image is None:
                    return {"error": "Could not read image file"}
            
            # Detect faces
            boxes, landms = self.detect_faces(image)
            
            if len(boxes) == 0:
                return {
                    "is_deepfake": False,
                    "confidence": 0.5,
                    "status": "no_face_detected",
                    "message": "No faces detected in the image"
                }
            
            # Take the first face
            box = boxes[0]
            landm = landms[0].reshape(5, 2)
            
            # Crop and normalize face
            face_img = norm_crop(image, landm, image_size=224)
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            
            # Convert to tensor
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                # Model 1: Xception
                i1 = F.interpolate(face_tensor, size=299, mode="bilinear")
                i1.sub_(0.5).mul_(2.0)  # Normalize to [-1, 1]
                o1 = self.model1(i1).softmax(-1)[:, 1].cpu().numpy()[0]
                
                # Model 2: WSDAN + Xception  
                i2 = (face_tensor - self.zhq_nm_avg) / self.zhq_nm_std
                o2, _, _ = self.model2(i2)
                o2 = o2.softmax(-1)[:, 1].cpu().numpy()[0]
                
                # Ensemble prediction (adjusted weights for 2 models)
                final_score = 0.3 * o1 + 0.7 * o2
            
            # Determine result
            is_deepfake = final_score > 0.5
            confidence = final_score if is_deepfake else (1 - final_score)
            
            return {
                "is_deepfake": bool(is_deepfake),
                "confidence": float(confidence),
                "status": "fake" if is_deepfake else "real",
                "scores": {
                    "xception": float(o1),
                    "wsdan_xception": float(o2),
                    "ensemble": float(final_score)
                },
                "faces_detected": len(boxes)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

# Initialize detector
detector = DeepfakeDetector()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models": "xception_only",
        "face_detector": "opencv",
        "device": str(detector.device)
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Run prediction
            result = detector.predict(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
            return jsonify({
                "filename": file.filename,
                "detection_result": result
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting UnMask backend (cleaned version)")
    logger.info("Using: Xception models + OpenCV face detection")
    app.run(debug=True, host='0.0.0.0', port=5000)
