#!/usr/bin/env python3
"""
Enhanced Deepfake Detection Server with dlib integration
"""

import os
import sys
import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
import logging
import traceback
from pathlib import Path

# Add Deepfake-Detection to path
current_dir = Path(__file__).parent
deepfake_dir = current_dir / "Deepfake-Detection"
sys.path.insert(0, str(deepfake_dir))

try:
    from network.models import model_selection
    from network.mesonet import Meso4, MesoInception4
    from network.xception import TransferModel
except ImportError as e:
    print(f"Warning: Could not import deepfake detection models: {e}")
    print("Some functionality may be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class DeepfakeDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize dlib face detector
        self.face_detector = dlib.get_frontal_face_detector()
        logger.info("dlib face detector initialized")
        
        # Initialize face predictor (if available)
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if os.path.exists(predictor_path):
            self.face_predictor = dlib.shape_predictor(predictor_path)
            logger.info("dlib face predictor loaded")
        else:
            self.face_predictor = None
            logger.warning("dlib face predictor not found. Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        
        # Load model
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the deepfake detection model"""
        try:
            model_path = deepfake_dir / "pretrained_model" / "FF++_c23.pth"
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}")
                return
            
            # Try to load Xception model
            self.model = TransferModel(name='xception', num_out_classes=2)
            self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info("Xception model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def detect_faces_dlib(self, image):
        """Detect faces using dlib"""
        if isinstance(image, str):
            image = cv2.imread(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        face_regions = []
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_regions.append((x, y, w, h))
        
        return face_regions
    
    def preprocess_face(self, face_image):
        """Preprocess face image for model input"""
        # Resize to model input size (usually 299x299 for Xception)
        face_resized = cv2.resize(face_image, (299, 299))
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        return face_tensor.to(self.device)
    
    def predict_deepfake(self, image_path):
        """Predict if image contains deepfake"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return {"error": "Could not load image"}
            
            # Detect faces using dlib
            faces = self.detect_faces_dlib(image)
            
            if not faces:
                return {
                    "prediction": "no_faces",
                    "confidence": 0.0,
                    "message": "No faces detected in the image"
                }
            
            results = []
            
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_region = image[y:y+h, x:x+w]
                
                if self.model is not None:
                    # Preprocess and predict
                    face_tensor = self.preprocess_face(face_region)
                    
                    with torch.no_grad():
                        outputs = self.model(face_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        fake_prob = probabilities[0][1].item()  # Probability of being fake
                        
                    prediction = "fake" if fake_prob > 0.5 else "real"
                    confidence = fake_prob if prediction == "fake" else (1 - fake_prob)
                else:
                    # Fallback without model
                    prediction = "unknown"
                    confidence = 0.0
                
                results.append({
                    "face_id": i + 1,
                    "prediction": prediction,
                    "confidence": confidence,
                    "bbox": {"x": x, "y": y, "width": w, "height": h}
                })
            
            # Overall result (worst case scenario)
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
            logger.error(f"Error in prediction: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

# Initialize detector
detector = DeepfakeDetector()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Enhanced Deepfake Detection Server with dlib",
        "model_loaded": detector.model is not None,
        "dlib_available": True,
        "face_predictor_available": detector.face_predictor is not None,
        "device": str(detector.device)
    })

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    """Main detection endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Perform detection
            result = detector.predict_deepfake(temp_path)
            return jsonify(result)
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error in detection endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify dlib functionality"""
    try:
        # Test dlib face detection with a simple test
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        faces = detector.detect_faces_dlib(test_image)
        
        return jsonify({
            "dlib_working": True,
            "test_faces_detected": len(faces),
            "model_available": detector.model is not None,
            "message": "dlib integration is working properly"
        })
    
    except Exception as e:
        return jsonify({
            "dlib_working": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("="*50)
    print("Enhanced Deepfake Detection Server with dlib")
    print("="*50)
    print(f"Using device: {detector.device}")
    print(f"Model loaded: {detector.model is not None}")
    print(f"dlib face detector: Available")
    print(f"dlib face predictor: {'Available' if detector.face_predictor else 'Not available'}")
    print("="*50)
    print("Server starting on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /        - Health check")
    print("  POST /detect  - Detect deepfakes in uploaded image")
    print("  GET  /test    - Test dlib functionality")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
