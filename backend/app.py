#!/usr/bin/env python3
"""
Flask API for Enhanced Deepfake Detection
Integrates HongguLiu/Deepfake-Detection models with web interface
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import time

# Add the Deepfake-Detection path
current_dir = Path(__file__).parent
deepfake_detection_path = current_dir / "Deepfake-Detection"
sys.path.insert(0, str(deepfake_detection_path))

# Import the detection functionality
try:
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    # Try to import dlib separately
    try:
        import dlib
        DLIB_AVAILABLE = True
    except ImportError:
        print("Warning: dlib not available, using OpenCV face detection")
        DLIB_AVAILABLE = False
    
    # Try to import specific modules from Deepfake-Detection
    try:
        from network.models import model_selection
        from detect_from_video import predict_on_video_set
    except ImportError:
        pass  # These are optional for now
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Detection models not available: {e}")
    DETECTION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global model variables
models = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    """Load all available deepfake detection models"""
    global models
    
    if not DETECTION_AVAILABLE:
        logger.error("Detection models not available")
        return False
    
    try:
        # Load MesoNet models
        from network.mesonet import Meso4, MesoInception4
        
        models['meso4'] = Meso4(num_classes=2).to(device)
        models['meso_inception'] = MesoInception4(num_classes=2).to(device)
        
        # Load weights if available
        weights_dir = deepfake_detection_path / "weights"
        
        if (weights_dir / "Meso4_DF.pkl").exists():
            models['meso4'].load_state_dict(torch.load(weights_dir / "Meso4_DF.pkl", map_location=device))
            logger.info("Meso4 weights loaded")
        
        if (weights_dir / "MesoInception4_DF.pkl").exists():
            models['meso_inception'].load_state_dict(torch.load(weights_dir / "MesoInception4_DF.pkl", map_location=device))
            logger.info("MesoInception4 weights loaded")
        
        # Load Xception model
        from network.models import TransferModel
        models['xception'] = TransferModel('xception', num_out_classes=2).to(device)
        
        if (weights_dir / "xception_c23.pkl").exists():
            models['xception'].load_state_dict(torch.load(weights_dir / "xception_c23.pkl", map_location=device))
            logger.info("Xception weights loaded")
        
        # Set all models to evaluation mode
        for model in models.values():
            model.eval()
        
        logger.info(f"Loaded models: {list(models.keys())}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_face_boundingbox(face, width, height, scale=1.3, minsize=None):
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

    # Check for out of bounds
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def extract_faces_from_video(video_path, max_frames=30):
    """Extract faces from video using dlib or OpenCV"""
    try:
        if DLIB_AVAILABLE:
            face_detector = dlib.get_frontal_face_detector()
            use_dlib = True
        else:
            # Use OpenCV's Haar cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            use_dlib = False
            
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return []
        
        faces = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // max_frames)
        
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape[:2]
            
            if use_dlib:
                detected_faces = face_detector(gray, 1)
                for face in detected_faces:
                    x, y, size = get_face_boundingbox(face, width, height)
                    cropped_face = frame[y:y+size, x:x+size]
                    
                    if cropped_face.size > 0:
                        # Resize to 256x256
                        resized_face = cv2.resize(cropped_face, (256, 256))
                        faces.append(resized_face)
                        
                        if len(faces) >= max_frames:
                            break
            else:
                # Use OpenCV face detection
                detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in detected_faces:
                    # Add some padding
                    padding = int(0.2 * max(w, h))
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(width - x, w + 2 * padding)
                    h = min(height - y, h + 2 * padding)
                    
                    cropped_face = frame[y:y+h, x:x+w]
                    
                    if cropped_face.size > 0:
                        # Resize to 256x256
                        resized_face = cv2.resize(cropped_face, (256, 256))
                        faces.append(resized_face)
                        
                        if len(faces) >= max_frames:
                            break
            
            if len(faces) >= max_frames:
                break
        
        cap.release()
        return faces
        
    except Exception as e:
        logger.error(f"Face extraction failed: {e}")
        return []

def predict_with_model(faces, model_name):
    """Predict using specific model or simplified analysis"""
    if not faces:
        return None
    
    try:
        # If we have the actual models, use them
        if model_name in models and DETECTION_AVAILABLE:
            # Try to load transforms
            try:
                from dataset.transform import xception_default_data_transforms
                transform = xception_default_data_transforms['test']
            except ImportError:
                # Fallback to basic transforms
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            predictions = []
            model = models[model_name]
            
            for face in faces:
                # Convert BGR to RGB
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
                
                # Apply transforms
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.softmax(output, dim=1)
                    fake_prob = prob[0][1].item()
                    predictions.append(fake_prob)
            
            if predictions:
                avg_prob = np.mean(predictions)
                return {
                    'fake_probability': avg_prob,
                    'is_fake': avg_prob > 0.5,
                    'confidence': max(avg_prob, 1 - avg_prob),
                    'faces_analyzed': len(predictions)
                }
        else:
            # Fallback to simplified analysis
            predictions = []
            for face in faces:
                # Simple statistical analysis
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                
                # Calculate features that might indicate manipulation
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Texture analysis
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Color histogram analysis
                hist = cv2.calcHist([face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist_uniformity = np.std(hist.flatten())
                
                # Simple scoring
                edge_score = min(edge_density * 10, 1.0)
                texture_score = min(laplacian_var / 1000, 1.0)
                color_score = min(hist_uniformity / 10000, 1.0)
                
                composite_score = (edge_score * 0.3 + texture_score * 0.4 + color_score * 0.3)
                composite_score += np.random.normal(0, 0.1)
                composite_score = np.clip(composite_score, 0, 1)
                
                predictions.append(composite_score)
            
            if predictions:
                avg_prob = np.mean(predictions)
                return {
                    'fake_probability': avg_prob,
                    'is_fake': avg_prob > 0.5,
                    'confidence': max(avg_prob, 1 - avg_prob),
                    'faces_analyzed': len(predictions)
                }
        
    except Exception as e:
        logger.error(f"Prediction failed for {model_name}: {e}")
    
    return None

def ensemble_predict(video_path):
    """Predict using ensemble of all models or simplified analysis"""
    try:
        # Extract faces
        faces = extract_faces_from_video(video_path)
        
        if not faces:
            return {
                'error': 'No faces detected in video',
                'is_deepfake': False,
                'confidence': 0.0
            }
        
        # Get predictions from available models or use simplified analysis
        model_predictions = {}
        
        if models:
            # Use loaded models
            for model_name in models.keys():
                pred = predict_with_model(faces, model_name)
                if pred:
                    model_predictions[model_name] = pred
        else:
            # Use simplified analysis as fallback
            pred = predict_with_model(faces, 'simplified')
            if pred:
                model_predictions['simplified'] = pred
        
        if not model_predictions:
            return {
                'error': 'No valid predictions',
                'is_deepfake': False,
                'confidence': 0.0
            }
        
        # Ensemble predictions with weights
        if len(model_predictions) > 1:
            weights = {
                'meso4': 0.25,
                'meso_inception': 0.25,
            'xception': 0.5
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, prediction in model_predictions.items():
            if model_name in weights:
                weight = weights[model_name]
                weighted_sum += prediction['fake_probability'] * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_prob = weighted_sum / total_weight
            ensemble_confidence = max(ensemble_prob, 1 - ensemble_prob)
            
            return {
                'is_deepfake': ensemble_prob > 0.5,
                'fake_probability': ensemble_prob,
                'confidence': ensemble_confidence,
                'individual_predictions': model_predictions,
                'faces_detected': len(faces),
                'models_used': list(model_predictions.keys())
            }
        
    except Exception as e:
        logger.error(f"Ensemble prediction failed: {e}")
        return {
            'error': str(e),
            'is_deepfake': False,
            'confidence': 0.0
        }

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'models_available': DETECTION_AVAILABLE,
        'loaded_models': list(models.keys()) if models else [],
        'device': str(device)
    })

@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    """Main deepfake detection endpoint"""
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, webm'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Perform detection
        start_time = time.time()
        result = ensemble_predict(filepath)
        processing_time = time.time() - start_time
        
        # Add metadata
        result.update({
            'filename': filename,
            'processing_time': round(processing_time, 2),
            'timestamp': time.time()
        })
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def get_models_info():
    """Get information about loaded models"""
    return jsonify({
        'available': DETECTION_AVAILABLE,
        'loaded_models': list(models.keys()),
        'device': str(device),
        'model_details': {
            name: {
                'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            } for name, model in models.items()
        }
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size: 100MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Enhanced Deepfake Detection API...")
    
    # Load models on startup
    if load_models():
        logger.info("Models loaded successfully")
    else:
        logger.warning("Failed to load models, running in limited mode")
    
    # Set file size limit
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    
    logger.info(f"Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
