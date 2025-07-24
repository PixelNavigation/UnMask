#!/usr/bin/env python3
"""
Enhanced Deepfake Detector integrating multiple models and approaches
Combines original UnMask models with HongguLiu/Deepfake-Detection models
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image as pil_image
import logging
from typing import Dict, List, Optional, Tuple
import dlib

# Add paths for both model systems
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'kaggle-dfdc'))
sys.path.insert(0, os.path.join(current_dir, 'Deepfake-Detection'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import original models
try:
    from DeepfakeDetector import DeepfakeDetector
    from kaggle_dfdc.face_utils import FaceDetector
    from kaggle_dfdc.model_def import WSDAN, xception
    ORIGINAL_AVAILABLE = True
except ImportError:
    logger.warning("Original UnMask models not available")
    ORIGINAL_AVAILABLE = False
    DeepfakeDetector = None

# Import new models from cloned repository
try:
    from network.models import model_selection, TransferModel
    from network.mesonet import Meso4, MesoInception4
    # Create a simple transform since we might not have the exact dataset module
    import torchvision.transforms as transforms
    xception_default_data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    HONGGU_AVAILABLE = True
except ImportError:
    logger.warning("HongguLiu models not available")
    HONGGU_AVAILABLE = False
    TransferModel = None
    Meso4 = None
    MesoInception4 = None
    xception_default_data_transforms = None

class EnhancedDeepfakeDetector:
    """
    Enhanced deepfake detector that combines multiple detection approaches:
    1. Original UnMask models (Xception + WSDAN + BiLSTM)
    2. HongguLiu models (MesoNet, enhanced Xception)
    3. Ensemble voting system
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Enhanced Detector initialized on {self.device}")
        
        # Initialize face detectors
        self.init_face_detectors()
        
        # Initialize all models
        self.init_original_models()
        self.init_honggu_models()
        
        # Model weights for ensemble
        self.model_weights = {
            'original_ensemble': 0.4,  # Original UnMask ensemble
            'meso4': 0.15,             # MesoNet Meso4
            'meso_inception': 0.15,    # MesoNet MesoInception4
            'xception_honggu': 0.3     # HongguLiu Xception
        }
        
    def init_face_detectors(self):
        """Initialize face detection systems"""
        # Original RetinaFace detector
        try:
            self.retinaface_detector = FaceDetector(device=str(self.device))
            logger.info("RetinaFace detector initialized")
        except Exception as e:
            logger.warning(f"RetinaFace detector failed: {e}")
            self.retinaface_detector = None
        
        # Dlib face detector (from HongguLiu repo)
        try:
            self.dlib_detector = dlib.get_frontal_face_detector()
            logger.info("Dlib face detector initialized")
        except Exception as e:
            logger.warning(f"Dlib detector failed: {e}")
            self.dlib_detector = None
    
    def init_original_models(self):
        """Initialize original UnMask models"""
        try:
            self.original_detector = DeepfakeDetector()
            logger.info("Original UnMask models loaded successfully")
            self.original_available = True
        except Exception as e:
            logger.warning(f"Original models failed to load: {e}")
            self.original_available = False
    
    def init_honggu_models(self):
        """Initialize HongguLiu models"""
        self.honggu_models = {}
        
        # Initialize MesoNet models
        try:
            self.honggu_models['meso4'] = Meso4(num_classes=2).to(self.device)
            self.honggu_models['meso_inception'] = MesoInception4(num_classes=2).to(self.device)
            
            # Load pretrained weights if available
            weights_dir = Path("Deepfake-Detection/weights")
            if weights_dir.exists():
                meso4_weights = weights_dir / "Meso4_DF.pkl"
                meso_inception_weights = weights_dir / "MesoInception4_DF.pkl"
                
                if meso4_weights.exists():
                    self.honggu_models['meso4'].load_state_dict(torch.load(meso4_weights, map_location=self.device))
                    logger.info("Meso4 weights loaded")
                
                if meso_inception_weights.exists():
                    self.honggu_models['meso_inception'].load_state_dict(torch.load(meso_inception_weights, map_location=self.device))
                    logger.info("MesoInception4 weights loaded")
            
        except Exception as e:
            logger.warning(f"MesoNet models failed to load: {e}")
        
        # Initialize enhanced Xception
        try:
            self.honggu_models['xception'] = TransferModel('xception', num_out_classes=2).to(self.device)
            
            # Load pretrained weights if available
            xception_weights = Path("Deepfake-Detection/weights/xception_c23.pkl")
            if xception_weights.exists():
                self.honggu_models['xception'].load_state_dict(torch.load(xception_weights, map_location=self.device))
                logger.info("HongguLiu Xception weights loaded")
                
        except Exception as e:
            logger.warning(f"HongguLiu Xception failed to load: {e}")
        
        # Set all models to evaluation mode
        for model in self.honggu_models.values():
            model.eval()
        
        logger.info(f"HongguLiu models loaded: {list(self.honggu_models.keys())}")
    
    def get_dlib_boundingbox(self, face, width, height, scale=1.3, minsize=None):
        """
        Generate bounding box for dlib face detection
        """
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
    
    def extract_faces_dlib(self, frame):
        """Extract faces using dlib detector"""
        if self.dlib_detector is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.dlib_detector(gray, 1)
            
            face_crops = []
            height, width = frame.shape[:2]
            
            for face in faces:
                x, y, size = self.get_dlib_boundingbox(face, width, height)
                cropped_face = frame[y:y+size, x:x+size]
                
                if cropped_face.size > 0:
                    # Resize to 256x256 for HongguLiu models
                    resized_face = cv2.resize(cropped_face, (256, 256))
                    face_crops.append(resized_face)
            
            return face_crops
            
        except Exception as e:
            logger.warning(f"Dlib face extraction failed: {e}")
            return []
    
    def preprocess_for_honggu_models(self, face_image, target_size=256):
        """Preprocess face image for HongguLiu models"""
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image_obj = pil_image.fromarray(face_rgb)
            
            # Apply transforms
            transform = xception_default_data_transforms['test']
            transformed = transform(pil_image_obj)
            
            # Add batch dimension
            return transformed.unsqueeze(0).to(self.device)
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return None
    
    def predict_with_honggu_models(self, face_crops):
        """Get predictions from HongguLiu models"""
        predictions = {}
        
        if not face_crops:
            return predictions
        
        for model_name, model in self.honggu_models.items():
            try:
                model_predictions = []
                
                for face_crop in face_crops:
                    # Preprocess
                    input_tensor = self.preprocess_for_honggu_models(face_crop)
                    if input_tensor is None:
                        continue
                    
                    # Predict
                    with torch.no_grad():
                        output = model(input_tensor)
                        prob = torch.softmax(output, dim=1)
                        fake_prob = prob[0][1].item()  # Probability of being fake
                        model_predictions.append(fake_prob)
                
                if model_predictions:
                    # Average predictions across all faces
                    avg_prediction = np.mean(model_predictions)
                    predictions[model_name] = {
                        'fake_probability': avg_prediction,
                        'is_fake': avg_prediction > 0.5,
                        'confidence': max(avg_prediction, 1 - avg_prediction)
                    }
                    
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
                continue
        
        return predictions
    
    def ensemble_predict(self, video_path: str, use_temporal: bool = True) -> Dict:
        """
        Enhanced prediction using ensemble of all available models
        """
        results = {
            'video_path': video_path,
            'models_used': [],
            'individual_predictions': {},
            'ensemble_result': {},
            'processing_info': {}
        }
        
        try:
            # 1. Get prediction from original UnMask models
            if self.original_available:
                try:
                    original_result = self.original_detector.predict(video_path, use_temporal=use_temporal)
                    results['individual_predictions']['original_ensemble'] = original_result
                    results['models_used'].append('original_ensemble')
                except Exception as e:
                    logger.warning(f"Original model prediction failed: {e}")
            
            # 2. Get predictions from HongguLiu models
            honggu_predictions = self.predict_video_honggu_models(video_path)
            results['individual_predictions'].update(honggu_predictions)
            results['models_used'].extend(honggu_predictions.keys())
            
            # 3. Ensemble the predictions
            ensemble_result = self.combine_predictions(results['individual_predictions'])
            results['ensemble_result'] = ensemble_result
            
            # 4. Add processing information
            results['processing_info'] = {
                'total_models': len(results['models_used']),
                'temporal_analysis': use_temporal,
                'face_detection_methods': ['retinaface', 'dlib']
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def predict_video_honggu_models(self, video_path: str) -> Dict:
        """Predict video using HongguLiu models"""
        predictions = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return predictions
            
            frame_predictions = {model_name: [] for model_name in self.honggu_models.keys()}
            frame_count = 0
            max_frames = 30  # Sample frames for efficiency
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // max_frames)
            
            for i in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract faces using dlib
                face_crops = self.extract_faces_dlib(frame)
                
                if face_crops:
                    # Get predictions for this frame
                    frame_pred = self.predict_with_honggu_models(face_crops)
                    
                    for model_name, pred in frame_pred.items():
                        frame_predictions[model_name].append(pred['fake_probability'])
                    
                    frame_count += 1
                    if frame_count >= max_frames:
                        break
            
            cap.release()
            
            # Aggregate frame predictions
            for model_name, probs in frame_predictions.items():
                if probs:
                    avg_prob = np.mean(probs)
                    predictions[model_name] = {
                        'fake_probability': avg_prob,
                        'is_fake': avg_prob > 0.5,
                        'confidence': max(avg_prob, 1 - avg_prob),
                        'frames_analyzed': len(probs)
                    }
        
        except Exception as e:
            logger.error(f"HongguLiu video prediction failed: {e}")
        
        return predictions
    
    def combine_predictions(self, individual_predictions: Dict) -> Dict:
        """Combine predictions from all models using weighted ensemble"""
        if not individual_predictions:
            return {'error': 'No predictions available'}
        
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            confidence_scores = []
            model_votes = []
            
            for model_name, prediction in individual_predictions.items():
                if model_name not in self.model_weights:
                    continue
                
                weight = self.model_weights[model_name]
                
                # Extract fake probability
                if model_name == 'original_ensemble':
                    fake_prob = prediction.get('fake_probability', 0.5)
                else:
                    fake_prob = prediction.get('fake_probability', 0.5)
                
                weighted_sum += fake_prob * weight
                total_weight += weight
                confidence_scores.append(prediction.get('confidence', 0.5))
                model_votes.append(1 if fake_prob > 0.5 else 0)
            
            if total_weight == 0:
                return {'error': 'No valid predictions with weights'}
            
            # Calculate ensemble results
            ensemble_fake_prob = weighted_sum / total_weight
            ensemble_is_fake = ensemble_fake_prob > 0.5
            ensemble_confidence = max(ensemble_fake_prob, 1 - ensemble_fake_prob)
            
            # Calculate consensus
            consensus = np.mean(model_votes) if model_votes else 0.5
            
            return {
                'is_deepfake': ensemble_is_fake,
                'fake_probability': ensemble_fake_prob,
                'confidence': ensemble_confidence,
                'consensus_score': consensus,
                'models_agreement': len([v for v in model_votes if v == int(ensemble_is_fake)]) / len(model_votes) if model_votes else 0,
                'avg_individual_confidence': np.mean(confidence_scores) if confidence_scores else 0.5
            }
            
        except Exception as e:
            logger.error(f"Prediction combination failed: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'original_models_available': self.original_available,
            'honggu_models_loaded': list(self.honggu_models.keys()),
            'face_detectors': [],
            'ensemble_weights': self.model_weights,
            'device': str(self.device)
        }
        
        if self.retinaface_detector:
            info['face_detectors'].append('retinaface')
        if self.dlib_detector:
            info['face_detectors'].append('dlib')
        
        return info

# Convenience function for quick detection
def detect_deepfake_enhanced(video_path: str, use_temporal: bool = True) -> Dict:
    """
    Quick deepfake detection using enhanced ensemble approach
    """
    detector = EnhancedDeepfakeDetector()
    return detector.ensemble_predict(video_path, use_temporal=use_temporal)

if __name__ == "__main__":
    # Test the enhanced detector
    detector = EnhancedDeepfakeDetector()
    print("Enhanced Deepfake Detector Info:")
    print(detector.get_model_info())
