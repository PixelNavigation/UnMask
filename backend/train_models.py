#!/usr/bin/env python3
"""
Model Training Script for FaceForensics++ Dataset
Fine-tune Xception and WSDAN models on the current dataset to improve performance
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T
import logging
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import required modules
kaggle_dfdc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kaggle-dfdc')
sys.path.insert(0, kaggle_dfdc_path)

from face_utils import FaceDetector, norm_crop
from model_def import WSDAN, xception
from DeepfakeDetector import DeepfakeDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceForensicsDataset(Dataset):
    """Dataset for FaceForensics++ faces"""
    
    def __init__(self, video_paths, labels, face_detector, transform=None, max_faces_per_video=10):
        self.video_paths = video_paths
        self.labels = labels
        self.face_detector = face_detector
        self.transform = transform
        self.max_faces_per_video = max_faces_per_video
        
        # Extract faces from all videos
        logger.info(f"Extracting faces from {len(video_paths)} videos...")
        self.faces = []
        self.face_labels = []
        
        for video_path, label in tqdm(zip(video_paths, labels), total=len(video_paths), desc="Extracting faces"):
            faces = self._extract_faces_from_video(video_path)
            # Limit faces per video to avoid class imbalance
            faces = faces[:self.max_faces_per_video]
            
            self.faces.extend(faces)
            self.face_labels.extend([label] * len(faces))
        
        logger.info(f"Extracted {len(self.faces)} total faces")
        logger.info(f"Real faces: {sum(1 for l in self.face_labels if l == 0)}")
        logger.info(f"Fake faces: {sum(1 for l in self.face_labels if l == 1)}")
    
    def _extract_faces_from_video(self, video_path):
        """Extract faces from a video"""
        faces = []
        cap = cv2.VideoCapture(video_path)
        
        # Sample frames from video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(20, frame_count)  # Sample up to 20 frames
        step = max(1, frame_count // sample_frames)
        
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Detect faces using standardized approach
                boxes, landms = self._detect_faces_standardized(frame)
                
                if len(boxes) > 0:
                    # Process first face only
                    landm = landms[0].reshape(5, 2)
                    face_img = norm_crop(frame, landm, image_size=224)
                    faces.append(face_img)
                    
                    if len(faces) >= self.max_faces_per_video:
                        break
                        
            except Exception as e:
                logger.warning(f"Error extracting face from frame: {e}")
                continue
        
        cap.release()
        return faces
    
    def _detect_faces_standardized(self, image):
        """Standardized face detection to match DeepfakeDetector"""
        # Resize image to standard size to avoid prior box mismatches
        original_shape = image.shape
        target_height, target_width = 640, 640
        
        # Resize while maintaining aspect ratio
        h, w = original_shape[:2]
        scale = min(target_height / h, target_width / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized_image = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        padded_image = np.zeros((target_height, target_width, 3), dtype=image.dtype)
        padded_image[:new_h, :new_w] = resized_image
        
        # Detect faces on standardized image
        boxes, landms = self.face_detector.detect(padded_image)
        
        # Scale coordinates back to original image size
        if len(boxes) > 0:
            boxes = boxes / scale
            landms = landms / scale
        
        return boxes, landms
    
    def __len__(self):
        return len(self.faces)
    
    def __getitem__(self, idx):
        face_img = self.faces[idx]
        label = self.face_labels[idx]
        
        # Convert to PIL for transforms
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        if self.transform:
            face_tensor = self.transform(face_pil)
        else:
            face_tensor = T.ToTensor()(face_pil)
        
        return face_tensor, label

class ModelTrainer:
    """Trainer for deepfake detection models"""
    
    def __init__(self, dataset_path, device='cuda'):
        self.dataset_path = Path(dataset_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize face detector
        self.face_detector = FaceDetector(device=str(self.device))
        
        # Initialize models
        self.setup_models()
        
        # Data transforms
        self.train_transform = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            T.RandomRotation(5),
        ])
        
        self.val_transform = T.Compose([
            T.ToTensor(),
        ])
    
    def setup_models(self):
        """Initialize models with existing checkpoints"""
        # Model 1: Standalone Xception
        self.model_xception = xception(num_classes=2, pretrained=False).to(self.device)
        
        # Load existing weights if available
        xception_weights_path = "kaggle-dfdc/model_def/xception-hg-2.pth"
        if os.path.exists(xception_weights_path):
            logger.info("Loading existing Xception weights...")
            ckpt = torch.load(xception_weights_path, map_location=self.device, weights_only=False)
            self.model_xception.load_state_dict(ckpt.get("state_dict", ckpt))
        
        # Model 2: WSDAN + Xception
        self.model_wsdan = WSDAN(num_classes=2, M=8, net="xception", pretrained=False).to(self.device)
        
        # Load existing WSDAN weights if available
        wsdan_weights_path = "kaggle-dfdc/model_def/ckpt_x.pth"
        if os.path.exists(wsdan_weights_path):
            logger.info("Loading existing WSDAN weights...")
            ckpt = torch.load(wsdan_weights_path, map_location=self.device, weights_only=False)
            self.model_wsdan.load_state_dict(ckpt.get("state_dict", ckpt))
        
        logger.info("Models initialized successfully")
    
    def load_dataset(self, max_videos_per_class=1000):
        """Load and prepare dataset using CSV metadata - FAKE VIDEOS ONLY"""
        logger.info("Loading dataset from CSV metadata...")
        
        # Find all CSV files in the dataset
        csv_folder = self.dataset_path / "csv"
        if not csv_folder.exists():
            logger.error("CSV folder not found! Looking for metadata files...")
            return self._load_dataset_fallback(max_videos_per_class)
        
        # Read all CSV files and combine them
        all_videos = []
        all_labels = []
        
        csv_files = list(csv_folder.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            logger.info(f"Reading {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"CSV columns: {df.columns.tolist()}")
                
                # Find the relevant columns (adapt based on actual CSV structure)
                video_col = None
                label_col = None
                
                # Look for exact matches first, then partial matches
                for col in df.columns:
                    if col.lower() in ['file path', 'file_path', 'filepath', 'video_path', 'video path', 'path']:
                        video_col = col
                        break
                    elif col.lower() in ['video', 'file', 'video_file', 'video file']:
                        video_col = col
                
                # If no exact match, look for partial matches but exclude "Unnamed" columns
                if video_col is None:
                    for col in df.columns:
                        if 'unnamed' not in col.lower() and ('video' in col.lower() or 'file' in col.lower() or 'path' in col.lower()):
                            video_col = col
                            break
                
                # Common column names for labels
                for col in df.columns:
                    if 'label' in col.lower() or 'real' in col.lower() or 'fake' in col.lower() or 'class' in col.lower():
                        label_col = col
                        break
                
                if video_col is None or label_col is None:
                    logger.warning(f"Could not identify video and label columns in {csv_file.name}")
                    logger.info(f"Available columns: {df.columns.tolist()}")
                    # Print first few rows to understand structure
                    logger.info(f"First few rows:\n{df.head()}")
                    continue
                
                # Process each row - ONLY FAKE VIDEOS
                for _, row in df.iterrows():
                    video_name = row[video_col]
                    label_value = row[label_col]
                    
                    # Convert video_name to string and handle various formats
                    if pd.isna(video_name):
                        continue  # Skip rows with missing video names
                    
                    video_name = str(video_name).strip()
                    if not video_name:
                        continue  # Skip empty video names
                    
                    # Only process videos with clear labels
                    if isinstance(label_value, str):
                        if label_value.lower() in ['real', 'original']:
                            is_fake = False
                        elif label_value.lower() in ['fake', 'deepfake', 'manipulated']:
                            is_fake = True
                        else:
                            continue  # Skip unclear labels
                    else:
                        is_fake = bool(label_value)
                    
                    # Find the actual video file
                    video_path = self._find_video_file(video_name)
                    if video_path:
                        all_videos.append(video_path)
                        all_labels.append(1 if is_fake else 0)  # 0 = real, 1 = fake
                
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {e}")
                continue
        
        if not all_videos:
            logger.warning("No videos found from CSV metadata, falling back to folder-based loading")
            return self._load_dataset_fallback(max_videos_per_class)
        
        # Separate real and fake videos
        real_videos = []
        fake_videos = []
        
        for video_path, label in zip(all_videos, all_labels):
            if 'original' in str(video_path).lower():
                real_videos.append(video_path)
            else:
                fake_videos.append(video_path)
        
        # Use balanced dataset: equal numbers of real and fake videos
        max_per_class = min(len(real_videos), len(fake_videos), max_videos_per_class // 2)
        
        # Select videos
        selected_real = real_videos[:max_per_class]
        selected_fake = fake_videos[:max_per_class]
        
        # Combine and create labels
        final_videos = selected_real + selected_fake
        final_labels = [0] * len(selected_real) + [1] * len(selected_fake)  # 0 = real, 1 = fake
        
        logger.info(f"Loaded {len(selected_real)} real videos, {len(selected_fake)} fake videos")
        logger.info(f"Total: {len(final_videos)} videos")
        
        # Split into train/validation
        train_videos, val_videos, train_labels, val_labels = train_test_split(
            final_videos, final_labels, test_size=0.2, random_state=42, stratify=final_labels
        )
        
        logger.info(f"Train: {len(train_videos)} videos, Val: {len(val_videos)} videos")
        logger.info(f"Training labels distribution - 0 (lower quality): {train_labels.count(0)}, 1 (higher quality): {train_labels.count(1)}")
        
        # Create datasets
        train_dataset = FaceForensicsDataset(
            train_videos, train_labels, self.face_detector, 
            transform=self.train_transform, max_faces_per_video=12  # More faces per video since we have fewer videos
        )
        
        val_dataset = FaceForensicsDataset(
            val_videos, val_labels, self.face_detector, 
            transform=self.val_transform, max_faces_per_video=8  # More faces for validation too
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_dataset, val_dataset
    
    def _find_video_file(self, video_name):
        """Find video file in the dataset"""
        # Clean up video name - ensure it's a string
        video_name = str(video_name).strip()
        
        # If it's already a relative path with folder, use it directly
        if '/' in video_name and not video_name.startswith('/'):
            full_path = self.dataset_path / video_name
            if full_path.exists():
                return str(full_path)
        
        # Handle path separators and extract just the filename if it's a full path
        if '/' in video_name or '\\' in video_name:
            video_name = os.path.basename(video_name)
        
        # Ensure .mp4 extension
        if not video_name.endswith('.mp4'):
            video_name += '.mp4'
        
        # Search in all possible folders
        folders_to_search = [
            "original",
            "DeepFakeDetection", 
            "Deepfakes",
            "Face2Face", 
            "FaceShifter",
            "FaceSwap", 
            "NeuralTextures"
        ]
        
        for folder_name in folders_to_search:
            folder_path = self.dataset_path / folder_name
            if folder_path.exists():
                video_path = folder_path / video_name
                if video_path.exists():
                    return str(video_path)
        
        logger.warning(f"Video file not found: {video_name}")
        return None
    
    def _load_dataset_fallback(self, max_videos_per_class=1000):
        """Fallback method to load dataset from folder structure - FAKE VIDEOS ONLY"""
        logger.info("Loading dataset from folder structure - fake videos only...")
        
        # Only load fake videos from deepfake folders
        fake_videos = []
        fake_folders = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
        videos_per_fake_folder = max_videos_per_class // len(fake_folders)
        
        for folder_name in fake_folders:
            folder_path = self.dataset_path / folder_name
            if folder_path.exists():
                folder_videos = [str(f) for f in folder_path.glob("*.mp4")][:videos_per_fake_folder]
                fake_videos.extend(folder_videos)
        
        logger.info(f"Found {len(fake_videos)} fake videos (no real videos)")
        
        # Create labels based on deepfake type quality
        all_videos = fake_videos
        all_labels = []
        for video_path in all_videos:
            if any(fake_type in video_path.lower() for fake_type in ['face2face', 'neuraltextures']):
                all_labels.append(0)  # Lower quality deepfakes
            else:
                all_labels.append(1)  # Higher quality deepfakes
        
        # Split into train/validation
        train_videos, val_videos, train_labels, val_labels = train_test_split(
            all_videos, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        logger.info(f"Train: {len(train_videos)} videos, Val: {len(val_videos)} videos")
        
        # Create datasets
        train_dataset = FaceForensicsDataset(
            train_videos, train_labels, self.face_detector, 
            transform=self.train_transform, max_faces_per_video=8
        )
        
        val_dataset = FaceForensicsDataset(
            val_videos, val_labels, self.face_detector, 
            transform=self.val_transform, max_faces_per_video=5
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_dataset, val_dataset
    
    def train_model(self, model, model_name, num_epochs=10, learning_rate=1e-4):
        """Train a single model"""
        logger.info(f"Training {model_name}...")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                if model_name == "WSDAN":
                    # For WSDAN, normalize input
                    zhq_nm_avg = torch.Tensor([.4479, .3744, .3473]).view(1, 3, 1, 1).expand_as(data[:, :, :1, :1]).to(self.device)
                    zhq_nm_std = torch.Tensor([.2537, .2502, .2424]).view(1, 3, 1, 1).expand_as(data[:, :, :1, :1]).to(self.device)
                    data_normalized = (data - zhq_nm_avg) / zhq_nm_std
                    output, _, _ = model(data_normalized)
                else:
                    # For Xception, resize to 299x299 and normalize
                    data_resized = torch.nn.functional.interpolate(data, size=299, mode='bilinear', align_corners=False)
                    data_resized = data_resized.sub_(0.5).mul_(2.0)  # Normalize to [-1, 1]
                    output = model(data_resized)
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Collect predictions for metrics
                pred = output.argmax(dim=1)
                train_preds.extend(pred.cpu().numpy())
                train_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for data, target in tqdm(self.val_loader, desc="Validation"):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if model_name == "WSDAN":
                        zhq_nm_avg = torch.Tensor([.4479, .3744, .3473]).view(1, 3, 1, 1).expand_as(data[:, :, :1, :1]).to(self.device)
                        zhq_nm_std = torch.Tensor([.2537, .2502, .2424]).view(1, 3, 1, 1).expand_as(data[:, :, :1, :1]).to(self.device)
                        data_normalized = (data - zhq_nm_avg) / zhq_nm_std
                        output, _, _ = model(data_normalized)
                    else:
                        data_resized = torch.nn.functional.interpolate(data, size=299, mode='bilinear', align_corners=False)
                        data_resized = data_resized.sub_(0.5).mul_(2.0)
                        output = model(data_resized)
                    
                    val_loss += criterion(output, target).item()
                    
                    pred = output.argmax(dim=1)
                    val_preds.extend(pred.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())
            
            # Calculate metrics
            train_acc = accuracy_score(train_targets, train_preds)
            val_acc = accuracy_score(val_targets, val_preds)
            val_precision = precision_score(val_targets, val_preds, zero_division=0)
            val_recall = recall_score(val_targets, val_preds, zero_division=0)
            val_f1 = f1_score(val_targets, val_preds, zero_division=0)
            
            # Update learning rate
            scheduler.step()
            
            # Save metrics
            train_losses.append(train_loss / len(self.train_loader))
            val_losses.append(val_loss / len(self.val_loader))
            val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            logger.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                       f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
    
    def save_models(self):
        """Save trained models"""
        os.makedirs("trained_models", exist_ok=True)
        
        # Save Xception
        torch.save({
            'state_dict': self.model_xception.state_dict(),
            'model_type': 'xception'
        }, "trained_models/xception_finetuned.pth")
        
        # Save WSDAN
        torch.save({
            'state_dict': self.model_wsdan.state_dict(),
            'model_type': 'wsdan'
        }, "trained_models/wsdan_finetuned.pth")
        
        logger.info("Models saved to 'trained_models' directory")
    
    def evaluate_models(self):
        """Evaluate both models on a diverse test set"""
        logger.info("Evaluating models on diverse test set...")
        
        detector = DeepfakeDetector()
        
        # Get test videos from all categories
        test_videos = []
        test_labels = []
        
        # Real videos from original folder
        original_folder = self.dataset_path / "original"
        if original_folder.exists():
            real_test = list(original_folder.glob("*.mp4"))[-30:]  # Last 30 videos
            test_videos.extend([str(v) for v in real_test])
            test_labels.extend([0] * len(real_test))
        
        # Fake videos from all deepfake categories (balanced)
        fake_folders = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
        videos_per_category = 5  # 5 videos per category = 30 fake videos total
        
        for folder_name in fake_folders:
            folder_path = self.dataset_path / folder_name
            if folder_path.exists():
                fake_test = list(folder_path.glob("*.mp4"))[-videos_per_category:]
                test_videos.extend([str(v) for v in fake_test])
                test_labels.extend([1] * len(fake_test))
        
        logger.info(f"Testing on {len(test_videos)} videos (Real: {test_labels.count(0)}, Fake: {test_labels.count(1)})")
        
        predictions = []
        confidences = []
        
        for video_path in tqdm(test_videos, desc="Testing"):
            try:
                result = detector.predict(video_path, use_temporal=False)
                pred = 1 if result.get("is_deepfake", False) else 0
                conf = result.get("confidence", 0.5)
                predictions.append(pred)
                confidences.append(conf)
            except Exception as e:
                logger.warning(f"Error testing {Path(video_path).name}: {e}")
                predictions.append(0)  # Default to real
                confidences.append(0.5)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        
        # Calculate per-category metrics
        logger.info("\n📊 Per-category performance:")
        
        # Real videos performance
        real_indices = [i for i, label in enumerate(test_labels) if label == 0]
        real_preds = [predictions[i] for i in real_indices]
        real_accuracy = accuracy_score([0] * len(real_preds), real_preds)
        logger.info(f"Real videos accuracy: {real_accuracy:.4f} ({len(real_indices)} videos)")
        
        # Fake videos performance by category
        fake_start_idx = len([i for i, label in enumerate(test_labels) if label == 0])
        
        for i, folder_name in enumerate(fake_folders):
            start_idx = fake_start_idx + (i * videos_per_category)
            end_idx = start_idx + videos_per_category
            
            if end_idx <= len(predictions):
                category_preds = predictions[start_idx:end_idx]
                category_labels = test_labels[start_idx:end_idx]
                
                if category_labels:
                    category_accuracy = accuracy_score(category_labels, category_preds)
                    logger.info(f"{folder_name} accuracy: {category_accuracy:.4f} ({len(category_labels)} videos)")
        
        logger.info(f"\n🎯 Overall Test Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'true_labels': test_labels,
            'confidences': confidences
        }

def main():
    """Main training function"""
    dataset_path = r"C:\Users\babel\Downloads\archive\FaceForensics++_C23"
    
    trainer = ModelTrainer(dataset_path)
    
    # Load dataset with balanced real/fake videos (smaller size to prevent overfitting)
    train_dataset, val_dataset = trainer.load_dataset(max_videos_per_class=100)
    
    # Train Xception model
    logger.info("=" * 50)
    logger.info("Training Xception Model")
    logger.info("=" * 50)
    xception_results = trainer.train_model(trainer.model_xception, "Xception", num_epochs=5, learning_rate=1e-4)
    
    # Train WSDAN model
    logger.info("=" * 50)
    logger.info("Training WSDAN Model")
    logger.info("=" * 50)
    wsdan_results = trainer.train_model(trainer.model_wsdan, "WSDAN", num_epochs=5, learning_rate=1e-4)
    
    # Save models
    trainer.save_models()
    
    # Evaluate models
    test_results = trainer.evaluate_models()
    
    # Save training results
    results = {
        'xception': xception_results,
        'wsdan': wsdan_results,
        'test_results': test_results
    }
    
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info(f"Best Xception validation accuracy: {xception_results['best_val_acc']:.4f}")
    logger.info(f"Best WSDAN validation accuracy: {wsdan_results['best_val_acc']:.4f}")
    logger.info(f"Test accuracy: {test_results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
