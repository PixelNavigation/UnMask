from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import torch
import re
import numpy as np
from PIL import Image

# Add the dfdc_deepfake_challenge directory to the path
sys.path.append('dfdc_model')

from dfdc_model.kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video
from dfdc_model.training.zoo.classifiers import DeepFakeClassifier
from enhanced_prediction import enhanced_predict_on_video, enhanced_predict_on_image

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the model and preprocessing pipeline
def load_models():
    models = []
    weights_dir = "dfdc_model/weights"
    # Update these model names based on your downloaded weights
    model_files = [
        "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40",
        # Add more model files if you have them
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for model_file in model_files:
        model_path = os.path.join(weights_dir, model_file)
        if os.path.exists(model_path):
            model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to(device)
            print(f"Loading state dict {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
            model.eval()
            del checkpoint
            if device == "cuda":
                models.append(model.half())
            else:
                models.append(model)
    
    return models

# Initialize the preprocessing pipeline
def initialize_preprocessing():
    frames_per_video = 32
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
    input_size = 380
    return face_extractor, input_size, frames_per_video

# Load models and initialize preprocessing
print("Loading models...")
models = load_models()
face_extractor, input_size, frames_per_video = initialize_preprocessing()
print(f"Loaded {len(models)} models")

def predict_on_image(image_path):
    """Predict on a single image using face extraction and model inference"""
    try:
        from dfdc_model.kernel_utils import isotropically_resize_image, put_to_center, normalize_transform
        import numpy as np
        import cv2
        
        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use MTCNN to detect faces
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(image)
        img_pil = img_pil.resize(size=[s // 2 for s in img_pil.size])
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = face_extractor.detector
        
        batch_boxes, probs = detector.detect(img_pil, landmarks=False)
        
        if batch_boxes is None:
            return 0.5  # No faces detected, return neutral
            
        # Process the largest/most confident face
        best_idx = 0
        if len(probs) > 1:
            best_idx = np.argmax(probs)
            
        bbox = batch_boxes[best_idx]
        if bbox is not None:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3
            crop = image[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            
            # Resize and center the face
            resized_face = isotropically_resize_image(crop, input_size)
            resized_face = put_to_center(resized_face, input_size)
            
            # Prepare for model inference
            x = torch.tensor(resized_face, device=device).float().unsqueeze(0)
            x = x.permute((0, 3, 1, 2))
            x = normalize_transform(x[0] / 255.).unsqueeze(0)
            
            # Run prediction
            with torch.no_grad():
                preds = []
                for model in models:
                    if device == "cuda":
                        y_pred = model(x.half())
                    else:
                        y_pred = model(x)
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    preds.append(y_pred.cpu().numpy())
                return np.mean(preds)
        
        return 0.5
    except Exception as e:
        print(f"Error predicting on image: {e}")
        return 0.5

@app.route('/')
def index():
    return jsonify({'message': 'Deepfake Detection Flask API is running.'})

@app.route('/analyze/image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Use enhanced image prediction with Grad-CAM
        result = enhanced_predict_on_image(filepath, models, input_size)
        
        label = 'fake' if result['prediction'] > 0.5 else 'real'
        response = {
            'label': label,
            'confidence': float(result['prediction']),
            'details': f'Image analyzed using {len(models)} models with face detection and Grad-CAM',
            'faces_detected': 1,
            'processing_time': 'N/A',
            'gradcam_image': result['gradcam_image'],
            'fake_frames': []  # Not applicable for images
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Use enhanced video prediction with frame analysis and Grad-CAM
        result = enhanced_predict_on_video(
            face_extractor=face_extractor,
            video_path=filepath,
            batch_size=frames_per_video,
            input_size=input_size,
            models=models,
            strategy=confident_strategy,
            apply_compression=False,
            return_frame_predictions=True
        )
        
        prediction = result['prediction']
        label = 'fake' if prediction > 0.5 else 'real'
        
        response = {
            'label': label,
            'confidence': float(prediction),
            'details': f'Video analyzed using {len(models)} models with confident strategy and frame analysis',
            'faces_detected': len(result.get('frame_predictions', [])),
            'processing_time': 'N/A',
            'fake_frames': result.get('fake_frames', []),
            'gradcam_image': result.get('gradcam_images', [None])[0],  # Return first Grad-CAM image
            'frame_predictions': result.get('frame_predictions', [])
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
