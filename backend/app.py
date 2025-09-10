from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import tempfile
import traceback
import logging
from PIL import Image

def _ensure_writable_caches():
    """
    Redirect all model / hub caches to writable tmp directories to avoid
    PermissionError inside container (e.g. attempts to write to '/.cache').
    Safe to call multiple times.
    """
    tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    base_cache = os.path.join(tmp_root, ".cache")
    subdirs = {
        "HF_HOME": "huggingface",
        "HF_HUB_CACHE": "huggingface",
        "TRANSFORMERS_CACHE": "huggingface",
        "TORCH_HOME": "torch",
        "TIMM_CACHE_DIR": "timm",
        "XDG_CACHE_HOME": "",  # root of the cache tree
    }
    for var, sub in subdirs.items():
        path = os.environ.get(var)
        if not path:
            path = os.path.join(base_cache, sub) if sub else base_cache
            os.environ[var] = path
        os.makedirs(path, exist_ok=True)
    # Quiet albumentations update check noise
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

_ensure_writable_caches()

# Add model directory before importing model utilities
sys.path.append('dfdc_model')

try:
    from dfdc_model.kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video
    from dfdc_model.training.zoo.classifiers import DeepFakeClassifier
except Exception as e:
    traceback.print_exc()
    with open(os.path.join(tempfile.gettempdir(), "startup_error.log"), "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
    VideoReader = FaceExtractor = confident_strategy = predict_on_video = None
    DeepFakeClassifier = None

app = Flask(__name__)
CORS(app)

# Choose a writable upload folder
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER") or os.path.join(tempfile.gettempdir(), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Choose an upload folder that is writable in container environments.
# Prefer environment override, then /tmp/uploads, then a tempdir fallback.
upload_folder = os.environ.get('UPLOAD_FOLDER', '/tmp/uploads')
try:
    os.makedirs(upload_folder, exist_ok=True)
except PermissionError:
    # Fallback to system temp directory if the preferred path is not writable
    fallback = os.path.join(tempfile.gettempdir(), 'unmask_uploads')
    try:
        os.makedirs(fallback, exist_ok=True)
        upload_folder = fallback
        print(f"Warning: preferred upload folder not writable, using fallback: {fallback}")
    except Exception as e:
        # As a last resort, set upload_folder to tempfile.gettempdir()
        upload_folder = tempfile.gettempdir()
        print(f"Critical: unable to create upload directories, using system temp: {upload_folder} ({e})")

app.config['UPLOAD_FOLDER'] = upload_folder

def cleanup_memory():
    """Clean up memory and GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def cleanup_file(filepath):
    """Clean up uploaded file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error removing file {filepath}: {e}")

def smart_prediction_strategy(predictions):
    """
    Improved prediction strategy that considers both count and confidence of fake frames
    """
    predictions = np.array(predictions)
    
    if len(predictions) == 0:
        return 0.5
    
    # Count frames above different thresholds
    fake_frames_50 = np.sum(predictions > 0.5)  # Standard threshold
    fake_frames_60 = np.sum(predictions > 0.6)  # Higher confidence
    fake_frames_70 = np.sum(predictions > 0.7)  # Very high confidence
    
    total_frames = len(predictions)
    fake_ratio = fake_frames_50 / total_frames
    
    print(f"Debug: {fake_frames_50}/{total_frames} frames suspicious ({fake_ratio:.2%})")
    print(f"Debug: Mean prediction: {np.mean(predictions):.3f}")
    
    # If majority of frames are suspicious (>50%), classify as fake
    if fake_ratio > 0.5:
        # Weight the prediction based on how many frames are suspicious
        if fake_frames_70 > total_frames * 0.3:  # 30%+ very high confidence
            return min(0.95, np.mean(predictions[predictions > 0.7]))
        elif fake_frames_60 > total_frames * 0.4:  # 40%+ high confidence  
            return min(0.9, np.mean(predictions[predictions > 0.6]))
        else:
            # Use weighted average favoring suspicious frames
            suspicious_mean = np.mean(predictions[predictions > 0.5])
            overall_mean = np.mean(predictions)
            return (suspicious_mean * 0.7 + overall_mean * 0.3)
    
    # If significant minority is suspicious (30-50%), be more conservative
    elif fake_ratio > 0.3:
        return min(0.7, np.mean(predictions))
    
    # Otherwise use standard mean
    else:
        return np.mean(predictions)

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
models = []
try:
    models = load_models()
    print(f"Loaded {len(models)} models")
except Exception:
    tb = traceback.format_exc()
    log_path = os.path.join(tempfile.gettempdir(), 'startup_error.log')
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('Model loading failed. Traceback:\n')
            f.write(tb)
    except Exception as write_err:
        print(f"Failed writing startup error log: {write_err}")
    print(f"Model loading failed; continuing with 0 models. Trace written to {log_path}")
face_extractor, input_size, frames_per_video = initialize_preprocessing()

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
                
                # Clean up intermediate tensors
                del x, y_pred
                cleanup_memory()
                
                return np.mean(preds)
        
        return 0.5
    except Exception as e:
        print(f"Error predicting on image: {e}")
        cleanup_memory()
        return 0.5

@app.route('/')
def index():
    # If a static frontend exists (Next.js export), serve it; otherwise return JSON
    index_path = os.path.join(app.static_folder or '', 'index.html')
    if app.static_folder and os.path.exists(index_path):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({'message': 'Deepfake Detection Flask API is running.'})

# Serve other static assets (JS/CSS) when using exported Next.js
@app.route('/<path:path>')
def static_proxy(path):
    if app.static_folder:
        file_path = os.path.join(app.static_folder, path)
        if os.path.exists(file_path):
            return send_from_directory(app.static_folder, path)
    return jsonify({'error': 'Not found'}), 404

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
            'original_image': result['original_image'],
            'fake_frames': []  # Not applicable for images
        }
        
        # Clean up uploaded file and memory
        cleanup_file(filepath)
        cleanup_memory()
        
        return jsonify(response)
    except Exception as e:
        cleanup_file(filepath)
        cleanup_memory()
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
            strategy=smart_prediction_strategy,  # Use our improved strategy
            apply_compression=False,
            return_frame_predictions=True
        )
        
        prediction = result['prediction']
        label = 'fake' if prediction > 0.5 else 'real'
        
        # Calculate some statistics for better details
        fake_frames = result.get('fake_frames', [])
        total_frames = len(result.get('frame_predictions', []))
        fake_ratio = len(fake_frames) / max(total_frames, 1) * 100
        
        # Get gradcam images
        gradcam_data = result.get('gradcam_images', [])
        gradcam_image = None
        original_image = None
        
        if gradcam_data and len(gradcam_data) > 0:
            if isinstance(gradcam_data[0], dict):
                gradcam_image = gradcam_data[0].get('gradcam')
                original_image = gradcam_data[0].get('original')
            else:
                # Fallback for old format
                gradcam_image = gradcam_data[0]
        
        response = {
            'label': label,
            'confidence': float(prediction),
            'details': f'Video analyzed: {len(fake_frames)}/{total_frames} frames flagged as suspicious ({fake_ratio:.1f}%). Using smart prediction strategy.',
            'faces_detected': len(result.get('frame_predictions', [])),
            'processing_time': 'N/A',
            'fake_frames': fake_frames,
            'gradcam_image': gradcam_image,
            'original_image': original_image,
            'frame_predictions': result.get('frame_predictions', [])
        }
        
        # Clean up uploaded file and memory
        cleanup_file(filepath)
        cleanup_memory()
        
        return jsonify(response)
    except Exception as e:
        cleanup_file(filepath)
        cleanup_memory()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
