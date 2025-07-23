"""
UnMask Backend API - Enhanced with Bi-LSTM Temporal Analysis
Using Xception models, OpenCV face detection, and Enhanced Temporal Bi-LSTM
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import tempfile
import logging
from pathlib import Path

# Import DeepfakeDetector class
from DeepfakeDetector import DeepfakeDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize detector
detector = DeepfakeDetector()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models": "xception_only",
        "face_detector": "opencv",
        "device": str(detector.device),
        "enhanced_features": [
            "Rich intermediate features",
            "Monte Carlo uncertainty estimation", 
            "Temporal smoothing with EMA",
            "Multi-head attention",
            "Enhanced video analysis"
        ],
        "temporal_ready": True
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
                "detection_result": result,
                "analysis_type": "basic"
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze_video_temporal', methods=['POST'])
def analyze_video_temporal():
    """
    Enhanced video analysis endpoint with temporal processing
    """
    try:
        # Check if video file is present
        if 'file' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['file']
        if video_file.filename == '':
            return jsonify({"error": "No video file selected"}), 400
        
        # Check file extension
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        file_ext = Path(video_file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return jsonify({"error": f"Unsupported file format. Allowed: {allowed_extensions}"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            video_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Perform enhanced temporal analysis
            result = detector.predict_video_temporal_enhanced(temp_path)
            
            # Add metadata
            result["analysis_type"] = "enhanced_temporal"
            result["filename"] = video_file.filename
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Video temporal analysis error: {e}")
        return jsonify({"error": f"Video analysis failed: {str(e)}"}), 500

@app.route('/api/analyze_frame_temporal', methods=['POST'])
def analyze_frame_temporal():
    """
    Real-time frame analysis with temporal context
    """
    try:
        # Check if image file is present
        if 'file' not in request.files:
            return jsonify({"error": "No frame image provided"}), 400
        
        frame_file = request.files['file']
        if frame_file.filename == '':
            return jsonify({"error": "No frame selected"}), 400
        
        # Read image data
        image_data = frame_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Perform enhanced real-time analysis
        result = detector.predict_realtime_frame_enhanced(frame)
        
        # Add metadata
        result["analysis_type"] = "enhanced_realtime"
        result["filename"] = frame_file.filename
        result["timestamp"] = np.datetime64('now').astype(str)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Frame temporal analysis error: {e}")
        return jsonify({"error": f"Frame analysis failed: {str(e)}"}), 500

@app.route('/api/reset_temporal_history', methods=['POST'])
def reset_temporal_history():
    """Reset the temporal frame history for real-time analysis"""
    try:
        detector.reset_temporal_history()
        return jsonify({
            "status": "success",
            "message": "Temporal history reset successfully"
        })
    except Exception as e:
        logger.error(f"Reset temporal history error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_model_info', methods=['GET'])
def get_model_info():
    """Get information about the enhanced temporal model"""
    return jsonify({
        "model_type": "Enhanced Temporal Bi-LSTM + Xception",
        "base_models": ["Xception", "WSDAN+Xception"],
        "face_detector": "OpenCV Haar Cascade",
        "device": str(detector.device),
        "features": {
            "rich_features": "512-dimensional intermediate embeddings",
            "uncertainty_modeling": "Monte Carlo Dropout",
            "temporal_smoothing": "Exponential Moving Average",
            "attention": "Multi-head attention mechanism",
            "sequence_length": detector.sequence_length,
            "no_retraining": "Uses existing trained models"
        },
        "improvements": [
            "✅ Rich intermediate features (not just final scores)",
            "✅ Dropout-based uncertainty modeling", 
            "✅ Temporal smoothing with EMA",
            "✅ Multi-head attention for better focus",
            "✅ Enhanced reliability scoring"
        ],
        "endpoints": {
            "/api/upload": "Basic single file analysis",
            "/api/analyze_video_temporal": "Enhanced video temporal analysis",
            "/api/analyze_frame_temporal": "Real-time frame analysis with history",
            "/api/reset_temporal_history": "Reset frame history",
            "/api/health": "Health check",
            "/api/get_model_info": "This endpoint"
        }
    })

if __name__ == '__main__':
    logger.info("🚀 Starting Enhanced UnMask Backend")
    logger.info("✅ Base Models: Xception + WSDAN+Xception + OpenCV")
    logger.info("✅ Enhanced Features: Temporal Bi-LSTM with Rich Features")
    logger.info("✅ Uncertainty Modeling: Monte Carlo Dropout")
    logger.info("✅ Temporal Smoothing: Exponential Moving Average")
    logger.info("✅ Multi-head Attention: 4 attention heads")
    logger.info("📡 Available Endpoints:")
    logger.info("   • POST /api/upload - Basic single file analysis")
    logger.info("   • POST /api/analyze_video_temporal - Enhanced video analysis")
    logger.info("   • POST /api/analyze_frame_temporal - Real-time with history")
    logger.info("   • POST /api/reset_temporal_history - Reset temporal context")
    logger.info("   • GET /api/health - Health check")
    logger.info("   • GET /api/get_model_info - Model information")
    logger.info("🌐 Server starting on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
