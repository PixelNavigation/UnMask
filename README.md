---
title: Unmask
emoji: 🕵️‍♂️
colorFrom: purple
colorTo: indigo
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
---


# UnMask - Enhanced Deepfake Detection

This project integrates multiple deepfake detection models with a modern web interface.

## 🚀 Quick Start

### Backend Setup
## Updated the model to tf_efficientnet_b7_ns_0_40
1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install opencv-python numpy flask flask-cors torch torchvision dlib
   ```

3. **Start the server:**
   
   **Option A: Using the batch file (Windows):**
   ```bash
   start_server.bat
   ```
   
   **Option B: Using Python directly:**
   ```bash
   python simple_server.py
   ```
   
   **Option C: Using the Flask API (if dependencies available):**
   ```bash
   python app.py
   ```

The backend will start on `http://localhost:8000` (simple server) or `http://localhost:5000` (Flask API).

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:3000`.

## 🎯 Features

### Detection Models
- **MesoNet (Meso4 & MesoInception4)**: Lightweight CNN architectures
- **Enhanced Xception**: Deep learning model optimized for deepfake detection
- **Ensemble Voting**: Combines multiple models for better accuracy

### Web Interface
- **Drag & Drop Upload**: Easy file upload interface
- **Real-time Processing**: Live feedback during analysis
- **Detailed Results**: Comprehensive detection metrics
- **Multi-model Analysis**: Shows individual model predictions

## 📊 API Endpoints

### Flask API (app.py)
- `GET /` - Server status and information
- `POST /api/detect` - Upload video for deepfake detection
- `GET /api/models` - Get information about loaded models

### Simple Server (simple_server.py)
- `GET /` - API status (JSON)
- `GET /detect` - Upload interface (HTML)
- `POST /upload` - Process uploaded video

## 🔧 Technical Details

### Supported Formats
- **Video**: MP4, AVI, MOV, MKV, WebM
- **Maximum Size**: 100MB
- **Processing**: Automatic face detection and analysis

### Models Integration
The project integrates the [HongguLiu/Deepfake-Detection](https://github.com/HongguLiu/Deepfake-Detection) repository, providing:

1. **MesoNet Models**: Specialized for deepfake detection
2. **Transfer Learning**: Pre-trained Xception networks
3. **Face Detection**: Automatic face extraction from videos
4. **Statistical Analysis**: Advanced frame-level analysis

### Architecture
```
frontend/          # Next.js React application
├── src/
│   ├── app/       # Main application pages
│   └── Components/ # Reusable UI components
│
backend/           # Python detection server
├── Deepfake-Detection/  # Cloned detection models
├── app.py         # Flask API server
├── simple_server.py     # Standalone HTTP server
└── requirements.txt     # Python dependencies
```

## 🛠️ Troubleshooting

### Backend Issues
1. **Python dependencies missing**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Port conflicts**: Change port in server files if needed

3. **Model loading errors**: Ensure the Deepfake-Detection repository is properly cloned

### Frontend Issues
1. **CORS errors**: Ensure backend server is running and accessible

2. **Upload failures**: Check file format and size limits

3. **Connection refused**: Verify backend server is running on correct port

## 📈 Performance

### Processing Speed
- **Face Detection**: ~1-2 seconds per video
- **Model Inference**: ~0.5-1 second per model
- **Total Analysis**: ~3-5 seconds average

### Accuracy
- **Ensemble Model**: Combines multiple approaches for better reliability
- **False Positive Rate**: Reduced through multi-model voting
- **Confidence Scoring**: Provides reliability metrics

## 🔮 Future Enhancements

1. **Additional Models**: Integration of more detection algorithms
2. **Batch Processing**: Multiple file uploads
3. **Real-time Streaming**: Live video analysis
4. **Model Training**: Custom model training interface
5. **Advanced Analytics**: Detailed forensic analysis

## 📝 Notes

- This is a research/educational tool - results should be verified
- Processing time depends on video length and system performance
- GPU acceleration available if PyTorch CUDA is installed
- Regular model updates recommended for best performance