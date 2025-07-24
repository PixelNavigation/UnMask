@echo off
echo Starting Enhanced Deepfake Detection Server...
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH!
    pause
    exit /b 1
)

echo.
echo Checking required packages...
python -c "import cv2, numpy, sys; print('OpenCV and NumPy are available')" 2>nul
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install opencv-python numpy
)

echo.
echo Starting server...
python simple_server.py

pause
