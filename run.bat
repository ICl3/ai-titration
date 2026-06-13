@echo off
title AI Titration System

echo ============================================
echo    AI Titration Control System - Launcher
echo ============================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

cd /d "%~dp0"

python -c "import flask, flask_socketio, cv2, torch, serial, scipy, numpy, PIL" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Missing packages detected. Installing...
    pip install flask flask-socketio opencv-python pyserial scipy numpy Pillow torch torchvision
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install. Check your network and try again.
        pause
        exit /b 1
    )
    echo Done.
)

echo.
echo Starting server...
echo Browser will open at http://localhost:5000
echo Press Ctrl+C to stop
echo ============================================
echo.

python Auto_Ctrl\web_app.py

pause
