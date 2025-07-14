@echo off
echo 🤖 AI Document Assistant - Windows Launcher
echo ==================================================

REM Set Python path
set PYTHONPATH=%cd%

REM Start backend server
echo.
echo 🚀 Starting Backend Server...
start cmd /k "python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000"

REM Wait a moment
timeout /t 3 /nobreak > nul

REM Start frontend
echo.
echo 🌐 Starting Frontend Interface...
python interface/ui.py

pause