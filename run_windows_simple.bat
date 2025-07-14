@echo off
echo 🤖 AI Document Assistant - Simple Windows Version
echo ===================================================

REM Check if port 8000 is in use
netstat -an | find "8000" > nul
if %ERRORLEVEL% == 0 (
    echo ⚠️  Port 8000 is already in use. Please stop other instances first.
    pause
    exit /b 1
)

REM Set Python path
set PYTHONPATH=%cd%

REM Start backend server
echo.
echo 🚀 Starting Backend Server...
start "Backend" cmd /k "python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000"

REM Wait a moment
echo ⏱️  Waiting for backend to start...
timeout /t 5 /nobreak > nul

REM Start Streamlit frontend
echo.
echo 🌐 Starting Streamlit Interface...
echo 📱 The interface will open at: http://localhost:8501
echo.
python -m streamlit run interface/simple_ui.py --server.port 8501 --server.address localhost

pause