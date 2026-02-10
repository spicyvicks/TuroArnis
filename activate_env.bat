@echo off
REM Quick activation script for GCN environment
echo Activating GCN environment (Python 3.12)...
call venv\Scripts\activate.bat
echo.
echo ✅ Environment activated!
echo.
echo Python version:
python --version
echo.
echo To run GCN FPS test:
echo   python test_gcn_fps.py --video path/to/video.mp4 --viewpoint front
echo.
