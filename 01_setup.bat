@echo off
echo ============================================
echo  VLM Hazard Dataset - Environment Setup
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found

REM Install pip packages
echo.
echo Installing required packages...
pip install yt-dlp opencv-python pandas tqdm Pillow requests

REM Check ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] ffmpeg not found!
    echo Download from: https://www.gyan.dev/ffmpeg/builds/
    echo Extract and add to PATH, then re-run this script.
    echo.
    echo Quick install via winget:
    echo   winget install ffmpeg
) else (
    echo [OK] ffmpeg found
)

echo.
echo Creating folder structure...
python -c "
import os
folders = [
    'phase1_opensource/PPE_violation/raw',
    'phase1_opensource/PPE_violation/frames',
    'phase1_opensource/fall_from_height/raw',
    'phase1_opensource/fall_from_height/frames',
    'phase1_opensource/machinery_danger/raw',
    'phase1_opensource/machinery_danger/frames',
    'phase1_opensource/near_miss/raw',
    'phase1_opensource/near_miss/frames',
    'phase2_industrial/PPE_violation/raw',
    'phase2_industrial/PPE_violation/frames',
    'phase2_industrial/fall_from_height/raw',
    'phase2_industrial/fall_from_height/frames',
    'phase2_industrial/machinery_danger/raw',
    'phase2_industrial/machinery_danger/frames',
    'phase2_industrial/near_miss/raw',
    'phase2_industrial/near_miss/frames',
    'your_mkv_files/raw',
    'your_mkv_files/frames',
]
for f in folders:
    os.makedirs(f, exist_ok=True)
    print(f'  Created: {f}')
print('Done.')
"

echo.
echo ============================================
echo  Setup complete!
echo  Next: Run 02_download_youtube.py
echo ============================================
pause
