@echo off
title ARGUS Surveillance Intelligence Live Platform
echo ==============================================================
echo       ARGUS - Adaptive Relational Graph Understanding
echo      Edge-Viable Causal Graph-Transformer Surveillance
echo ==============================================================
echo.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    pause
    exit /b 1
)

set "CKPT=outputs\checkpoints_rlvd_full\best.pt"
if not exist "%CKPT%" (
    echo [ERROR] Checkpoint not found at %CKPT%
    echo Please train a model first or run web_demo.py with --checkpoint / --allow-random-model.
    pause
    exit /b 1
)

echo [*] Initializing ARGUS Platform at http://localhost:7860...
".venv\Scripts\python.exe" web_demo.py --port 7860 --host 127.0.0.1 --checkpoint "%CKPT%" --open-browser

if errorlevel 1 (
    echo.
    echo [ERROR] Web demo exited with an error code.
    pause
)
