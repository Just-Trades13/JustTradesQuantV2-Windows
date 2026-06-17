@echo off
REM ============================================================
REM  Just Trades Quant V2 - Launcher (Windows)
REM  Double-click to start the app. Run setup.bat first if you
REM  have not installed the dependencies yet.
REM ============================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo Starting Just Trades Quant V2...
echo (A browser tab will open. Close this window to stop the app.)
streamlit run app.py

pause
