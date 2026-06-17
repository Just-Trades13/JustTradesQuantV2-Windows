@echo off
REM ============================================================
REM  Just Trades Quant V2 - First-time setup (Windows)
REM  Creates a virtual environment and installs dependencies.
REM  Run this ONCE before using run.bat.
REM ============================================================

cd /d "%~dp0"

echo.
echo Checking for Python...
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found on this machine.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo and make sure "Add python.exe to PATH" is checked.
    pause
    exit /b 1
)

echo Creating virtual environment (.venv)...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create the virtual environment.
    pause
    exit /b 1
)

echo Installing dependencies (this can take a few minutes)...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete. Double-click run.bat to start the app.
echo ============================================================
pause
