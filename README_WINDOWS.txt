Just Trades Quant V2 — Windows Quick Start
============================================

REQUIREMENTS
  - Windows 10 or 11
  - Python 3.10 or newer  (https://www.python.org/downloads/)
    During install, CHECK the box "Add python.exe to PATH".

FIRST TIME
  1. Copy this whole folder onto the Windows machine.
  2. Double-click  setup.bat   (creates the environment + installs everything).
     Wait for "Setup complete."

EVERY TIME AFTER
  3. Double-click  run.bat   to start the app.
     A browser tab opens automatically. Close the black window to stop it.

API KEY
  - The AI Analyst feature needs an Anthropic API key.
  - It lives in the file named ".env"  (ANTHROPIC_API_KEY=...).
  - Open .env in Notepad to change it if needed.

TROUBLESHOOTING
  - "Python was not found" -> reinstall Python with "Add to PATH" checked,
    then re-run setup.bat.
  - If run.bat says the environment is missing, run setup.bat first.
