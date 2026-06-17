"""
Standalone launcher for Just Trades Quant V2.
PyInstaller bundles this as the .exe entry point. It boots the Streamlit
server in-process and opens the default browser to the app.
"""
import os
import sys
import time
import threading
import webbrowser


def resolve_path(rel):
    """Resolve a path inside the PyInstaller bundle (or source tree in dev)."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)


def _open_browser():
    # Give the server a moment to bind the port, then open the tab.
    time.sleep(4)
    webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
    # Streamlit reads its target + flags from argv, same as the CLI.
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("app.py"),
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]

    threading.Thread(target=_open_browser, daemon=True).start()

    import streamlit.web.cli as stcli
    sys.exit(stcli.main())
