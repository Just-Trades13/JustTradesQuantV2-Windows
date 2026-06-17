# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Just Trades Quant V2 (Windows standalone .exe)
# Build with:  pyinstaller JustTradesQuant.spec   (must be run ON Windows)

from PyInstaller.utils.hooks import copy_metadata, collect_data_files, collect_submodules

# Streamlit resolves a lot at runtime via importlib.metadata + ships static
# assets — both must be force-collected or the frozen app fails to start.
datas = []
datas += copy_metadata("streamlit")
datas += copy_metadata("pandas")
datas += copy_metadata("numpy")
datas += copy_metadata("plotly")
datas += copy_metadata("scipy")
datas += copy_metadata("yfinance")
datas += collect_data_files("streamlit")
datas += collect_data_files("plotly")

# App's own files that must ride along inside the bundle.
datas += [
    ("app.py", "."),
    ("pine_interpreter.py", "."),
    ("pine_optimizer.py", "."),
    ("run_audit.py", "."),
    (".streamlit", ".streamlit"),
    ("data_cache", "data_cache"),
    ("strategies", "strategies"),
    ("saved_strategies", "saved_strategies"),
    ("just_trades_logo.png", "."),
    (".env", "."),
]

hiddenimports = []
hiddenimports += collect_submodules("streamlit")
hiddenimports += collect_submodules("plotly")
hiddenimports += [
    "pine_interpreter",
    "pine_optimizer",
    "run_audit",
    "google.generativeai",
    "anthropic",
]

block_cipher = None

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="JustTradesQuantV2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,          # keep a console window so users see startup/errors
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="JustTradesQuantV2",
)
