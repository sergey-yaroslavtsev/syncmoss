# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# ── Hidden imports ──────────────────────────────────────────────
hiddenimports = collect_submodules('scipy')
hiddenimports += collect_submodules('PySide6')
# Ensure numba and llvmlite submodules are included in the frozen app
hiddenimports += collect_submodules('numba')
hiddenimports += collect_submodules('llvmlite')
# GUI smoke driver lives in tests/syncmoss_test.py; bundling it lets the frozen
# binary self-test via `SYNCmoss --test` (see syncmoss/main.py).
hiddenimports += ['syncmoss_test']
scipy_data = collect_data_files('scipy')
# llvmlite carries native data/bitcode that should be collected
llvmlite_data = collect_data_files('llvmlite')

syncmoss_dir = os.path.abspath(os.path.join(SPECPATH, '..', 'syncmoss'))
project_dir  = os.path.abspath(os.path.join(SPECPATH, '..'))
tests_dir    = os.path.abspath(os.path.join(SPECPATH, '..', 'tests'))

# Only Python library data (scipy, etc.) goes through PyInstaller.
# Application resources (icons/, parameters/, themes, etc.) are copied
# next to the .exe by bundle.py after the build.
datas = scipy_data + llvmlite_data

# SYNCmoss only uses PySide6.QtWidgets / QtGui / QtCore. Excluding the large,
# unused Qt frameworks keeps the bundle small; excludes win over the
# collect_submodules('PySide6') above. The CI "--test" smoke run guards against
# accidentally excluding something the app actually needs.
qt_excludes = [
    'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineQuick',
    'PySide6.QtWebChannel', 'PySide6.QtWebSockets', 'PySide6.QtWebView',
    'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.QtQuick3D', 'PySide6.QtQuickWidgets',
    'PySide6.QtQuickControls2',
    'PySide6.Qt3DCore', 'PySide6.Qt3DRender', 'PySide6.Qt3DInput', 'PySide6.Qt3DAnimation',
    'PySide6.Qt3DLogic', 'PySide6.Qt3DExtras',
    'PySide6.QtCharts', 'PySide6.QtDataVisualization', 'PySide6.QtGraphs',
    'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets', 'PySide6.QtSpatialAudio',
    'PySide6.QtPdf', 'PySide6.QtPdfWidgets',
    'PySide6.QtBluetooth', 'PySide6.QtNfc', 'PySide6.QtPositioning', 'PySide6.QtSensors',
    'PySide6.QtSerialPort', 'PySide6.QtSerialBus',
    'PySide6.QtTest', 'PySide6.QtSql', 'PySide6.QtDesigner', 'PySide6.QtHelp',
    'PySide6.QtUiTools', 'PySide6.QtNetworkAuth', 'PySide6.QtRemoteObjects',
    'PySide6.QtTextToSpeech', 'PySide6.QtScxml', 'PySide6.QtStateMachine',
]
# pytest is only used by the source test-suite; it must not be pulled into the
# frozen binary (the smoke driver imports it lazily / guarded).
other_excludes = ['tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'pytest', '_pytest', 'pluggy']

block_cipher = None

a = Analysis(
    [os.path.join(syncmoss_dir, 'main.py')],
    pathex=[project_dir, tests_dir],
    hooksconfig={
        'matplotlib': {
            'backends': 'all',
        },
    },
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=qt_excludes + other_excludes,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── GUI exe (no console window) ────────────────────────────────
exe_gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SYNCmoss',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=os.path.join(syncmoss_dir, 'icons', 'icon_r.ico'),
)

# ── Console/debug exe ──────────────────────────────────────────
exe_console = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SYNCmoss_console',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=os.path.join(syncmoss_dir, 'icons', 'icon_r.ico'),
)

coll = COLLECT(
    exe_gui,
    exe_console,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SYNCmoss',
)
