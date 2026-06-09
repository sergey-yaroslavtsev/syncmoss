# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# ── Hidden imports ──────────────────────────────────────────────
# numba/llvmlite/scipy do a lot of dynamic importing, so collect their submodules
# (this keeps core modules like scipy._cyutility that the hooks miss). The big
# unused scipy subpackages are pruned again via scipy_excludes below; numba's own
# test-suite is filtered out (not needed at runtime, and it emits a wall of
# "hidden import not found" warnings for its CUDA tests).
hiddenimports = collect_submodules('numba', filter=lambda name: 'tests' not in name)
hiddenimports += collect_submodules('llvmlite')
hiddenimports += collect_submodules('scipy')
# Only the Qt modules SYNCmoss actually uses. We deliberately do NOT
# collect_submodules('PySide6'): that drags in the QtQuick / QtQml / Qt3D /
# WebEngine frameworks (100+ MB) the app never uses. The PySide6 PyInstaller
# hook bundles the Qt frameworks/plugins matching just these imports.
hiddenimports += ['PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets']
# GUI smoke driver lives in tests/syncmoss_test.py; bundling it lets the frozen
# binary self-test via `SYNCmoss --test` (see syncmoss/main.py).
hiddenimports += ['syncmoss_test']

# scipy is only needed for scipy.linalg.eig (the import-time "dummy" in
# models.py); scipy_excludes below drops the big unused subpackages.
# llvmlite carries native data/bitcode that must be collected.
llvmlite_data = collect_data_files('llvmlite')

syncmoss_dir = os.path.abspath(os.path.join(SPECPATH, '..', 'syncmoss'))
project_dir  = os.path.abspath(os.path.join(SPECPATH, '..'))
tests_dir    = os.path.abspath(os.path.join(SPECPATH, '..', 'tests'))

# Application resources (icons/, parameters/, themes, etc.) are copied next to
# the .exe by bundle.py after the build.
datas = llvmlite_data

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
# scipy is only used for scipy.linalg.eig; exclude the big unused subpackages.
# (linalg, _lib, special and sparse are kept: scipy.linalg's import chain may
# touch special/sparse.)
scipy_excludes = [
    'scipy.stats', 'scipy.optimize', 'scipy.spatial', 'scipy.interpolate',
    'scipy.integrate', 'scipy.fft', 'scipy.fftpack', 'scipy.ndimage',
    'scipy.signal', 'scipy.io', 'scipy.cluster', 'scipy.odr',
    'scipy.datasets', 'scipy.misc',
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
            # Only the backends SYNCmoss uses (Qt for the GUI, Agg/SVG/PDF for
            # saving). 'all' pulls every backend incl. the tk one.
            'backends': ['Agg', 'QtAgg', 'SVG', 'PDF'],
        },
    },
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=qt_excludes + scipy_excludes + other_excludes,
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
