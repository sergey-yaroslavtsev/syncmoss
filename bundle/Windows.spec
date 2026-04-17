# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# ── Hidden imports ──────────────────────────────────────────────
hiddenimports = collect_submodules('scipy')
hiddenimports += collect_submodules('PySide6')
# Ensure numba and llvmlite submodules are included in the frozen app
hiddenimports += collect_submodules('numba')
hiddenimports += collect_submodules('llvmlite')
scipy_data = collect_data_files('scipy')
# llvmlite carries native data/bitcode that should be collected
llvmlite_data = collect_data_files('llvmlite')

src_dir = os.path.abspath(os.path.join(SPECPATH, '..', 'src'))

# Only Python library data (scipy, etc.) goes through PyInstaller.
# Application resources (icons/, parameters/, themes, etc.) are copied
# next to the .exe by bundle.py after the build.
datas = scipy_data + llvmlite_data

block_cipher = None

a = Analysis(
    [os.path.join(src_dir, 'main.py')],
    pathex=[src_dir],
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
    excludes=[],
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
    icon=os.path.join(src_dir, 'icons', 'icon_r.ico'),
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
    icon=os.path.join(src_dir, 'icons', 'icon_r.ico'),
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
