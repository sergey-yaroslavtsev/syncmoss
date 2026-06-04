# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# Hidden imports
hiddenimports = collect_submodules('scipy')
hiddenimports += collect_submodules('PySide6')
hiddenimports += collect_submodules('numba')
hiddenimports += collect_submodules('llvmlite')

scipy_data = collect_data_files('scipy')
llvmlite_data = collect_data_files('llvmlite')

syncmoss_dir = os.path.abspath(os.path.join(SPECPATH, '..', 'syncmoss'))

datas = scipy_data + llvmlite_data

block_cipher = None

a = Analysis(
    [os.path.join(syncmoss_dir, 'main.py')],
    pathex=[syncmoss_dir],
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

exe_gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SYNCmoss',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe_gui,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SYNCmoss',
)

app = BUNDLE(
    coll,
    name='SYNCmoss.app',
    icon=None,
    bundle_identifier=None,
)
