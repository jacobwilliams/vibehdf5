# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for building vibehdf5 standalone executable.

Usage:
    pyinstaller vibehdf5.spec

This will create a bundled application in the dist/ directory.
"""

import sys
import os
import sysconfig
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# On macOS, we need to explicitly include the Python library
binaries_extra = []
if sys.platform == 'darwin':
    # Get Python library location
    python_lib_dir = sysconfig.get_config_var('LIBDIR')
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Find the Python dylib
    python_dylib = None
    if python_lib_dir and os.path.exists(python_lib_dir):
        # Try different naming conventions
        for name in [
            f'libpython{python_version}.dylib',
            f'libpython{python_version}t.dylib',  # Free-threaded build
            f'libpython{sys.version_info.major}.dylib',
        ]:
            full_path = os.path.join(python_lib_dir, name)
            if os.path.exists(full_path):
                python_dylib = full_path
                print(f"Found Python library: {python_dylib}")
                binaries_extra.append((python_dylib, '.'))
                break

    if not python_dylib:
        print(f"Warning: Could not find Python dylib in {python_lib_dir}")
        print(f"Looking for: libpython{python_version}.dylib")

# Collect all necessary data files
datas = []

# Collect matplotlib data files (needed for fonts, styles, etc.)
datas += collect_data_files('matplotlib', include_py_files=False)

# Collect h5py data files if any
datas += collect_data_files('h5py', include_py_files=False)

# Collect all hidden imports needed by the application
hiddenimports = []

# Qt backend modules
hiddenimports += collect_submodules('PySide6')
hiddenimports += collect_submodules('qtpy')

# Matplotlib backends
hiddenimports += [
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_qt',
    'matplotlib.backends.backend_agg',
]

# Pandas dependencies
hiddenimports += [
    'pandas',
    'pandas._libs',
    'pandas._libs.tslibs',
]

# NumPy dependencies
hiddenimports += [
    'numpy.core._multiarray_umath',
    'numpy.core._multiarray_tests',
]

# h5py dependencies
hiddenimports += [
    'h5py.defs',
    'h5py.utils',
    'h5py.h5ac',
    'h5py._proxy',
]

a = Analysis(
    ['run_vibehdf5.py'],  # Use dedicated entry point instead of __main__.py
    pathex=[],
    binaries=binaries_extra,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_macos.py'] if sys.platform == 'darwin' else [],
    excludes=[
        'tkinter',  # Only exclude tkinter since we're using Qt
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# For macOS app bundle
if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='vibehdf5',
        debug=False,  # Disable debug for production
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,  # Disable UPX compression (can cause issues)
        console=False,  # No console window for GUI app
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=None,  # Add icon path here if you have one
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,  # Disable UPX
        upx_exclude=[],
        name='vibehdf5',
    )

    app = BUNDLE(
        coll,
        name='vibehdf5.app',
        icon='media/icon.icns',  # macOS icon file
        bundle_identifier='com.github.jacobwilliams.vibehdf5',
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': '1.0.2',
            'CFBundleVersion': '1.0.2',
            'CFBundleName': 'vibehdf5',
            'CFBundleDisplayName': 'vibehdf5 - HDF5 Viewer',
            'CFBundleIdentifier': 'com.github.jacobwilliams.vibehdf5',
            'NSHumanReadableCopyright': 'Copyright © 2025 Jacob Williams',
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'HDF5 File',
                    'CFBundleTypeRole': 'Viewer',
                    'LSItemContentTypes': ['org.hdfgroup.hdf5'],
                    'LSHandlerRank': 'Default',
                }
            ],
        },
    )

# For Windows and Linux single-file executable
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='vibehdf5',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,  # No console window
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=None,  # Add icon path here if you have one
    )
