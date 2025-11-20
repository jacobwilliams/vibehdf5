# Building Standalone Executable

This guide explains how to build a standalone executable of vibehdf5 that can be distributed and run without requiring Python to be installed.

## Prerequisites

Install PyInstaller:

```bash
pip install pyinstaller
```

Or using pixi:

```bash
pixi add pyinstaller
```

## Quick Build

### Using the Build Script (Recommended)

Make the script executable and run it:

```bash
chmod +x build_executable.sh
./build_executable.sh
```

### Manual Build

Run PyInstaller directly:

```bash
pyinstaller vibehdf5.spec
```

## Output Locations

### macOS
- **Location**: `dist/vibehdf5.app`
- **Type**: Application bundle
- **Run**: `open dist/vibehdf5.app`
- **Install**: `cp -r dist/vibehdf5.app /Applications/`

### Linux
- **Location**: `dist/vibehdf5`
- **Type**: Single executable file
- **Run**: `./dist/vibehdf5`
- **Install**: Copy to `/usr/local/bin/` or `~/.local/bin/`

### Windows
- **Location**: `dist/vibehdf5.exe`
- **Type**: Single executable file
- **Run**: Double-click or run from command line
- **Install**: Copy to desired location

## Distribution

### macOS
To distribute the macOS app:

1. **Create DMG (recommended)**:
   ```bash
   # Install create-dmg if needed
   brew install create-dmg

   # Create DMG
   create-dmg \
     --volname "vibehdf5" \
     --window-pos 200 120 \
     --window-size 800 400 \
     --icon-size 100 \
     --icon "vibehdf5.app" 200 190 \
     --hide-extension "vibehdf5.app" \
     --app-drop-link 600 185 \
     "vibehdf5-installer.dmg" \
     "dist/"
   ```

2. **Or create ZIP**:
   ```bash
   cd dist
   zip -r vibehdf5-macos.zip vibehdf5.app
   ```

### Linux
Create a tarball:

```bash
cd dist
tar -czf vibehdf5-linux.tar.gz vibehdf5
```

### Windows
Create a ZIP file:

```bash
cd dist
# Windows: Use built-in compression or 7-Zip
# Linux/macOS: zip vibehdf5-windows.zip vibehdf5.exe
```

## Customization

### Adding an Icon

1. Create or obtain an icon file:
   - macOS: `.icns` file
   - Windows: `.ico` file
   - Linux: `.png` file (usually 256x256 or 512x512)

2. Edit `vibehdf5.spec` and update the `icon` parameter:
   ```python
   icon='path/to/your/icon.icns',  # macOS
   icon='path/to/your/icon.ico',   # Windows
   ```

### Reducing File Size

The executable can be quite large (100-200 MB) due to included libraries. To reduce size:

1. **Remove unused modules**: Edit the `excludes` list in `vibehdf5.spec`
2. **Use UPX compression** (already enabled in spec file)
3. **Consider one-folder mode** instead of one-file for faster startup

### Debug Build

To create a build with console output for debugging:

1. Edit `vibehdf5.spec`
2. Change `console=False` to `console=True`
3. Rebuild

## Troubleshooting

### Missing Modules

If you get import errors when running the executable:

1. Add the missing module to `hiddenimports` in `vibehdf5.spec`
2. Rebuild

### Qt Plugin Errors

If you encounter Qt platform plugin errors:

```bash
# Add to hiddenimports in vibehdf5.spec
'PySide6.QtCore',
'PySide6.QtGui',
'PySide6.QtWidgets',
```

### Library Loading Errors

On Linux, if you get library loading errors:

```bash
# Install required system libraries
sudo apt-get install libxcb-xinerama0  # or similar
```

### Performance Issues

The first launch may be slow as the executable unpacks libraries. Subsequent launches are faster.

## Advanced Configuration

### Building for Different Python Versions

PyInstaller uses the Python version it's installed with. To target a specific version:

```bash
python3.10 -m PyInstaller vibehdf5.spec
```

### Cross-Platform Building

PyInstaller does **not** support cross-compilation. You must build on the target platform:
- Build macOS apps on macOS
- Build Windows executables on Windows
- Build Linux executables on Linux

### CI/CD Integration

For automated builds, see `.github/workflows/` for examples of building in GitHub Actions.

## File Size Expectations

Typical executable sizes:
- **macOS**: ~150-200 MB (app bundle)
- **Windows**: ~100-150 MB (single .exe)
- **Linux**: ~100-150 MB (single binary)

The large size includes:
- Python interpreter
- Qt libraries
- NumPy, Pandas, Matplotlib
- HDF5 libraries
- All dependencies

## Testing the Executable

1. **Test on clean system**: Run on a machine without Python installed
2. **Test file opening**: Try opening various HDF5 files
3. **Test all features**: CSV viewing, plotting, filtering, etc.
4. **Check error handling**: Ensure errors are displayed properly

## Support

For issues or questions:
- GitHub Issues: https://github.com/jacobwilliams/vibehdf5/issues
- Check PyInstaller docs: https://pyinstaller.org/en/stable/
