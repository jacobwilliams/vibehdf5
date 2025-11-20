# Quick Start: Building vibehdf5 Executable

This guide gets you up and running with building a standalone executable in under 5 minutes.

## Prerequisites

- Python 3.10 or later installed
- PyInstaller installed (or will be installed automatically)
- You're on the platform you want to build for (macOS, Linux, or Windows)

## Build in 3 Steps

### 1. Navigate to the Project Directory

```bash
cd /path/to/vibehdf5
```

### 2. Run the Build Script

```bash
./build_executable.sh
```

The script will:
- Install PyInstaller if needed
- Clean any previous builds
- Run PyInstaller with the spec file
- Report success and show output location

### 3. Test the Executable

**macOS:**
```bash
open dist/vibehdf5.app
```

**Linux:**
```bash
./dist/vibehdf5
```

**Windows:**
```cmd
dist\vibehdf5.exe
```

## That's It!

Your standalone executable is ready to distribute. Users can run it without having Python installed.

## Quick Distribution

**macOS - Create ZIP:**
```bash
cd dist
zip -r vibehdf5-macos.zip vibehdf5.app
```

**Linux - Create tarball:**
```bash
cd dist
tar -czf vibehdf5-linux.tar.gz vibehdf5
```

**Windows - Create ZIP:**
Use Windows Explorer or:
```cmd
powershell Compress-Archive dist\vibehdf5.exe vibehdf5-windows.zip
```

## Common Issues

### "pyinstaller: command not found"
```bash
pip install pyinstaller
```

### "Permission denied: ./build_executable.sh"
```bash
chmod +x build_executable.sh
```

### Build fails with import errors
Check that all dependencies are installed:
```bash
pip install -e .
```

### Executable won't run
Try building with console enabled for debugging:
1. Edit `vibehdf5.spec`
2. Change `console=False` to `console=True`
3. Rebuild: `./build_executable.sh`

## Next Steps

For more advanced options (custom icons, size optimization, etc.), see [BUILD_EXECUTABLE.md](BUILD_EXECUTABLE.md).

## File Sizes

Expect executables around:
- **macOS**: ~150-200 MB
- **Linux**: ~100-150 MB
- **Windows**: ~100-150 MB

This includes Python, Qt, NumPy, Pandas, Matplotlib, and all dependencies.

## Supported Platforms

- macOS 10.13+ (High Sierra or later)
- Linux (most distributions with glibc 2.17+)
- Windows 10 or later

Build on each platform separately - PyInstaller doesn't cross-compile.
