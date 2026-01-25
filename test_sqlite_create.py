"""Quick test to see where the SQLite creation fails."""

import os
import tempfile

print("Starting test...")

try:
    print("Importing backend_factory...")
    from vibehdf5.backend_factory import create_backend
    print("✓ Import successful")

    # Create temp file path
    filepath = os.path.join(tempfile.gettempdir(), "test_error.db")
    print(f"Creating backend for: {filepath}")

    backend = create_backend(filepath, "sqlite")
    print(f"✓ Backend created: {type(backend)}")

    print("Calling backend.create()...")
    backend.create(filepath)
    print("✓ Create successful")

    print("Calling backend.close()...")
    backend.close()
    print("✓ Close successful")

    print("\n✅ All steps completed successfully!")

    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"✓ Cleaned up: {filepath}")

except Exception as e:
    import traceback
    print(f"\n❌ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
