#!/usr/bin/env python3
"""
Generate a simple icon for vibehdf5.
Creates both .icns (macOS) and .ico (Windows) formats.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon():
    """Create a simple HDF5-themed icon."""
    # Create base image (1024x1024 for high res)
    size = 1024
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background gradient (blue theme for data/files)
    # for y in range(size):
    #     color_value = int(50 + (y / size) * 100)
    #     draw.rectangle([(0, y), (size, y+1)],
    #                   fill=(30, color_value, 150, 255))

    # Draw a stylized folder/file icon
    margin = size // 8
    folder_top = margin
    folder_left = margin
    folder_right = size - margin
    folder_bottom = size - margin

    # Folder shape
    draw.rounded_rectangle(
        [(folder_left, folder_top + 50), (folder_right, folder_bottom)],
        radius=40,
        fill=(70, 140, 200, 255),
        outline=(255, 255, 255, 200),
        width=8
    )

    # Tab
    tab_width = size // 3
    draw.rounded_rectangle(
        [(folder_left, folder_top), (folder_left + tab_width, folder_top + 80)],
        radius=20,
        fill=(70, 140, 200, 255),
        outline=(255, 255, 255, 200),
        width=8
    )

    # HDF5 text
    try:
        font_size = size // 6
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()

    text = "VibeHdf5"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (size - text_width) // 2
    text_y = (size - text_height) // 2 + 50

    # Text shadow
    draw.text((text_x + 4, text_y + 4), text, font=font, fill=(0, 0, 0, 150))
    # Text
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))

    # Save as PNG first
    png_path = 'icon.png'
    img.save(png_path, 'PNG')
    print(f"Created {png_path}")

    # Create .icns for macOS
    try:
        # Save multiple sizes for .icns
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        icon_dir = 'icon.iconset'
        os.makedirs(icon_dir, exist_ok=True)

        for s in sizes:
            resized = img.resize((s, s), Image.Resampling.LANCZOS)
            resized.save(f'{icon_dir}/icon_{s}x{s}.png')
            if s <= 512:  # Also create @2x versions
                resized_2x = img.resize((s*2, s*2), Image.Resampling.LANCZOS)
                resized_2x.save(f'{icon_dir}/icon_{s}x{s}@2x.png')

        # Convert to .icns using iconutil (macOS only)
        os.system(f'iconutil -c icns {icon_dir} -o vibehdf5.icns')
        print("Created vibehdf5.icns (macOS)")

        # Clean up iconset directory
        import shutil
        shutil.rmtree(icon_dir)

    except Exception as e:
        print(f"Could not create .icns: {e}")

    # Create .ico for Windows
    try:
        sizes_ico = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        ico_images = [img.resize(s, Image.Resampling.LANCZOS) for s in sizes_ico]
        ico_images[0].save('vibehdf5.ico', format='ICO', sizes=sizes_ico)
        print("Created vibehdf5.ico (Windows)")
    except Exception as e:
        print(f"Could not create .ico: {e}")

if __name__ == '__main__':
    try:
        create_icon()
        print("\nIcon files created successfully!")
        print("Add to spec file: icon='vibehdf5.icns' (macOS) or icon='vibehdf5.ico' (Windows)")
    except ImportError:
        print("Error: Pillow library required. Install with: pip install Pillow")
