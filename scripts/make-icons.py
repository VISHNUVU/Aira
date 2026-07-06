#!/usr/bin/env python3
"""
make-icons.py — generate the macOS icon set from icon_source_1024.png.

Produces, into src-tauri/icons/:
    32x32.png  128x128.png  128x128@2x.png  icon.png (512)  icon.ico  icon.icns

Run standalone (`python3 scripts/make-icons.py`) or from build-macos.sh.
Requires Pillow:  pip install pillow
"""
import os
import sys

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "icon_source_1024.png")
OUT = os.path.join(ROOT, "src-tauri", "icons")


def main() -> int:
    if not os.path.isfile(SRC):
        sys.exit(f"source not found: {SRC}")
    os.makedirs(OUT, exist_ok=True)
    src = Image.open(SRC).convert("RGBA")
    if src.size != (1024, 1024):
        src = src.resize((1024, 1024), Image.LANCZOS)

    def resized(px: int) -> Image.Image:
        return src.resize((px, px), Image.LANCZOS)

    # Standard Tauri PNG set.
    png_sizes = {
        "32x32.png": 32,
        "128x128.png": 128,
        "128x128@2x.png": 256,
        "icon.png": 512,
    }
    for name, px in png_sizes.items():
        resized(px).save(os.path.join(OUT, name), format="PNG")
        print(f"  wrote {name} ({px}px)")

    # Windows .ico (multi-size, harmless to keep cross-platform).
    src.save(
        os.path.join(OUT, "icon.ico"),
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print("  wrote icon.ico")

    # macOS .icns — Pillow writes a valid multi-resolution ICNS directly,
    # which sidesteps iconutil's temp-dir requirements.
    src.save(os.path.join(OUT, "icon.icns"), format="ICNS")
    print("  wrote icon.icns")
    print("✓ icon set complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
