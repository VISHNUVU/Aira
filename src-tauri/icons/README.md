# App icons

Tauri needs real icon files here before `tauri build`. Generate them from a
single source PNG (1024×1024 recommended) with:

```bash
npm run tauri icon path/to/source.png
# or:  cargo tauri icon path/to/source.png
```

This produces `32x32.png`, `128x128.png`, `128x128@2x.png`, `icon.icns` (macOS),
and `icon.ico` (Windows), which `tauri.conf.json` references.

Until then, dev mode (`tauri dev`) runs without custom icons.
