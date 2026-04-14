import os
from utils.config import SUPPORTED_EXTENSIONS


def scan_images(directory: str) -> list[str]:
    """Return sorted list of supported image filenames in directory."""
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )


def browse_directory() -> str:
    """Open native directory picker dialog. Returns selected path or empty string."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        path = filedialog.askdirectory()
        root.destroy()
        return path.replace('\\', '/') if path else ''
    except Exception:
        return ''
