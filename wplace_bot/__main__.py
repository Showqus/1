"""Точка входа: python -m wplace_bot"""
from __future__ import annotations

import logging
import sys


def main() -> int:
    from .winapi import set_dpi_awareness

    dpi = set_dpi_awareness()  # до создания любых окон
    from .app_logging import app_paths, setup_logging

    paths = app_paths()
    log_path = setup_logging(paths)
    log = logging.getLogger("main")
    log.info("DPI awareness: %s", dpi)
    if "--selftest" in sys.argv:
        return selftest(log)
    try:
        import tkinter as tk

        from .config import load_config
        from .gui import App

        cfg = load_config(paths.config)
        root = tk.Tk()
        App(root, paths, cfg, log_path)
        root.mainloop()
        return 0
    except Exception as e:  # noqa: BLE001
        log.critical("Программа упала при запуске", exc_info=True)
        _fatal_box(f"Wplace Bot не смог запуститься:\n{e!r}\n\nЛог: {log_path}")
        return 1


def selftest(log: logging.Logger) -> int:
    """Проверка, что exe собран правильно (запускается в CI: WplaceBot.exe --selftest)."""
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            res = fn()
            log.info("SELFTEST %s: OK %s", name, "" if res is None else res)
        except Exception:  # noqa: BLE001
            ok = False
            log.exception("SELFTEST %s: FAIL", name)

    def palette():
        from .palette import COLORS, ColorMatcher

        m = ColorMatcher()
        assert all(m.classify_one(c.rgb) == c.id for c in COLORS)
        return len(COLORS)

    def image():
        from PIL import Image

        from .image_prep import prepare_target
        from .palette import FREE_IDS

        t = prepare_target(Image.new("RGBA", (8, 8), (237, 28, 36, 255)), 4, 4, FREE_IDS)
        assert (t == 7).all()

    def screen():
        from .screen import MssScreen

        s = MssScreen()
        rect = s.virtual_rect()
        img = s.grab((rect[0], rect[1], 16, 16))
        return f"virtual={rect} shot={img.shape}"

    def mouse():
        from .winapi import default_input

        inp = default_input()
        return f"cursor={inp.get_cursor()} idle={inp.idle_seconds():.1f}s"

    def tk_window():
        import tkinter as tk

        root = tk.Tk()
        root.update()
        root.destroy()

    def gui_import():
        from . import gui

        assert gui.App

    check("palette", palette)
    check("image", image)
    check("screen", screen)
    check("mouse", mouse)
    check("tk", tk_window)
    check("gui_import", gui_import)
    log.info("SELFTEST %s", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def _fatal_box(text: str) -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "Wplace Bot — ошибка", 0x10)
    else:
        print(text, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
