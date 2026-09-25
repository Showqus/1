"""Логи, перехват ошибок и сборка отчёта для разработчика."""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass

from . import __version__
from .winapi import system_info

log = logging.getLogger(__name__)

KEEP_LOGS = 15


@dataclass
class Paths:
    base: str
    logs: str
    screens: str
    config: str


def _writable(d: str) -> bool:
    try:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".write_test")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def app_paths() -> Paths:
    """Папка рядом с exe; если туда нельзя писать — %LOCALAPPDATA%\\WplaceBot."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not _writable(base):
        root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        base = os.path.join(root, "WplaceBot")
        os.makedirs(base, exist_ok=True)
    logs = os.path.join(base, "logs")
    screens = os.path.join(logs, "screens")
    os.makedirs(screens, exist_ok=True)
    return Paths(base=base, logs=logs, screens=screens, config=os.path.join(base, "config.json"))


def setup_logging(paths: Paths) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(paths.logs, f"wplace_bot_{ts}.log")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(threadName)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    root.addHandler(fh)
    if sys.stderr is not None and not getattr(sys, "frozen", False):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        root.addHandler(sh)
    for noisy in ("PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.INFO)

    for old in sorted(glob.glob(os.path.join(paths.logs, "wplace_bot_*.log")))[:-KEEP_LOGS]:
        try:
            os.remove(old)
        except OSError:
            pass

    def excepthook(exc_type, exc, tb):
        logging.getLogger("crash").critical(
            "Необработанная ошибка:\n%s", "".join(traceback.format_exception(exc_type, exc, tb))
        )

    def thread_excepthook(args):
        logging.getLogger("crash").critical(
            "Необработанная ошибка в потоке %s:\n%s",
            args.thread.name if args.thread else "?",
            "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
        )

    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook

    log.info("=" * 70)
    log.info("Wplace Bot %s, запуск %s", __version__, time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Исполняемый файл: %s (frozen=%s)", sys.executable, getattr(sys, "frozen", False))
    log.info("Папка программы: %s", paths.base)
    log.info("Система: %s", system_info())
    return log_path


def make_report(paths: Paths, note: str = "") -> str:
    """Собрать zip с логами, настройками и отладочными снимками."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(paths.base, f"report_{ts}.zip")
    logs = sorted(glob.glob(os.path.join(paths.logs, "wplace_bot_*.log")))[-5:]
    shots = sorted(glob.glob(os.path.join(paths.screens, "*.png")))[-25:]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in logs:
            z.write(p, os.path.join("logs", os.path.basename(p)))
        for p in shots:
            z.write(p, os.path.join("screens", os.path.basename(p)))
        if os.path.exists(paths.config):
            z.write(paths.config, "config.json")
        info = {"version": __version__, "created": ts, "system": system_info(), "note": note}
        z.writestr("info.json", json.dumps(info, ensure_ascii=False, indent=2, default=str))
    log.info("Отчёт сохранён: %s (логов %d, снимков %d)", out, len(logs), len(shots))
    return out
