"""Снимки экрана и чтение цветов клеток."""
from __future__ import annotations

import logging
import threading

import numpy as np

from .geometry import Grid, Rect

log = logging.getLogger(__name__)


class ScreenSource:
    def grab(self, rect: Rect) -> np.ndarray:
        """Снимок прямоугольника экрана в виде массива (h, w, 3) uint8 RGB."""
        raise NotImplementedError

    def virtual_rect(self) -> Rect:
        """Прямоугольник всего виртуального экрана (все мониторы)."""
        raise NotImplementedError


class MssScreen(ScreenSource):
    """Снимки через библиотеку mss. Экземпляр mss свой для каждого потока."""

    def __init__(self):
        self._local = threading.local()

    def _sct(self):
        sct = getattr(self._local, "sct", None)
        if sct is None:
            import mss

            factory = getattr(mss, "MSS", None) or mss.mss  # в mss 10 mss.mss() устарел
            sct = factory()
            self._local.sct = sct
        return sct

    def grab(self, rect: Rect) -> np.ndarray:
        left, top, width, height = (int(v) for v in rect)
        if width <= 0 or height <= 0:
            raise ValueError(f"Пустая область снимка: {rect}")
        shot = self._sct().grab({"left": left, "top": top, "width": width, "height": height})
        arr = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return arr[:, :, 2::-1].copy()

    def virtual_rect(self) -> Rect:
        mon = self._sct().monitors[0]
        return mon["left"], mon["top"], mon["width"], mon["height"]

    def monitors(self) -> list[dict]:
        return list(self._sct().monitors)


def sample_radius(pitch: float) -> int:
    """Сколько экранных пикселей вокруг центра клетки усреднять."""
    return max(0, min(3, int(pitch * 0.2)))


def sample_cells(img: np.ndarray, img_rect: Rect, grid: Grid, width: int, height: int) -> np.ndarray:
    """Цвета клеток холста (height, width, 3) по снимку ``img`` области ``img_rect``.

    Берётся медиана квадрата вокруг центра клетки — так мелкие обводки/курсор не мешают.
    """
    left, top = img_rect[0], img_rect[1]
    ih, iw = img.shape[:2]
    cx = np.rint(grid.x0 + np.arange(width) * grid.pitch - left).astype(np.int64)
    cy = np.rint(grid.y0 + np.arange(height) * grid.pitch - top).astype(np.int64)
    r = sample_radius(grid.pitch)
    d = np.arange(-r, r + 1)
    yy = np.clip(cy[:, None, None, None] + d[None, None, :, None], 0, ih - 1)
    xx = np.clip(cx[None, :, None, None] + d[None, None, None, :], 0, iw - 1)
    yy, xx = np.broadcast_arrays(yy, xx)
    patches = img[yy, xx]  # (H, W, k, k, 3)
    if r == 0:
        return patches[:, :, 0, 0, :].astype(np.uint8)
    return np.median(patches.reshape(height, width, -1, 3), axis=2).astype(np.uint8)


def sample_point(img: np.ndarray, img_rect: Rect, x: float, y: float, radius: int) -> np.ndarray:
    left, top = img_rect[0], img_rect[1]
    ih, iw = img.shape[:2]
    cx, cy = int(round(x)) - left, int(round(y)) - top
    y0, y1 = max(0, cy - radius), min(ih, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(iw, cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return np.array([0, 0, 0], dtype=np.uint8)
    return np.median(img[y0:y1, x0:x1].reshape(-1, 3), axis=0).astype(np.uint8)
