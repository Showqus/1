"""Поиск кнопок цветов (свотчей) палитры wplace на снимке экрана."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from .geometry import Rect
from .palette import BY_ID, COLORS

log = logging.getLogger(__name__)


@dataclass
class Swatch:
    x: int  # центр на экране
    y: int
    w: int
    h: int

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    @classmethod
    def from_list(cls, v) -> "Swatch":
        x, y, w, h = (int(a) for a in v)
        return cls(x, y, w, h)


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Связные области маски: (площадь, x0, y0, x1, y1)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    out = []
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        seen[sy, sx] = True
        q = deque([(sy, sx)])
        area = 0
        x0 = x1 = sx
        y0 = y1 = sy
        while q:
            y, x = q.popleft()
            area += 1
            x0, x1 = min(x0, x), max(x1, x)
            y0, y1 = min(y0, y), max(y1, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append((area, x0, y0, x1, y1))
    return out


def label_image(img: np.ndarray, tolerance: float) -> np.ndarray:
    """Для каждого пикселя — id ближайшего цвета палитры или 0, если дальше допуска."""
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3).astype(np.int32)
    pal = np.array([c.rgb for c in COLORS], dtype=np.int32)
    ids = np.array([c.id for c in COLORS], dtype=np.int16)
    out = np.zeros(len(flat), dtype=np.int16)
    tol2 = tolerance * tolerance
    chunk = 50000
    for s in range(0, len(flat), chunk):
        part = flat[s : s + chunk]
        d = ((part[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
        k = np.argmin(d, axis=1)
        best = d[np.arange(len(part)), k]
        out[s : s + chunk] = np.where(best <= tol2, ids[k], 0)
    return out.reshape(h, w)


def find_swatches(
    img: np.ndarray, rect: Rect, tolerance: float = 6, min_side: int = 6
) -> dict[int, Swatch]:
    """Найти на снимке палитры кнопки всех цветов.

    Кнопка цвета — это почти квадратная заливка точным цветом палитры. Берётся
    область, похожая на кнопку (не слишком вытянутая и не огромная). Если цвет
    совпадает с фоном панели (обычно белый), он может не найтись — тогда его
    можно указать вручную.
    """
    left, top = rect[0], rect[1]
    labels = label_image(img, tolerance)
    candidates: dict[int, list[tuple[int, int, int, int, int]]] = {}
    for c in COLORS:
        mask = labels == c.id
        if mask.sum() < min_side * min_side // 2:
            continue
        comps = []
        for area, x0, y0, x1, y1 in _components(mask):
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            if bw < min_side or bh < min_side:
                continue
            if max(bw, bh) > 3 * min(bw, bh):
                continue
            if area < 0.35 * bw * bh:
                continue
            comps.append((area, x0, y0, x1, y1))
        if comps:
            candidates[c.id] = comps

    if not candidates:
        return {}

    # Типичный размер кнопки — медиана по лучшим кандидатам. Отбрасываем то,
    # что сильно больше (фон панели, картинки) или меньше (мелкие детали).
    best_sizes = []
    for comps in candidates.values():
        area, x0, y0, x1, y1 = max(comps)
        best_sizes.append(max(x1 - x0 + 1, y1 - y0 + 1))
    typical = float(np.median(best_sizes))

    found: dict[int, Swatch] = {}
    for cid, comps in candidates.items():
        good = [
            comp for comp in comps
            if 0.5 * typical <= max(comp[3] - comp[1] + 1, comp[4] - comp[2] + 1) <= 1.6 * typical
        ]
        if not good:
            continue
        area, x0, y0, x1, y1 = max(good)
        found[cid] = Swatch(
            x=left + (x0 + x1) // 2,
            y=top + (y0 + y1) // 2,
            w=x1 - x0 + 1,
            h=y1 - y0 + 1,
        )
    log.info(
        "Поиск палитры: найдено %d цветов, типичный размер кнопки %.0f px; не найдены: %s",
        len(found), typical, [cid for cid in BY_ID if cid not in found],
    )
    return found


def palette_visible_ratio(
    img: np.ndarray, rect: Rect, swatches: dict[int, Swatch], tolerance: int = 12
) -> float:
    """Доля известных кнопок цветов, которые сейчас видны на снимке."""
    if not swatches:
        return 0.0
    left, top = rect[0], rect[1]
    ih, iw = img.shape[:2]
    a = img.astype(np.int16)
    visible = 0
    for cid, sw in swatches.items():
        rx = max(1, sw.w // 4)
        ry = max(1, sw.h // 4)
        cx, cy = sw.x - left, sw.y - top
        y0, y1 = max(0, cy - ry), min(ih, cy + ry + 1)
        x0, x1 = max(0, cx - rx), min(iw, cx + rx + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        patch = a[y0:y1, x0:x1]
        match = np.abs(patch - np.array(BY_ID[cid].rgb, dtype=np.int16)).max(axis=2) <= tolerance
        if match.mean() >= 0.25:
            visible += 1
    return visible / len(swatches)
