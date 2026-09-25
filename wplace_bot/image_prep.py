"""Подготовка картинки: масштабирование и перевод в палитру wplace."""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from .palette import BY_ID, ids_rgb, nearest_ids

log = logging.getLogger(__name__)

SKIP = -1  # клетка не рисуется (прозрачный пиксель картинки)
MAX_DITHER_PIXELS = 250_000


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    img.load()
    log.info("Картинка загружена: %s, размер %sx%s, режим %s", path, img.width, img.height, img.mode)
    return img.convert("RGBA")


def fit_size(orig_w: int, orig_h: int, width: int, height: int, keep_aspect: bool) -> tuple[int, int]:
    """Итоговый размер в пикселях холста. 0 в ширине/высоте означает «по пропорции»."""
    if width <= 0 and height <= 0:
        return orig_w, orig_h
    if keep_aspect or width <= 0 or height <= 0:
        if width > 0:
            height = max(1, round(orig_h * width / orig_w))
        else:
            width = max(1, round(orig_w * height / orig_h))
    return int(width), int(height)


def prepare_target(
    img: Image.Image,
    width: int,
    height: int,
    allowed_ids: list[int],
    resample: str = "nearest",
    dither: bool = False,
    alpha_threshold: int = 128,
) -> np.ndarray:
    """Вернуть массив (H, W) с id цветов палитры; SKIP для прозрачных пикселей."""
    if not allowed_ids:
        raise ValueError("Не выбрано ни одного цвета палитры")
    rs = Image.Resampling.NEAREST if resample == "nearest" else Image.Resampling.LANCZOS
    if (width, height) != img.size:
        img = img.resize((width, height), rs)
    arr = np.asarray(img, dtype=np.uint8)
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    if dither and width * height <= MAX_DITHER_PIXELS:
        ids = _dither_fs(rgb.astype(np.float64), allowed_ids)
    else:
        if dither:
            log.warning("Картинка слишком большая для дизеринга (%d px) — дизеринг выключен", width * height)
        ids = nearest_ids(rgb.reshape(-1, 3), allowed_ids).reshape(height, width)
    ids = ids.astype(np.int16)
    ids[alpha < alpha_threshold] = SKIP
    return ids


def _dither_fs(rgb: np.ndarray, allowed_ids: list[int]) -> np.ndarray:
    """Дизеринг Флойда–Стейнберга (ошибка распределяется в RGB)."""
    h, w, _ = rgb.shape
    pal = ids_rgb(allowed_ids)
    pal_ids = np.asarray(allowed_ids, dtype=np.int16)
    out = np.zeros((h, w), dtype=np.int16)
    work = rgb.copy()
    for y in range(h):
        row = work[y]
        for x in range(w):
            old = np.clip(row[x], 0, 255)
            k = int(np.argmin(((pal - old) ** 2).sum(axis=1)))
            out[y, x] = pal_ids[k]
            err = old - pal[k]
            if x + 1 < w:
                row[x + 1] += err * (7 / 16)
            if y + 1 < h:
                nxt = work[y + 1]
                if x > 0:
                    nxt[x - 1] += err * (3 / 16)
                nxt[x] += err * (5 / 16)
                if x + 1 < w:
                    nxt[x + 1] += err * (1 / 16)
    return out


def zone_target(width: int, height: int, color_id: int) -> np.ndarray:
    if color_id not in BY_ID:
        raise ValueError(f"Неизвестный цвет {color_id}")
    return np.full((height, width), color_id, dtype=np.int16)


def render_target(target: np.ndarray) -> Image.Image:
    """Картинка (RGBA) из массива id — для предпросмотра."""
    h, w = target.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for cid in np.unique(target):
        if cid == SKIP:
            continue
        mask = target == cid
        out[mask, :3] = BY_ID[int(cid)].rgb
        out[mask, 3] = 255
    return Image.fromarray(out, "RGBA")


def color_counts(target: np.ndarray) -> list[tuple[int, int]]:
    """[(id, количество)] по убыванию количества, без SKIP."""
    ids, counts = np.unique(target[target != SKIP], return_counts=True)
    pairs = sorted(zip(ids.tolist(), counts.tolist()), key=lambda p: -p[1])
    return [(int(i), int(c)) for i, c in pairs]
