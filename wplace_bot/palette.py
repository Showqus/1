"""Палитра wplace.live и сопоставление цветов."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PaletteColor:
    id: int
    name: str
    name_ru: str
    rgb: tuple[int, int, int]
    premium: bool

    @property
    def hex(self) -> str:
        return "#%02x%02x%02x" % self.rgb

    @property
    def label(self) -> str:
        return f"{self.id}. {self.name_ru} ({self.name})"


# id 0 (Transparent) не рисуется ботом и в список не входит.
_RAW = [
    (1, "Black", "Чёрный", (0, 0, 0), False),
    (2, "Dark Gray", "Тёмно-серый", (60, 60, 60), False),
    (3, "Gray", "Серый", (120, 120, 120), False),
    (4, "Light Gray", "Светло-серый", (210, 210, 210), False),
    (5, "White", "Белый", (255, 255, 255), False),
    (6, "Deep Red", "Бордовый", (96, 0, 24), False),
    (7, "Red", "Красный", (237, 28, 36), False),
    (8, "Orange", "Оранжевый", (255, 127, 39), False),
    (9, "Gold", "Золотой", (246, 170, 9), False),
    (10, "Yellow", "Жёлтый", (249, 221, 59), False),
    (11, "Light Yellow", "Светло-жёлтый", (255, 250, 188), False),
    (12, "Dark Green", "Тёмно-зелёный", (14, 185, 104), False),
    (13, "Green", "Зелёный", (19, 230, 123), False),
    (14, "Light Green", "Светло-зелёный", (135, 255, 94), False),
    (15, "Dark Teal", "Тёмно-бирюзовый", (12, 129, 110), False),
    (16, "Teal", "Бирюзовый", (16, 174, 166), False),
    (17, "Light Teal", "Светло-бирюзовый", (19, 225, 190), False),
    (18, "Dark Blue", "Тёмно-синий", (40, 80, 158), False),
    (19, "Blue", "Синий", (64, 147, 228), False),
    (20, "Cyan", "Голубой", (96, 247, 242), False),
    (21, "Indigo", "Индиго", (107, 80, 246), False),
    (22, "Light Indigo", "Светлый индиго", (153, 177, 251), False),
    (23, "Dark Purple", "Тёмно-фиолетовый", (120, 12, 153), False),
    (24, "Purple", "Фиолетовый", (170, 56, 185), False),
    (25, "Light Purple", "Светло-фиолетовый", (224, 159, 249), False),
    (26, "Dark Pink", "Тёмно-розовый", (203, 0, 122), False),
    (27, "Pink", "Розовый", (236, 31, 128), False),
    (28, "Light Pink", "Светло-розовый", (243, 141, 169), False),
    (29, "Dark Brown", "Тёмно-коричневый", (104, 70, 52), False),
    (30, "Brown", "Коричневый", (149, 104, 42), False),
    (31, "Beige", "Бежевый", (248, 178, 119), False),
    (32, "Medium Gray", "Средне-серый", (170, 170, 170), True),
    (33, "Dark Red", "Тёмно-красный", (165, 14, 30), True),
    (34, "Light Red", "Светло-красный", (250, 128, 114), True),
    (35, "Dark Orange", "Тёмно-оранжевый", (228, 92, 26), True),
    (36, "Light Tan", "Светло-песочный", (214, 181, 148), True),
    (37, "Dark Goldenrod", "Тёмно-золотистый", (156, 132, 49), True),
    (38, "Goldenrod", "Золотистый", (197, 173, 49), True),
    (39, "Light Goldenrod", "Светло-золотистый", (232, 212, 95), True),
    (40, "Dark Olive", "Тёмно-оливковый", (74, 107, 58), True),
    (41, "Olive", "Оливковый", (90, 148, 74), True),
    (42, "Light Olive", "Светло-оливковый", (132, 197, 115), True),
    (43, "Dark Cyan", "Тёмно-циановый", (15, 121, 159), True),
    (44, "Light Cyan", "Светло-циановый", (187, 250, 242), True),
    (45, "Light Blue", "Светло-синий", (125, 199, 255), True),
    (46, "Dark Indigo", "Тёмный индиго", (77, 49, 184), True),
    (47, "Dark Slate Blue", "Тёмный сланцево-синий", (74, 66, 132), True),
    (48, "Slate Blue", "Сланцево-синий", (122, 113, 196), True),
    (49, "Light Slate Blue", "Светлый сланцево-синий", (181, 174, 241), True),
    (50, "Light Brown", "Светло-коричневый", (219, 164, 99), True),
    (51, "Dark Beige", "Тёмно-бежевый", (209, 128, 81), True),
    (52, "Light Beige", "Светло-бежевый", (255, 197, 165), True),
    (53, "Dark Peach", "Тёмно-персиковый", (155, 82, 73), True),
    (54, "Peach", "Персиковый", (209, 128, 120), True),
    (55, "Light Peach", "Светло-персиковый", (250, 182, 164), True),
    (56, "Dark Tan", "Тёмно-песочный", (123, 99, 82), True),
    (57, "Tan", "Песочный", (156, 132, 107), True),
    (58, "Dark Slate", "Тёмный сланец", (51, 57, 65), True),
    (59, "Slate", "Сланец", (109, 117, 141), True),
    (60, "Light Slate", "Светлый сланец", (179, 185, 209), True),
    (61, "Dark Stone", "Тёмный камень", (109, 100, 63), True),
    (62, "Stone", "Камень", (148, 140, 107), True),
    (63, "Light Stone", "Светлый камень", (205, 197, 158), True),
]

COLORS: list[PaletteColor] = [PaletteColor(*row) for row in _RAW]
BY_ID: dict[int, PaletteColor] = {c.id: c for c in COLORS}
ALL_IDS: list[int] = [c.id for c in COLORS]
FREE_IDS: list[int] = [c.id for c in COLORS if not c.premium]
PREMIUM_IDS: list[int] = [c.id for c in COLORS if c.premium]


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255) -> CIELAB (D65). Работает с массивами формы (..., 3)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = c @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    l = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([l, a, b], axis=-1)


def ids_rgb(ids: list[int]) -> np.ndarray:
    return np.array([BY_ID[i].rgb for i in ids], dtype=np.float64)


def nearest_ids(pixels_rgb: np.ndarray, ids: list[int], perceptual: bool = True) -> np.ndarray:
    """Для каждого цвета (N, 3) вернуть id ближайшего цвета палитры из ``ids``."""
    ids_arr = np.asarray(ids, dtype=np.int16)
    pal = ids_rgb(ids)
    px = np.asarray(pixels_rgb, dtype=np.float64).reshape(-1, 3)
    if perceptual:
        pal = rgb_to_lab(pal)
    out = np.empty(len(px), dtype=np.int16)
    chunk = 40000
    for start in range(0, len(px), chunk):
        part = px[start : start + chunk]
        if perceptual:
            part = rgb_to_lab(part)
        d = ((part[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
        out[start : start + chunk] = ids_arr[np.argmin(d, axis=1)]
    return out


class ColorMatcher:
    """Определяет, каким цветом палитры является цвет, снятый с экрана.

    Цвета на экране должны совпадать с палитрой почти точно, поэтому используется
    обычное евклидово расстояние в RGB и допуск ``tolerance``. Если ближайший цвет
    дальше допуска — возвращается -1 (пустая клетка/карта/что-то непонятное).
    """

    def __init__(self, tolerance: float = 18.0, ids: list[int] | None = None):
        self.ids = np.asarray(ids or ALL_IDS, dtype=np.int16)
        self.rgb = ids_rgb(list(self.ids))
        self.tolerance = float(tolerance)

    def classify(self, colors: np.ndarray) -> np.ndarray:
        arr = np.asarray(colors, dtype=np.float64)
        shape = arr.shape[:-1]
        flat = arr.reshape(-1, 3)
        d = ((flat[:, None, :] - self.rgb[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d, axis=1)
        best = np.sqrt(d[np.arange(len(flat)), idx])
        res = np.where(best <= self.tolerance, self.ids[idx], -1).astype(np.int16)
        return res.reshape(shape)

    def classify_one(self, rgb) -> int:
        return int(self.classify(np.asarray(rgb, dtype=np.float64).reshape(1, 3))[0])
