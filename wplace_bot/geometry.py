"""Сетка холста на экране и автоопределение размера пикселя."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

Rect = tuple[int, int, int, int]  # left, top, width, height


@dataclass
class Grid:
    """Клетка (i, j) холста имеет центр в (x0 + i*pitch, y0 + j*pitch) на экране."""

    x0: float
    y0: float
    pitch: float

    def center(self, i: float, j: float) -> tuple[float, float]:
        return self.x0 + i * self.pitch, self.y0 + j * self.pitch

    def cell_at(self, x: float, y: float) -> tuple[int, int]:
        return round((x - self.x0) / self.pitch), round((y - self.y0) / self.pitch)

    def rect(self, width: int, height: int, margin: int = 0) -> Rect:
        """Прямоугольник экрана, покрывающий все клетки (с запасом ``margin`` px)."""
        half = self.pitch / 2
        left = math.floor(self.x0 - half) - margin
        top = math.floor(self.y0 - half) - margin
        right = math.ceil(self.x0 + (width - 1) * self.pitch + half) + margin
        bottom = math.ceil(self.y0 + (height - 1) * self.pitch + half) + margin
        return left, top, right - left, bottom - top


def union_rect(*rects: Rect) -> Rect:
    l = min(r[0] for r in rects)
    t = min(r[1] for r in rects)
    r_ = max(r[0] + r[2] for r in rects)
    b = max(r[1] + r[3] for r in rects)
    return l, t, r_ - l, b - t


def clip_rect(r: Rect, bounds: Rect) -> Rect:
    l = max(r[0], bounds[0])
    t = max(r[1], bounds[1])
    rr = min(r[0] + r[2], bounds[0] + bounds[2])
    b = min(r[1] + r[3], bounds[1] + bounds[3])
    return l, t, max(1, rr - l), max(1, b - t)


def point_in_rect(x: float, y: float, r: Rect) -> bool:
    return r[0] <= x < r[0] + r[2] and r[1] <= y < r[1] + r[3]


def rects_overlap(a: Rect, b: Rect) -> bool:
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def point_rect(x: float, y: float, radius: int) -> Rect:
    return int(round(x)) - radius, int(round(y)) - radius, 2 * radius + 1, 2 * radius + 1


@dataclass
class PitchResult:
    pitch: float
    phase_x: float  # экранная координата (в пределах снимка) одной из границ клеток по X
    phase_y: float
    confidence: float  # 0..1, насколько уверенно найдена сетка

    @property
    def ok(self) -> bool:
        return self.confidence >= 0.35


def _edge_profiles(img: np.ndarray, threshold: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Сколько границ цвета между соседними столбцами/строками.

    Учитываются только границы, у которых хотя бы с одной стороны цвет палитры:
    так края подложки-карты и интерфейса почти не мешают.
    """
    from .palette_finder import label_image  # локальный импорт, чтобы избежать цикла

    a = img.astype(np.int16)
    pal = label_image(img, 6) > 0
    dx = (np.abs(a[:, 1:] - a[:, :-1]).sum(axis=2) > threshold) & (pal[:, 1:] | pal[:, :-1])
    dy = (np.abs(a[1:, :] - a[:-1, :]).sum(axis=2) > threshold) & (pal[1:, :] | pal[:-1, :])
    return dx.sum(axis=0).astype(np.float64), dy.sum(axis=1).astype(np.float64)


def _axis_score(prof: np.ndarray, dil: np.ndarray, total: float, pitch: float, phase_step: float):
    """Оценка сетки с шагом ``pitch`` по одной оси (лучшая по фазе).

    Оценка = «покрытие» × «плотность»:
    * покрытие — какая доля всех границ лежит на линиях сетки (±1 px), за вычетом
      случайного совпадения. У кратного шага (2p, 3p…) часть границ оказывается
      между линиями, и покрытие падает;
    * плотность — сколько границ в среднем на линии сетки относительно среднего по
      снимку. У дробного шага (p/2…) половина линий пустая, и плотность падает.
    Возвращает (оценка, фаза, покрытие).
    """
    n = len(prof)
    phases = np.arange(0.0, pitch, phase_step)
    ks = np.arange(0, int(n / pitch) + 2)
    b = phases[:, None] + ks[None, :] * pitch  # координаты границ клеток
    idx = np.floor(b - 0.5).astype(np.int64)  # индекс «границы между столбцами idx и idx+1»
    valid = (idx >= 0) & (idx < n)
    cnt = valid.sum(axis=1)
    ci = np.clip(idx, 0, n - 1)
    line_sum = np.where(valid, prof[ci], 0.0).sum(axis=1)
    covered = np.where(valid, dil[ci], 0.0).sum(axis=1)
    chance = min(0.95, 3.0 / pitch)
    cov = np.clip((covered / total - chance) / (1 - chance), 0.0, 1.0)
    density = line_sum / np.maximum(cnt, 1) / (total / n)
    sc = np.where(cnt >= 3, cov * density, -np.inf)
    best = int(np.argmax(sc))
    return float(sc[best]), float(phases[best]), float(cov[best])


def detect_pitch(img: np.ndarray, pmin: float = 4.0, pmax: float = 80.0) -> PitchResult:
    """Найти размер пикселя холста по снимку экрана (h, w, 3).

    Ищем периодическую сетку, на линиях которой лежат границы между пикселями
    цветов палитры. Нужен участок, на котором уже есть нарисованные пиксели.
    """
    h, w = img.shape[:2]
    pmax = min(pmax, max(w, h) / 6)
    if h < 16 or w < 16 or pmax <= pmin:
        return PitchResult(0.0, 0.0, 0.0, 0.0)
    raw_x, raw_y = _edge_profiles(img)
    if raw_x.sum() < 20 or raw_y.sum() < 20:
        log.info("Автоопределение пикселя: на снимке почти нет пикселей цветов палитры")
        return PitchResult(0.0, 0.0, 0.0, 0.0)
    dil_k = np.array([1.0, 1.0, 1.0])
    axes = []
    for raw in (raw_x, raw_y):
        axes.append({
            "wide": np.convolve(raw, [0.5, 1.0, 0.5], mode="same"),  # допуск ±1 px для грубого поиска
            "narrow": np.convolve(raw, [0.25, 1.0, 0.25], mode="same"),
            "dil": np.convolve(raw, dil_k, mode="same"),
            "total": float(raw.sum()),
        })

    def score(p: float, step: float, kind: str = "wide") -> tuple[float, float, float, float]:
        rx = _axis_score(axes[0][kind], axes[0]["dil"], axes[0]["total"], p, step)
        ry = _axis_score(axes[1][kind], axes[1]["dil"], axes[1]["total"], p, step)
        return rx[0] + ry[0], rx[1], ry[1], (rx[2] + ry[2]) / 2

    # Шаг перебора геометрический: ошибка шага накапливается по ширине снимка,
    # поэтому относительная точность должна быть ~0.5 px на всю ширину.
    span = max(len(raw_x), len(raw_y))
    ratio = 1 + 0.5 / span
    coarse = pmin * ratio ** np.arange(int(math.log(pmax / pmin) / math.log(ratio)) + 1)
    scores = np.array([score(p, 0.5)[0] for p in coarse])
    best_p = float(coarse[int(np.argmax(scores))])

    # Страховка: если в 2..8 раз меньший шаг объясняет границы почти так же хорошо,
    # то найден кратный шаг — берём меньший.
    base_score = score(best_p, 0.25)[0]
    for n in range(8, 1, -1):
        sub = best_p / n
        if sub < pmin:
            continue
        local = sub * ratio ** np.arange(-3, 4)
        local_scores = [score(p, 0.25)[0] for p in local]
        if max(local_scores) >= 0.62 * base_score:
            log.debug("Автоопределение: шаг %.3f кратен %.3f (x%d)", best_p, sub, n)
            best_p = float(local[int(np.argmax(local_scores))])
            break

    fine = np.linspace(best_p / ratio ** 4, best_p * ratio ** 4, 81)
    fine_vals = np.array([score(p, 0.05, "narrow")[0] for p in fine])
    # Границы на экране целочисленные, поэтому целый диапазон шагов объясняет их
    # одинаково хорошо — берём середину этого диапазона.
    ties = fine[fine_vals >= fine_vals.max() - 1e-6 * abs(fine_vals.max())]
    pitch = float(np.median(ties))
    s_best, fx, fy, cov = score(pitch, 0.05, "narrow")

    # Уверенность: доля границ, объяснённых сеткой, и отрыв от лучшего «соперника» —
    # шага, не кратного найденному. Соперников берём только среди шагов, дающих не
    # меньше ~16 клеток на снимке: у крупных шагов мало выборок, их оценка шумная.
    rivals = [
        float(sc) for p, sc in zip(coarse, scores)
        if p <= span / 16 and abs(p / pitch - 1) > 0.03 and not _is_harmonic(p, pitch)
    ]
    rival = max(rivals) if rivals else 0.0
    separation = max(0.0, min(1.0, (s_best - rival) / (abs(s_best) + 1e-9)))
    conf = cov * separation
    log.info(
        "Автоопределение пикселя: шаг=%.3f фаза=(%.2f, %.2f) оценка=%.3f покрытие=%.2f соперник=%.3f уверенность=%.2f",
        pitch, fx, fy, s_best, cov, rival, conf,
    )
    return PitchResult(pitch, fx, fy, conf)


def _is_harmonic(p: float, base: float) -> bool:
    """p — кратный или дробный (n/m, m ≤ 3) шаг от base."""
    r = p / base
    for m in (1, 2, 3):
        n = round(r * m)
        if n >= 1 and abs(r * m - n) < 0.04 * max(1, n) ** 0.5 and (n, m) != (1, 1):
            return True
    return False


def snap_to_center(coord: float, phase: float, pitch: float) -> float:
    """Сдвинуть экранную координату в центр клетки сетки, в которую она попадает.

    ``phase`` — абсолютная координата одной из границ клеток (как её вернул
    detect_pitch плюс левый/верхний край снимка). Экранный пиксель с индексом x
    занимает отрезок [x, x+1), поэтому переводим в «непрерывные» координаты и обратно.
    """
    cont = coord + 0.5
    k = math.floor((cont - phase) / pitch)
    return phase + (k + 0.5) * pitch - 0.5
