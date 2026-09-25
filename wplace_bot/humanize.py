"""«Человеческое» поведение: плавные движения мыши, случайные задержки, порядок пикселей."""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .winapi import InputDevice

log = logging.getLogger(__name__)

MIN_CLICK_INTERVAL = 0.6  # сек между кликами (системный порог двойного клика — 0.5 с)


@dataclass
class HumanProfile:
    click_delay_min: float = 0.35  # пауза между пикселями, сек
    click_delay_max: float = 1.40
    mouse_speed: float = 1.0  # множитель скорости движения мыши
    click_jitter: float = 0.25  # разброс точки клика внутри пикселя (доля размера пикселя)
    hesitation_chance: float = 0.06  # вероятность «задуматься» перед пикселем
    hesitation_min: float = 0.8
    hesitation_max: float = 3.0
    rest_every_min: int = 18  # короткий отдых каждые N пикселей
    rest_every_max: int = 40
    rest_min: float = 3.0
    rest_max: float = 12.0
    overshoot_chance: float = 0.12  # вероятность «промахнуться» и поправиться


def skewed(rng: random.Random, lo: float, hi: float, a: float = 2.0, b: float = 4.0) -> float:
    """Случайное число из [lo, hi], чаще ближе к lo — как реакция человека."""
    if hi <= lo:
        return lo
    return lo + rng.betavariate(a, b) * (hi - lo)


def min_jerk(t: float) -> float:
    return t * t * t * (10 - 15 * t + 6 * t * t)


def bezier_path(
    rng: random.Random, start: tuple[float, float], end: tuple[float, float], steps: int
) -> list[tuple[float, float]]:
    """Кривая Безье от start до end с плавным разгоном и торможением."""
    (x0, y0), (x3, y3) = start, end
    dx, dy = x3 - x0, y3 - y0
    dist = math.hypot(dx, dy)
    if dist < 1:
        return [end]
    nx, ny = -dy / dist, dx / dist  # нормаль к направлению
    bend = dist * rng.uniform(0.03, 0.22) * rng.choice((-1, 1))
    bend2 = bend * rng.uniform(0.3, 1.1)
    p1 = (x0 + dx * rng.uniform(0.2, 0.4) + nx * bend, y0 + dy * rng.uniform(0.2, 0.4) + ny * bend)
    p2 = (x0 + dx * rng.uniform(0.6, 0.85) + nx * bend2, y0 + dy * rng.uniform(0.6, 0.85) + ny * bend2)
    pts = []
    for k in range(1, steps + 1):
        t = min_jerk(k / steps)
        u = 1 - t
        x = u ** 3 * x0 + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * x3
        y = u ** 3 * y0 + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * y3
        if k < steps:  # лёгкое дрожание руки, кроме последней точки
            x += rng.gauss(0, 0.35)
            y += rng.gauss(0, 0.35)
        pts.append((x, y))
    return pts


def move_duration(rng: random.Random, dist: float, target_size: float, speed: float) -> float:
    """Время движения по закону Фиттса с разбросом."""
    target_size = max(target_size, 2.0)
    t = rng.uniform(0.07, 0.14) + rng.uniform(0.08, 0.15) * math.log2(1 + dist / target_size)
    return max(0.04, t / max(speed, 0.1))


class HumanMouse:
    """Двигает мышь и кликает «по-человечески».

    ``sleep`` — функция ожидания бота (учитывает паузу/стоп), ``checkpoint`` вызывается
    перед каждым шагом движения и может выбросить исключение для паузы/остановки.
    """

    def __init__(
        self,
        inp: InputDevice,
        profile: HumanProfile,
        rng: random.Random,
        sleep: Callable[[float], None],
        checkpoint: Callable[[], None],
        dry_run: bool = False,
        now: Callable[[], float] = time.monotonic,
    ):
        self.inp = inp
        self.p = profile
        self.rng = rng
        self.sleep = sleep
        self.checkpoint = checkpoint
        self.dry_run = dry_run
        self.now = now
        self.expected: tuple[int, int] | None = None
        self._last_click = -1e9

    # --- контроль вмешательства пользователя ---
    def sync(self) -> None:
        """Запомнить текущую позицию курсора как «нашу»."""
        self.expected = self.inp.get_cursor()

    def user_moved(self, tolerance: int = 6) -> bool:
        if self.expected is None:
            return False
        x, y = self.inp.get_cursor()
        return abs(x - self.expected[0]) > tolerance or abs(y - self.expected[1]) > tolerance

    def _set(self, x: float, y: float) -> None:
        xi, yi = int(round(x)), int(round(y))
        self.inp.set_cursor(xi, yi)
        self.expected = (xi, yi)

    # --- движения ---
    def move_to(self, x: float, y: float, target_size: float = 10.0) -> None:
        self.checkpoint()
        start = self.inp.get_cursor()
        dist = math.hypot(x - start[0], y - start[1])
        if dist < 1.5:
            self._set(x, y)
            return
        if dist > 150 and self.rng.random() < self.p.overshoot_chance:
            over = self.rng.uniform(0.03, 0.08)
            ox = x + (x - start[0]) * over + self.rng.gauss(0, 2)
            oy = y + (y - start[1]) * over + self.rng.gauss(0, 2)
            self._glide(start, (ox, oy), target_size)
            self.sleep(self.rng.uniform(0.05, 0.16))
            self._glide((ox, oy), (x, y), target_size)
        else:
            self._glide(start, (x, y), target_size)

    def _glide(self, start, end, target_size: float) -> None:
        dist = math.hypot(end[0] - start[0], end[1] - start[1])
        duration = move_duration(self.rng, dist, target_size, self.p.mouse_speed)
        step_t = self.rng.uniform(0.008, 0.016)
        steps = max(2, int(duration / step_t))
        for px, py in bezier_path(self.rng, start, end, steps):
            self.checkpoint()
            self._set(px, py)
            self.sleep(duration / steps)

    def click(self) -> None:
        self.checkpoint()
        self.sleep(skewed(self.rng, 0.04, 0.22))  # прицеливание
        # Два клика быстрее ~0.5 с браузер может принять за двойной — карта приблизится.
        gap = self.now() - self._last_click
        if gap < MIN_CLICK_INTERVAL:
            self.sleep(MIN_CLICK_INTERVAL - gap + self.rng.uniform(0.02, 0.12))
        self.checkpoint()
        if self.dry_run:
            log.debug("[пробный режим] клик в %s пропущен", self.expected)
            return
        self.inp.mouse_down()
        try:
            self.sleep(self.rng.uniform(0.045, 0.13))
        finally:
            self.inp.mouse_up()
            self._last_click = self.now()

    def jitter_point(self, cx: float, cy: float, size: float) -> tuple[float, float]:
        """Случайная точка внутри пикселя, не слишком близко к краю."""
        max_off = max(0.0, min(size * self.p.click_jitter, size / 2 - 1.5))
        return cx + self.rng.uniform(-max_off, max_off), cy + self.rng.uniform(-max_off, max_off)


# ---------------------------------------------------------------------------
# Порядок постановки пикселей

ORDER_MODES = {
    "colors_nearest": "По цветам, соседние подряд (как человек)",
    "rows": "Построчно «змейкой»",
    "random": "Случайно",
}


def plan_order(
    todo: list[tuple[int, int, int]],
    mode: str,
    rng: random.Random,
    limit: int,
    start: tuple[float, float] | None = None,
) -> list[tuple[int, int, int]]:
    """Выбрать до ``limit`` пикселей (x, y, цвет) из ``todo`` в «человеческом» порядке."""
    if not todo or limit <= 0:
        return []
    if mode == "random":
        items = list(todo)
        rng.shuffle(items)
        return items[:limit]
    if mode == "rows":
        items = sorted(todo, key=lambda t: (t[1], t[0] if t[1] % 2 == 0 else -t[0]))
        return items[:limit]
    return _plan_colors_nearest(todo, rng, limit, start)


def _plan_colors_nearest(todo, rng, limit, start):
    by_color: dict[int, list[tuple[int, int]]] = {}
    for x, y, c in todo:
        by_color.setdefault(c, []).append((x, y))
    # Начинаем с самого массового цвета (иногда — со второго, для разнообразия).
    colors = sorted(by_color, key=lambda c: -len(by_color[c]))
    if len(colors) > 1 and rng.random() < 0.25:
        colors[0], colors[1] = colors[1], colors[0]

    out: list[tuple[int, int, int]] = []
    pos = np.array(start if start is not None else by_color[colors[0]][0], dtype=np.float64)
    for c in colors:
        if len(out) >= limit:
            break
        pts = np.array(by_color[c], dtype=np.float64)
        alive = np.ones(len(pts), dtype=bool)
        while alive.any() and len(out) < limit:
            idx = np.nonzero(alive)[0]
            d = ((pts[idx] - pos) ** 2).sum(axis=1)
            k = min(3, len(idx))
            near = idx[np.argpartition(d, k - 1)[:k]] if len(idx) > k else idx
            near = near[np.argsort(((pts[near] - pos) ** 2).sum(axis=1))]
            weights = [0.75, 0.18, 0.07][: len(near)]
            pick = int(rng.choices(list(near), weights=weights)[0])
            alive[pick] = False
            pos = pts[pick]
            out.append((int(pts[pick][0]), int(pts[pick][1]), c))
    return out
