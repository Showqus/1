"""Основной цикл бота: подходы по зарядам, постановка пикселей, проверки, паузы."""
from __future__ import annotations

import glob
import logging
import math
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from .config import Behavior
from .geometry import Grid, Rect, clip_rect, point_in_rect, point_rect, rects_overlap, union_rect
from .humanize import HumanMouse, plan_order, skewed
from .image_prep import SKIP
from .palette import BY_ID, ColorMatcher
from .palette_finder import Swatch, palette_visible_ratio
from .screen import ScreenSource, sample_cells, sample_point, sample_radius
from .winapi import InputDevice, beep

log = logging.getLogger(__name__)

MAX_DEBUG_SHOTS = 40


class StopRequested(Exception):
    pass


class PauseRequested(Exception):
    def __init__(self, message: str, alert: bool = False):
        super().__init__(message)
        self.message = message
        self.alert = alert


class JobError(Exception):
    """Ошибка настроек/калибровки — показывается пользователю как есть."""


@dataclass
class Job:
    grid: Grid
    target: np.ndarray  # (H, W) id цвета или SKIP
    swatches: dict[int, Swatch]
    paint_button: tuple[int, int]
    submit_button: tuple[int, int]
    palette_rect: Rect

    @property
    def width(self) -> int:
        return int(self.target.shape[1])

    @property
    def height(self) -> int:
        return int(self.target.shape[0])

    def target_rect(self, margin: int = 2) -> Rect:
        return self.grid.rect(self.width, self.height, margin)

    def needed_colors(self) -> list[int]:
        return sorted(int(c) for c in np.unique(self.target) if c != SKIP)

    def validate(self, virtual_screen: Rect | None = None) -> None:
        if self.grid.pitch < 2:
            raise JobError("Размер пикселя на экране слишком маленький (меньше 2 px). Приблизьте карту.")
        missing = [c for c in self.needed_colors() if c not in self.swatches]
        if missing:
            names = ", ".join(BY_ID[c].label for c in missing)
            raise JobError(f"Не найдены на экране кнопки цветов: {names}. Найдите палитру заново или укажите их вручную.")
        if rects_overlap(self.target_rect(0), self.palette_rect):
            raise JobError(
                "Область рисования пересекается с панелью палитры — палитра закроет часть рисунка. "
                "Сдвиньте карту так, чтобы рисунок был выше палитры, и откалибруйте заново."
            )
        for name, (bx, by) in (("Paint", self.paint_button), ("подтверждения", self.submit_button)):
            if point_in_rect(bx, by, self.target_rect(0)):
                raise JobError(f"Кнопка {name} попадает внутрь области рисунка — проверьте калибровку.")
        if virtual_screen is not None:
            l, t, w, h = self.target_rect(0)
            vl, vt, vw, vh = virtual_screen
            if l < vl or t < vt or l + w > vl + vw or t + h > vt + vh:
                raise JobError("Рисунок не помещается на экран. Уменьшите картинку или отдалите карту.")

    def describe(self) -> str:
        g = self.grid
        return (
            f"рисунок {self.width}x{self.height}, пикселей к рисованию {(self.target != SKIP).sum()}, "
            f"сетка x0={g.x0:.2f} y0={g.y0:.2f} шаг={g.pitch:.3f}, область {self.target_rect(0)}, "
            f"цвета {self.needed_colors()}, кнопка Paint {self.paint_button}, подтверждение {self.submit_button}, "
            f"палитра {self.palette_rect}, найдено кнопок цветов {len(self.swatches)}"
        )


class RealClock:
    slice = 0.05

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, sec: float) -> None:
        time.sleep(max(0.0, sec))


class ChargeModel:
    """Оценка текущих зарядов: +1 каждые ``rate`` секунд, но не больше максимума."""

    def __init__(self, max_charges: int, rate: float, start: float, now: float):
        self.max = max(1, int(max_charges))
        self.rate = max(1.0, float(rate))
        self.value = max(0.0, min(float(start), self.max))
        self.t = now

    def current(self, now: float) -> float:
        return min(self.max, self.value + (now - self.t) / self.rate)

    def consume(self, n: int, now: float, exhausted: bool = False) -> None:
        self.value = 0.0 if exhausted else max(0.0, self.current(now) - n)
        self.t = now

    def time_until(self, need: float, now: float) -> float:
        need = min(need, self.max)
        return max(0.0, (need - self.current(now)) * self.rate)


def fmt_seconds(sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class PaintBot:
    def __init__(
        self,
        job: Job,
        behavior: Behavior,
        screen: ScreenSource,
        inp: InputDevice,
        clock=None,
        rng: random.Random | None = None,
        on_event: Callable[..., None] | None = None,
        debug_dir: str | None = None,
    ):
        self.job = job
        self.b = behavior
        self.screen = screen
        self.inp = inp
        self.clock = clock or RealClock()
        self.rng = rng or random.Random()
        self.on_event = on_event
        self.debug_dir = debug_dir
        self.matcher = ColorMatcher(behavior.color_tolerance)
        self.mouse = HumanMouse(
            inp, behavior.human_profile(), self.rng, self._sleep, self._checkpoint_mouse,
            dry_run=behavior.dry_run, now=self.clock.now,
        )
        self.charges = ChargeModel(
            behavior.max_charges, behavior.seconds_per_charge, behavior.start_charges, self.clock.now()
        )
        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()
        self._resume_evt = threading.Event()
        self._mouse_check = behavior.pause_on_user_mouse
        self.selected: int | None = None
        self.reference: np.ndarray | None = None
        self.ref_classes: np.ndarray | None = None
        self.last_remaining: int | None = None
        self.total = int((job.target != SKIP).sum())
        self.stats = {"placed": 0, "batches": 0, "verify_failed": 0}
        self._planned_batch = 1
        self._failed_batches = 0
        self._debug_budget = {}
        self._vr: Rect | None = None
        self._browser_window = 0
        self.union = union_rect(
            job.target_rect(2),
            job.palette_rect,
            point_rect(*job.paint_button, 4),
            point_rect(*job.submit_button, 4),
        )

    # ------------------------------------------------------------------ управление
    def request_stop(self) -> None:
        self._stop_evt.set()
        self._resume_evt.set()

    def request_pause(self) -> None:
        self._pause_evt.set()

    def request_resume(self) -> None:
        self._pause_evt.clear()
        self._resume_evt.set()

    @property
    def paused(self) -> bool:
        return self._pause_evt.is_set()

    def _emit(self, kind: str, **data) -> None:
        if self.on_event:
            try:
                self.on_event(kind, **data)
            except Exception:  # noqa: BLE001
                log.exception("Ошибка в обработчике события %s", kind)

    def _checkpoint(self) -> None:
        if self._stop_evt.is_set():
            raise StopRequested()
        if self._pause_evt.is_set():
            raise PauseRequested("Пауза по запросу пользователя")

    def _checkpoint_mouse(self) -> None:
        self._checkpoint()
        x, y = self.inp.get_cursor()
        if x <= 1 and y <= 1 and self.mouse.expected != (x, y):
            raise StopRequested("курсор отведён в левый верхний угол экрана (аварийная остановка)")
        if self._mouse_check and self.mouse.user_moved():
            raise PauseRequested("Вы сдвинули мышь — бот на паузе. Нажмите «Продолжить» (или горячую клавишу паузы).")

    def _sleep(self, sec: float) -> None:
        end = self.clock.now() + max(0.0, sec)
        while True:
            self._checkpoint()
            rem = end - self.clock.now()
            if rem <= 0:
                return
            self.clock.sleep(min(rem, self.clock.slice))

    def _countdown(self, total: float, label: str) -> None:
        end = self.clock.now() + total
        while True:
            rem = end - self.clock.now()
            if rem <= 0:
                break
            self._emit("countdown", seconds=rem, label=label)
            self._sleep(min(1.0, rem))
        self._emit("countdown", seconds=0, label="")

    # ------------------------------------------------------------------ главный цикл
    def run(self) -> None:
        self._emit("state", state="running")
        try:
            self._startup()
            while True:
                try:
                    if self._cycle():
                        break
                except PauseRequested as e:
                    self._handle_pause(e)
            log.info("Готово: рисунок завершён. Поставлено пикселей за сессию: %d", self.stats["placed"])
            self._emit("finished", message="Готово! Рисунок совпадает с картинкой.")
            if self.b.sound:
                beep("info")
        except StopRequested as e:
            log.info("Бот остановлен%s", f": {e}" if str(e) else "")
            self._emit("stopped", message="Остановлено" + (f": {e}" if str(e) else ""))
        except JobError as e:
            log.error("Ошибка настройки: %s", e)
            self._emit("error", message=str(e))
            if self.b.sound:
                beep("error")
        except Exception as e:  # noqa: BLE001
            log.exception("Непредвиденная ошибка в работе бота")
            self._debug_shot("crash")
            self._emit("error", message=f"Непредвиденная ошибка: {e!r}. Подробности в логе — сохраните отчёт.")
            if self.b.sound:
                beep("error")
        finally:
            log.info("Статистика: %s", self.stats)
            self._emit("state", state="idle")

    def _startup(self) -> None:
        log.info("Старт бота: %s", self.job.describe())
        log.info("Поведение: %s", self.b)
        vr = None
        try:
            vr = self.screen.virtual_rect()
            log.info("Виртуальный экран: %s", vr)
        except Exception:  # noqa: BLE001
            log.warning("Не удалось узнать размер экрана", exc_info=True)
        self.job.validate(vr)
        if vr is not None:
            self.union = clip_rect(self.union, vr)
            self._vr = vr

        # Самопроверка: совпадают ли координаты установки и чтения курсора (проблемы DPI).
        cur = self.inp.get_cursor()
        probe = (cur[0] + 1, cur[1]) if cur[0] < 5000 else (cur[0] - 1, cur[1])
        self.inp.set_cursor(*probe)
        after = self.inp.get_cursor()
        self.inp.set_cursor(*cur)
        if after != probe:
            log.warning(
                "Самопроверка курсора: поставили %s, прочитали %s — возможно, проблемы с масштабом Windows (DPI). "
                "Отключаю паузу при движении мыши.", probe, after,
            )
            self._mouse_check = False

        cx, cy = self.job.grid.center((self.job.width - 1) / 2, (self.job.height - 1) / 2)
        title = self.inp.window_title_at(int(cx), int(cy))
        self._browser_window = self.inp.window_at(int(cx), int(cy))
        log.info("Окно под рисунком: %r (id %s)", title, self._browser_window)
        if title and "wplace" not in title.lower():
            log.warning("Под областью рисования окно %r — похоже, это не вкладка wplace", title)
        self._debug_shot("start")

    def _cycle(self) -> bool:
        self._wait_for_charges()
        self._wait_user_idle()
        if self._batch():
            if not self.b.guard_mode:
                return True
            self._guard_wait()
        return False

    def _handle_pause(self, e: PauseRequested) -> None:
        self._pause_evt.set()
        self._resume_evt.clear()
        log.warning("Пауза: %s", e.message)
        self._emit("paused", message=e.message, alert=e.alert)
        if e.alert and self.b.sound:
            beep("warning")
        while not self._resume_evt.wait(0.2):
            pass
        if self._stop_evt.is_set():
            raise StopRequested()
        log.info("Продолжаю работу после паузы")
        self.selected = None
        self._emit("state", state="running")

    # ------------------------------------------------------------------ ожидания
    def _choose_batch_size(self) -> int:
        mx = max(1, int(self.b.max_charges))
        n = mx
        if self.b.random_batch:
            lo = max(1, math.ceil(mx * min(1.0, max(0.1, self.b.batch_min_fraction))))
            n = self.rng.randint(lo, mx)
        if self.last_remaining is not None:
            n = min(n, max(1, self.last_remaining))
        return n

    def _wait_for_charges(self) -> None:
        n = self._choose_batch_size()
        self._planned_batch = n
        now = self.clock.now()
        need = self.charges.time_until(n, now)
        wait = 0.0
        if need > 0:
            wait = need * self.rng.uniform(1.0, 1.08)
        if self.stats["batches"] > 0:
            wait += self.rng.uniform(self.b.extra_wait_min, max(self.b.extra_wait_min, self.b.extra_wait_max))
        if wait > 0.5:
            log.info(
                "Жду %s: зарядов сейчас ~%.1f, для следующего подхода нужно %d",
                fmt_seconds(wait), self.charges.current(now), n,
            )
            self._countdown(wait, "До следующего подхода")

    def _wait_user_idle(self) -> None:
        need = self.b.idle_before_batch
        if need <= 0:
            return
        told = False
        while True:
            idle = self.inp.idle_seconds()
            if idle >= need:
                return
            if not told:
                log.info("Жду, пока пользователь не трогает мышь/клавиатуру %.0f сек", need)
                self._emit("status", text="Жду, пока вы отпустите мышь и клавиатуру…")
                told = True
            self._sleep(max(0.2, min(1.0, need - idle)))

    def _guard_wait(self) -> None:
        sec = max(30.0, self.b.guard_interval_min * 60 * self.rng.uniform(0.8, 1.3))
        log.info("Режим охраны: следующая проверка через %s", fmt_seconds(sec))
        self._emit("status", text="Рисунок готов. Режим охраны: жду и проверяю снова.")
        self._countdown(sec, "Охрана: следующая проверка")

    # ------------------------------------------------------------------ подход
    def _grab(self, rect: Rect) -> np.ndarray:
        last = None
        for attempt in range(5):
            try:
                return self.screen.grab(rect)
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("Не удалось сделать снимок экрана %s (попытка %d): %r", rect, attempt + 1, e)
                self._sleep(1.5)
        raise PauseRequested(f"Не получается сделать снимок экрана ({last!r}). Экран заблокирован?", alert=True)

    def _palette_open(self, img: np.ndarray | None = None, rect: Rect | None = None) -> float:
        if img is None:
            rect = self.job.palette_rect
            img = self._grab(rect)
        return palette_visible_ratio(img, rect, self.job.swatches)

    def _check_windows(self) -> None:
        """Клики уйдут не туда, если браузер перекрыт другим окном — проверяем заранее."""
        if not self._browser_window:
            return
        job = self.job
        pr = job.palette_rect
        points = {
            "рисунок": job.grid.center((job.width - 1) / 2, (job.height - 1) / 2),
            "кнопка Paint": job.paint_button,
            "кнопка подтверждения": job.submit_button,
            "палитра": (pr[0] + pr[2] / 2, pr[1] + pr[3] / 2),
        }
        for name, (x, y) in points.items():
            hwnd = self.inp.window_at(int(x), int(y))
            if hwnd and hwnd != self._browser_window:
                title = self.inp.window_title_at(int(x), int(y))
                log.warning("Место «%s» перекрыто окном %r (id %s)", name, title, hwnd)
                raise PauseRequested(
                    f"Место «{name}» закрыто другим окном: «{title}». Сделайте окно браузера с wplace видимым "
                    "(не двигая карту) и нажмите «Продолжить».",
                    alert=True,
                )

    def _batch(self) -> bool:
        job = self.job
        self._check_windows()
        self.mouse.sync()
        img = self._grab(self.union)
        ratio = self._palette_open(img, self.union)
        log.debug("Палитра видна на %.0f%%", ratio * 100)
        if ratio < 0.6:
            self._open_palette()
            img = self._grab(self.union)

        colors = sample_cells(img, self.union, job.grid, job.width, job.height)
        classes = self.matcher.classify(colors)
        self._check_alignment(colors, classes, img)
        need = (job.target != SKIP) & (classes != job.target)
        remaining = int(need.sum())
        unknown = int(((classes == -1) & (job.target != SKIP)).sum())
        self.last_remaining = remaining
        self._emit_progress(remaining)
        log.info(
            "Осталось поставить %d из %d пикселей (из них пустых/нераспознанных клеток: %d)",
            remaining, self.total, unknown,
        )
        if remaining == 0:
            return True

        ys, xs = np.nonzero(need)
        todo = [(int(x), int(y), int(job.target[y, x])) for y, x in zip(ys, xs)]
        cx, cy = job.grid.cell_at(*self.inp.get_cursor())
        start = (min(max(cx, 0), job.width - 1), min(max(cy, 0), job.height - 1))
        plan = plan_order(todo, self.b.order, self.rng, self._planned_batch, start)
        self.stats["batches"] += 1
        log.info(
            "Подход #%d: план %d пикселей, цвета %s",
            self.stats["batches"], len(plan), sorted({c for _, _, c in plan}),
        )
        self._emit("status", text=f"Рисую: подход #{self.stats['batches']}, {len(plan)} пикс.")

        placed, exhausted = self._paint(plan)
        self.charges.consume(len(placed), self.clock.now(), exhausted)
        self.last_remaining = max(0, remaining - len(placed))

        if not placed:
            self._failed_batches += 1
            log.warning("За подход не поставлено ни одного пикселя (%d подряд)", self._failed_batches)
            self._debug_shot("no_pixels")
            if self._failed_batches >= 3:
                self._failed_batches = 0
                raise PauseRequested(
                    "Три подхода подряд не удалось поставить ни одного пикселя. Проверьте калибровку "
                    "(размер пикселя, положение рисунка, кнопки цветов) и что на аккаунте есть заряды.",
                    alert=True,
                )
            return False

        self._failed_batches = 0
        self.stats["placed"] += len(placed)
        self._submit(placed)
        if self.reference is not None:
            for x, y, c in placed:
                self.reference[y, x] = BY_ID[c].rgb
                self.ref_classes[y, x] = c
        self._emit_progress(self.last_remaining)
        return False

    def _emit_progress(self, remaining: int) -> None:
        self._emit(
            "progress",
            done=self.total - remaining,
            total=self.total,
            placed=self.stats["placed"],
            eta=remaining * self.b.seconds_per_charge,
        )

    def _check_alignment(self, colors: np.ndarray, classes: np.ndarray, img: np.ndarray) -> None:
        """Проверка, что карта не сдвинулась с прошлого подхода.

        Два признака: (1) сырые цвета клеток почти не изменились (или стали нужными);
        (2) клетки, которые в прошлый раз уже были правильными (в т.ч. поставленные
        нами), остались правильными. При сдвиге карты оба признака резко падают.
        """
        target = self.job.target
        if self.reference is None or self.reference.shape != colors.shape:
            self.reference = colors.copy()
            self.ref_classes = classes.copy()
            return
        diff = np.abs(colors.astype(np.int16) - self.reference.astype(np.int16)).max(axis=2)
        ok = (diff <= self.b.color_tolerance) | (classes == target)
        ratio = float(ok.mean())
        anchors = (self.ref_classes == target) & (target != SKIP)
        n_anchor = int(anchors.sum())
        kept = float((classes[anchors] == target[anchors]).mean()) if n_anchor else 1.0
        log.debug("Проверка положения карты: цвета совпадают %.1f%%, опорных пикселей %d, на месте %.1f%%",
                  ratio * 100, n_anchor, kept * 100)
        moved = (ok.size >= 16 and ratio < self.b.align_threshold) or (n_anchor >= 8 and kept < 0.5)
        if moved:
            self._debug_shot("map_moved", img=img)
            raise PauseRequested(
                f"Картинка под рисунком сильно изменилась (цвета совпадают на {ratio:.0%}, уже нарисованное на месте "
                f"на {kept:.0%}). Похоже, карту сдвинули или изменили масштаб, либо окно браузера чем-то перекрыто. "
                "Верните карту точно на место и нажмите «Продолжить», или остановите бота, откалибруйте заново "
                "и запустите снова.",
                alert=True,
            )
        self.reference = colors.copy()
        self.ref_classes = classes.copy()

    def _paint(self, plan: list[tuple[int, int, int]]) -> tuple[list[tuple[int, int, int]], bool]:
        grid = self.job.grid
        pitch = grid.pitch
        queue = deque((x, y, c, 0) for x, y, c in plan)
        placed: list[tuple[int, int, int]] = []
        pending = None
        fail_streak = 0
        fails = 0
        exhausted = False
        count = 0
        next_rest = self.rng.randint(self.b.rest_every_min, max(self.b.rest_every_min, self.b.rest_every_max))

        while queue:
            x, y, c, attempt = queue[0]
            if c != self.selected:
                self._select_color(c)
            tx, ty = self.mouse.jitter_point(*grid.center(x, y), pitch)
            self.mouse.move_to(tx, ty, target_size=pitch)
            if pending is not None:
                if self._verify_cell(pending):
                    placed.append(pending[:3])
                    fail_streak = 0
                else:
                    fails += 1
                    px, py, pc, patt = pending
                    pending = None
                    if patt < 1:
                        log.info("Пиксель (%d,%d) не поставился — пробую ещё раз", px, py)
                        queue.appendleft((px, py, pc, patt + 1))
                        self.selected = None
                        continue
                    fail_streak += 1
                    if fail_streak >= 2 or fails >= 6:
                        exhausted = True
                        log.warning("Пиксели перестали ставиться — похоже, закончились заряды")
                        break
                pending = None
            queue.popleft()
            self.mouse.click()
            pending = (x, y, c, attempt)
            count += 1

            self._sleep(skewed(self.rng, self.b.click_delay_min, self.b.click_delay_max))
            if self.rng.random() < self.b.hesitation_chance:
                self._sleep(self.rng.uniform(self.b.hesitation_min, self.b.hesitation_max))
            if queue and count >= next_rest:
                rest = self.rng.uniform(self.b.rest_min, self.b.rest_max)
                log.debug("Короткий отдых %.1f сек", rest)
                self._sleep(rest)
                next_rest = count + self.rng.randint(
                    self.b.rest_every_min, max(self.b.rest_every_min, self.b.rest_every_max)
                )

        # Последний пиксель проверяем, уведя курсор к кнопке подтверждения.
        while pending is not None:
            sx, sy = self.job.submit_button
            self.mouse.move_to(sx + self.rng.uniform(-4, 4), sy + self.rng.uniform(-3, 3), target_size=20)
            if self._verify_cell(pending):
                placed.append(pending[:3])
                break
            px, py, pc, patt = pending
            pending = None
            if patt < 1 and not exhausted:
                log.info("Последний пиксель (%d,%d) не поставился — пробую ещё раз", px, py)
                self.selected = None
                self._select_color(pc)
                tx, ty = self.mouse.jitter_point(*grid.center(px, py), pitch)
                self.mouse.move_to(tx, ty, target_size=pitch)
                self.mouse.click()
                pending = (px, py, pc, patt + 1)
            else:
                exhausted = True
        log.info("Поставлено в подходе: %d из %d, заряды кончились: %s", len(placed), len(plan), exhausted)
        return placed, exhausted

    def _select_color(self, cid: int) -> None:
        sw = self.job.swatches[cid]
        size = max(4, min(sw.w, sw.h))
        tx, ty = self.mouse.jitter_point(sw.x, sw.y, size)
        self.mouse.move_to(tx, ty, target_size=size)
        self.mouse.click()
        self.selected = cid
        log.debug("Выбран цвет %s", BY_ID[cid].label)
        self._sleep(skewed(self.rng, 0.15, 0.6))

    def _verify_cell(self, item) -> bool:
        if not self.b.verify_pixels or self.b.dry_run:
            return True
        x, y, c = item[:3]
        cx, cy = self.job.grid.center(x, y)
        r = max(1, sample_radius(self.job.grid.pitch))
        rect = point_rect(cx, cy, r + 1)
        img = self._grab(rect)
        rgb = sample_point(img, rect, cx, cy, r)
        got = self.matcher.classify_one(rgb)
        if got == c:
            return True
        self.stats["verify_failed"] += 1
        got_name = BY_ID[got].label if got in BY_ID else "не цвет палитры"
        log.info(
            "Проверка пикселя (%d,%d): нужен %s, на экране %s rgb=%s",
            x, y, BY_ID[c].label, got_name, tuple(int(v) for v in rgb),
        )
        self._debug_shot("verify_failed", limit=3)
        return False

    def _open_palette(self) -> None:
        for attempt in (1, 2):
            x, y = self.job.paint_button
            self.mouse.move_to(x + self.rng.uniform(-4, 4), y + self.rng.uniform(-3, 3), target_size=30)
            self.mouse.click()
            self._sleep(self.rng.uniform(1.0, 1.8))
            ratio = self._palette_open()
            log.info("Открываю палитру (попытка %d): видно %.0f%% кнопок цветов", attempt, ratio * 100)
            if ratio >= 0.6:
                self.selected = None
                return
            if self.b.dry_run:
                log.info("[пробный режим] палитра не видна — продолжаю без неё")
                return
            self._sleep(self.rng.uniform(1.0, 2.0))
            if self._palette_open() >= 0.6:
                self.selected = None
                return
        self._debug_shot("palette_not_open")
        raise PauseRequested(
            "Не получилось открыть палитру кнопкой «Paint». Проверьте, что вкладка wplace открыта, вы вошли в "
            "аккаунт и позиции кнопки «Paint»/палитры откалиброваны. Затем нажмите «Продолжить».",
            alert=True,
        )

    def _submit(self, placed: list[tuple[int, int, int]]) -> None:
        x, y = self.job.submit_button
        self.mouse.move_to(x + self.rng.uniform(-4, 4), y + self.rng.uniform(-3, 3), target_size=30)
        self.mouse.click()
        self.selected = None
        log.info("Нажата кнопка подтверждения: отправляю %d пикселей", len(placed))
        self._sleep(self.rng.uniform(1.2, 2.2))
        if self.b.dry_run or not self.b.palette_closes_after_submit:
            self._sleep(self.rng.uniform(0.5, 1.5))
            return
        deadline = self.clock.now() + self.b.submit_timeout
        while self._palette_open() >= 0.3:
            if self.clock.now() >= deadline:
                self._debug_shot("submit_stuck")
                raise PauseRequested(
                    f"После нажатия «Paint» палитра не закрылась за {self.b.submit_timeout:.0f} сек. Возможно, "
                    "появилась проверка «я не робот» или ошибка сайта. Разберитесь в браузере вручную "
                    "(пройдите проверку / нажмите Paint), затем нажмите «Продолжить».",
                    alert=True,
                )
            self._sleep(0.5)
        self._sleep(self.rng.uniform(0.6, 1.2))
        # Проверяем, что пиксели остались на своих местах после отправки.
        ok, img, rect = self._placed_ok(placed)
        if ok < 0.7 * len(placed):
            log.warning("После отправки на месте %d из %d пикселей — перепроверяю", ok, len(placed))
            self._sleep(self.rng.uniform(2.5, 4.0))
            ok, img, rect = self._placed_ok(placed)
        log.info("После отправки на месте %d из %d пикселей", ok, len(placed))
        if ok < 0.7 * len(placed):
            self._debug_shot("after_submit", img=img, rect=rect)
            raise PauseRequested(
                f"После отправки на своих местах только {ok} из {len(placed)} пикселей. Похоже, карту сдвинули "
                "(или изменили масштаб), либо сайт не принял пиксели. Проверьте браузер: если карта на месте — "
                "нажмите «Продолжить», иначе остановите бота, откалибруйте заново и запустите снова.",
                alert=True,
            )

    def _placed_ok(self, placed):
        rect = self.job.target_rect(2)
        if self._vr is not None:
            rect = clip_rect(rect, self._vr)
        img = self._grab(rect)
        colors = sample_cells(img, rect, self.job.grid, self.job.width, self.job.height)
        classes = self.matcher.classify(colors)
        ok = sum(1 for px, py, c in placed if classes[py, px] == c)
        return ok, img, rect

    # ------------------------------------------------------------------ отладка
    def _debug_shot(self, name: str, img: np.ndarray | None = None, rect: Rect | None = None, limit: int = 10) -> None:
        if not self.debug_dir:
            return
        used = self._debug_budget.get(name, 0)
        if used >= limit:
            return
        self._debug_budget[name] = used + 1
        try:
            rect = rect or self.union
            if img is None:
                img = self.screen.grab(rect)
            os.makedirs(self.debug_dir, exist_ok=True)
            pic = Image.fromarray(img).convert("RGB")
            draw = ImageDraw.Draw(pic)
            ox, oy = rect[0], rect[1]

            def box(r, color):
                draw.rectangle([r[0] - ox, r[1] - oy, r[0] + r[2] - ox - 1, r[1] + r[3] - oy - 1], outline=color)

            box(self.job.target_rect(0), (255, 0, 255))
            box(self.job.palette_rect, (0, 128, 255))
            for sx, sy in (self.job.paint_button, self.job.submit_button):
                draw.ellipse([sx - ox - 5, sy - oy - 5, sx - ox + 5, sy - oy + 5], outline=(0, 200, 0), width=2)
            for sw in self.job.swatches.values():
                draw.line([sw.x - ox - 3, sw.y - oy, sw.x - ox + 3, sw.y - oy], fill=(255, 0, 0))
                draw.line([sw.x - ox, sw.y - oy - 3, sw.x - ox, sw.y - oy + 3], fill=(255, 0, 0))
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.debug_dir, f"{ts}_{name}.png")
            pic.save(path)
            log.info("Снимок для отладки: %s (область экрана %s)", os.path.basename(path), rect)
            files = sorted(glob.glob(os.path.join(self.debug_dir, "*.png")))
            for old in files[:-MAX_DEBUG_SHOTS]:
                os.remove(old)
        except Exception:  # noqa: BLE001
            log.warning("Не удалось сохранить отладочный снимок", exc_info=True)
