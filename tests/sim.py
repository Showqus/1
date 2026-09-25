"""Симулятор wplace для тестов: экран + мышь + логика палитры и зарядов."""
from __future__ import annotations

import math

import numpy as np

from wplace_bot.geometry import Rect
from wplace_bot.palette import COLORS
from wplace_bot.palette_finder import Swatch
from wplace_bot.screen import ScreenSource
from wplace_bot.winapi import InputDevice

LUT = np.zeros((64, 3), dtype=np.uint8)
for _c in COLORS:
    LUT[_c.id] = _c.rgb


class FakeClock:
    slice = 0.5

    def __init__(self):
        self.t = 1000.0

    def now(self) -> float:
        return self.t

    def sleep(self, sec: float) -> None:
        self.t += max(0.0, sec)


class FakeWplace(ScreenSource, InputDevice):
    W, H = 1600, 1000
    PANEL = (250, 800, 1100, 185)  # палитра: x, y, w, h
    SW, GAP = 24, 8
    PAINT_BTN = (740, 940, 120, 40)  # кнопка Paint при закрытой палитре
    SUBMIT_BTN = (740, 935, 120, 40)  # кнопка Paint внутри палитры

    def __init__(self, clock: FakeClock, pitch=7.37, offset=(100.3, 60.8), cells=(160, 90), seed=1,
                 charges=30, max_charges=30, art=True):
        self.clock = clock
        self.rng = np.random.default_rng(seed)
        self.pitch = pitch
        self.off = list(offset)
        self.cw, self.ch = cells
        self.board = np.zeros((self.ch, self.cw), dtype=np.int16)
        if art:
            for _ in range(60):
                x, y = self.rng.integers(0, self.cw - 8), self.rng.integers(0, self.ch - 8)
                w, h = self.rng.integers(2, 8, size=2)
                self.board[y : y + h, x : x + w] = self.rng.integers(1, 64)
        self.pending: dict[tuple[int, int], int] = {}
        # «карта» под холстом — случайные прямоугольники не из палитры (без периодичности)
        self.map = np.empty((self.H, self.W, 3), dtype=np.uint8)
        self.map[:] = (170, 205, 233)
        for _ in range(400):
            x, y = self.rng.integers(-50, self.W), self.rng.integers(-50, self.H)
            w, h = self.rng.integers(10, 160, size=2)
            col = self.rng.integers(60, 200, size=3)
            col[2] = 233
            self.map[max(0, y) : y + h, max(0, x) : x + w] = col
        self.palette_open = False
        self.selected: int | None = None
        self.max_charges = max_charges
        self.charges = float(charges)
        self.charge_t = clock.now()
        self.pos = (5, 5)
        self._down = False
        self.clicks: list[tuple[int, int, str]] = []
        self.submits = 0
        self.captcha_next = 0  # сколько следующих отправок «зависнет» на капче
        self.captcha_active = False
        self.user_jerk_at: int | None = None  # через сколько set_cursor «пользователь» дёрнет мышь
        self._sets = 0
        self.overlay: tuple[int, int, int, int] | None = None  # прямоугольник чужого окна поверх браузера
        self.swatch_rects: dict[int, tuple[int, int, int, int]] = {}
        self._panel = self._render_panel()

    # ---------------------------------------------------------------- рендер
    def _render_panel(self) -> np.ndarray:
        x0, y0, w, h = self.PANEL
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        for k, c in enumerate(COLORS):
            r, col = divmod(k, 32)
            sx = 20 + col * (self.SW + self.GAP)
            sy = 15 + r * (self.SW + self.GAP)
            img[sy : sy + self.SW, sx : sx + self.SW] = c.rgb
            if c.premium:  # «замочек»
                img[sy + 9 : sy + 15, sx + 9 : sx + 15] = (40, 40, 40)
            self.swatch_rects[c.id] = (x0 + sx, y0 + sy, self.SW, self.SW)
        bx, by, bw, bh = self.SUBMIT_BTN
        img[by - y0 : by - y0 + bh, bx - x0 : bx - x0 + bw] = (30, 90, 250)
        return img

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor((x + 0.5 - self.off[0]) / self.pitch), math.floor((y + 0.5 - self.off[1]) / self.pitch))

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        return self.off[0] + (i + 0.5) * self.pitch - 0.5, self.off[1] + (j + 0.5) * self.pitch - 0.5

    def truth_swatches(self) -> dict[int, Swatch]:
        return {cid: Swatch(x + w // 2, y + h // 2, w, h) for cid, (x, y, w, h) in self.swatch_rects.items()}

    def grab(self, rect: Rect) -> np.ndarray:
        l, t, w, h = (int(v) for v in rect)
        xs = np.arange(l, l + w)
        ys = np.arange(t, t + h)
        out = np.zeros((h, w, 3), dtype=np.uint8)
        inside_x = (xs >= 0) & (xs < self.W)
        inside_y = (ys >= 0) & (ys < self.H)
        cx = np.clip(xs, 0, self.W - 1)
        cy = np.clip(ys, 0, self.H - 1)
        out[:] = self.map[cy[:, None], cx[None, :]]
        ci = np.floor((xs + 0.5 - self.off[0]) / self.pitch).astype(int)
        cj = np.floor((ys + 0.5 - self.off[1]) / self.pitch).astype(int)
        disp = self.board.copy()
        for (i, j), c in self.pending.items():
            disp[j, i] = c
        vi = (ci >= 0) & (ci < self.cw)
        vj = (cj >= 0) & (cj < self.ch)
        cells = disp[np.clip(cj, 0, self.ch - 1)[:, None], np.clip(ci, 0, self.cw - 1)[None, :]]
        mask = (cells > 0) & vi[None, :] & vj[:, None]
        out[mask] = LUT[cells[mask]]
        if self.palette_open:
            self._paste(out, l, t, self._panel, self.PANEL[0], self.PANEL[1])
        else:
            bx, by, bw, bh = self.PAINT_BTN
            self._paste(out, l, t, np.full((bh, bw, 3), (30, 90, 250), dtype=np.uint8), bx, by)
        out[~inside_y, :] = 0
        out[:, ~inside_x] = 0
        return out

    @staticmethod
    def _paste(out, l, t, img, x, y):
        h, w = out.shape[:2]
        ih, iw = img.shape[:2]
        x0, y0 = max(l, x), max(t, y)
        x1, y1 = min(l + w, x + iw), min(t + h, y + ih)
        if x0 < x1 and y0 < y1:
            out[y0 - t : y1 - t, x0 - l : x1 - l] = img[y0 - y : y1 - y, x0 - x : x1 - x]

    def virtual_rect(self) -> Rect:
        return 0, 0, self.W, self.H

    # ---------------------------------------------------------------- мышь
    def get_cursor(self):
        return self.pos

    def set_cursor(self, x, y):
        self.pos = (int(x), int(y))
        self._sets += 1
        if self.user_jerk_at is not None and self._sets >= self.user_jerk_at:
            self.user_jerk_at = None
            self.pos = (self.pos[0] + 60, self.pos[1] + 40)

    def mouse_down(self):
        self._down = True

    def mouse_up(self):
        if self._down:
            self._click(*self.pos)
        self._down = False

    def idle_seconds(self) -> float:
        return 1e9

    def window_title_at(self, x, y) -> str:
        return "Другое окно" if self.overlay and self._in(self.overlay, x, y) else "Wplace - Google Chrome"

    def window_at(self, x, y) -> int:
        return 2 if self.overlay and self._in(self.overlay, x, y) else 1

    @staticmethod
    def _in(rect, x, y) -> bool:
        rx, ry, rw, rh = rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def _regen(self):
        now = self.clock.now()
        self.charges = min(self.max_charges, self.charges + (now - self.charge_t) / 30.0)
        self.charge_t = now

    def _click(self, x, y):
        self._regen()
        if not self.palette_open:
            if self._in(self.PAINT_BTN, x, y):
                self.palette_open = True
                self.clicks.append((x, y, "open"))
            else:
                self.clicks.append((x, y, "noop"))
            return
        if self._in(self.PANEL, x, y):
            for cid, r in self.swatch_rects.items():
                if self._in(r, x, y):
                    self.selected = cid
                    self.clicks.append((x, y, "color"))
                    return
            if self._in(self.SUBMIT_BTN, x, y):
                self.clicks.append((x, y, "submit"))
                if self.captcha_next > 0:
                    self.captcha_next -= 1
                    self.captcha_active = True
                    return
                self.commit()
                return
            self.clicks.append((x, y, "panel"))
            return
        i, j = self.cell_of(x, y)
        if 0 <= i < self.cw and 0 <= j < self.ch and self.selected:
            if (i, j) in self.pending or self.charges >= 1:
                if (i, j) not in self.pending:
                    self.charges -= 1
                self.pending[(i, j)] = self.selected
                self.clicks.append((x, y, "pixel"))
                return
        self.clicks.append((x, y, "pixel-fail"))

    def commit(self):
        for (i, j), c in self.pending.items():
            self.board[j, i] = c
        self.pending.clear()
        self.palette_open = False
        self.captcha_active = False
        self.submits += 1
