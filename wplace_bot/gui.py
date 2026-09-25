"""Графический интерфейс (tkinter)."""
from __future__ import annotations

import logging
import os
import queue
import random
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from . import APP_NAME, __version__
from .app_logging import Paths, make_report
from .bot import Job, JobError, PaintBot, fmt_seconds
from .config import AppConfig, Behavior, save_config
from .geometry import Grid, detect_pitch, snap_to_center
from .humanize import ORDER_MODES, HumanMouse
from .image_prep import SKIP, color_counts, fit_size, load_image, prepare_target, render_target, zone_target
from .palette import BY_ID, COLORS, FREE_IDS, PREMIUM_IDS, ColorMatcher
from .palette_finder import Swatch, find_swatches
from .screen import MssScreen, sample_cells
from .winapi import default_input, key_pressed

log = logging.getLogger(__name__)

HOTKEY_CHOICES = [f"F{i}" for i in range(1, 13)] + ["PAUSE", "END", "HOME", "INSERT", "SCROLLLOCK"]

# (раздел, поле, подпись, тип, мин, макс, шаг)
BEHAVIOR_FIELDS = [
    ("Скорость и задержки", "click_delay_min", "Пауза между пикселями: от, сек", "float", 0.05, 60, 0.05),
    ("Скорость и задержки", "click_delay_max", "Пауза между пикселями: до, сек", "float", 0.05, 120, 0.05),
    ("Скорость и задержки", "mouse_speed", "Скорость мыши (1 = обычная)", "float", 0.2, 4, 0.1),
    ("Скорость и задержки", "click_jitter", "Разброс точки клика в пикселе (0–0.45)", "float", 0, 0.45, 0.05),
    ("Скорость и задержки", "overshoot_chance", "Шанс «промахнуться» и поправиться", "float", 0, 1, 0.01),
    ("Скорость и задержки", "hesitation_chance", "Шанс задуматься перед пикселем", "float", 0, 1, 0.01),
    ("Скорость и задержки", "hesitation_min", "Задуматься: от, сек", "float", 0, 60, 0.1),
    ("Скорость и задержки", "hesitation_max", "Задуматься: до, сек", "float", 0, 120, 0.1),
    ("Скорость и задержки", "rest_every_min", "Отдых каждые N пикселей: от", "int", 1, 10000, 1),
    ("Скорость и задержки", "rest_every_max", "Отдых каждые N пикселей: до", "int", 1, 10000, 1),
    ("Скорость и задержки", "rest_min", "Длительность отдыха: от, сек", "float", 0, 600, 0.5),
    ("Скорость и задержки", "rest_max", "Длительность отдыха: до, сек", "float", 0, 1200, 0.5),
    ("Заряды", "max_charges", "Максимум зарядов на аккаунте", "int", 1, 100000, 1),
    ("Заряды", "start_charges", "Зарядов сейчас (при старте)", "int", 0, 100000, 1),
    ("Заряды", "seconds_per_charge", "Секунд на восстановление 1 заряда", "float", 1, 3600, 1),
    ("Заряды", "random_batch", "Тратить случайное число зарядов за подход", "bool"),
    ("Заряды", "batch_min_fraction", "Минимум зарядов за подход (доля)", "float", 0.1, 1, 0.05),
    ("Заряды", "extra_wait_min", "Доп. ожидание между подходами: от, сек", "float", 0, 86400, 5),
    ("Заряды", "extra_wait_max", "Доп. ожидание между подходами: до, сек", "float", 0, 86400, 5),
    ("Проверки и безопасность", "order", "Порядок пикселей", "choice"),
    ("Проверки и безопасность", "verify_pixels", "Проверять каждый поставленный пиксель", "bool"),
    ("Проверки и безопасность", "color_tolerance", "Допуск цвета при проверке", "float", 1, 80, 1),
    ("Проверки и безопасность", "align_threshold", "Порог «карта сдвинулась» (0–1)", "float", 0, 1, 0.05),
    ("Проверки и безопасность", "idle_before_batch", "Перед подходом ждать бездействия мыши, сек", "float", 0, 600, 1),
    ("Проверки и безопасность", "pause_on_user_mouse", "Пауза, если я двигаю мышь", "bool"),
    ("Проверки и безопасность", "palette_closes_after_submit", "Проверять, что после «Paint» палитра закрылась", "bool"),
    ("Проверки и безопасность", "submit_timeout", "Ждать закрытия палитры, сек", "float", 2, 300, 1),
    ("Проверки и безопасность", "guard_mode", "Охранять рисунок после завершения", "bool"),
    ("Проверки и безопасность", "guard_interval_min", "Проверять рисунок раз в N минут", "float", 1, 1440, 1),
    ("Проверки и безопасность", "sound", "Звук при паузе и ошибке", "bool"),
    ("Проверки и безопасность", "minimize_on_start", "Сворачивать окно при старте", "bool"),
    ("Проверки и безопасность", "dry_run", "Пробный режим (мышь двигается, клики не делаются)", "bool"),
]

CALIB_POINTS = [
    ("origin", "Центр ЛЕВОГО ВЕРХНЕГО пикселя рисунка"),
    ("corner2", "Центр ПРАВОГО НИЖНЕГО пикселя (для зоны; для картинки — по желанию)"),
    ("paint_button", "Кнопка «Paint» внизу экрана (открывает палитру)"),
    ("submit_button", "Кнопка подтверждения «Paint» при открытой палитре"),
    ("palette_tl", "Палитра: левый верхний угол панели с цветами"),
    ("palette_br", "Палитра: правый нижний угол панели с цветами"),
]


class TkLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__(logging.INFO)
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.q.put(("log", self.format(record), record.levelno))
        except Exception:  # noqa: BLE001
            pass


class HotkeyThread(threading.Thread):
    def __init__(self, get_keys, post):
        super().__init__(name="hotkeys", daemon=True)
        self.get_keys = get_keys
        self.post = post
        self.stop_evt = threading.Event()

    def run(self):
        prev: dict[str, bool] = {}
        while not self.stop_evt.wait(0.03):
            try:
                for action, key in self.get_keys().items():
                    down = key_pressed(key)
                    if down and not prev.get(action):
                        self.post(("hotkey", action))
                    prev[action] = down
            except Exception:  # noqa: BLE001
                log.debug("Ошибка опроса горячих клавиш", exc_info=True)
                time.sleep(0.5)


def contrast_fg(rgb) -> str:
    r, g, b = rgb
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#ffffff"


def checker(size: tuple[int, int], cell: int = 8) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, (235, 235, 235))
    d = ImageDraw.Draw(img)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if (x // cell + y // cell) % 2:
                d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(205, 205, 205))
    return img


class App:
    def __init__(self, root: tk.Tk, paths: Paths, cfg: AppConfig, log_path: str):
        self.root = root
        self.paths = paths
        self.cfg = cfg
        self.log_path = log_path
        self.screen = MssScreen()
        self.inp = default_input()
        self.events: queue.Queue = queue.Queue()
        self.bot: PaintBot | None = None
        self.bot_thread: threading.Thread | None = None
        self.source_img: Image.Image | None = None
        self.target: np.ndarray | None = None
        self.capture_key: str | None = None
        self._preview_photo = None
        self._busy = False

        logging.getLogger().addHandler(TkLogHandler(self.events))
        root.title(f"{APP_NAME} {__version__}")
        root.geometry("1040x760")
        root.minsize(900, 640)
        root.report_callback_exception = self._tk_exception
        self._init_style()
        self._build()
        self._load_into_ui()

        self.hotkeys = HotkeyThread(
            lambda: {
                "capture": self.cfg.hotkeys.capture,
                "pause": self.cfg.hotkeys.pause,
                "stop": self.cfg.hotkeys.stop,
            },
            self.events.put,
        )
        self.hotkeys.start()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(100, self._poll)
        if self.cfg.image.image_path and os.path.exists(self.cfg.image.image_path):
            root.after(200, lambda: self._load_image(self.cfg.image.image_path, keep_size=True))
        else:
            root.after(200, self._refresh_preview)

    # ================================================================== UI
    def _init_style(self):
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Big.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")
        style.configure("Warn.TLabel", foreground="#b00020", font=("Segoe UI", 10, "bold"))
        style.configure("Ok.TLabel", foreground="#1b7a1b")

    def _build(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self.nb = nb
        self.tab_img = ttk.Frame(nb, padding=8)
        self.tab_cal = ttk.Frame(nb, padding=8)
        self.tab_beh = ttk.Frame(nb, padding=8)
        self.tab_run = ttk.Frame(nb, padding=8)
        nb.add(self.tab_img, text="1. Что рисовать")
        nb.add(self.tab_cal, text="2. Калибровка")
        nb.add(self.tab_beh, text="3. Поведение")
        nb.add(self.tab_run, text="4. Запуск и лог")
        self._build_image_tab()
        self._build_calib_tab()
        self._build_behavior_tab()
        self._build_run_tab()

    # ---------------------------------------------------------------- вкладка 1
    def _build_image_tab(self):
        t = self.tab_img
        left = ttk.Frame(t)
        left.pack(side="left", fill="y")
        right = ttk.Frame(t)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.v_mode = tk.StringVar(value="image")
        mf = ttk.LabelFrame(left, text="Режим", padding=6)
        mf.pack(fill="x")
        ttk.Radiobutton(mf, text="Нарисовать картинку", variable=self.v_mode, value="image",
                        command=self._refresh_preview).pack(anchor="w")
        ttk.Radiobutton(mf, text="Залить выделенную зону одним цветом", variable=self.v_mode, value="zone",
                        command=self._refresh_preview).pack(anchor="w")

        f = ttk.LabelFrame(left, text="Картинка", padding=6)
        f.pack(fill="x", pady=(8, 0))
        ttk.Button(f, text="Выбрать картинку…", command=self._choose_image).grid(row=0, column=0, sticky="w")
        self.l_path = ttk.Label(f, text="не выбрана", width=38, style="Hint.TLabel")
        self.l_path.grid(row=0, column=1, columnspan=3, sticky="w", padx=6)

        ttk.Label(f, text="Ширина, пикс.:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.v_w = tk.IntVar(value=0)
        self.v_h = tk.IntVar(value=0)
        sw = ttk.Spinbox(f, from_=1, to=2000, textvariable=self.v_w, width=7, command=lambda: self._size_changed("w"))
        sw.grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(f, text="Высота:").grid(row=1, column=2, sticky="e", pady=(6, 0))
        sh = ttk.Spinbox(f, from_=1, to=2000, textvariable=self.v_h, width=7, command=lambda: self._size_changed("h"))
        sh.grid(row=1, column=3, sticky="w", pady=(6, 0))
        sw.bind("<Return>", lambda e: self._size_changed("w"))
        sw.bind("<FocusOut>", lambda e: self._size_changed("w"))
        sh.bind("<Return>", lambda e: self._size_changed("h"))
        sh.bind("<FocusOut>", lambda e: self._size_changed("h"))
        self.v_keep = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="сохранять пропорции", variable=self.v_keep).grid(row=2, column=1, columnspan=3, sticky="w")
        ttk.Button(f, text="Исходный размер", command=self._orig_size).grid(row=2, column=0, sticky="w")

        ttk.Label(f, text="Масштабирование:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.v_resample = tk.StringVar()
        self.resample_map = {"Пиксель-арт (чёткие пиксели)": "nearest", "Фото (сглаживание)": "smooth"}
        cb = ttk.Combobox(f, textvariable=self.v_resample, values=list(self.resample_map), state="readonly", width=28)
        cb.grid(row=3, column=1, columnspan=3, sticky="w", pady=(6, 0))
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_preview())
        self.v_dither = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Дизеринг (смешивать цвета точками)", variable=self.v_dither,
                        command=self._refresh_preview).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Button(f, text="Мои платные цвета…", command=self._premium_dialog).grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.l_premium = ttk.Label(f, text="", style="Hint.TLabel")
        self.l_premium.grid(row=5, column=1, columnspan=3, sticky="w", padx=6, pady=(6, 0))
        ttk.Button(f, text="Обновить предпросмотр", command=self._refresh_preview).grid(row=6, column=0, sticky="w", pady=(8, 0))

        z = ttk.LabelFrame(left, text="Заливка зоны", padding=6)
        z.pack(fill="x", pady=(8, 0))
        ttk.Label(z, text="Цвет:").grid(row=0, column=0, sticky="w")
        self.v_zone_color = tk.StringVar()
        self.color_labels = {c.label: c.id for c in COLORS}
        zc = ttk.Combobox(z, textvariable=self.v_zone_color, values=list(self.color_labels), state="readonly", width=34)
        zc.grid(row=0, column=1, sticky="w", padx=4)
        zc.bind("<<ComboboxSelected>>", lambda e: self._refresh_preview())
        self.l_zone_swatch = tk.Label(z, text="    ", relief="solid", bd=1)
        self.l_zone_swatch.grid(row=0, column=2, padx=4)
        ttk.Label(z, text="Размер зоны задаётся двумя углами\nна вкладке «Калибровка».", style="Hint.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        ttk.Label(right, text="Предпросмотр (в цветах палитры wplace):").pack(anchor="w")
        self.preview = tk.Canvas(right, width=420, height=340, bg="#dddddd", highlightthickness=0)
        self.preview.pack(anchor="w", pady=4)
        self.l_info = ttk.Label(right, text="", justify="left", wraplength=440)
        self.l_info.pack(anchor="w")
        ttk.Label(right, text="Цвета рисунка:").pack(anchor="w", pady=(6, 0))
        lbf = ttk.Frame(right)
        lbf.pack(anchor="w", fill="both", expand=True)
        self.lb_colors = tk.Listbox(lbf, height=8, width=44, activestyle="none")
        lsb = ttk.Scrollbar(lbf, command=self.lb_colors.yview)
        self.lb_colors.configure(yscrollcommand=lsb.set)
        self.lb_colors.pack(side="left", fill="both", expand=True)
        lsb.pack(side="left", fill="y")

    # ---------------------------------------------------------------- вкладка 2
    def _build_calib_tab(self):
        t = self.tab_cal
        hint = (
            "1) Откройте wplace.live в браузере, войдите в аккаунт, приблизьте карту так, чтобы весь рисунок был на экране "
            "и пиксели были крупными (от 6 px).\n"
            "2) Для каждой строки нажмите «Указать», наведите мышь на нужное место в браузере и нажмите {cap}.\n"
            "3) Откройте палитру (кнопка Paint) и нажмите «Найти цвета». Потом «Проверить калибровку».\n"
            "После калибровки НЕ двигайте и не масштабируйте карту."
        ).format(cap=self.cfg.hotkeys.capture)
        self.l_cal_hint = ttk.Label(t, text=hint, justify="left", wraplength=980)
        self.l_cal_hint.pack(anchor="w")

        g = ttk.Frame(t)
        g.pack(fill="x", pady=8)
        self.cal_labels: dict[str, ttk.Label] = {}
        for row, (key, text) in enumerate(CALIB_POINTS):
            ttk.Label(g, text=text).grid(row=row, column=0, sticky="w", pady=2)
            lab = ttk.Label(g, text="—", width=16)
            lab.grid(row=row, column=1, sticky="w", padx=8)
            self.cal_labels[key] = lab
            ttk.Button(g, text="Указать", command=lambda k=key: self._begin_capture(k)).grid(row=row, column=2, padx=2)
            if key == "corner2":
                ttk.Button(g, text="Сбросить", command=self._clear_corner2).grid(row=row, column=3, padx=2)

        pf = ttk.Frame(t)
        pf.pack(fill="x", pady=(4, 0))
        ttk.Label(pf, text="Размер пикселя холста на экране, px:").pack(side="left")
        self.v_pitch = tk.DoubleVar(value=0.0)
        ttk.Spinbox(pf, from_=0, to=200, increment=0.01, textvariable=self.v_pitch, width=9).pack(side="left", padx=6)
        ttk.Button(pf, text="Определить автоматически", command=self._auto_pitch).pack(side="left", padx=2)
        ttk.Button(pf, text="Рассчитать по двум углам", command=self._pitch_from_corners).pack(side="left", padx=2)
        self.l_pitch = ttk.Label(pf, text="", style="Hint.TLabel")
        self.l_pitch.pack(side="left", padx=8)

        sf = ttk.LabelFrame(t, text="Цвета палитры", padding=6)
        sf.pack(fill="x", pady=8)
        ttk.Button(sf, text="Найти цвета в палитре (палитра должна быть открыта)", command=self._find_palette).grid(
            row=0, column=0, sticky="w")
        self.l_swatches = ttk.Label(sf, text="", justify="left", wraplength=560)
        self.l_swatches.grid(row=0, column=1, columnspan=3, sticky="w", padx=8)
        ttk.Label(sf, text="Указать цвет вручную:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.v_manual_color = tk.StringVar()
        ttk.Combobox(sf, textvariable=self.v_manual_color, values=list(self.color_labels), state="readonly",
                     width=34).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Button(sf, text="Указать", command=self._begin_manual_swatch).grid(row=1, column=2, sticky="w", pady=(6, 0))

        bf = ttk.Frame(t)
        bf.pack(fill="x", pady=4)
        ttk.Button(bf, text="Проверить калибровку (снимок с сеткой)", command=self._check_calibration).pack(side="left")
        ttk.Button(bf, text="Показать углы рисунка мышью", command=self._show_corners).pack(side="left", padx=8)

        self.l_capture = ttk.Label(t, text="", style="Warn.TLabel", wraplength=980)
        self.l_capture.pack(anchor="w", pady=(8, 0))

    # ---------------------------------------------------------------- вкладка 3
    def _build_behavior_tab(self):
        t = self.tab_beh
        cols = [ttk.Frame(t), ttk.Frame(t)]
        cols[0].pack(side="left", fill="both", expand=True)
        cols[1].pack(side="left", fill="both", expand=True, padx=(10, 0))
        section_col = {"Скорость и задержки": 0, "Заряды": 1, "Проверки и безопасность": 1}
        frames: dict[str, ttk.LabelFrame] = {}
        self.bvars: dict[str, tuple[tk.Variable, str]] = {}
        self.order_map = {v: k for k, v in ORDER_MODES.items()}
        rows: dict[str, int] = {}
        for spec in BEHAVIOR_FIELDS:
            section, name, label, kind = spec[:4]
            if section not in frames:
                fr = ttk.LabelFrame(cols[section_col[section]], text=section, padding=6)
                fr.pack(fill="x", pady=(0, 8))
                frames[section] = fr
                rows[section] = 0
            fr = frames[section]
            r = rows[section]
            rows[section] += 1
            if kind == "bool":
                var = tk.BooleanVar()
                ttk.Checkbutton(fr, text=label, variable=var).grid(row=r, column=0, columnspan=2, sticky="w")
            elif kind == "choice":
                var = tk.StringVar()
                row = ttk.Frame(fr)
                row.grid(row=r, column=0, columnspan=2, sticky="w")
                ttk.Label(row, text=label).pack(side="left")
                ttk.Combobox(row, textvariable=var, values=list(self.order_map), state="readonly", width=36).pack(
                    side="left", padx=4)
            else:
                lo, hi, step = spec[4:7]
                var = tk.DoubleVar() if kind == "float" else tk.IntVar()
                ttk.Label(fr, text=label).grid(row=r, column=0, sticky="w")
                ttk.Spinbox(fr, from_=lo, to=hi, increment=step, textvariable=var, width=9).grid(
                    row=r, column=1, sticky="w", padx=4, pady=1)
            self.bvars[name] = (var, kind)

        hk = ttk.LabelFrame(cols[0], text="Горячие клавиши", padding=6)
        hk.pack(fill="x")
        self.v_hk = {}
        for r, (name, label) in enumerate(
            (("capture", "Запомнить позицию курсора"), ("pause", "Пауза / продолжить"), ("stop", "Стоп"))
        ):
            ttk.Label(hk, text=label).grid(row=r, column=0, sticky="w")
            v = tk.StringVar()
            cb = ttk.Combobox(hk, textvariable=v, values=HOTKEY_CHOICES, width=12)
            cb.grid(row=r, column=1, sticky="w", padx=4)
            cb.bind("<<ComboboxSelected>>", lambda e: self._save())
            self.v_hk[name] = v
        ttk.Label(hk, text="Аварийная остановка: увести курсор в левый верхний угол экрана.", style="Hint.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(cols[0], text="Сбросить поведение к стандартному", command=self._reset_behavior).pack(anchor="w", pady=8)

    # ---------------------------------------------------------------- вкладка 4
    def _build_run_tab(self):
        t = self.tab_run
        bf = ttk.Frame(t)
        bf.pack(fill="x")
        self.b_start = ttk.Button(bf, text="▶  Старт", command=self._start)
        self.b_start.pack(side="left")
        self.b_pause = ttk.Button(bf, text="⏸  Пауза", command=self._toggle_pause, state="disabled")
        self.b_pause.pack(side="left", padx=6)
        self.b_stop = ttk.Button(bf, text="■  Стоп", command=self._stop, state="disabled")
        self.b_stop.pack(side="left")
        ttk.Button(bf, text="Сохранить отчёт для разработчика", command=self._report).pack(side="right")
        ttk.Button(bf, text="Открыть папку с логами", command=lambda: self._open_path(self.paths.logs)).pack(
            side="right", padx=6)

        self.l_state = ttk.Label(t, text="Остановлен", style="Big.TLabel")
        self.l_state.pack(anchor="w", pady=(10, 0))
        self.l_alert = ttk.Label(t, text="", style="Warn.TLabel", wraplength=980, justify="left")
        self.l_alert.pack(anchor="w")
        self.pb = ttk.Progressbar(t, maximum=100)
        self.pb.pack(fill="x", pady=6)
        self.l_progress = ttk.Label(t, text="")
        self.l_progress.pack(anchor="w")
        self.l_countdown = ttk.Label(t, text="")
        self.l_countdown.pack(anchor="w")

        lf = ttk.Frame(t)
        lf.pack(fill="both", expand=True, pady=(8, 0))
        self.txt_log = tk.Text(lf, height=20, wrap="word", state="disabled", font=("Consolas", 9))
        sb = ttk.Scrollbar(lf, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)
        self.txt_log.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.txt_log.tag_configure("warn", foreground="#a15c00")
        self.txt_log.tag_configure("err", foreground="#b00020")

    # ================================================================== конфиг <-> UI
    def _load_into_ui(self):
        im = self.cfg.image
        self.v_mode.set(im.mode if im.mode in ("image", "zone") else "image")
        self.v_w.set(im.width)
        self.v_h.set(im.height)
        self.v_keep.set(im.keep_aspect)
        inv = {v: k for k, v in self.resample_map.items()}
        self.v_resample.set(inv.get(im.resample, next(iter(self.resample_map))))
        self.v_dither.set(im.dither)
        self.v_zone_color.set(BY_ID.get(im.zone_color, BY_ID[1]).label)
        self._update_premium_label()
        self.l_path.configure(text=os.path.basename(im.image_path) if im.image_path else "не выбрана")

        b = self.cfg.behavior
        for name, (var, kind) in self.bvars.items():
            val = getattr(b, name)
            if kind == "choice":
                var.set(ORDER_MODES.get(val, ORDER_MODES["colors_nearest"]))
            else:
                var.set(val)
        for name, v in self.v_hk.items():
            v.set(getattr(self.cfg.hotkeys, name))
        self.v_pitch.set(round(self.cfg.calib.pitch, 3))
        self._refresh_calib_labels()

    def _collect(self) -> AppConfig:
        """Считать значения из интерфейса в self.cfg. Бросает ValueError с понятным текстом."""
        im = self.cfg.image
        im.mode = self.v_mode.get()
        im.keep_aspect = bool(self.v_keep.get())
        im.resample = self.resample_map.get(self.v_resample.get(), "nearest")
        im.dither = bool(self.v_dither.get())
        im.zone_color = self.color_labels.get(self.v_zone_color.get(), 1)
        try:
            im.width = int(self.v_w.get())
            im.height = int(self.v_h.get())
        except (tk.TclError, ValueError):
            raise ValueError("Размер картинки указан неверно")
        b = self.cfg.behavior
        for name, (var, kind) in self.bvars.items():
            label = next(s[2] for s in BEHAVIOR_FIELDS if s[1] == name)
            try:
                if kind == "choice":
                    setattr(b, name, self.order_map.get(var.get(), "colors_nearest"))
                elif kind == "bool":
                    setattr(b, name, bool(var.get()))
                elif kind == "int":
                    setattr(b, name, int(var.get()))
                else:
                    setattr(b, name, float(var.get()))
            except (tk.TclError, ValueError):
                raise ValueError(f"Неверное значение в поле «{label}»")
        if b.click_delay_max < b.click_delay_min:
            raise ValueError("Пауза между пикселями: «до» меньше, чем «от»")
        if b.extra_wait_max < b.extra_wait_min:
            raise ValueError("Доп. ожидание: «до» меньше, чем «от»")
        for name, v in self.v_hk.items():
            key = v.get().strip().upper() or getattr(self.cfg.hotkeys, name)
            setattr(self.cfg.hotkeys, name, key)
        try:
            self.cfg.calib.pitch = float(self.v_pitch.get())
        except (tk.TclError, ValueError):
            raise ValueError("Размер пикселя указан неверно")
        return self.cfg

    def _save(self, quiet: bool = True):
        try:
            self._collect()
        except ValueError as e:
            if not quiet:
                messagebox.showerror(APP_NAME, str(e))
            return False
        try:
            save_config(self.cfg, self.paths.config)
        except OSError:
            log.exception("Не удалось сохранить настройки")
        return True

    def _reset_behavior(self):
        self.cfg.behavior = Behavior()
        self._load_into_ui()
        log.info("Настройки поведения сброшены к стандартным")

    # ================================================================== картинка
    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="Выберите картинку",
            filetypes=[("Картинки", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"), ("Все файлы", "*.*")],
        )
        if path:
            self._load_image(path, keep_size=False)

    def _load_image(self, path: str, keep_size: bool):
        try:
            self.source_img = load_image(path)
        except Exception as e:  # noqa: BLE001
            log.exception("Не удалось открыть картинку %s", path)
            messagebox.showerror(APP_NAME, f"Не удалось открыть картинку:\n{e}")
            return
        self.cfg.image.image_path = path
        self.l_path.configure(text=os.path.basename(path))
        if not keep_size or self.v_w.get() <= 0 or self.v_h.get() <= 0:
            w, h = self.source_img.size
            if max(w, h) > 200:
                k = 200 / max(w, h)
                w, h = max(1, round(w * k)), max(1, round(h * k))
            self.v_w.set(w)
            self.v_h.set(h)
        if not keep_size:  # пользователь сам выбрал файл — значит, рисуем картинку
            self.v_mode.set("image")
        self._refresh_preview()

    def _orig_size(self):
        if self.source_img:
            self.v_w.set(self.source_img.width)
            self.v_h.set(self.source_img.height)
            self._refresh_preview()

    def _size_changed(self, which: str):
        if not self.source_img:
            return
        try:
            w, h = int(self.v_w.get()), int(self.v_h.get())
        except (tk.TclError, ValueError):
            return
        if self.v_keep.get():
            ow, oh = self.source_img.size
            if which == "w" and w > 0:
                self.v_h.set(fit_size(ow, oh, w, 0, True)[1])
            elif which == "h" and h > 0:
                self.v_w.set(fit_size(ow, oh, 0, h, True)[0])
        self._refresh_preview()

    def _allowed_ids(self) -> list[int]:
        return FREE_IDS + [i for i in self.cfg.image.premium_ids if i in BY_ID]

    def _update_premium_label(self):
        n = len(self.cfg.image.premium_ids)
        self.l_premium.configure(text=f"используются {len(FREE_IDS)} бесплатных + {n} платных")

    def _premium_dialog(self):
        top = tk.Toplevel(self.root)
        top.title("Платные цвета")
        top.transient(self.root)
        ttk.Label(top, text="Отметьте платные цвета, которые у вас куплены:").grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=6)
        vars_ = {}
        for k, cid in enumerate(PREMIUM_IDS):
            c = BY_ID[cid]
            v = tk.BooleanVar(value=cid in self.cfg.image.premium_ids)
            vars_[cid] = v
            r, col = divmod(k, 2)
            fr = ttk.Frame(top)
            fr.grid(row=r + 1, column=col, sticky="w", padx=8)
            tk.Label(fr, text="   ", bg=c.hex, relief="solid", bd=1).pack(side="left")
            ttk.Checkbutton(fr, text=c.label, variable=v).pack(side="left", padx=4)

        def ok():
            self.cfg.image.premium_ids = [cid for cid, v in vars_.items() if v.get()]
            self._update_premium_label()
            top.destroy()
            self._refresh_preview()

        ttk.Button(top, text="Готово", command=ok).grid(row=20, column=0, columnspan=2, pady=8)
        top.grab_set()

    def _refresh_preview(self):
        try:
            self._collect()
        except ValueError:
            pass
        cid = self.cfg.image.zone_color
        self.l_zone_swatch.configure(bg=BY_ID[cid].hex if cid in BY_ID else "#ffffff")
        self.preview.delete("all")
        self.lb_colors.delete(0, "end")
        target = None
        try:
            if self.cfg.image.mode == "zone":
                w, h = self._zone_size()
                target = zone_target(w, h, cid) if w and h else None
                if target is None:
                    self.l_info.configure(text="Зона ещё не задана: укажите два угла на вкладке «Калибровка».")
            else:
                self.target = self._prepare_image_target()
                target = self.target
                if target is None:
                    self.l_info.configure(text="Картинка не выбрана.")
        except Exception as e:  # noqa: BLE001
            log.exception("Ошибка подготовки предпросмотра")
            self.l_info.configure(text=f"Ошибка: {e}")
            return
        if target is None:
            return
        img = render_target(target)
        h, w = target.shape
        if w <= 420 and h <= 340:
            scale = max(1, min(420 // w, 340 // h))
        else:
            scale = min(420 / w, 340 / h)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        shown = img.resize(size, Image.Resampling.NEAREST)
        bg = checker(size)
        bg.paste(shown, (0, 0), shown)
        self._preview_photo = ImageTk.PhotoImage(bg)
        self.preview.create_image(0, 0, anchor="nw", image=self._preview_photo)
        counts = color_counts(target)
        total = sum(c for _, c in counts)
        self.l_info.configure(
            text=f"Размер: {w}×{h}, пикселей для рисования: {total}, цветов: {len(counts)}\n"
            f"При 1 заряде в {self.cfg.behavior.seconds_per_charge:.0f} с это ≈ "
            f"{fmt_seconds(total * self.cfg.behavior.seconds_per_charge)} (без учёта уже нарисованного)."
        )
        for k, (cid, cnt) in enumerate(counts):
            c = BY_ID[cid]
            self.lb_colors.insert("end", f"  {c.label} — {cnt}")
            self.lb_colors.itemconfig(k, background=c.hex, foreground=contrast_fg(c.rgb))

    def _prepare_image_target(self) -> np.ndarray | None:
        if self.source_img is None:
            return None
        im = self.cfg.image
        w, h = fit_size(self.source_img.width, self.source_img.height, im.width, im.height, False)
        if w * h > 1_000_000:
            raise ValueError("Слишком большой размер (больше 1 млн пикселей)")
        return prepare_target(self.source_img, w, h, self._allowed_ids(), im.resample, im.dither, im.alpha_threshold)

    def _zone_size(self) -> tuple[int, int]:
        c = self.cfg.calib
        if not c.origin or not c.corner2 or c.pitch <= 0:
            return 0, 0
        w = round(abs(c.corner2[0] - c.origin[0]) / c.pitch) + 1
        h = round(abs(c.corner2[1] - c.origin[1]) / c.pitch) + 1
        return int(w), int(h)

    # ================================================================== калибровка
    def _fmt_point(self, p) -> str:
        return "—" if not p else f"({p[0]:.0f}, {p[1]:.0f})" if isinstance(p[0], int) else f"({p[0]:.1f}, {p[1]:.1f})"

    def _refresh_calib_labels(self):
        c = self.cfg.calib
        for key, _ in CALIB_POINTS:
            self.cal_labels[key].configure(text=self._fmt_point(getattr(c, key)))
        n = len(c.swatches)
        needed = self._needed_colors()
        missing = [BY_ID[i].name_ru for i in needed if str(i) not in c.swatches]
        text = f"Найдено кнопок цветов: {n} из {len(COLORS)}."
        if needed:
            text += " Для рисунка нужны все найдены." if not missing else f" Для рисунка НЕ найдены: {', '.join(missing)}."
        self.l_swatches.configure(text=text, style="Ok.TLabel" if needed and not missing else "TLabel")

    def _needed_colors(self) -> list[int]:
        if self.v_mode.get() == "zone":
            return [self.color_labels.get(self.v_zone_color.get(), 1)]
        if self.target is not None:
            return [int(c) for c in np.unique(self.target) if c != SKIP]
        return []

    def _begin_capture(self, key: str):
        self.capture_key = key
        text = dict(CALIB_POINTS).get(key, key)
        if key.startswith("swatch:"):
            text = f"Кнопка цвета {BY_ID[int(key[7:])].label}"
        self.l_capture.configure(
            text=f"Наведите курсор: {text} — и нажмите {self.cfg.hotkeys.capture}. "
            "(Нажмите «Указать» у другой строки, чтобы переключиться.)"
        )

    def _begin_manual_swatch(self):
        cid = self.color_labels.get(self.v_manual_color.get())
        if not cid:
            messagebox.showinfo(APP_NAME, "Сначала выберите цвет в списке.")
            return
        self._begin_capture(f"swatch:{cid}")

    def _do_capture(self):
        if not self.capture_key:
            return
        x, y = self.inp.get_cursor()
        key = self.capture_key
        self.capture_key = None
        c = self.cfg.calib
        if key.startswith("swatch:"):
            cid = int(key[7:])
            sizes = [v[2] for v in c.swatches.values()] or [16]
            s = int(np.median(sizes))
            c.swatches[str(cid)] = [x, y, s, s]
            log.info("Кнопка цвета %s указана вручную: (%d, %d)", BY_ID[cid].label, x, y)
        else:
            setattr(c, key, [x, y])
            log.info("Калибровка: %s = (%d, %d)", key, x, y)
            if key in ("origin", "corner2"):
                self._snap_points()
        self.l_capture.configure(text=f"Запомнено: ({x}, {y})")
        self._refresh_calib_labels()
        self._refresh_preview()
        self._save()

    def _clear_corner2(self):
        self.cfg.calib.corner2 = None
        self._refresh_calib_labels()
        self._refresh_preview()

    def _snap_points(self):
        """Если сетка известна по автоопределению — подвинуть точки в центры пикселей."""
        ph = getattr(self, "_phase", None)
        c = self.cfg.calib
        if not ph or c.pitch <= 0 or abs(ph[2] - c.pitch) > 1e-6:
            return
        for key in ("origin", "corner2"):
            p = getattr(c, key)
            if p:
                setattr(c, key, [round(snap_to_center(p[0], ph[0], c.pitch), 2),
                                 round(snap_to_center(p[1], ph[1], c.pitch), 2)])

    def _grab_hidden(self, rect):
        """Снимок экрана со спрятанным окном программы."""
        self.root.withdraw()
        for w in self.root.winfo_children():
            if isinstance(w, tk.Toplevel):
                w.withdraw()
        self.root.update()
        time.sleep(0.4)
        try:
            return self.screen.grab(rect)
        finally:
            self.root.deiconify()
            for w in self.root.winfo_children():
                if isinstance(w, tk.Toplevel):
                    w.deiconify()

    def _auto_pitch(self):
        c = self.cfg.calib
        if not c.origin:
            messagebox.showinfo(APP_NAME, "Сначала укажите левый верхний пиксель рисунка.")
            return
        vl, vt, vw, vh = self.screen.virtual_rect()
        half = 400
        left = max(vl, int(c.origin[0]) - half)
        top = max(vt, int(c.origin[1]) - half)
        right = min(vl + vw, int(c.origin[0]) + half)
        bottom = min(vt + vh, int(c.origin[1]) + half)
        if c.palette_tl and c.palette_br and c.palette_tl[1] > c.origin[1]:
            bottom = min(bottom, min(c.palette_tl[1], c.palette_br[1]) - 2)
        rect = (left, top, right - left, bottom - top)
        img = self._grab_hidden(rect)
        self._save_debug_image(img, "pitch_detect")
        res = detect_pitch(img)
        if not res.ok:
            self.l_pitch.configure(text=f"не уверен (уверенность {res.confidence:.2f})")
            messagebox.showwarning(
                APP_NAME,
                "Не удалось уверенно определить размер пикселя.\n\n"
                "Автоопределению нужны уже нарисованные пиксели рядом с рисунком. Можно:\n"
                "• указать правый нижний пиксель и нажать «Рассчитать по двум углам»;\n"
                "• ввести размер вручную и проверить его кнопкой «Проверить калибровку».\n\n"
                f"Найденное значение: {res.pitch:.3f} px (уверенность {res.confidence:.2f}).",
            )
            return
        c.pitch = round(res.pitch, 4)
        self.v_pitch.set(c.pitch)
        self._phase = (left + res.phase_x, top + res.phase_y, c.pitch)
        self._snap_points()
        self.l_pitch.configure(text=f"найдено {res.pitch:.3f} px (уверенность {res.confidence:.2f})")
        self._refresh_calib_labels()
        self._refresh_preview()
        self._save()

    def _pitch_from_corners(self):
        try:
            self._collect()
        except ValueError as e:
            messagebox.showerror(APP_NAME, str(e))
            return
        c = self.cfg.calib
        if not c.origin or not c.corner2:
            messagebox.showinfo(APP_NAME, "Укажите левый верхний и правый нижний пиксели рисунка.")
            return
        if self.v_mode.get() != "image" or self.target is None:
            messagebox.showinfo(APP_NAME, "Расчёт по двум углам работает для картинки: размер берётся из картинки.")
            return
        h, w = self.target.shape
        vals = []
        if w > 1:
            vals.append(abs(c.corner2[0] - c.origin[0]) / (w - 1))
        if h > 1:
            vals.append(abs(c.corner2[1] - c.origin[1]) / (h - 1))
        if not vals:
            return
        if len(vals) == 2 and abs(vals[0] - vals[1]) > 0.04 * max(vals):
            messagebox.showwarning(
                APP_NAME,
                f"По горизонтали выходит {vals[0]:.3f} px, по вертикали {vals[1]:.3f} px — сильно отличаются. "
                "Проверьте, что углы указаны на правильных пикселях и размер картинки верный.",
            )
        c.pitch = round(sum(vals) / len(vals), 4)
        self.v_pitch.set(c.pitch)
        self.l_pitch.configure(text=f"по углам: {c.pitch:.3f} px")
        log.info("Размер пикселя по двум углам: %s -> %.4f", vals, c.pitch)
        self._save()

    def _palette_rect(self):
        c = self.cfg.calib
        if not c.palette_tl or not c.palette_br:
            return None
        l, r = sorted((int(c.palette_tl[0]), int(c.palette_br[0])))
        t, b = sorted((int(c.palette_tl[1]), int(c.palette_br[1])))
        return l, t, max(1, r - l), max(1, b - t)

    def _find_palette(self):
        rect = self._palette_rect()
        if not rect:
            messagebox.showinfo(APP_NAME, "Сначала укажите левый верхний и правый нижний углы палитры.")
            return
        img = self._grab_hidden(rect)
        found = find_swatches(img, rect)
        self._save_debug_image(img, "palette", marks=[(s.x - rect[0], s.y - rect[1]) for s in found.values()])
        if len(found) < 5:
            messagebox.showwarning(
                APP_NAME,
                f"Найдено только {len(found)} цветов. Откройте палитру (кнопка Paint) и проверьте, что углы "
                "палитры указаны верно, затем попробуйте снова.",
            )
        keep_manual = {k: v for k, v in self.cfg.calib.swatches.items() if int(k) not in found}
        self.cfg.calib.swatches = {str(k): v.to_list() for k, v in found.items()}
        if keep_manual and found:
            # Старые ручные точки оставляем для цветов, которые автопоиск не нашёл.
            self.cfg.calib.swatches.update(keep_manual)
        self._refresh_calib_labels()
        self._save()

    def _save_debug_image(self, img: np.ndarray, name: str, marks=()):
        try:
            pic = Image.fromarray(img).convert("RGB")
            d = ImageDraw.Draw(pic)
            for x, y in marks:
                d.line([x - 4, y, x + 4, y], fill=(255, 0, 0))
                d.line([x, y - 4, x, y + 4], fill=(255, 0, 0))
            path = os.path.join(self.paths.screens, f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.png")
            pic.save(path)
            log.debug("Снимок сохранён: %s", path)
        except Exception:  # noqa: BLE001
            log.warning("Не удалось сохранить снимок", exc_info=True)

    def _build_job(self) -> Job:
        self._collect()
        c = self.cfg.calib
        if not c.origin:
            raise JobError("Не указан левый верхний пиксель рисунка (вкладка «Калибровка»).")
        if c.pitch <= 0:
            raise JobError("Не задан размер пикселя на экране (вкладка «Калибровка»).")
        if not c.paint_button or not c.submit_button:
            raise JobError("Не указаны кнопки «Paint» (вкладка «Калибровка»).")
        prect = self._palette_rect()
        if not prect:
            raise JobError("Не указаны углы палитры (вкладка «Калибровка»).")
        if not c.swatches:
            raise JobError("Не найдены цвета палитры: нажмите «Найти цвета в палитре».")
        x0, y0 = c.origin
        if self.cfg.image.mode == "zone":
            if not c.corner2:
                raise JobError("Для заливки зоны укажите правый нижний пиксель (вкладка «Калибровка»).")
            w, h = self._zone_size()
            x0, y0 = min(c.origin[0], c.corner2[0]), min(c.origin[1], c.corner2[1])
            target = zone_target(w, h, self.cfg.image.zone_color)
        else:
            self.target = self._prepare_image_target()
            if self.target is None:
                raise JobError("Не выбрана картинка (вкладка «Что рисовать»).")
            target = self.target
        swatches = {int(k): Swatch.from_list(v) for k, v in c.swatches.items()}
        return Job(
            grid=Grid(float(x0), float(y0), float(c.pitch)),
            target=target,
            swatches=swatches,
            paint_button=(int(c.paint_button[0]), int(c.paint_button[1])),
            submit_button=(int(c.submit_button[0]), int(c.submit_button[1])),
            palette_rect=prect,
        )

    def _check_calibration(self):
        try:
            job = self._build_job()
        except (JobError, ValueError) as e:
            messagebox.showerror(APP_NAME, str(e))
            return
        CalibrationView(self, job)

    def _show_corners(self):
        try:
            job = self._build_job()
        except (JobError, ValueError) as e:
            messagebox.showerror(APP_NAME, str(e))
            return
        pts = [job.grid.center(0, 0), job.grid.center(job.width - 1, 0),
               job.grid.center(job.width - 1, job.height - 1), job.grid.center(0, job.height - 1)]

        def run():
            mouse = HumanMouse(self.inp, self.cfg.behavior.human_profile(), random.Random(), time.sleep, lambda: None)
            for x, y in pts + pts[:1]:
                mouse.move_to(x, y, job.grid.pitch)
                time.sleep(0.7)

        threading.Thread(target=run, name="show-corners", daemon=True).start()

    # ================================================================== запуск
    def _running(self) -> bool:
        return self.bot_thread is not None and self.bot_thread.is_alive()

    def _start(self):
        if self._running():
            return
        try:
            job = self._build_job()
            job.validate(self.screen.virtual_rect())
        except (JobError, ValueError) as e:
            messagebox.showerror(APP_NAME, str(e))
            return
        self._save()
        hk = self.cfg.hotkeys
        n = int((job.target != SKIP).sum())
        if not messagebox.askokcancel(
            APP_NAME,
            f"Рисунок {job.width}×{job.height}, пикселей: {n}, цветов: {len(job.needed_colors())}.\n\n"
            "Во время подхода не трогайте мышь (иначе бот встанет на паузу).\n"
            f"Пауза — {hk.pause}, стоп — {hk.stop}, аварийно — увести курсор в левый верхний угол.\n\n"
            "Начать?",
        ):
            return
        self.l_alert.configure(text="")
        self.bot = PaintBot(
            job, self.cfg.behavior, self.screen, self.inp,
            on_event=lambda kind, **d: self.events.put(("bot", kind, d)),
            debug_dir=self.paths.screens,
        )
        self.bot_thread = threading.Thread(target=self.bot.run, name="bot", daemon=True)
        self.bot_thread.start()
        self.nb.select(self.tab_run)
        self._set_buttons(True)
        if self.cfg.behavior.minimize_on_start:
            self.root.after(300, self.root.iconify)

    def _toggle_pause(self):
        if not self.bot or not self._running():
            return
        if self.bot.paused:
            self.l_alert.configure(text="")
            if self.cfg.behavior.minimize_on_start:
                self.root.iconify()
                self.root.after(400, self.bot.request_resume)
            else:
                self.bot.request_resume()
        else:
            self.bot.request_pause()
            self.l_state.configure(text="Ставлю на паузу…")

    def _stop(self):
        if self.bot:
            self.bot.request_stop()
            self.l_state.configure(text="Останавливаю…")

    def _set_buttons(self, running: bool):
        self.b_start.configure(state="disabled" if running else "normal")
        self.b_pause.configure(state="normal" if running else "disabled", text="⏸  Пауза")
        self.b_stop.configure(state="normal" if running else "disabled")

    def _show_window(self):
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(500, lambda: self.root.attributes("-topmost", False))
        except tk.TclError:
            pass

    # ================================================================== события
    def _poll(self):
        try:
            for _ in range(300):
                try:
                    ev = self.events.get_nowait()
                except queue.Empty:
                    break
                self._handle_event(ev)
        except Exception:  # noqa: BLE001
            log.exception("Ошибка обработки события интерфейса")
        self.root.after(100, self._poll)

    def _handle_event(self, ev):
        kind = ev[0]
        if kind == "log":
            _, text, level = ev
            self._append_log(text, level)
        elif kind == "hotkey":
            action = ev[1]
            if action == "capture":
                self._do_capture()
            elif action == "pause":
                self._toggle_pause()
            elif action == "stop":
                self._stop()
        elif kind == "bot":
            self._handle_bot_event(ev[1], ev[2])

    def _handle_bot_event(self, kind: str, d: dict):
        if kind == "state":
            if d.get("state") == "running":
                self.l_state.configure(text="Работает")
                self.b_pause.configure(text="⏸  Пауза")
            elif d.get("state") == "idle":
                self._set_buttons(False)
                self.l_countdown.configure(text="")
        elif kind == "paused":
            self.l_state.configure(text="Пауза")
            self.b_pause.configure(text="▶  Продолжить")
            self.l_alert.configure(text=d.get("message", ""))
            self.nb.select(self.tab_run)
            self._show_window()
        elif kind == "progress":
            total = max(1, d["total"])
            self.pb.configure(value=100.0 * d["done"] / total)
            self.l_progress.configure(
                text=f"Готово {d['done']} из {d['total']} пикселей • поставлено за сессию: {d['placed']} • "
                f"осталось примерно {fmt_seconds(d['eta'])}"
            )
        elif kind == "countdown":
            sec = d.get("seconds", 0)
            self.l_countdown.configure(text=f"{d.get('label', '')}: {fmt_seconds(sec)}" if sec > 0 else "")
        elif kind == "status":
            self.l_countdown.configure(text=d.get("text", ""))
        elif kind in ("finished", "stopped", "error"):
            self.l_state.configure(text={"finished": "Готово", "stopped": "Остановлен", "error": "Ошибка"}[kind])
            if kind == "error":
                self.l_alert.configure(text=d.get("message", ""))
            self._show_window()

    def _append_log(self, text: str, level: int):
        tag = "err" if level >= logging.ERROR else "warn" if level >= logging.WARNING else ""
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", text + "\n", tag)
        lines = int(self.txt_log.index("end-1c").split(".")[0])
        if lines > 3000:
            self.txt_log.delete("1.0", f"{lines - 3000}.0")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    # ================================================================== прочее
    def _report(self):
        self._save()
        try:
            path = make_report(self.paths)
        except Exception as e:  # noqa: BLE001
            log.exception("Не удалось собрать отчёт")
            messagebox.showerror(APP_NAME, f"Не удалось собрать отчёт: {e}")
            return
        messagebox.showinfo(
            APP_NAME,
            f"Отчёт сохранён:\n{path}\n\nОтправьте этот zip-файл разработчику. В нём логи, настройки и снимки "
            "области рисования (другие части экрана туда не попадают).",
        )
        self._open_path(os.path.dirname(path))

    def _open_path(self, path: str):
        try:
            if sys.platform == "win32":
                os.startfile(path)  # noqa: S606
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:  # noqa: BLE001
            log.warning("Не удалось открыть %s", path, exc_info=True)

    def _tk_exception(self, exc, val, tb):
        log.error("Ошибка в интерфейсе", exc_info=(exc, val, tb))
        messagebox.showerror(APP_NAME, f"Ошибка: {val}\n\nПодробности записаны в лог.")

    def _on_close(self):
        if self._running():
            if not messagebox.askyesno(APP_NAME, "Бот работает. Остановить и выйти?"):
                return
            self.bot.request_stop()
            self.bot_thread.join(timeout=3)
        self._save()
        self.hotkeys.stop_evt.set()
        log.info("Выход из программы")
        self.root.destroy()


class CalibrationView:
    """Окно проверки калибровки: снимок экрана с сеткой и призраком картинки."""

    MAX_W, MAX_H = 900, 600

    def __init__(self, app: App, job: Job):
        self.app = app
        self.job = job
        self.top = tk.Toplevel(app.root)
        self.top.title("Проверка калибровки")
        self.top.transient(app.root)
        self.v_pitch = tk.DoubleVar(value=job.grid.pitch)
        self.v_x = tk.DoubleVar(value=job.grid.x0)
        self.v_y = tk.DoubleVar(value=job.grid.y0)
        self.v_grid = tk.BooleanVar(value=True)
        self.v_ghost = tk.BooleanVar(value=True)
        self.photo = None

        bar = ttk.Frame(self.top, padding=6)
        bar.pack(fill="x")
        ttk.Label(bar, text="Размер пикселя:").pack(side="left")
        ttk.Spinbox(bar, from_=1, to=200, increment=0.01, textvariable=self.v_pitch, width=8,
                    command=self.redraw).pack(side="left", padx=4)
        ttk.Label(bar, text="X:").pack(side="left")
        ttk.Spinbox(bar, from_=-10000, to=10000, increment=0.5, textvariable=self.v_x, width=8,
                    command=self.redraw).pack(side="left", padx=4)
        ttk.Label(bar, text="Y:").pack(side="left")
        ttk.Spinbox(bar, from_=-10000, to=10000, increment=0.5, textvariable=self.v_y, width=8,
                    command=self.redraw).pack(side="left", padx=4)
        ttk.Checkbutton(bar, text="сетка", variable=self.v_grid, command=self.redraw).pack(side="left", padx=4)
        ttk.Checkbutton(bar, text="картинка", variable=self.v_ghost, command=self.redraw).pack(side="left", padx=4)
        ttk.Button(bar, text="Новый снимок", command=self.snap).pack(side="left", padx=4)
        ttk.Button(bar, text="Сохранить", command=self.apply).pack(side="left", padx=4)
        self.canvas = tk.Canvas(self.top, width=self.MAX_W, height=self.MAX_H, bg="#333333", highlightthickness=0)
        self.canvas.pack(padx=6, pady=6)
        self.l_stats = ttk.Label(self.top, text="", padding=6, justify="left")
        self.l_stats.pack(anchor="w")
        ttk.Label(
            self.top,
            text="Маленькие квадраты — цвет, который должен быть в пикселе; линии — границы пикселей. "
            "Если линии не совпадают с настоящими пикселями — поправьте размер/X/Y и нажмите «Сохранить».",
            style="Hint.TLabel", wraplength=880, padding=6,
        ).pack(anchor="w")
        self.snap()

    def grid(self) -> Grid:
        return Grid(float(self.v_x.get()), float(self.v_y.get()), float(self.v_pitch.get()))

    def snap(self):
        g = self.grid()
        margin = int(3 * g.pitch) + 4
        self.rect = g.rect(self.job.width, self.job.height, margin)
        self.img = self.app._grab_hidden(self.rect)
        self.redraw()

    def redraw(self):
        try:
            g = self.grid()
        except (tk.TclError, ValueError):
            return
        if g.pitch <= 0:
            return
        base = Image.fromarray(self.img).convert("RGBA")
        h, w = self.img.shape[:2]
        scale = min(self.MAX_W / w, self.MAX_H / h, 8.0)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        pic = base.resize(size, Image.Resampling.NEAREST)
        over = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        ox, oy = self.rect[0], self.rect[1]
        tw, th = self.job.width, self.job.height

        def sx(x):
            return (x - ox + 0.5) * scale

        def sy(y):
            return (y - oy + 0.5) * scale

        if self.v_grid.get() and g.pitch * scale >= 3:
            for i in range(tw + 1):
                x = sx(g.x0 + (i - 0.5) * g.pitch)
                d.line([x, sy(g.y0 - 0.5 * g.pitch), x, sy(g.y0 + (th - 0.5) * g.pitch)], fill=(255, 0, 255, 150))
            for j in range(th + 1):
                y = sy(g.y0 + (j - 0.5) * g.pitch)
                d.line([sx(g.x0 - 0.5 * g.pitch), y, sx(g.x0 + (tw - 0.5) * g.pitch), y], fill=(255, 0, 255, 150))
        else:
            d.rectangle([sx(g.x0 - 0.5 * g.pitch), sy(g.y0 - 0.5 * g.pitch),
                         sx(g.x0 + (tw - 0.5) * g.pitch), sy(g.y0 + (th - 0.5) * g.pitch)], outline=(255, 0, 255, 255))
        if self.v_ghost.get() and g.pitch * scale >= 3:
            r = max(1.0, g.pitch * scale * 0.18)
            for j in range(th):
                for i in range(tw):
                    cid = int(self.job.target[j, i])
                    if cid == SKIP:
                        continue
                    cx, cy = sx(g.x0 + i * g.pitch), sy(g.y0 + j * g.pitch)
                    d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=BY_ID[cid].rgb + (255,), outline=(0, 0, 0, 160))
        pic = Image.alpha_composite(pic, over)
        self.photo = ImageTk.PhotoImage(pic)
        self.canvas.delete("all")
        self.canvas.configure(width=size[0], height=size[1])
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        colors = sample_cells(self.img, self.rect, g, tw, th)
        classes = ColorMatcher(self.app.cfg.behavior.color_tolerance).classify(colors)
        mask = self.job.target != SKIP
        good = int(((classes == self.job.target) & mask).sum())
        unknown = int(((classes == -1) & mask).sum())
        total = int(mask.sum())
        self.l_stats.configure(
            text=f"Уже совпадает с картинкой: {good} из {total} ({good / max(1, total):.0%}); "
            f"нужно поставить: {total - good}; пустых/нераспознанных клеток: {unknown}."
        )

    def apply(self):
        g = self.grid()
        c = self.app.cfg.calib
        dx, dy = g.x0 - self.job.grid.x0, g.y0 - self.job.grid.y0
        c.pitch = round(g.pitch, 4)
        if c.origin:
            c.origin = [round(c.origin[0] + dx, 2), round(c.origin[1] + dy, 2)]
        if c.corner2:
            c.corner2 = [round(c.corner2[0] + dx, 2), round(c.corner2[1] + dy, 2)]
        self.app.v_pitch.set(c.pitch)
        self.app._refresh_calib_labels()
        self.app._save()
        self.job.grid = g
        log.info("Калибровка исправлена вручную: x0=%.2f y0=%.2f шаг=%.4f", g.x0, g.y0, g.pitch)
