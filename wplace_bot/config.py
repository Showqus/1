"""Настройки программы (config.json рядом с exe)."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .humanize import HumanProfile

log = logging.getLogger(__name__)


@dataclass
class ImageSettings:
    mode: str = "image"  # "image" — картинка, "zone" — заливка зоны одним цветом
    image_path: str = ""
    width: int = 0  # 0 — по пропорции/исходный размер
    height: int = 0
    keep_aspect: bool = True
    resample: str = "nearest"  # "nearest" (пиксель-арт) | "smooth" (фото)
    dither: bool = False
    premium_ids: list[int] = field(default_factory=list)  # купленные платные цвета
    alpha_threshold: int = 128
    zone_color: int = 1


@dataclass
class Calibration:
    origin: list[float] | None = None  # центр левого верхнего пикселя рисунка
    corner2: list[float] | None = None  # центр правого нижнего пикселя (для зоны)
    pitch: float = 0.0  # размер пикселя холста на экране
    paint_button: list[int] | None = None  # кнопка «Paint», открывающая палитру
    submit_button: list[int] | None = None  # кнопка подтверждения внутри палитры
    palette_tl: list[int] | None = None
    palette_br: list[int] | None = None
    swatches: dict[str, list[int]] = field(default_factory=dict)  # id цвета -> [x, y, w, h]


@dataclass
class Behavior:
    click_delay_min: float = 0.35
    click_delay_max: float = 1.40
    mouse_speed: float = 1.0
    click_jitter: float = 0.25
    hesitation_chance: float = 0.06
    hesitation_min: float = 0.8
    hesitation_max: float = 3.0
    rest_every_min: int = 18
    rest_every_max: int = 40
    rest_min: float = 3.0
    rest_max: float = 12.0
    overshoot_chance: float = 0.12

    max_charges: int = 30  # максимум зарядов на аккаунте
    start_charges: int = 30  # сколько зарядов сейчас (при старте)
    seconds_per_charge: float = 30.0
    random_batch: bool = True  # не всегда тратить все заряды за подход
    batch_min_fraction: float = 0.6
    extra_wait_min: float = 5.0  # дополнительное случайное ожидание между подходами, сек
    extra_wait_max: float = 90.0

    order: str = "colors_nearest"
    verify_pixels: bool = True
    color_tolerance: float = 18.0
    align_threshold: float = 0.6
    idle_before_batch: float = 3.0  # ждать, пока пользователь не трогает мышь N сек
    pause_on_user_mouse: bool = True
    palette_closes_after_submit: bool = True
    submit_timeout: float = 15.0
    guard_mode: bool = False
    guard_interval_min: float = 5.0  # минуты
    sound: bool = True
    minimize_on_start: bool = True
    dry_run: bool = False

    def human_profile(self) -> HumanProfile:
        names = {f.name for f in dataclasses.fields(HumanProfile)}
        return HumanProfile(**{k: v for k, v in dataclasses.asdict(self).items() if k in names})


@dataclass
class Hotkeys:
    capture: str = "F8"
    pause: str = "F9"
    stop: str = "F10"


@dataclass
class AppConfig:
    image: ImageSettings = field(default_factory=ImageSettings)
    calib: Calibration = field(default_factory=Calibration)
    behavior: Behavior = field(default_factory=Behavior)
    hotkeys: Hotkeys = field(default_factory=Hotkeys)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        cfg = cls()
        for section in ("image", "calib", "behavior", "hotkeys"):
            obj = getattr(cfg, section)
            values = data.get(section) or {}
            for f in dataclasses.fields(obj):
                if f.name in values:
                    setattr(obj, f.name, _coerce(values[f.name], getattr(obj, f.name)))
        return cfg


def _coerce(value, default):
    """Привести значение из JSON к типу значения по умолчанию (защита от ручных правок)."""
    if value is None or default is None:
        return value
    try:
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return str(value)
    except (TypeError, ValueError):
        log.warning("Некорректное значение в настройках: %r, используется %r", value, default)
        return default
    return value


def load_config(path: str) -> AppConfig:
    if not os.path.exists(path):
        log.info("Файл настроек не найден, используются значения по умолчанию: %s", path)
        return AppConfig()
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = AppConfig.from_dict(json.load(fh))
        log.info("Настройки загружены: %s", path)
        return cfg
    except Exception:  # noqa: BLE001
        log.exception("Не удалось прочитать настройки %s — используются значения по умолчанию", path)
        try:
            os.replace(path, path + ".broken")
        except OSError:
            pass
        return AppConfig()


def save_config(cfg: AppConfig, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
