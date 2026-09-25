"""Дымовой тест интерфейса (нужен дисплей: на Linux запускать через xvfb-run)."""
import logging
import os
import time

import numpy as np
import pytest
from PIL import Image

tk = pytest.importorskip("tkinter")

from wplace_bot import gui as gui_mod  # noqa: E402
from wplace_bot.app_logging import Paths, make_report  # noqa: E402
from wplace_bot.config import AppConfig  # noqa: E402

from .sim import FakeClock, FakeWplace  # noqa: E402


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("нет дисплея")
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


def pump(root, sec=0.3):
    end = time.time() + sec
    while time.time() < end:
        root.update()
        time.sleep(0.01)


def test_gui_end_to_end(root, tmp_path, monkeypatch):
    logging.getLogger().setLevel(logging.INFO)
    paths = Paths(str(tmp_path), str(tmp_path / "logs"), str(tmp_path / "logs" / "screens"), str(tmp_path / "config.json"))
    os.makedirs(paths.screens)
    img_path = tmp_path / "art.png"
    arr = np.zeros((10, 14, 4), dtype=np.uint8)
    arr[..., :3] = (237, 28, 36)
    arr[:, 7:, :3] = (64, 147, 228)
    arr[..., 3] = 255
    Image.fromarray(arr, "RGBA").save(img_path)

    sim = FakeWplace(FakeClock(), pitch=9.3, offset=(40.2, 30.6))
    cfg = AppConfig()
    cfg.image.image_path = str(img_path)
    cfg.behavior.sound = False
    cfg.behavior.idle_before_batch = 0
    cfg.behavior.minimize_on_start = False
    app = gui_mod.App(root, paths, cfg, str(tmp_path / "x.log"))
    app.screen = sim
    app.inp = sim
    pump(root, 0.5)
    assert app.target is not None and app.target.shape == (10, 14)

    # калибровка через «горячую клавишу»
    def capture(key, x, y):
        app._begin_capture(key)
        sim.pos = (int(x), int(y))
        app._do_capture()

    x0, y0 = sim.cell_center(30, 20)
    capture("origin", x0, y0)
    capture("paint_button", 800, 960)
    capture("submit_button", 800, 955)
    capture("palette_tl", sim.PANEL[0], sim.PANEL[1])
    capture("palette_br", sim.PANEL[0] + sim.PANEL[2] - 1, sim.PANEL[1] + sim.PANEL[3] - 1)

    monkeypatch.setattr(gui_mod.messagebox, "showwarning", lambda *a, **k: None)
    monkeypatch.setattr(gui_mod.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(gui_mod.messagebox, "showerror", lambda *a, **k: pytest.fail(f"showerror: {a}"))
    app._auto_pitch()
    assert abs(app.cfg.calib.pitch - 9.3) < 0.02
    sim.palette_open = True
    app._find_palette()
    sim.palette_open = False
    assert len(app.cfg.calib.swatches) >= 60

    view = gui_mod.CalibrationView(app, app._build_job())
    pump(root)
    view.v_pitch.set(9.31)
    view.redraw()
    view.apply()
    view.top.destroy()

    # переключение режима «зона»
    capture("corner2", *sim.cell_center(35, 24))
    app.v_mode.set("zone")
    app._refresh_preview()
    job = app._build_job()
    assert job.target.shape == (5, 6)
    app.v_mode.set("image")
    app._refresh_preview()

    # запуск бота из интерфейса (на симуляторе, с быстрыми часами)
    monkeypatch.setattr(gui_mod.messagebox, "askokcancel", lambda *a, **k: True)
    orig_bot = gui_mod.PaintBot

    def fast_bot(*a, **k):
        k["clock"] = sim.clock
        return orig_bot(*a, **k)

    monkeypatch.setattr(gui_mod, "PaintBot", fast_bot)
    app.cfg.behavior.extra_wait_min = 0
    app.cfg.behavior.extra_wait_max = 1
    app._load_into_ui()
    app._start()
    deadline = time.time() + 60
    while app._running() and time.time() < deadline:
        pump(root, 0.1)
    end = time.time() + 20  # на быстрых часах событий обратного отсчёта очень много
    while not app.events.empty() and time.time() < end:
        pump(root, 0.1)
    pump(root, 0.2)
    assert not app._running()
    assert app.l_state.cget("text") == "Готово", app.txt_log.get("1.0", "end")[-3000:]
    reg = sim.board[20:30, 30:44]
    assert (reg == app.target).all()

    assert os.path.exists(paths.config)
    report = make_report(paths)
    assert os.path.exists(report)
    app._on_close()
