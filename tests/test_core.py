import random

import numpy as np
import pytest
from PIL import Image

from wplace_bot.bot import ChargeModel
from wplace_bot.config import AppConfig
from wplace_bot.geometry import Grid, detect_pitch, snap_to_center
from wplace_bot.humanize import HumanMouse, HumanProfile, bezier_path, plan_order
from wplace_bot.image_prep import SKIP, fit_size, prepare_target
from wplace_bot.palette import BY_ID, COLORS, FREE_IDS, ColorMatcher, nearest_ids
from wplace_bot.palette_finder import find_swatches, palette_visible_ratio
from wplace_bot.screen import sample_cells
from wplace_bot.winapi import DummyInput

from .sim import FakeClock, FakeWplace


def test_palette_is_complete():
    assert len(COLORS) == 63
    assert len(FREE_IDS) == 31
    assert BY_ID[7].rgb == (237, 28, 36)
    assert len({c.rgb for c in COLORS}) == 63


def test_matcher_exact_and_unknown():
    m = ColorMatcher(tolerance=18)
    for c in COLORS:
        assert m.classify_one(c.rgb) == c.id
    assert m.classify_one((10, 250, 10)) == -1


def test_nearest_ids_exact():
    rgb = np.array([c.rgb for c in COLORS])
    assert list(nearest_ids(rgb, [c.id for c in COLORS])) == [c.id for c in COLORS]


def test_prepare_target_alpha_and_palette():
    arr = np.zeros((4, 6, 4), dtype=np.uint8)
    arr[..., :3] = (237, 28, 36)
    arr[..., 3] = 255
    arr[0, 0, 3] = 0
    img = Image.fromarray(arr, "RGBA")
    t = prepare_target(img, 6, 4, FREE_IDS)
    assert t.shape == (4, 6)
    assert t[0, 0] == SKIP
    assert (t[t != SKIP] == 7).all()
    t2 = prepare_target(img, 12, 8, FREE_IDS, dither=True)
    assert t2.shape == (8, 12)


def test_fit_size():
    assert fit_size(200, 100, 50, 0, True) == (50, 25)
    assert fit_size(200, 100, 0, 10, True) == (20, 10)
    assert fit_size(200, 100, 30, 30, False) == (30, 30)
    assert fit_size(200, 100, 0, 0, True) == (200, 100)


@pytest.mark.parametrize("pitch", [5.3, 7.37, 12.0, 23.41])
def test_detect_pitch_on_rendered_canvas(pitch):
    sim = FakeWplace(FakeClock(), pitch=pitch, offset=(33.7, 21.2), cells=(int(700 / pitch), int(600 / pitch)))
    sim.board[:] = np.random.default_rng(3).integers(0, 64, size=sim.board.shape)
    img = sim.grab((0, 0, 700, 600))
    res = detect_pitch(img)
    assert res.ok, res
    assert abs(res.pitch - pitch) < 0.02, res
    # центр клетки после «прилипания» совпадает с настоящим центром
    cx, cy = sim.cell_center(5, 7)
    assert abs(snap_to_center(cx + 1, res.phase_x, res.pitch) - cx) < 1.0
    assert abs(snap_to_center(cy - 1, res.phase_y, res.pitch) - cy) < 1.0


def test_detect_pitch_fails_on_empty_map():
    sim = FakeWplace(FakeClock(), art=False)
    res = detect_pitch(sim.grab((0, 0, 700, 600)))
    assert not res.ok


def test_sample_cells_reads_board():
    sim = FakeWplace(FakeClock(), pitch=9.1)
    grid = Grid(*sim.cell_center(10, 5), 9.1)
    rect = grid.rect(20, 10, 2)
    colors = sample_cells(sim.grab(rect), rect, grid, 20, 10)
    classes = ColorMatcher().classify(colors)
    expected = sim.board[5:15, 10:30]
    assert ((classes == expected) | (expected == 0)).all()


def test_find_swatches_in_sim_panel():
    sim = FakeWplace(FakeClock())
    sim.palette_open = True
    rect = sim.PANEL
    found = find_swatches(sim.grab(rect), rect)
    truth = sim.truth_swatches()
    # белая кнопка на белом фоне сливается — её можно указать вручную
    assert set(found) >= set(truth) - {5}
    for cid, sw in found.items():
        assert abs(sw.x - truth[cid].x) <= 2 and abs(sw.y - truth[cid].y) <= 2
    assert palette_visible_ratio(sim.grab(rect), rect, found) > 0.9
    sim.palette_open = False
    assert palette_visible_ratio(sim.grab(rect), rect, found) < 0.2


def test_plan_order_unique_and_limited():
    rng = random.Random(1)
    todo = [(x, y, 1 + (x + y) % 3) for x in range(30) for y in range(20)]
    for mode in ("colors_nearest", "rows", "random"):
        plan = plan_order(todo, mode, rng, 50, (0, 0))
        assert len(plan) == 50
        assert len(set(plan)) == 50
        assert set(plan) <= set(todo)
    plan = plan_order(todo, "colors_nearest", rng, 10_000)
    assert sorted(plan) == sorted(todo)


def test_bezier_ends_at_target():
    pts = bezier_path(random.Random(2), (0, 0), (300, 120), 30)
    assert pts[-1] == (300, 120)
    assert len(pts) == 30


def test_human_mouse_moves_and_clicks():
    inp = DummyInput()
    m = HumanMouse(inp, HumanProfile(), random.Random(3), lambda s: None, lambda: None)
    m.move_to(500, 300, 10)
    m.click()
    assert inp.pos == (500, 300)
    assert inp.clicks == [(500, 300)]
    assert not m.user_moved()
    inp.pos = (520, 300)
    assert m.user_moved()


def test_charge_model():
    cm = ChargeModel(30, 30, 10, now=0)
    assert cm.current(0) == 10
    assert cm.current(300) == 20
    assert cm.current(10_000) == 30
    cm.consume(5, now=300)
    assert cm.current(300) == 15
    assert cm.time_until(20, 300) == pytest.approx(150)
    cm.consume(1, now=300, exhausted=True)
    assert cm.current(300) == 0


def test_config_roundtrip():
    cfg = AppConfig()
    cfg.calib.origin = [10.5, 20.25]
    cfg.calib.swatches = {"7": [1, 2, 3, 4]}
    cfg.behavior.max_charges = 55
    data = cfg.to_dict()
    data["behavior"]["unknown_field"] = 1
    data["behavior"]["mouse_speed"] = "1.5"
    cfg2 = AppConfig.from_dict(data)
    assert cfg2.calib.origin == [10.5, 20.25]
    assert cfg2.calib.swatches == {"7": [1, 2, 3, 4]}
    assert cfg2.behavior.max_charges == 55
    assert cfg2.behavior.mouse_speed == 1.5
    assert cfg2.behavior.human_profile().mouse_speed == 1.5
