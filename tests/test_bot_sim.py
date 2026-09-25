"""Полный прогон бота на симуляторе wplace."""
import random

import numpy as np

from wplace_bot.bot import Job, PaintBot
from wplace_bot.config import Behavior
from wplace_bot.geometry import Grid
from wplace_bot.image_prep import SKIP
from wplace_bot.palette import FREE_IDS

from .sim import FakeClock, FakeWplace


def make(seed=0, target_size=(18, 12), at=(20, 10), sim_kwargs=None, **beh):
    clock = FakeClock()
    sim = FakeWplace(clock, seed=seed, **(sim_kwargs or {}))
    rng = np.random.default_rng(seed)
    w, h = target_size
    target = rng.choice(FREE_IDS[:8], size=(h, w)).astype(np.int16)
    target[0, 0] = SKIP
    sim.board[at[1] + 2, at[0] + 3] = int(target[2, 3])  # один пиксель уже правильный
    x0, y0 = sim.cell_center(*at)
    job = Job(
        grid=Grid(x0 + 0.4, y0 - 0.3, sim.pitch),
        target=target,
        swatches=sim.truth_swatches(),
        paint_button=(sim.PAINT_BTN[0] + 60, sim.PAINT_BTN[1] + 20),
        submit_button=(sim.SUBMIT_BTN[0] + 60, sim.SUBMIT_BTN[1] + 20),
        palette_rect=sim.PANEL,
    )
    b = Behavior(sound=False, idle_before_batch=0, extra_wait_min=1, extra_wait_max=5, **beh)
    events = []
    bot = PaintBot(job, b, sim, sim, clock=clock, rng=random.Random(seed),
                   on_event=lambda k, **d: events.append((k, d)))
    return sim, job, bot, events


def region(sim, job, at=(20, 10)):
    h, w = job.target.shape
    return sim.board[at[1] : at[1] + h, at[0] : at[0] + w]


def assert_drawn(sim, job, at=(20, 10)):
    reg = region(sim, job, at)
    mask = job.target != SKIP
    assert (reg[mask] == job.target[mask]).all()


def test_full_run_draws_image():
    sim, job, bot, events = make()
    bot.run()
    kinds = [k for k, _ in events]
    assert "finished" in kinds, events[-5:]
    assert_drawn(sim, job)
    # пиксель вне рисунка и SKIP-пиксель не трогали
    assert sim.board[10, 20] == 0 or sim.board[10, 20] == FakeWplace(FakeClock(), seed=0).board[10, 20]
    n = int((job.target != SKIP).sum()) - 1
    pixel_clicks = sum(1 for c in sim.clicks if c[2] == "pixel")
    assert pixel_clicks == n
    assert not any(c[2] in ("noop", "panel") for c in sim.clicks)
    assert sim.submits >= n // 30


def test_charges_run_out_early():
    sim, job, bot, events = make(seed=1, sim_kwargs={"charges": 7}, start_charges=30, max_charges=30)
    bot.run()
    assert "finished" in [k for k, _ in events]
    assert_drawn(sim, job)
    fails = sum(1 for c in sim.clicks if c[2] == "pixel-fail")
    assert fails >= 1


def test_captcha_pauses_and_resumes():
    sim, job, bot, events = make(seed=2)
    sim.captcha_next = 1

    def on_event(kind, **d):
        events.append((kind, d))
        if kind == "paused":
            assert "проверка" in d["message"] or "Paint" in d["message"]
            sim.commit()  # «пользователь» прошёл проверку
            bot.request_resume()

    bot.on_event = on_event
    bot.run()
    kinds = [k for k, _ in events]
    assert "paused" in kinds and "finished" in kinds
    assert_drawn(sim, job)


def test_map_moved_pauses():
    sim, job, bot, events = make(seed=3, target_size=(40, 20))
    orig_commit = sim.commit

    def commit_and_shift():
        orig_commit()
        sim.off[0] += 37  # карту сдвинули после первого подхода
        sim.off[1] += 23

    sim.commit = commit_and_shift

    def on_event(kind, **d):
        events.append((kind, d))
        if kind == "paused":
            bot.request_stop()

    bot.on_event = on_event
    bot.run()
    paused = [d for k, d in events if k == "paused"]
    assert paused and "карту сдвинули" in paused[0]["message"], paused
    assert "stopped" in [k for k, _ in events]


def test_user_mouse_move_pauses_then_continues():
    sim, job, bot, events = make(seed=4)
    sim.user_jerk_at = 200

    def on_event(kind, **d):
        events.append((kind, d))
        if kind == "paused":
            bot.request_resume()

    bot.on_event = on_event
    bot.run()
    paused = [d for k, d in events if k == "paused"]
    assert paused and "сдвинули мышь" in paused[0]["message"]
    assert "finished" in [k for k, _ in events]
    assert_drawn(sim, job)


def test_zone_fill_and_guard_off():
    sim, job, bot, events = make(seed=5, target_size=(10, 6))
    job.target[:] = 13
    bot.total = int((job.target != SKIP).sum())
    bot.run()
    assert "finished" in [k for k, _ in events]
    assert (region(sim, job) == 13).all()


def test_wrong_calibration_does_not_spam_forever():
    sim, job, bot, events = make(seed=6)
    # Сетка сдвинута на полпикселя по размеру — проверка пикселей будет падать.
    job.grid = Grid(job.grid.x0, job.grid.y0, sim.pitch * 1.5)

    def on_event(kind, **d):
        events.append((kind, d))
        if kind == "paused":
            bot.request_stop()

    bot.on_event = on_event
    bot.run()
    kinds = [k for k, _ in events]
    assert "stopped" in kinds or "finished" in kinds or "error" in kinds


def test_other_window_on_top_pauses():
    sim, job, bot, events = make(seed=7)
    orig_commit = sim.commit

    def commit_and_cover():
        orig_commit()
        sim.overlay = (700, 900, 300, 100)  # после первого подхода кнопку Paint закрыло окно

    sim.commit = commit_and_cover

    def on_event(kind, **d):
        events.append((kind, d))
        if kind == "paused":
            sim.overlay = None  # пользователь убрал окно
            bot.request_resume()

    bot.on_event = on_event
    bot.run()
    paused = [d for k, d in events if k == "paused"]
    assert paused and "закрыто другим окном" in paused[0]["message"], paused
    assert "finished" in [k for k, _ in events]
    assert_drawn(sim, job)


def test_button_inside_target_rejected():
    import pytest

    from wplace_bot.bot import JobError

    sim, job, bot, events = make(seed=8)
    job.paint_button = tuple(int(v) for v in job.grid.center(3, 3))
    with pytest.raises(JobError):
        job.validate()
