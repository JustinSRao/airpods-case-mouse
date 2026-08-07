"""Offline sanity checks for the motion pipeline. No camera, no real cursor.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe -m scripts.self_check

Mouse *button* events are never exercised here on purpose -- a synthetic click
would land on whatever window happens to be under the pointer. Button
bookkeeping is verified with the send path stubbed out.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import replace

import numpy as np

from src.config.settings import AppSettings
from src.mouse.cursor_filter import EmaFilter, NoFilter, OneEuroFilter, build_filter
from src.mouse.motion_mapper import MotionMapper
from src.mouse import mouse_controller as mc
from src.tracking.hand_features import (
    ANCHOR_STRATEGY_NAMES,
    Landmark,
    compute_anchor,
    fingertip_relative_to_palm,
    palm_width,
)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


def synthetic_hand(offset: np.ndarray = None, scale: float = 1.0) -> np.ndarray:
    """A crude but geometrically sane right hand in pixel space."""
    base = np.zeros((21, 2), dtype=np.float64)
    base[Landmark.WRIST] = (100, 200)
    base[Landmark.INDEX_MCP] = (80, 150)
    base[Landmark.MIDDLE_MCP] = (100, 145)
    base[Landmark.RING_MCP] = (120, 150)
    base[Landmark.PINKY_MCP] = (140, 158)
    base[Landmark.INDEX_TIP] = (78, 100)
    base[Landmark.MIDDLE_TIP] = (100, 95)
    centre = base[Landmark.WRIST]
    base = centre + (base - centre) * scale
    if offset is not None:
        base = base + offset
    return base


print("\n[1] settings")
settings = AppSettings.load()
check("defaults load", settings.camera.width == 640, f"got {settings.camera.width}")
check("dshow backend default", settings.camera.backend == "dshow")
check("anchor strategy valid", settings.anchor.strategy in ANCHOR_STRATEGY_NAMES)

print("\n[2] hand features")
hand = synthetic_hand()
width = palm_width(hand)
check("palm width positive", width > 0, f"got {width:.2f}")

# Scale invariance: a hand twice as large should give twice the palm width,
# and identical palm-relative fingertip geometry.
big = synthetic_hand(scale=2.0)
check("palm width scales", abs(palm_width(big) / width - 2.0) < 1e-6)
rel_small = fingertip_relative_to_palm(hand, Landmark.INDEX_TIP)
rel_big = fingertip_relative_to_palm(big, Landmark.INDEX_TIP)
check(
    "fingertip-relative is scale invariant",
    np.allclose(rel_small, rel_big, atol=1e-6),
    f"{rel_small} vs {rel_big}",
)

# Translation invariance: sliding the whole hand must not change the
# palm-relative fingertip position. This is the property that separates
# "moving the case" from "clicking".
moved = synthetic_hand(offset=np.array([37.0, -21.0]))
rel_moved = fingertip_relative_to_palm(moved, Landmark.INDEX_TIP)
check(
    "fingertip-relative is translation invariant",
    np.allclose(rel_small, rel_moved, atol=1e-6),
    f"{rel_small} vs {rel_moved}",
)

for strategy in ANCHOR_STRATEGY_NAMES:
    anchor = compute_anchor(hand, strategy)
    check(f"anchor '{strategy}' returns 2D", anchor.shape == (2,))

print("\n[3] filters")
for name, filt in (
    ("none", NoFilter()),
    ("ema", EmaFilter(0.5)),
    ("one_euro", OneEuroFilter()),
):
    filt.reset()
    out = None
    for _ in range(40):
        out = filt.apply(np.array([10.0, 20.0]), 1 / 30)
    check(f"{name} converges to constant input", np.allclose(out, [10, 20], atol=0.2), f"got {out}")

noisy = EmaFilter(0.3)
values = [noisy.apply(np.array([5.0 + (i % 2) * 2.0, 0.0]), 1 / 30) for i in range(60)]
spread = max(v[0] for v in values[-10:]) - min(v[0] for v in values[-10:])
check("ema attenuates alternating noise", spread < 1.0, f"residual spread {spread:.3f}")

try:
    build_filter("nope", smoothing=0.5, min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    check("build_filter rejects unknown name", False, "no error raised")
except ValueError:
    check("build_filter rejects unknown name", True)

print("\n[4] motion mapper")
mapper = MotionMapper(settings.cursor)
first = mapper.update(np.array([100.0, 100.0]), 80.0, 1 / 30)
check("first frame emits no motion", first.dx == 0 and first.dy == 0)

# A stationary hand must not creep the cursor.
mapper.reset()
mapper.update(np.array([100.0, 100.0]), 80.0, 1 / 30)
total = 0.0
for _ in range(60):
    total += abs(mapper.update(np.array([100.0, 100.0]), 80.0, 1 / 30).dx)
check("stationary hand produces zero drift", total == 0.0, f"drifted {total:.4f}px")

def sweep(cursor_settings, delta: np.ndarray) -> tuple[float, float]:
    """Run a steady sweep and return the final (dx, dy)."""
    m = MotionMapper(cursor_settings)
    m.update(np.array([100.0, 100.0]), 80.0, 1 / 30)
    last = None
    for i in range(1, 20):
        last = m.update(np.array([100.0, 100.0]) + delta * i, 80.0, 1 / 30)
    return (last.dx, last.dy)


RIGHT = np.array([4.0, 0.0])
IMAGE_UP = np.array([0.0, -4.0])

# Axis signs are tested against explicit settings rather than whatever the
# config currently says, so flipping a default cannot silently pass.
plain = replace(settings.cursor, invert_x=False, invert_y=False)
dx, _ = sweep(plain, RIGHT)
check("rightward hand -> positive dx", dx > 0, f"dx={dx:.2f}")
_, dy = sweep(plain, IMAGE_UP)
check("uninverted: up in image -> negative dy", dy < 0, f"dy={dy:.2f}")

flipped_y = replace(settings.cursor, invert_x=False, invert_y=True)
_, dy_flipped = sweep(flipped_y, IMAGE_UP)
check("invert_y flips the sign", dy_flipped > 0, f"dy={dy_flipped:.2f}")

flipped_x = replace(settings.cursor, invert_x=True, invert_y=False)
dx_flipped, _ = sweep(flipped_x, RIGHT)
check("invert_x flips the sign", dx_flipped < 0, f"dx={dx_flipped:.2f}")

# The shipped default is invert_y=True, measured on the reference setup
# rather than derived: the raw camera-Y sign gave the opposite of what the
# hand was doing. Assert the default carries that, so a config edit that
# silently reverts it fails here instead of in the user's hand.
shipped = settings.cursor
check(
    "default config ships invert_y enabled",
    shipped.invert_y is True,
    f"invert_y={shipped.invert_y}",
)
_, dy_shipped = sweep(shipped, IMAGE_UP)
_, dy_raw = sweep(plain, IMAGE_UP)
check(
    "default config inverts Y relative to raw camera motion",
    dy_shipped * dy_raw < 0,
    f"shipped={dy_shipped:.2f} raw={dy_raw:.2f}",
)

# Scale invariance of the whole mapping: the same motion expressed as a
# fraction of palm width must give the same cursor delta at any hand size.
def travel(scale: float) -> float:
    m = MotionMapper(settings.cursor)
    m.reset()
    m.update(np.array([0.0, 0.0]), 80.0 * scale, 1 / 30)
    return sum(
        m.update(np.array([i * 4.0 * scale, 0.0]), 80.0 * scale, 1 / 30).dx
        for i in range(1, 30)
    )


near, far = travel(1.0), travel(2.0)
check(
    "cursor travel is independent of hand distance",
    abs(near - far) / max(abs(near), 1e-9) < 0.02,
    f"{near:.1f}px vs {far:.1f}px",
)

# Velocity clamp.
mapper.reset()
mapper.update(np.array([0.0, 0.0]), 80.0, 1 / 30)
huge = mapper.update(np.array([5000.0, 0.0]), 80.0, 1 / 30)
max_step = settings.cursor.max_velocity / 30
check("velocity clamp engages", huge.clamped and abs(huge.dx) <= max_step + 1e-6,
      f"dx={huge.dx:.1f} limit={max_step:.1f}")

print("\n[5] hand selection")
from src.tracking.hand_tracker import SELECTION_MODES, HandObservation, HandTracker  # noqa: E402


def fake_hand(mean_x: float, handedness: str, score: float = 0.97) -> HandObservation:
    pts = synthetic_hand(offset=np.array([mean_x - 118.0, 0.0]))
    return HandObservation(
        handedness=handedness,
        score=score,
        landmarks_px=pts.astype(np.float32),
        landmarks_norm=np.zeros((21, 3), dtype=np.float32),
        world_landmarks=np.zeros((21, 3), dtype=np.float32),
    )


# The real failure seen on camera: MediaPipe confidently labels the physically
# LEFT hand "Right". Position-based selection must ignore that entirely.
left_hand_mislabelled_right = fake_hand(150.0, "Right", 0.96)
right_hand_mislabelled_left = fake_hand(500.0, "Left", 0.98)
both = [left_hand_mislabelled_right, right_hand_mislabelled_left]

tracker_settings = AppSettings.load().tracking
tracker_settings.selection = "rightmost"
picker = HandTracker(tracker_settings)
chosen = picker.select_hand(both)
check(
    "rightmost picks the hand on the right despite a wrong label",
    chosen is right_hand_mislabelled_left,
    f"picked {chosen.handedness if chosen else None}",
)

tracker_settings.selection = "leftmost"
check("leftmost picks the other one", picker.select_hand(both) is left_hand_mislabelled_right)

tracker_settings.selection = "handedness"
tracker_settings.target_handedness = "Right"
check(
    "handedness mode follows the label (and so picks wrong here)",
    picker.select_hand(both) is left_hand_mislabelled_right,
)

tracker_settings.selection = "rightmost"
check("selection with no hands returns None", picker.select_hand([]) is None)
check("selection with one hand returns it", picker.select_hand([both[0]]) is both[0])

tracker_settings.selection = "bogus"
try:
    picker.select_hand(both)
    check("unknown selection mode raises", False, "no error")
except ValueError:
    check("unknown selection mode raises", True)

check("default selection is a known mode",
      AppSettings.load().tracking.selection in SELECTION_MODES)

print("\n[6] multi-press toggle (P x5 within 5s)")
from src.hotkeys import MultiPressDetector  # noqa: E402
from src.main import TOGGLE_PRESS_COUNT, TOGGLE_PRESS_WINDOW  # noqa: E402

det = MultiPressDetector(TOGGLE_PRESS_COUNT, TOGGLE_PRESS_WINDOW)
fired = [det.register(t) for t in (0.0, 0.2, 0.4, 0.6, 0.8)]
check("five fast taps fire once", fired == [False, False, False, False, True], f"{fired}")

det.reset()
fired = [det.register(t) for t in (0.0, 0.2, 0.4, 0.6)]
check("four taps do not fire", not any(fired), f"{fired}")

# Taps spread beyond the window must never accumulate into a trigger.
det.reset()
slow = [det.register(t) for t in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)]
check("slow taps outside the window never fire", not any(slow), f"{slow}")

# Four taps, then a long gap, then four more: neither burst reaches five.
det.reset()
straddle = [det.register(t) for t in (0.0, 0.5, 1.0, 1.5, 20.0, 20.5, 21.0, 21.5)]
check("stale taps expire before a new burst", not any(straddle), f"{straddle}")

# A second burst after a successful trigger must need five fresh taps.
det.reset()
for t in (0.0, 0.1, 0.2, 0.3, 0.4):
    det.register(t)
again = [det.register(t) for t in (0.5, 0.6, 0.7, 0.8)]
check("counter clears after firing", not any(again), f"{again}")
check("fifth tap of second burst fires", det.register(0.9) is True)

det.reset()
det.register(0.0)
det.register(0.1)
check("progress reports partial count", det.progress == (2, 5), f"{det.progress}")
det.expire(10.0)
check("expire decays stale progress", det.progress == (0, 5), f"{det.progress}")

print("\n[7] mouse controller (send path stubbed)")
check("INPUT struct is the size Windows expects",
      ctypes.sizeof(mc._INPUT) in (28, 40),
      f"sizeof={ctypes.sizeof(mc._INPUT)}")

controller = mc.MouseController()
sent: list[tuple[int, int, int]] = []
controller._send = lambda flags, dx=0, dy=0: sent.append((flags, dx, dy))  # type: ignore[method-assign]

# Sub-pixel accumulation: four 0.4px steps must yield exactly 1 pixel total.
steps = [controller.move_relative(0.4, 0.0) for _ in range(4)]
check("sub-pixel motion accumulates", sum(s[0] for s in steps) == 1, f"steps={steps}")
check("no event sent for zero-pixel move", len(sent) == 1, f"sent={len(sent)}")

controller.press(mc.MouseButton.LEFT)
controller.press(mc.MouseButton.LEFT)
check("duplicate press is idempotent", len(controller.held_buttons) == 1)
controller.press(mc.MouseButton.RIGHT)
check("both buttons can be held", len(controller.held_buttons) == 2)
controller.release_all()
check("release_all clears everything", len(controller.held_buttons) == 0)
controller.release_all()
check("release_all is safe when nothing held", len(controller.held_buttons) == 0)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("All checks passed.")
