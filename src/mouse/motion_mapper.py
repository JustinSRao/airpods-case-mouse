"""Turns camera-space anchor motion into screen-pixel cursor motion.

Pipeline for one frame:

    anchor (px)
      -> smoothing filter
      -> delta against previous filtered anchor
      -> divide by palm width      (scale-invariant "hand units")
      -> dead zone                 (kill stationary jitter)
      -> acceleration gain         (precision when slow, reach when fast)
      -> sensitivity + axis signs  (hand units -> screen pixels)
      -> velocity clamp            (never fling the cursor across the screen)

Dividing by palm width before anything else is what makes the mapping
*relative and scale-free*: leaning closer to the camera makes the hand bigger
in pixels, but the ratio of movement to palm width is unchanged, so the cursor
responds identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.config.settings import CursorSettings
from src.mouse.cursor_filter import CursorFilter, build_filter

log = logging.getLogger(__name__)


@dataclass
class MotionResult:
    """Diagnostics for one mapping step, for the debug HUD."""

    delta_px: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hand_speed: float = 0.0  # palm-widths per second
    gain: float = 1.0
    in_dead_zone: bool = False
    clamped: bool = False

    @property
    def dx(self) -> float:
        return float(self.delta_px[0])

    @property
    def dy(self) -> float:
        return float(self.delta_px[1])


class MotionMapper:
    """Stateful anchor -> cursor-delta converter."""

    def __init__(self, settings: CursorSettings) -> None:
        self._settings = settings
        self._filter: CursorFilter = build_filter(
            settings.filter,
            smoothing=settings.smoothing,
            min_cutoff=settings.one_euro_min_cutoff,
            beta=settings.one_euro_beta,
            d_cutoff=settings.one_euro_d_cutoff,
        )
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        """Clear history so the next frame produces no motion.

        Called whenever the hand is lost or control is toggled: without this,
        reacquiring the hand somewhere else in frame would produce one huge
        delta and hurl the cursor across the screen.
        """
        self._filter.reset()
        self._previous = None

    def update(self, anchor_px: np.ndarray, scale_px: float, dt: float) -> MotionResult:
        """Map this frame's anchor position to a screen-pixel cursor delta."""
        if dt <= 0.0:
            return MotionResult()

        s = self._settings
        filtered = self._filter.apply(np.asarray(anchor_px, dtype=np.float64), dt)

        if self._previous is None:
            self._previous = filtered
            return MotionResult()

        delta_hand = (filtered - self._previous) / scale_px
        self._previous = filtered

        magnitude = float(np.linalg.norm(delta_hand))
        if magnitude < s.dead_zone:
            return MotionResult(hand_speed=magnitude / dt, in_dead_zone=True)

        # Subtract the dead zone instead of thresholding outright, so motion
        # ramps up from zero rather than jumping the moment it is exceeded.
        delta_hand *= (magnitude - s.dead_zone) / magnitude

        hand_speed = magnitude / dt
        gain = 1.0 + s.acceleration * hand_speed

        axis_scale = np.array(
            [
                s.sensitivity * s.x_sensitivity * (-1.0 if s.invert_x else 1.0),
                s.sensitivity * s.y_sensitivity * (-1.0 if s.invert_y else 1.0),
            ]
        )
        delta_px = delta_hand * gain * axis_scale

        clamped = False
        max_step = s.max_velocity * dt
        step_magnitude = float(np.linalg.norm(delta_px))
        if step_magnitude > max_step:
            delta_px *= max_step / step_magnitude
            clamped = True

        return MotionResult(
            delta_px=delta_px,
            hand_speed=hand_speed,
            gain=gain,
            in_dead_zone=False,
            clamped=clamped,
        )
