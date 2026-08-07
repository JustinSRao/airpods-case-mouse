"""Swappable 2D smoothing filters for the tracking anchor.

We filter the anchor *position* rather than the frame-to-frame delta. Position
filtering is what these algorithms are designed for, and differentiating a
smoothed position gives a smoother delta than smoothing a noisy delta does.

Every filter implements ``apply(value, dt) -> value`` and ``reset()``. Swapping
one for another is a single config change.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class CursorFilter(ABC):
    """Base class for 2D position filters."""

    @abstractmethod
    def apply(self, value: np.ndarray, dt: float) -> np.ndarray:
        """Filter one sample. ``dt`` is seconds since the previous sample."""

    @abstractmethod
    def reset(self) -> None:
        """Forget all history (call when tracking is lost and reacquired)."""


class NoFilter(CursorFilter):
    """Pass-through, for measuring how much smoothing actually costs."""

    def apply(self, value: np.ndarray, dt: float) -> np.ndarray:
        return value

    def reset(self) -> None:
        return


class EmaFilter(CursorFilter):
    """Exponential moving average.

    ``alpha`` is the weight of the newest sample: 1.0 is no smoothing, and
    smaller values smooth harder at the cost of lag. Simple and predictable,
    but the lag/smoothness tradeoff is fixed regardless of how fast the hand
    is moving -- which is exactly what One Euro fixes.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._state: np.ndarray | None = None

    def apply(self, value: np.ndarray, dt: float) -> np.ndarray:
        if self._state is None:
            self._state = value.astype(np.float64).copy()
        else:
            self._state += self._alpha * (value - self._state)
        return self._state.copy()

    def reset(self) -> None:
        self._state = None


class _LowPass:
    """Scalar exponential low-pass with an externally supplied alpha."""

    def __init__(self) -> None:
        self.value: float | None = None

    def apply(self, sample: float, alpha: float) -> float:
        if self.value is None:
            self.value = sample
        else:
            self.value += alpha * (sample - self.value)
        return self.value

    def reset(self) -> None:
        self.value = None


class OneEuroFilter(CursorFilter):
    """One Euro filter (Casiez, Roussel & Vogel, CHI 2012).

    An adaptive low-pass: the cutoff frequency rises with the observed speed,
    so a stationary hand is smoothed heavily (no jitter) while a fast movement
    is barely smoothed at all (no lag). That is the right tradeoff for a
    pointing device, which is why it is the standard choice for this problem.

    Parameters:
        min_cutoff: Cutoff in Hz at zero speed. Lower = less jitter at rest.
        beta:       How aggressively the cutoff tracks speed. Raise this if
                    fast movements feel laggy.
        d_cutoff:   Cutoff for the internal speed estimate.
    """

    def __init__(
        self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x = [_LowPass(), _LowPass()]
        self._dx = [_LowPass(), _LowPass()]
        self._prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        # Standard discrete low-pass coefficient for a given cutoff and rate.
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, value: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0.0:
            dt = 1e-3

        out = np.empty(2, dtype=np.float64)
        for axis in range(2):
            sample = float(value[axis])
            previous = float(self._prev[axis]) if self._prev is not None else sample

            speed = (sample - previous) / dt
            smoothed_speed = self._dx[axis].apply(
                speed, self._alpha(self._d_cutoff, dt)
            )
            cutoff = self._min_cutoff + self._beta * abs(smoothed_speed)
            out[axis] = self._x[axis].apply(sample, self._alpha(cutoff, dt))

        self._prev = value.astype(np.float64).copy()
        return out

    def reset(self) -> None:
        for low_pass in (*self._x, *self._dx):
            low_pass.reset()
        self._prev = None


def build_filter(name: str, *, smoothing: float, min_cutoff: float, beta: float, d_cutoff: float) -> CursorFilter:
    """Construct the filter named in the config."""
    key = name.lower()
    if key == "none":
        return NoFilter()
    if key == "ema":
        return EmaFilter(alpha=smoothing)
    if key == "one_euro":
        return OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    raise ValueError(
        f"Unknown cursor filter {name!r}; expected 'ema', 'one_euro' or 'none'"
    )
