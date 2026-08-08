"""Finger press state machine.

A press is a *transition*, not a threshold comparison on one frame. Comparing
per-frame would chatter around the boundary and fire a burst of clicks, so
this is an explicit two-state machine with:

* **hysteresis** -- the press and release thresholds differ, so noise inside
  the gap cannot flip the state back and forth;
* **debounce** -- a minimum dwell time in each state, which kills the
  remaining chatter from a single noisy frame;
* **a forced release** -- so a lost hand can never strand a held button.

False-positive clicks are much worse than missed presses: a stray click lands
on whatever is under the cursor, while a missed press just needs repeating.
Everything here is biased accordingly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


class PressState(Enum):
    UP = "UP"
    DOWN = "DOWN"


class PressEvent(Enum):
    """Emitted only on the frame the state actually changes."""

    PRESSED = "PRESSED"
    RELEASED = "RELEASED"


@dataclass
class PressThresholds:
    """Thresholds in whatever units the active metric uses.

    ``press`` must be strictly greater than ``release``; the gap between them
    is the hysteresis band.
    """

    press: float
    release: float
    min_state_duration: float = 0.05

    def validate(self, name: str) -> None:
        if self.press <= self.release:
            raise ValueError(
                f"{name}: press threshold ({self.press}) must be greater than "
                f"release threshold ({self.release}); without a gap there is no "
                "hysteresis and the state will chatter"
            )


class PressDetector:
    """Tracks one finger's press state from a scalar metric."""

    def __init__(self, name: str, thresholds: PressThresholds) -> None:
        thresholds.validate(name)
        self._name = name
        self._thresholds = thresholds
        self._state = PressState.UP
        self._last_change = 0.0
        self._metric = 0.0

    @property
    def state(self) -> PressState:
        return self._state

    @property
    def metric(self) -> float:
        """Last metric value seen, for the debug HUD."""
        return self._metric

    @property
    def thresholds(self) -> PressThresholds:
        return self._thresholds

    def update(self, metric: float, now: float) -> PressEvent | None:
        """Feed one frame's metric; returns an event only on a transition."""
        self._metric = metric

        # Debounce: refuse to change state again too soon after the last one.
        if now - self._last_change < self._thresholds.min_state_duration:
            return None

        if self._state is PressState.UP and metric > self._thresholds.press:
            self._state = PressState.DOWN
            self._last_change = now
            return PressEvent.PRESSED

        if self._state is PressState.DOWN and metric < self._thresholds.release:
            self._state = PressState.UP
            self._last_change = now
            return PressEvent.RELEASED

        return None

    def force_release(self, now: float) -> PressEvent | None:
        """Drop to UP regardless of thresholds or debounce.

        Called when tracking is lost. Returns RELEASED if the state actually
        changed, so the caller knows to send the matching mouse-up.
        """
        if self._state is PressState.UP:
            return None
        self._state = PressState.UP
        self._last_change = now
        log.debug("%s: forced release", self._name)
        return PressEvent.RELEASED
