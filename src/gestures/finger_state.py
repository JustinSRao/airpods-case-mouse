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


class BaselineTracker:
    """Slow-moving estimate of a finger's *resting* metric value.

    Absolute thresholds do not work here. Measured on a real hand, the resting
    value of every candidate metric drifts by more than a press changes it --
    posture shifts over a few seconds swamp the signal. What is stable is the
    *deviation* from where the finger has been sitting recently.

    The baseline is frozen while the finger is down. Otherwise it would climb
    to meet the pressed value and the press would "fade out" during a long
    drag, releasing on its own.
    """

    def __init__(self, time_constant: float, signal_time_constant: float = 0.0) -> None:
        if time_constant <= 0:
            raise ValueError(f"time_constant must be positive, got {time_constant}")
        if signal_time_constant < 0:
            raise ValueError(
                f"signal_time_constant must be >= 0, got {signal_time_constant}"
            )
        if signal_time_constant >= time_constant:
            raise ValueError(
                f"signal_time_constant ({signal_time_constant}) must be shorter than "
                f"time_constant ({time_constant}); otherwise the fast filter removes "
                "the press before the baseline can measure it"
            )
        self._time_constant = time_constant
        self._signal_time_constant = signal_time_constant
        self._baseline: float | None = None
        self._smoothed: float | None = None

    @property
    def value(self) -> float | None:
        return self._baseline

    @property
    def ready(self) -> bool:
        return self._baseline is not None

    def update(self, metric: float, dt: float, frozen: bool) -> float:
        """Advance the filters and return the current deviation."""
        if self._smoothed is None or self._baseline is None:
            self._smoothed = metric
            self._baseline = metric
            return 0.0

        # Fast lane: knock down per-frame landmark jitter. Landmark noise is
        # essentially white, so a short average cuts it by roughly sqrt(N),
        # while a press -- a sustained shift over ~1s -- passes through almost
        # untouched. Together with the slow baseline this is a band-pass tuned
        # to the timescale of a press.
        if self._signal_time_constant > 0 and dt > 0:
            alpha = dt / (self._signal_time_constant + dt)
            self._smoothed += alpha * (metric - self._smoothed)
        else:
            self._smoothed = metric

        # Slow lane: where the finger has been resting lately.
        if not frozen and dt > 0:
            alpha = dt / (self._time_constant + dt)
            self._baseline += alpha * (self._smoothed - self._baseline)

        return self._smoothed - self._baseline

    def reset(self) -> None:
        self._baseline = None
        self._smoothed = None


class PressEvent(Enum):
    """Emitted only on the frame the state actually changes."""

    PRESSED = "PRESSED"
    RELEASED = "RELEASED"


@dataclass
class PressThresholds:
    """Deviation-from-baseline thresholds for one finger.

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


class PressStateMachine:
    """Hysteresis + debounce over a scalar. No filtering, no baseline.

    Split out from PressDetector so calibration can replay it over recorded
    deviations and measure what actually matters -- false clicks and missed
    presses -- rather than guessing from percentile overlap.
    """

    def __init__(self, thresholds: PressThresholds, name: str = "finger") -> None:
        thresholds.validate(name)
        self._thresholds = thresholds
        self._state = PressState.UP
        self._last_change = 0.0

    @property
    def state(self) -> PressState:
        return self._state

    def update(self, deviation: float, now: float) -> PressEvent | None:
        if now - self._last_change < self._thresholds.min_state_duration:
            return None

        if self._state is PressState.UP and deviation > self._thresholds.press:
            self._state = PressState.DOWN
            self._last_change = now
            return PressEvent.PRESSED

        if self._state is PressState.DOWN and deviation < self._thresholds.release:
            self._state = PressState.UP
            self._last_change = now
            return PressEvent.RELEASED

        return None

    def force_release(self, now: float) -> PressEvent | None:
        if self._state is PressState.UP:
            return None
        self._state = PressState.UP
        self._last_change = now
        return PressEvent.RELEASED


class PressDetector:
    """Tracks one finger's press state from its deviation off a baseline.

    Thresholds are in *deviation* units, not raw metric units: how far the
    finger has moved from where it has been resting, not where it is.
    """

    def __init__(
        self,
        name: str,
        thresholds: PressThresholds,
        baseline_time_constant: float = 1.0,
        signal_time_constant: float = 0.0,
    ) -> None:
        thresholds.validate(name)
        self._name = name
        self._thresholds = thresholds
        self._baseline = BaselineTracker(
            baseline_time_constant, signal_time_constant
        )
        self._machine = PressStateMachine(thresholds, name)
        self._last_update: float | None = None
        self._deviation = 0.0
        self._raw = 0.0

    @property
    def state(self) -> PressState:
        return self._machine.state

    @property
    def metric(self) -> float:
        """Current deviation from baseline -- what the thresholds compare to."""
        return self._deviation

    @property
    def raw_metric(self) -> float:
        return self._raw

    @property
    def baseline(self) -> float | None:
        return self._baseline.value

    @property
    def thresholds(self) -> PressThresholds:
        return self._thresholds

    def update(self, metric: float, now: float) -> PressEvent | None:
        """Feed one frame's raw metric; returns an event only on a transition."""
        dt = 0.0 if self._last_update is None else max(now - self._last_update, 0.0)
        self._last_update = now
        self._raw = metric

        # Freeze the baseline while pressed, so a long hold cannot decay away.
        self._deviation = self._baseline.update(
            metric, dt, frozen=self._machine.state is PressState.DOWN
        )
        return self._machine.update(self._deviation, now)

    def reset(self) -> None:
        """Forget the baseline (call when tracking is lost and reacquired)."""
        self._baseline.reset()
        self._last_update = None
        self._deviation = 0.0

    def force_release(self, now: float) -> PressEvent | None:
        """Drop to UP regardless of thresholds or debounce.

        Called when tracking is lost. Returns RELEASED if the state actually
        changed, so the caller knows to send the matching mouse-up.
        """
        event = self._machine.force_release(now)
        if event is not None:
            log.debug("%s: forced release", self._name)
        return event
