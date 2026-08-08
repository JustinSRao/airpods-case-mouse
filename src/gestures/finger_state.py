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


class RateTracker:
    """Smoothed time derivative of a metric, in units per second.

    On a rigid object a held press and a rest are geometrically the same --
    the finger is in the same place either way, so no measurement of *level*
    can tell them apart. All the information is in the movement between them.
    Differentiating amplifies per-frame noise, so the value is smoothed before
    differencing and the derivative is smoothed again after.
    """

    def __init__(self, signal_time_constant: float, rate_time_constant: float) -> None:
        if signal_time_constant < 0 or rate_time_constant < 0:
            raise ValueError("time constants must be >= 0")
        self._signal_tc = signal_time_constant
        self._rate_tc = rate_time_constant
        self._smoothed: float | None = None
        self._rate = 0.0

    @property
    def rate(self) -> float:
        return self._rate

    def update(self, value: float, dt: float) -> float:
        if self._smoothed is None:
            self._smoothed = value
            return 0.0
        if dt <= 0:
            return self._rate

        previous = self._smoothed
        alpha = dt / (self._signal_tc + dt) if self._signal_tc > 0 else 1.0
        self._smoothed += alpha * (value - self._smoothed)

        instantaneous = (self._smoothed - previous) / dt
        beta = dt / (self._rate_tc + dt) if self._rate_tc > 0 else 1.0
        self._rate += beta * (instantaneous - self._rate)
        return self._rate

    def reset(self) -> None:
        self._smoothed = None
        self._rate = 0.0


@dataclass
class RateThresholds:
    """Signed rate thresholds for transient press detection.

    ``press`` is positive (finger moving in the pressing direction) and
    ``release`` is negative (moving back). They are not a hysteresis pair on
    one level -- they are two opposite-direction events.
    """

    press: float
    release: float
    min_state_duration: float = 0.08

    def validate(self, name: str) -> None:
        if self.press <= 0:
            raise ValueError(f"{name}: press rate must be positive, got {self.press}")
        if self.release >= 0:
            raise ValueError(
                f"{name}: release rate must be negative, got {self.release}"
            )


class TransientPressStateMachine:
    """Latches on a movement in one direction, unlatches on the opposite one.

    Unlike the level machine, the state persists with no ongoing evidence:
    once a press transient is seen the button stays down until a release
    transient arrives. That is the only workable model when the held state is
    invisible, and it is why the tracking-loss release path matters even more
    here -- nothing else will spontaneously clear it.
    """

    def __init__(self, thresholds: RateThresholds, name: str = "finger") -> None:
        thresholds.validate(name)
        self._thresholds = thresholds
        self._state = PressState.UP
        self._last_change = 0.0

    @property
    def state(self) -> PressState:
        return self._state

    def update(self, rate: float, now: float) -> PressEvent | None:
        if now - self._last_change < self._thresholds.min_state_duration:
            return None

        if self._state is PressState.UP and rate > self._thresholds.press:
            self._state = PressState.DOWN
            self._last_change = now
            return PressEvent.PRESSED

        if self._state is PressState.DOWN and rate < self._thresholds.release:
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

    def force_release(self, now: float) -> PressEvent | None:
        """Drop to UP regardless of thresholds or debounce.

        Called when tracking is lost. Returns RELEASED if the state actually
        changed, so the caller knows to send the matching mouse-up.
        """
        event = self._machine.force_release(now)
        if event is not None:
            log.debug("%s: forced release", self._name)
        return event

    def reset(self) -> None:
        """Forget the baseline (call when tracking is lost and reacquired)."""
        self._baseline.reset()
        self._last_update = None
        self._deviation = 0.0


class TransientPressDetector:
    """Press detection from movement rate. Same interface as PressDetector."""

    def __init__(
        self,
        name: str,
        thresholds: RateThresholds,
        signal_time_constant: float = 0.06,
        rate_time_constant: float = 0.06,
    ) -> None:
        self._name = name
        self._thresholds = thresholds
        self._rate = RateTracker(signal_time_constant, rate_time_constant)
        self._machine = TransientPressStateMachine(thresholds, name)
        self._last_update: float | None = None
        self._raw = 0.0

    @property
    def state(self) -> PressState:
        return self._machine.state

    @property
    def metric(self) -> float:
        """Current rate -- what the thresholds compare against."""
        return self._rate.rate

    @property
    def raw_metric(self) -> float:
        return self._raw

    @property
    def baseline(self) -> float | None:
        return None

    @property
    def thresholds(self) -> RateThresholds:
        return self._thresholds

    def update(self, metric: float, now: float) -> PressEvent | None:
        dt = 0.0 if self._last_update is None else max(now - self._last_update, 0.0)
        self._last_update = now
        self._raw = metric
        return self._machine.update(self._rate.update(metric, dt), now)

    def force_release(self, now: float) -> PressEvent | None:
        event = self._machine.force_release(now)
        if event is not None:
            log.debug("%s: forced release", self._name)
        return event

    def reset(self) -> None:
        self._rate.reset()
        self._last_update = None
