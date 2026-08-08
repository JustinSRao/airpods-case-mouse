"""Measure your press and write per-finger metric + thresholds to config.json.

    .\\.venv\\Scripts\\python.exe -m scripts.calibrate_press

Never sends mouse input -- it only records. Two phases per finger:

  1. REST        -- hand on the case, finger relaxed, do not press.
  2. PRESS+HOLD  -- press the finger down and HOLD it for the whole phase.

Holding matters. An earlier version asked for repeated press-and-release,
which meant the "press" samples contained released frames too, so the two
distributions overlapped by construction and no threshold could ever be
found.

Every candidate metric is recorded on the same frames and ranked by how
cleanly it separates the two phases. There is no reason to assume the same
metric wins for every finger or camera angle, so the winner is chosen per
finger from the data.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass

import cv2

from src.camera.camera_manager import CameraManager
from src.config.settings import USER_CONFIG_PATH, AppSettings
from src.gestures.finger_state import (
    BaselineTracker,
    PressEvent,
    PressState,
    PressStateMachine,
    PressThresholds,
    RateTracker,
)
from src.tracking.hand_features import PRESS_METRIC_NAMES, press_metric
from src.tracking.hand_tracker import HandTracker

log = logging.getLogger("calibrate")

# Where in the rest->press gap each threshold sits. Press sits above the
# midpoint so ordinary resting jitter cannot reach it; the space down to
# release is the hysteresis band.
PRESS_FRACTION = 0.55
RELEASE_FRACTION = 0.30

# Tail percentiles used for the separation test. Comparing the extremes that
# face each other is what guarantees the states do not overlap in practice.
REST_TAIL = 0.95
PRESS_TAIL = 0.05

# Cap on the timestep used when replaying the baseline, so gaps left by
# discarded settle frames cannot advance it in one large jump.
MAX_REPLAY_DT = 1.0 / 25.0

# Minimum effect size before a metric is allowed to win, on top of simulating
# cleanly. With only a handful of press segments, a threshold search can fit
# noise: on recordings containing no press at all, simulation alone declared a
# clean threshold in roughly a quarter of trials. Those recordings have d'
# near zero, so requiring real separation as well removes them while leaving
# genuine signal (measured d' ~3 at 720p) untouched.
MIN_DPRIME = 1.5

# Where the rate threshold sits between quiet noise and the weakest observed
# transient. Biased low-ish so a soft press still registers, but the clean
# test still requires it to clear the noise ceiling outright.
RATE_FRACTION = 0.45

# A transient must beat quiet noise by at least this factor to be trusted.
MIN_RATE_MARGIN = 2.5

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def percentile(ordered: list[float], fraction: float) -> float:
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def run_cycles(
    camera: CameraManager,
    tracker: HandTracker,
    finger: str,
    start_time: float,
    cycles: int,
    rest_seconds: float,
    press_seconds: float,
    settle: float,
) -> tuple[dict[str, list[tuple[float, float, bool]]], list[tuple[float, bool]]]:
    """Alternate RELEASE and PRESS several times, recording every metric.

    Cycling rather than one long phase of each is deliberate: it exercises the
    baseline the same way real use does, and repeated transitions expose a
    press that a single sustained hold would let the baseline absorb.

    Frames within ``settle`` seconds of a phase change are dropped, since the
    hand is mid-transition and belongs to neither class.
    """
    recorded: dict[str, list[tuple[float, float, bool]]] = {
        name: [] for name in PRESS_METRIC_NAMES
    }
    cues: list[tuple[float, bool]] = []

    schedule: list[tuple[str, float, bool]] = []
    for _ in range(cycles):
        schedule.append(("RELEASE - lift the finger off", rest_seconds, False))
        schedule.append(("PRESS - lift then press down", press_seconds, True))

    for label, duration, pressing in schedule:
        phase_start = time.perf_counter()
        cues.append((phase_start, pressing))
        deadline = phase_start + duration
        while True:
            now = time.perf_counter()
            remaining = deadline - now
            if remaining <= 0:
                break
            frame = camera.read()
            if frame is None:
                continue

            hand = tracker.select_hand(
                tracker.process(frame, int((now - start_time) * 1000))
            )
            # Transient mode needs the moment of movement, which is exactly
            # what the settle window used to throw away. Keep every frame and
            # let the analysis decide which part of each phase to use.
            in_settle = settle > 0 and (now - phase_start) < settle
            if hand is not None and not in_settle:
                for name in PRESS_METRIC_NAMES:
                    recorded[name].append(
                        (
                            now,
                            press_metric(
                                hand.landmarks_px,
                                hand.world_landmarks,
                                finger,
                                name,
                            ),
                            pressing,
                        )
                    )

            banner = (60, 40, 140) if pressing else (40, 40, 40)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), banner, -1)
            cv2.putText(
                frame,
                f"[{finger}]  {label}",
                (10, 34),
                _FONT,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if hand is None:
                status, colour = "NO HAND DETECTED", (60, 60, 240)
            elif in_settle:
                status, colour = f"{remaining:4.1f}s  (settling)", (0, 190, 255)
            else:
                status, colour = (
                    f"{remaining:4.1f}s  recording",
                    (80, 220, 100),
                )
            cv2.putText(frame, status, (10, 74), _FONT, 0.6, colour, 1, cv2.LINE_AA)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt

    return recorded, cues


def countdown(camera: CameraManager, message: str, seconds: float = 3.0) -> None:
    deadline = time.perf_counter() + seconds
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        frame = camera.read()
        if frame is None:
            continue
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (25, 25, 25), -1)
        cv2.putText(frame, message, (10, 30), _FONT, 0.6, (0, 190, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"starting in {remaining:.0f}...",
            (10, 68),
            _FONT,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt


def replay_deviations(
    samples: list[tuple[float, float, bool]],
    time_constant: float,
    signal_time_constant: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Re-run the runtime baseline algorithm over recorded samples.

    ``samples`` is (timestamp, raw_metric, is_press_phase). Returns the
    deviations seen during rest and during press, so the thresholds are
    derived from exactly the quantity the detector will compare at runtime
    rather than from raw values it never sees.
    """
    tracker = BaselineTracker(time_constant, signal_time_constant)
    series: list[tuple[float, float, bool]] = []
    previous_time: float | None = None

    for timestamp, value, pressing in samples:
        dt = 0.0 if previous_time is None else max(timestamp - previous_time, 0.0)
        previous_time = timestamp
        # Settle frames are discarded, so consecutive timestamps can straddle
        # a long gap. Feeding that raw would advance the baseline by a huge
        # alpha in one step -- right at a phase change, which is exactly where
        # it would swallow the press. Treat the recording as continuous.
        dt = min(dt, MAX_REPLAY_DT)

        # Freeze on THIS sample's label. Using the previous one let the
        # baseline take one un-frozen step toward the already-pressed value.
        deviation = tracker.update(value, dt, frozen=pressing)
        series.append((timestamp, deviation, pressing))

    return series


def split_by_label(
    series: list[tuple[float, float, bool]]
) -> tuple[list[float], list[float]]:
    rest = [d for _, d, pressing in series if not pressing]
    press = [d for _, d, pressing in series if pressing]
    return rest, press


def segments(series: list[tuple[float, float, bool]]) -> list[tuple[int, int, bool]]:
    """Contiguous runs of the same label, as (start, end, pressing)."""
    runs: list[tuple[int, int, bool]] = []
    start = 0
    for i in range(1, len(series) + 1):
        if i == len(series) or series[i][2] != series[start][2]:
            runs.append((start, i, series[start][2]))
            start = i
    return runs


@dataclass
class ThresholdResult:
    """What a candidate threshold pair actually does to the recording."""

    press: float
    release: float
    false_clicks: int
    detected_presses: int
    total_presses: int
    missed_releases: int

    @property
    def clean(self) -> bool:
        """No stray clicks, every press caught, every one let go again."""
        return (
            self.false_clicks == 0
            and self.detected_presses == self.total_presses
            and self.missed_releases == 0
        )

    @property
    def margin(self) -> float:
        return self.press - self.release


def simulate(
    series: list[tuple[float, float, bool]],
    press_threshold: float,
    release_threshold: float,
    min_state_duration: float,
) -> ThresholdResult:
    """Run the real state machine over the recording and count what happens.

    This is the objective that matters. Percentile overlap is a proxy; a
    false click is the actual cost, and hysteresis plus debounce change the
    answer enough that the proxy is misleading near the boundary.
    """
    machine = PressStateMachine(
        PressThresholds(press_threshold, release_threshold, min_state_duration)
    )
    runs = segments(series)
    detected = set()
    false_clicks = 0
    missed_releases = 0

    for run_index, (start, end, pressing) in enumerate(runs):
        for i in range(start, end):
            timestamp, deviation, _ = series[i]
            event = machine.update(deviation, timestamp)
            if event is PressEvent.PRESSED:
                if pressing:
                    detected.add(run_index)
                else:
                    false_clicks += 1
        # By the end of a rest run the button must be back up, or a press
        # leaked past its release and would still be held in real use.
        if not pressing and machine.state is PressState.DOWN:
            missed_releases += 1

    total_presses = sum(1 for _, _, pressing in runs if pressing)
    return ThresholdResult(
        press=press_threshold,
        release=release_threshold,
        false_clicks=false_clicks,
        detected_presses=len(detected),
        total_presses=total_presses,
        missed_releases=missed_releases,
    )


def replay_rates(
    samples: list[tuple[float, float, bool]],
    signal_tc: float,
    rate_tc: float,
) -> list[tuple[float, float, bool]]:
    """Re-run the runtime rate tracker over recorded samples."""
    tracker = RateTracker(signal_tc, rate_tc)
    series: list[tuple[float, float, bool]] = []
    previous_time: float | None = None
    for timestamp, value, pressing in samples:
        dt = 0.0 if previous_time is None else max(timestamp - previous_time, 0.0)
        previous_time = timestamp
        series.append((timestamp, tracker.update(value, min(dt, MAX_REPLAY_DT)), pressing))
    return series


@dataclass
class RateResult:
    """Thresholds for transient mode plus how well they separate."""

    press: float
    release: float
    down_peak: float
    up_peak: float
    quiet_peak: float

    @property
    def clean(self) -> bool:
        return self.press > self.quiet_peak and -self.release > self.quiet_peak

    @property
    def margin(self) -> float:
        """Smallest headroom between a real transient and quiet noise."""
        return min(self.down_peak, self.up_peak) / max(self.quiet_peak, 1e-9)


def analyse_rates(
    series: list[tuple[float, float, bool]],
    events: list[tuple[float, bool]],
    event_window: float,
    quiet_lead: float,
) -> RateResult | None:
    """Compare rate during cued transients against rate while holding still.

    ``events`` is (cue_time, is_press). Within ``event_window`` seconds of a
    cue the finger is moving; ``quiet_lead`` seconds before the next cue it is
    settled. Thresholds go between the two.
    """
    down_peaks: list[float] = []
    up_peaks: list[float] = []
    quiet: list[float] = []

    for index, (cue_time, is_press) in enumerate(events):
        next_cue = events[index + 1][0] if index + 1 < len(events) else float("inf")
        window = [
            rate
            for t, rate, _ in series
            if cue_time <= t < min(cue_time + event_window, next_cue)
        ]
        settled = [
            rate for t, rate, _ in series if next_cue - quiet_lead <= t < next_cue
        ]
        if not window:
            continue
        # A press should drive the rate positive, a release negative.
        (down_peaks if is_press else up_peaks).append(
            max(window) if is_press else -min(window)
        )
        quiet.extend(abs(rate) for rate in settled)

    if not down_peaks or not up_peaks or len(quiet) < 20:
        return None

    # Use the weakest transient, not the average: the threshold has to catch
    # the softest press the user actually makes.
    down_peak = min(down_peaks)
    up_peak = min(up_peaks)
    quiet_peak = percentile(sorted(quiet), 0.99)

    press = quiet_peak + (down_peak - quiet_peak) * RATE_FRACTION
    release = -(quiet_peak + (up_peak - quiet_peak) * RATE_FRACTION)
    return RateResult(press, release, down_peak, up_peak, quiet_peak)


def best_thresholds(
    series: list[tuple[float, float, bool]], min_state_duration: float
) -> ThresholdResult | None:
    """Search threshold pairs for one that clicks exactly when it should.

    Among clean candidates, prefer the widest hysteresis margin: that is the
    one with most room before noise starts producing stray clicks.
    """
    rest, press = split_by_label(series)
    if len(rest) < 20 or len(press) < 20:
        return None

    low = min(rest)
    high = max(press)
    if not high > low:
        return None

    candidates = [low + (high - low) * f / 40.0 for f in range(1, 40)]
    best: ThresholdResult | None = None
    for press_threshold in candidates:
        for release_threshold in candidates:
            if release_threshold >= press_threshold:
                continue
            result = simulate(
                series, press_threshold, release_threshold, min_state_duration
            )
            if not result.clean:
                continue
            if best is None or result.margin > best.margin:
                best = result
    return best


class MetricScore:
    """How well one metric separated rest from press."""

    def __init__(self, name: str, rest: list[float], press: list[float]) -> None:
        self.name = name
        rest_sorted = sorted(rest)
        press_sorted = sorted(press)

        self.rest_median = statistics.median(rest_sorted)
        self.press_median = statistics.median(press_sorted)
        self.rest_tail = percentile(rest_sorted, REST_TAIL)
        self.press_tail = percentile(press_sorted, PRESS_TAIL)
        self.gap = self.press_tail - self.rest_tail

        # d-prime: separation in units of pooled spread. Scale-free, so metrics
        # measured in degrees and in dimensionless ratios can be compared.
        rest_var = statistics.pvariance(rest_sorted) if len(rest_sorted) > 1 else 0.0
        press_var = statistics.pvariance(press_sorted) if len(press_sorted) > 1 else 0.0
        pooled = ((rest_var + press_var) / 2.0) ** 0.5
        self.dprime = (
            (self.press_median - self.rest_median) / pooled if pooled > 1e-12 else 0.0
        )

        # Normalise the gap by spread too, so it is comparable across metrics.
        self.normalised_gap = self.gap / pooled if pooled > 1e-12 else 0.0

    @property
    def usable(self) -> bool:
        """Press must sit above rest AND the facing tails must not overlap."""
        return self.gap > 0 and self.dprime > 0

    def thresholds(self) -> tuple[float, float]:
        return (
            self.rest_tail + self.gap * PRESS_FRACTION,
            self.rest_tail + self.gap * RELEASE_FRACTION,
        )

    def row(self) -> str:
        flag = "ok " if self.usable else "OVERLAP"
        return (
            f"    {self.name:<14} rest~{self.rest_median:8.2f}  "
            f"press~{self.press_median:8.2f}  gap={self.gap:8.2f}  "
            f"d'={self.dprime:6.2f}  {flag}"
        )


def analyse_transient(
    recorded: dict[str, list[tuple[float, float, bool]]],
    cues: list[tuple[float, bool]],
    finger: str,
    args: argparse.Namespace,
) -> tuple[str, float, float] | None:
    """Rank metrics by how far their movement rate clears quiet noise."""
    print(f"\n  {finger}: {len(cues)} cued transients")
    print("  (rate of change during the movement vs while holding still)\n")
    print(
        f"    {'metric':<17}{'down':>9}{'up':>9}{'quiet':>9}"
        f"{'margin':>8}{'ok':>5}"
    )

    ranked: list[tuple[float, str, RateResult]] = []
    for name in PRESS_METRIC_NAMES:
        series = replay_rates(
            recorded[name], args.rate_signal_time_constant, args.rate_time_constant
        )
        result = analyse_rates(series, cues, args.event_window, args.quiet_lead)
        if result is None:
            print(f"    {name:<17}{'-':>9}{'-':>9}{'-':>9}{'-':>8}{'no':>5}")
            continue
        ok = result.clean and result.margin >= MIN_RATE_MARGIN
        print(
            f"    {name:<17}{result.down_peak:>9.2f}{result.up_peak:>9.2f}"
            f"{result.quiet_peak:>9.2f}{result.margin:>8.2f}"
            f"{'YES' if ok else 'no':>5}"
        )
        if ok:
            ranked.append((result.margin, name, result))

    if args.metric:
        ranked = [r for r in ranked if r[1] == args.metric]
    if not ranked:
        print(
            f"    !! {finger}: no metric's movement clears the noise by "
            f"{MIN_RATE_MARGIN}x."
        )
        return None

    ranked.sort(reverse=True)
    margin, name, result = ranked[0]
    print(
        f"\n    -> '{name}': press rate > {result.press:.3f}, "
        f"release rate < {result.release:.3f}   (margin {margin:.1f}x noise)"
    )
    return (name, result.press, result.release)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest-seconds", type=float, default=2.5)
    parser.add_argument("--press-seconds", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument(
        "--settle",
        type=float,
        default=0.6,
        help="Seconds discarded after each phase change (reaction time).",
    )
    parser.add_argument("--baseline-time-constant", type=float, default=None)
    parser.add_argument("--signal-time-constant", type=float, default=None)
    parser.add_argument("--min-state-duration", type=float, default=None)
    parser.add_argument("--mode", choices=("transient", "level"), default=None)
    parser.add_argument("--rate-signal-time-constant", type=float, default=None)
    parser.add_argument("--rate-time-constant", type=float, default=None)
    parser.add_argument(
        "--event-window",
        type=float,
        default=1.0,
        help="Seconds after each cue in which the movement is expected.",
    )
    parser.add_argument(
        "--quiet-lead",
        type=float,
        default=0.7,
        help="Seconds before the next cue treated as settled/still.",
    )
    parser.add_argument("--fingers", default="index,middle")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--metric",
        default=None,
        help="Force a specific metric instead of picking the best one.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")

    settings = AppSettings.load()
    fingers = [f.strip() for f in args.fingers.split(",") if f.strip()]
    if args.baseline_time_constant is None:
        args.baseline_time_constant = settings.gestures.baseline_time_constant
    if args.signal_time_constant is None:
        args.signal_time_constant = settings.gestures.signal_time_constant
    if args.min_state_duration is None:
        args.min_state_duration = settings.gestures.min_state_duration
    if args.mode is None:
        args.mode = settings.gestures.mode
    if args.rate_signal_time_constant is None:
        args.rate_signal_time_constant = settings.gestures.rate_signal_time_constant
    if args.rate_time_constant is None:
        args.rate_time_constant = settings.gestures.rate_time_constant

    print(f"\nCalibrating: {', '.join(fingers)}")
    print(f"{args.cycles} relax/press cycles per finger. Follow the banner.")
    print("Sit exactly as you will when using the mouse. ESC aborts.\n")

    chosen: dict[str, tuple[str, float, float]] = {}

    try:
        with (
            CameraManager(settings.camera) as camera,
            HandTracker(settings.tracking) as tracker,
        ):
            start_time = time.perf_counter()
            for finger in fingers:
                countdown(
                    camera,
                    f"[{finger}]  follow the banner: relax / press, x{args.cycles}",
                    4.0,
                )
                recorded, cues = run_cycles(
                    camera,
                    tracker,
                    finger,
                    start_time,
                    cycles=args.cycles,
                    rest_seconds=args.rest_seconds,
                    press_seconds=args.press_seconds,
                    # Transient mode must keep the moment of movement.
                    settle=0.0 if args.mode == "transient" else args.settle,
                )

                if args.mode == "transient":
                    picked = analyse_transient(recorded, cues, finger, args)
                    if picked is not None:
                        chosen[finger] = picked
                    continue

                n_rest = sum(1 for s in recorded[PRESS_METRIC_NAMES[0]] if not s[2])
                n_press = sum(1 for s in recorded[PRESS_METRIC_NAMES[0]] if s[2])
                print(f"\n  {finger}: {n_rest} rest samples, {n_press} press samples")
                print("  (deviation from baseline; thresholds chosen by simulating")
                print("   the real state machine and counting false clicks)\n")
                print(
                    f"    {'metric':<17}{'d-prime':>8}{'clean':>7}"
                    f"{'press>':>9}{'release<':>10}{'margin':>8}"
                )

                evaluated = []
                for name in PRESS_METRIC_NAMES:
                    series = replay_deviations(
                        recorded[name],
                        args.baseline_time_constant,
                        args.signal_time_constant,
                    )
                    rest_dev, press_dev = split_by_label(series)
                    if len(rest_dev) < 20 or len(press_dev) < 20:
                        continue
                    score = MetricScore(name, rest_dev, press_dev)
                    # Gate on effect size before trusting the threshold search,
                    # which can otherwise fit noise across a few segments.
                    result = (
                        best_thresholds(series, args.min_state_duration)
                        if score.dprime >= MIN_DPRIME
                        else None
                    )
                    evaluated.append((score, result))

                if not evaluated:
                    print("    !! too few samples - was the hand tracked throughout?")
                    continue

                # Rank by whether a clean threshold exists first, then by how
                # much hysteresis margin it leaves, then by raw separation.
                evaluated.sort(
                    key=lambda pair: (
                        pair[1] is not None,
                        pair[1].margin if pair[1] else 0.0,
                        pair[0].dprime,
                    ),
                    reverse=True,
                )
                for score, result in evaluated:
                    if result is None:
                        print(
                            f"    {score.name:<17}{score.dprime:>8.2f}{'no':>7}"
                            f"{'-':>9}{'-':>10}{'-':>8}"
                        )
                    else:
                        print(
                            f"    {score.name:<17}{score.dprime:>8.2f}{'YES':>7}"
                            f"{result.press:>9.3f}{result.release:>10.3f}"
                            f"{result.margin:>8.3f}"
                        )

                if args.metric:
                    match = [p for p in evaluated if p[0].name == args.metric]
                    if not match or match[0][1] is None:
                        print(f"    !! forced metric {args.metric!r} has no clean threshold")
                        continue
                    picked_score, picked_result = match[0]
                else:
                    picked_score, picked_result = evaluated[0]
                    if picked_result is None:
                        print(
                            f"    !! {finger}: no metric gives a clean threshold. "
                            "See the note printed at the end."
                        )
                        continue

                print(
                    f"\n    -> '{picked_score.name}': press > {picked_result.press:.3f}, "
                    f"release < {picked_result.release:.3f}   "
                    f"({picked_result.detected_presses}/{picked_result.total_presses} "
                    f"presses caught, {picked_result.false_clicks} false clicks)"
                )
                chosen[finger] = (
                    picked_score.name,
                    picked_result.press,
                    picked_result.release,
                )
    except KeyboardInterrupt:
        print("\nAborted.")
        return 1
    finally:
        cv2.destroyAllWindows()

    if not chosen:
        print(
            "\nNothing calibrated, nothing written.\n"
            "If every metric overlapped, the press is not mechanically visible\n"
            "to the camera. Usually one of:\n"
            "  - the finger barely moves; try an exaggerated press to confirm\n"
            "    the pipeline works, then reduce\n"
            "  - the hand is too far away (want palm width 80-110 px)\n"
            "  - the camera cannot see the finger bend; tilt the screen so the\n"
            "    fingers are more side-on rather than straight end-on\n"
        )
        return 1

    existing: dict = {}
    if USER_CONFIG_PATH.is_file():
        with USER_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)

    gestures = existing.setdefault("gestures", {})
    gestures["mode"] = args.mode
    for finger, (metric, press_value, release_value) in chosen.items():
        gestures[f"{finger}_metric"] = metric
        if args.mode == "transient":
            gestures[f"{finger}_press_rate"] = round(press_value, 4)
            gestures[f"{finger}_release_rate"] = round(release_value, 4)
        else:
            gestures[f"{finger}_press_threshold"] = round(press_value, 3)
            gestures[f"{finger}_release_threshold"] = round(release_value, 3)
    gestures["enabled"] = len(chosen) == len(fingers)

    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USER_CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
        fh.write("\n")

    print(f"\nWrote {USER_CONFIG_PATH}")
    print(f"clicking enabled: {gestures['enabled']}")
    if not gestures["enabled"]:
        missing = sorted(set(fingers) - set(chosen))
        print(f"  (still uncalibrated: {', '.join(missing)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
