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

import cv2

from src.camera.camera_manager import CameraManager
from src.config.settings import USER_CONFIG_PATH, AppSettings
from src.gestures.finger_state import BaselineTracker
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
) -> dict[str, list[tuple[float, float, bool]]]:
    """Alternate REST and PRESS several times, recording every metric.

    Cycling rather than one long phase of each is deliberate: it exercises the
    baseline the same way real use does, and repeated transitions expose a
    press that a single sustained hold would let the baseline absorb.

    Frames within ``settle`` seconds of a phase change are dropped, since the
    hand is mid-transition and belongs to neither class.
    """
    recorded: dict[str, list[tuple[float, float, bool]]] = {
        name: [] for name in PRESS_METRIC_NAMES
    }

    schedule: list[tuple[str, float, bool]] = []
    for _ in range(cycles):
        schedule.append(("RELAX - do not press", rest_seconds, False))
        schedule.append(("PRESS DOWN now", press_seconds, True))

    for label, duration, pressing in schedule:
        phase_start = time.perf_counter()
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
            in_settle = (now - phase_start) < settle
            if hand is not None and not in_settle:
                for name in PRESS_METRIC_NAMES:
                    recorded[name].append(
                        (
                            now,
                            press_metric(hand.world_landmarks, finger, name),
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

    return recorded


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
    samples: list[tuple[float, float, bool]], time_constant: float
) -> tuple[list[float], list[float]]:
    """Re-run the runtime baseline algorithm over recorded samples.

    ``samples`` is (timestamp, raw_metric, is_press_phase). Returns the
    deviations seen during rest and during press, so the thresholds are
    derived from exactly the quantity the detector will compare at runtime
    rather than from raw values it never sees.
    """
    tracker = BaselineTracker(time_constant)
    rest_dev: list[float] = []
    press_dev: list[float] = []
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

        (press_dev if pressing else rest_dev).append(deviation)

    return rest_dev, press_dev


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
                recorded = run_cycles(
                    camera,
                    tracker,
                    finger,
                    start_time,
                    cycles=args.cycles,
                    rest_seconds=args.rest_seconds,
                    press_seconds=args.press_seconds,
                    settle=args.settle,
                )

                scores = []
                for name in PRESS_METRIC_NAMES:
                    rest_dev, press_dev = replay_deviations(
                        recorded[name], args.baseline_time_constant
                    )
                    if len(rest_dev) >= 20 and len(press_dev) >= 20:
                        scores.append(MetricScore(name, rest_dev, press_dev))

                n_rest = sum(1 for s in recorded[PRESS_METRIC_NAMES[0]] if not s[2])
                n_press = sum(1 for s in recorded[PRESS_METRIC_NAMES[0]] if s[2])
                print(f"\n  {finger}: {n_rest} rest samples, {n_press} press samples")
                print("  (thresholds are deviation from the rolling baseline)")
                if not scores:
                    print("    !! too few samples - was the hand tracked throughout?")
                    continue
                scores.sort(key=lambda s: s.normalised_gap, reverse=True)
                for score in scores:
                    print(score.row())

                if args.metric:
                    picked = next(s for s in scores if s.name == args.metric)
                    if not picked.usable:
                        print(f"    !! forced metric {args.metric!r} does not separate")
                        continue
                else:
                    usable = [s for s in scores if s.usable]
                    if not usable:
                        print(
                            f"    !! {finger}: NO metric separates rest from press. "
                            "See the note printed at the end."
                        )
                        continue
                    picked = usable[0]

                press_threshold, release_threshold = picked.thresholds()
                print(
                    f"    -> using '{picked.name}': "
                    f"press > {press_threshold:.2f}, release < {release_threshold:.2f}"
                )
                chosen[finger] = (picked.name, press_threshold, release_threshold)
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
    for finger, (metric, press_threshold, release_threshold) in chosen.items():
        gestures[f"{finger}_metric"] = metric
        gestures[f"{finger}_press_threshold"] = round(press_threshold, 3)
        gestures[f"{finger}_release_threshold"] = round(release_threshold, 3)
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
