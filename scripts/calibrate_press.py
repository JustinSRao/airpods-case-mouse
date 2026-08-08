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

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def percentile(ordered: list[float], fraction: float) -> float:
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def collect(
    camera: CameraManager,
    tracker: HandTracker,
    prompt: str,
    subtitle: str,
    seconds: float,
    finger: str,
    start_time: float,
) -> dict[str, list[float]]:
    """Record every candidate metric for a fixed duration."""
    samples: dict[str, list[float]] = {name: [] for name in PRESS_METRIC_NAMES}
    deadline = time.perf_counter() + seconds

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        frame = camera.read()
        if frame is None:
            continue

        hand = tracker.select_hand(
            tracker.process(frame, int((time.perf_counter() - start_time) * 1000))
        )
        if hand is not None:
            for name in PRESS_METRIC_NAMES:
                samples[name].append(press_metric(hand.world_landmarks, finger, name))

        count = len(samples[PRESS_METRIC_NAMES[0]])
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (25, 25, 25), -1)
        cv2.putText(frame, prompt, (10, 26), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (10, 52), _FONT, 0.5, (0, 190, 255), 1, cv2.LINE_AA)
        status = (
            f"{remaining:4.1f}s   samples {count:4d}"
            if hand is not None
            else f"{remaining:4.1f}s   NO HAND DETECTED"
        )
        cv2.putText(
            frame,
            status,
            (10, 80),
            _FONT,
            0.55,
            (80, 220, 100) if hand is not None else (60, 60, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    return samples


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
    parser.add_argument("--rest-seconds", type=float, default=5.0)
    parser.add_argument("--press-seconds", type=float, default=5.0)
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

    print(f"\nCalibrating: {', '.join(fingers)}")
    print("Sit exactly as you will when using the mouse. ESC aborts.\n")

    chosen: dict[str, tuple[str, float, float]] = {}

    try:
        with (
            CameraManager(settings.camera) as camera,
            HandTracker(settings.tracking) as tracker,
        ):
            start_time = time.perf_counter()
            for finger in fingers:
                countdown(camera, f"[{finger}]  REST - relax, do NOT press")
                rest = collect(
                    camera,
                    tracker,
                    f"[{finger}]  REST",
                    "relax the finger, do NOT press",
                    args.rest_seconds,
                    finger,
                    start_time,
                )

                countdown(camera, f"[{finger}]  PRESS - hold it DOWN the whole time")
                press = collect(
                    camera,
                    tracker,
                    f"[{finger}]  PRESS and HOLD",
                    "keep it pressed for the whole phase",
                    args.press_seconds,
                    finger,
                    start_time,
                )

                n_rest = len(rest[PRESS_METRIC_NAMES[0]])
                n_press = len(press[PRESS_METRIC_NAMES[0]])
                print(f"\n  {finger}: {n_rest} rest samples, {n_press} press samples")
                if n_rest < 20 or n_press < 20:
                    print("    !! too few samples - was the hand tracked throughout?")
                    continue

                scores = [
                    MetricScore(name, rest[name], press[name])
                    for name in PRESS_METRIC_NAMES
                ]
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
