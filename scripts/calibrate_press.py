"""Measure your index/middle press and write thresholds to config/config.json.

    .\\.venv\\Scripts\\python.exe -m scripts.calibrate_press

Never sends mouse input -- it only records. Two phases per finger:

  1. REST   -- hand on the case, fingers relaxed. Captures your natural
               resting geometry, which is often already quite curled.
  2. PRESS  -- press and hold, release, repeat. Captures how far the metric
               actually travels when you mean it.

Thresholds are then placed inside the gap between those two distributions,
biased high so that a stray click (which lands on whatever is under the
cursor) is much less likely than a missed press (which you just repeat).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time

import cv2
import numpy as np

from src.camera.camera_manager import CameraManager
from src.config.settings import USER_CONFIG_PATH, AppSettings
from src.tracking.hand_features import finger_flexion, fingertip_drop
from src.tracking.hand_tracker import HandTracker

log = logging.getLogger("calibrate")

# Where in the rest->press gap each threshold sits. Press is placed well above
# the midpoint so ordinary resting jitter cannot reach it; release sits lower,
# and the space between them is the hysteresis band.
PRESS_FRACTION = 0.55
RELEASE_FRACTION = 0.30

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def metric_value(hand, metric: str, finger: str) -> float:
    if metric == "flexion":
        return finger_flexion(hand.world_landmarks, finger)
    if metric == "drop":
        return fingertip_drop(hand.world_landmarks, finger)
    raise ValueError(f"Unknown metric {metric!r}; expected 'flexion' or 'drop'")


def collect(
    camera: CameraManager,
    tracker: HandTracker,
    prompt: str,
    seconds: float,
    metric: str,
    finger: str,
    start_time: float,
) -> list[float]:
    """Show a prompt and record metric samples for a fixed duration."""
    samples: list[float] = []
    deadline = time.perf_counter() + seconds

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        frame = camera.read()
        if frame is None:
            continue

        timestamp_ms = int((time.perf_counter() - start_time) * 1000)
        hand = tracker.select_hand(tracker.process(frame, timestamp_ms))

        value = None
        if hand is not None:
            value = metric_value(hand, metric, finger)
            samples.append(value)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 74), (25, 25, 25), -1)
        cv2.putText(frame, prompt, (10, 26), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        status = (
            f"{remaining:4.1f}s   samples {len(samples):4d}   value {value:8.2f}"
            if value is not None
            else f"{remaining:4.1f}s   NO HAND DETECTED"
        )
        colour = (80, 220, 100) if value is not None else (60, 60, 240)
        cv2.putText(frame, status, (10, 58), _FONT, 0.55, colour, 1, cv2.LINE_AA)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    return samples


def countdown(camera: CameraManager, message: str, seconds: float = 3.0) -> None:
    """Give the user time to get into position between phases."""
    deadline = time.perf_counter() + seconds
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        frame = camera.read()
        if frame is None:
            continue
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 74), (25, 25, 25), -1)
        cv2.putText(frame, message, (10, 26), _FONT, 0.6, (0, 190, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"starting in {remaining:.0f}...",
            (10, 58),
            _FONT,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt


def summarise(name: str, samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p05": ordered[int(len(ordered) * 0.05)],
        "median": statistics.median(ordered),
        "p95": ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)],
        "max": ordered[-1],
    }


def derive_thresholds(
    rest: list[float], press: list[float], finger: str
) -> tuple[float, float] | None:
    """Place press/release inside the gap between the two distributions."""
    rest_stats = summarise("rest", rest)
    press_stats = summarise("press", press)

    # Use the tails that face each other: the highest resting values and the
    # lowest pressing values. If those overlap, the metric cannot separate the
    # two states and no threshold will work.
    rest_high = rest_stats["p95"]
    press_low = press_stats["p05"]

    print(f"\n  {finger} REST : " + "  ".join(
        f"{k}={v:.2f}" for k, v in rest_stats.items() if k != "n"
    ))
    print(f"  {finger} PRESS: " + "  ".join(
        f"{k}={v:.2f}" for k, v in press_stats.items() if k != "n"
    ))
    print(f"  separation: rest p95 = {rest_high:.2f} -> press p05 = {press_low:.2f}")

    gap = press_low - rest_high
    if gap <= 0:
        print(
            f"  !! {finger}: rest and press OVERLAP (gap {gap:.2f}). "
            "No threshold can separate them."
        )
        return None

    press_threshold = rest_high + gap * PRESS_FRACTION
    release_threshold = rest_high + gap * RELEASE_FRACTION
    print(
        f"  -> press > {press_threshold:.2f}, release < {release_threshold:.2f} "
        f"(gap {gap:.2f})"
    )
    return press_threshold, release_threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default=None, help="flexion | drop")
    parser.add_argument("--rest-seconds", type=float, default=5.0)
    parser.add_argument("--press-seconds", type=float, default=8.0)
    parser.add_argument(
        "--fingers",
        default="index,middle",
        help="Comma-separated fingers to calibrate.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Measure but do not write config."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    settings = AppSettings.load()
    metric = args.metric or settings.gestures.metric
    fingers = [f.strip() for f in args.fingers.split(",") if f.strip()]

    print(f"\nCalibrating {', '.join(fingers)} using metric '{metric}'.")
    print("Sit exactly as you will when using the mouse. ESC aborts.\n")

    results: dict[str, tuple[float, float]] = {}

    try:
        with (
            CameraManager(settings.camera) as camera,
            HandTracker(settings.tracking) as tracker,
        ):
            start_time = time.perf_counter()
            for finger in fingers:
                countdown(camera, f"[{finger}] REST - hand on case, relaxed")
                rest = collect(
                    camera,
                    tracker,
                    f"[{finger}] REST - do NOT press",
                    args.rest_seconds,
                    metric,
                    finger,
                    start_time,
                )

                countdown(
                    camera, f"[{finger}] PRESS - press and release repeatedly"
                )
                press = collect(
                    camera,
                    tracker,
                    f"[{finger}] PRESS and release, over and over",
                    args.press_seconds,
                    metric,
                    finger,
                    start_time,
                )

                if len(rest) < 20 or len(press) < 20:
                    print(
                        f"  !! {finger}: not enough samples "
                        f"(rest {len(rest)}, press {len(press)}). "
                        "Was your hand tracked the whole time?"
                    )
                    continue

                derived = derive_thresholds(rest, press, finger)
                if derived is not None:
                    results[finger] = derived
    except KeyboardInterrupt:
        print("\nAborted.")
        return 1
    finally:
        cv2.destroyAllWindows()

    if not results:
        print("\nNothing calibrated. Nothing written.")
        return 1

    if args.dry_run:
        print("\n--dry-run: config not written.")
        return 0

    existing: dict = {}
    if USER_CONFIG_PATH.is_file():
        with USER_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)

    gestures = existing.setdefault("gestures", {})
    gestures["metric"] = metric
    for finger, (press_threshold, release_threshold) in results.items():
        gestures[f"{finger}_press_threshold"] = round(press_threshold, 3)
        gestures[f"{finger}_release_threshold"] = round(release_threshold, 3)
    # Only enable clicking once every requested finger actually separated.
    gestures["enabled"] = len(results) == len(fingers)

    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USER_CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
        fh.write("\n")

    print(f"\nWrote {USER_CONFIG_PATH}")
    print(f"clicking enabled: {gestures['enabled']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
