"""Live view of every press metric, so you can see which one reacts.

    .\\.venv\\Scripts\\python.exe -m scripts.press_monitor

Never sends mouse input. For each finger and each candidate metric it shows
the raw value, the deviation from the rolling baseline (the quantity the
detector actually thresholds), and a bar with a peak marker.

Press your finger and watch which row moves. If nothing moves, no amount of
threshold tuning will help and the problem is physical -- the press is not
producing visible finger motion.

Keys:  R  reset baselines and peaks     ESC  quit
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
import time
from collections import deque

import cv2
import numpy as np

from src.camera.camera_manager import CameraManager
from src.config.settings import AppSettings
from src.gestures.finger_state import BaselineTracker
from src.tracking.hand_features import PRESS_METRIC_NAMES, press_metric
from src.tracking.hand_tracker import HandTracker

_FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_WIDTH = 620
ROW_HEIGHT = 24
BAR_LEFT = 250
BAR_WIDTH = 240

# Decay the observed peak so an old spike does not permanently rescale things.
PEAK_HALF_LIFE = 4.0

# Frames of history used to estimate each metric's noise floor.
NOISE_WINDOW = 150

# Bars are drawn on a FIXED scale in noise units, not auto-scaled. That is the
# whole point: jitter must look small and a real press must look big.
Z_FULL_SCALE = 8.0
# Deviation worth this many sigma is plausibly a real press rather than noise.
Z_SIGNIFICANT = 3.0


class MetricRow:
    """Baseline, deviation, and the deviation expressed in noise units.

    Showing raw deviation scaled to its own peak is useless here: a row that
    is pure jitter fills the bar exactly like a row carrying real signal. What
    matters is deviation *relative to that metric's own noise floor*, which is
    also the only way to compare degrees against dimensionless ratios.

    The noise floor is estimated from consecutive-frame differences. Landmark
    jitter is frame-to-frame white noise while a press unfolds over many
    frames, so differencing isolates the noise and a press barely contributes.
    The median absolute difference is used (times 1.4826/sqrt(2), the robust
    Gaussian sigma estimator) so that occasional large real movements do not
    inflate the estimate.
    """

    def __init__(
        self, finger: str, metric: str, time_constant: float, signal_tc: float
    ) -> None:
        self.finger = finger
        self.metric = metric
        self.tracker = BaselineTracker(time_constant, signal_tc)
        self.raw = 0.0
        self.deviation = 0.0
        self.peak_z = 0.0
        self._previous: float | None = None
        self._diffs: deque[float] = deque(maxlen=NOISE_WINDOW)

    @property
    def sigma(self) -> float:
        if len(self._diffs) < 20:
            return 0.0
        return 1.4826 * statistics.median(self._diffs) / math.sqrt(2.0)

    @property
    def z(self) -> float:
        sigma = self.sigma
        return self.deviation / sigma if sigma > 1e-12 else 0.0

    def update(self, value: float, dt: float) -> None:
        self.raw = value
        self.deviation = self.tracker.update(value, dt, frozen=False)
        if self._previous is not None:
            self._diffs.append(abs(value - self._previous))
        self._previous = value
        decay = 0.5 ** (dt / PEAK_HALF_LIFE)
        self.peak_z = max(self.peak_z * decay, abs(self.z))

    def reset(self) -> None:
        self.tracker.reset()
        self._diffs.clear()
        self._previous = None
        self.peak_z = 0.0


def draw_panel(rows: list[MetricRow], height: int, tracked: bool) -> np.ndarray:
    panel = np.full((height, PANEL_WIDTH, 3), 22, dtype=np.uint8)

    header = (
        "DEVIATION IN NOISE UNITS (sigma)" if tracked else "NO HAND DETECTED"
    )
    cv2.putText(
        panel,
        header,
        (10, 20),
        _FONT,
        0.48,
        (255, 255, 255) if tracked else (60, 60, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"bar = +/-{Z_FULL_SCALE:.0f} sigma   green past {Z_SIGNIFICANT:.0f}",
        (10, 37),
        _FONT,
        0.38,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel, "R reset   ESC quit", (10, 52), _FONT, 0.38, (150, 150, 150), 1, cv2.LINE_AA
    )

    y = 72
    current_finger = ""
    for row in rows:
        if row.finger != current_finger:
            current_finger = row.finger
            cv2.putText(
                panel, current_finger.upper(), (10, y + 10), _FONT, 0.46,
                (0, 190, 255), 1, cv2.LINE_AA,
            )
            y += 20

        z = row.z
        significant = abs(z) >= Z_SIGNIFICANT
        colour = (80, 220, 100) if significant else (150, 150, 150)

        cv2.putText(panel, row.metric, (22, y + 12), _FONT, 0.4, colour, 1, cv2.LINE_AA)

        centre = BAR_LEFT + BAR_WIDTH // 2
        cv2.line(panel, (BAR_LEFT, y + 8), (BAR_LEFT + BAR_WIDTH, y + 8), (48, 48, 48), 1)
        # Mark the significance threshold, so "is this real" is a visual check.
        for sign in (-1, 1):
            x = centre + int(sign * (Z_SIGNIFICANT / Z_FULL_SCALE) * (BAR_WIDTH // 2))
            cv2.line(panel, (x, y + 1), (x, y + 15), (70, 70, 70), 1)
        cv2.line(panel, (centre, y), (centre, y + 16), (100, 100, 100), 1)

        extent = int((z / Z_FULL_SCALE) * (BAR_WIDTH // 2))
        extent = max(-BAR_WIDTH // 2, min(BAR_WIDTH // 2, extent))
        if extent:
            cv2.rectangle(panel, (centre, y + 3), (centre + extent, y + 13), colour, -1)

        cv2.putText(
            panel,
            f"{z:+6.1f}s  pk{row.peak_z:5.1f}",
            (BAR_LEFT + BAR_WIDTH + 8, y + 12),
            _FONT,
            0.37,
            colour,
            1,
            cv2.LINE_AA,
        )
        y += ROW_HEIGHT

    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingers", default="index,middle")
    parser.add_argument("--baseline-time-constant", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")

    settings = AppSettings.load()
    time_constant = (
        args.baseline_time_constant
        if args.baseline_time_constant is not None
        else settings.gestures.baseline_time_constant
    )
    fingers = [f.strip() for f in args.fingers.split(",") if f.strip()]
    rows = [
        MetricRow(finger, metric, time_constant, settings.gestures.signal_time_constant)
        for finger in fingers
        for metric in PRESS_METRIC_NAMES
    ]

    print("\nPress your finger and watch which row moves. R resets, ESC quits.\n")

    with (
        CameraManager(settings.camera) as camera,
        HandTracker(settings.tracking) as tracker,
    ):
        start = time.perf_counter()
        previous = start
        while True:
            frame = camera.read()
            if frame is None:
                continue
            now = time.perf_counter()
            dt = now - previous
            previous = now

            hand = tracker.select_hand(
                tracker.process(frame, int((now - start) * 1000))
            )
            if hand is not None:
                for row in rows:
                    row.update(
                        press_metric(
                            hand.landmarks_px,
                            hand.world_landmarks,
                            row.finger,
                            row.metric,
                        ),
                        dt,
                    )

            height = max(frame.shape[0], 82 + len(rows) * ROW_HEIGHT + len(fingers) * 20)
            if frame.shape[0] != height:
                frame = cv2.copyMakeBorder(
                    frame, 0, height - frame.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(22, 22, 22)
                )
            cv2.imshow(
                "Press monitor",
                np.hstack([frame, draw_panel(rows, height, hand is not None)]),
            )

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                for row in rows:
                    row.reset()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
