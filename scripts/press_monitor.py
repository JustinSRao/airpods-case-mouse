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
import time

import cv2
import numpy as np

from src.camera.camera_manager import CameraManager
from src.config.settings import AppSettings
from src.gestures.finger_state import BaselineTracker
from src.tracking.hand_features import PRESS_METRIC_NAMES, press_metric
from src.tracking.hand_tracker import HandTracker

_FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_WIDTH = 560
ROW_HEIGHT = 30
BAR_LEFT = 250
BAR_WIDTH = 280

# Decay the observed peak so an old spike does not permanently rescale the bar.
PEAK_HALF_LIFE = 4.0


class MetricRow:
    """Baseline, current deviation and a decaying peak for one metric."""

    def __init__(self, finger: str, metric: str, time_constant: float) -> None:
        self.finger = finger
        self.metric = metric
        self.tracker = BaselineTracker(time_constant)
        self.raw = 0.0
        self.deviation = 0.0
        self.peak = 1e-6

    def update(self, value: float, dt: float) -> None:
        self.raw = value
        self.deviation = self.tracker.update(value, dt, frozen=False)
        decay = 0.5 ** (dt / PEAK_HALF_LIFE)
        self.peak = max(self.peak * decay, abs(self.deviation), 1e-6)

    def reset(self) -> None:
        self.tracker.reset()
        self.peak = 1e-6


def draw_panel(rows: list[MetricRow], height: int, tracked: bool) -> np.ndarray:
    panel = np.full((height, PANEL_WIDTH, 3), 22, dtype=np.uint8)

    header = "PRESS METRICS - deviation from baseline" if tracked else "NO HAND DETECTED"
    cv2.putText(
        panel,
        header,
        (10, 22),
        _FONT,
        0.5,
        (255, 255, 255) if tracked else (60, 60, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel, "R reset   ESC quit", (10, 42), _FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA
    )

    # The row deviating most right now is almost certainly the useful metric.
    best = max(rows, key=lambda r: abs(r.deviation) / r.peak if r.peak else 0.0)

    y = 66
    current_finger = ""
    for row in rows:
        if row.finger != current_finger:
            current_finger = row.finger
            cv2.putText(
                panel, current_finger.upper(), (10, y), _FONT, 0.5,
                (0, 190, 255), 1, cv2.LINE_AA,
            )
            y += 22

        highlight = row is best and abs(row.deviation) > 0.2 * row.peak
        colour = (80, 220, 100) if highlight else (185, 185, 185)

        cv2.putText(
            panel, row.metric, (22, y + 14), _FONT, 0.42, colour, 1, cv2.LINE_AA
        )
        cv2.putText(
            panel,
            f"{row.raw:8.2f}",
            (135, y + 14),
            _FONT,
            0.42,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

        # Bar centred on zero, scaled to this metric's recent peak.
        centre = BAR_LEFT + BAR_WIDTH // 2
        cv2.line(panel, (BAR_LEFT, y + 9), (BAR_LEFT + BAR_WIDTH, y + 9), (55, 55, 55), 1)
        cv2.line(panel, (centre, y), (centre, y + 18), (90, 90, 90), 1)
        extent = int((row.deviation / row.peak) * (BAR_WIDTH // 2)) if row.peak else 0
        extent = max(-BAR_WIDTH // 2, min(BAR_WIDTH // 2, extent))
        if extent:
            cv2.rectangle(
                panel,
                (centre, y + 3),
                (centre + extent, y + 15),
                colour,
                -1,
            )
        cv2.putText(
            panel,
            f"{row.deviation:+7.2f}  pk{row.peak:6.2f}",
            (BAR_LEFT + BAR_WIDTH + 8, y + 14),
            _FONT,
            0.38,
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
        MetricRow(finger, metric, time_constant)
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
                        press_metric(hand.world_landmarks, row.finger, row.metric), dt
                    )

            height = max(frame.shape[0], 66 + len(rows) * ROW_HEIGHT + len(fingers) * 22)
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
