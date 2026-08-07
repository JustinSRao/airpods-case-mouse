"""Headless pipeline benchmark: real capture + real inference, no cursor control.

Measures where the per-frame time actually goes, so we know whether inference
or capture is the bottleneck before reaching for threads.

    .\\.venv\\Scripts\\python.exe -m scripts.bench --seconds 10
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time

import numpy as np

from src.camera.camera_manager import CameraManager
from src.config.settings import AppSettings
from src.tracking.hand_features import compute_anchor, hand_scale, palm_width
from src.tracking.hand_tracker import HandTracker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--backend", default=None, help="dshow | msmf | any")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    settings = AppSettings.load()
    if args.camera_index is not None:
        settings.camera.index = args.camera_index
    if args.backend is not None:
        settings.camera.backend = args.backend
    if args.width is not None:
        settings.camera.width = args.width
    if args.height is not None:
        settings.camera.height = args.height

    capture_ms: list[float] = []
    inference_ms: list[float] = []
    frame_ms: list[float] = []
    anchors: list[np.ndarray] = []
    palm_widths: list[float] = []
    hand_frames = 0
    total_frames = 0

    with (
        CameraManager(settings.camera) as camera,
        HandTracker(settings.tracking) as tracker,
    ):
        print(f"Capturing for {args.seconds:.0f}s at {camera.actual_size} ...")
        start = time.perf_counter()
        previous = start

        while time.perf_counter() - start < args.seconds:
            t0 = time.perf_counter()
            frame = camera.read()
            t1 = time.perf_counter()
            if frame is None:
                continue

            observations = tracker.process(frame, int((t1 - start) * 1000))
            t2 = time.perf_counter()

            hand = tracker.select_hand(observations)
            if hand is not None:
                hand_frames += 1
                anchors.append(
                    compute_anchor(hand.landmarks_px, settings.anchor.strategy)
                )
                palm_widths.append(palm_width(hand.landmarks_px))

            total_frames += 1
            capture_ms.append((t1 - t0) * 1000)
            inference_ms.append((t2 - t1) * 1000)
            frame_ms.append((t2 - previous) * 1000)
            previous = t2

    elapsed = time.perf_counter() - start

    def stats(name: str, values: list[float]) -> None:
        if not values:
            print(f"  {name:<12} (no samples)")
            return
        ordered = sorted(values)
        p95 = ordered[int(len(ordered) * 0.95) - 1]
        print(
            f"  {name:<12} mean {statistics.mean(values):6.2f} ms   "
            f"median {statistics.median(values):6.2f} ms   p95 {p95:6.2f} ms"
        )

    print(f"\nFrames: {total_frames} in {elapsed:.1f}s -> {total_frames / elapsed:.1f} FPS")
    print(f"Right hand detected in {hand_frames}/{total_frames} frames "
          f"({100 * hand_frames / max(total_frames, 1):.0f}%)\n")
    stats("capture", capture_ms)
    stats("inference", inference_ms)
    stats("full frame", frame_ms)

    if palm_widths:
        print(f"\nPalm width: mean {statistics.mean(palm_widths):.1f} px "
              f"(min {min(palm_widths):.1f}, max {max(palm_widths):.1f})")

    if len(anchors) > 5:
        # Jitter of the anchor while it was visible, in palm-width units --
        # this is the number the dead zone has to sit above.
        arr = np.array(anchors)
        deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        scale = statistics.mean(palm_widths)
        print(
            f"Anchor step: median {statistics.median(deltas) / scale:.5f} pw   "
            f"p95 {sorted(deltas)[int(len(deltas) * 0.95) - 1] / scale:.5f} pw   "
            f"(current dead_zone = {settings.cursor.dead_zone})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
