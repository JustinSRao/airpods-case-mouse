"""MediaPipe hand-landmark tracking.

MediaPipe 1.0.0 removed the legacy ``mediapipe.solutions.hands`` module that
most tutorials use; the supported entry point is now the Tasks API
(``mediapipe.tasks.python.vision.HandLandmarker``) driven by a downloaded
``.task`` model bundle.

We run in VIDEO mode rather than LIVE_STREAM. VIDEO is synchronous, so a frame
and its landmarks stay together and the main loop keeps a single, easy-to-
reason-about ordering. LIVE_STREAM's async callback only pays off once
inference is the bottleneck, which it is not at 30 FPS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from src.config.settings import TrackingSettings

log = logging.getLogger(__name__)

NUM_LANDMARKS = 21

# Bone list for drawing the skeleton, taken from the MediaPipe task metadata
# so it stays correct if the model's topology ever changes.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
)


class HandTrackerError(RuntimeError):
    """Raised when the landmark model cannot be loaded."""


@dataclass(frozen=True)
class HandObservation:
    """One detected hand in one frame.

    Attributes:
        handedness: "Left" or "Right" as reported by MediaPipe.
        score: Handedness classification confidence in [0, 1].
        landmarks_px: (21, 2) float array in image pixel coordinates.
        landmarks_norm: (21, 3) array; x/y in [0, 1] image fractions, z is
            depth relative to the wrist in roughly the same scale as x.
        world_landmarks: (21, 3) array in metres, origin at the hand's
            geometric centre. Already translation-invariant, which makes it
            the natural feature space for press detection in Milestone 3.
    """

    handedness: str
    score: float
    landmarks_px: np.ndarray
    landmarks_norm: np.ndarray
    world_landmarks: np.ndarray


class HandTracker:
    """Thin wrapper over ``vision.HandLandmarker`` in VIDEO mode."""

    def __init__(self, settings: TrackingSettings) -> None:
        self._settings = settings
        self._landmarker: vision.HandLandmarker | None = None
        self._last_timestamp_ms = -1

    def open(self) -> None:
        model_path = Path(self._settings.model_path)
        if not model_path.is_file():
            raise HandTrackerError(
                f"Hand landmark model not found at {model_path}.\n"
                "Download it with:  .\\scripts\\download_model.ps1"
            )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self._settings.num_hands,
            min_hand_detection_confidence=self._settings.min_hand_detection_confidence,
            min_hand_presence_confidence=self._settings.min_hand_presence_confidence,
            min_tracking_confidence=self._settings.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        log.info("Hand landmarker loaded: %s", model_path.name)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandObservation]:
        """Detect hands in a BGR frame.

        ``timestamp_ms`` must increase strictly between calls; VIDEO mode
        rejects out-of-order timestamps.
        """
        if self._landmarker is None:
            raise HandTrackerError("process() called before open()")

        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        observations: list[HandObservation] = []
        for hand_index, landmarks in enumerate(result.hand_landmarks):
            categories = result.handedness[hand_index]
            category = categories[0] if categories else None

            norm = np.array(
                [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
            )
            pixels = norm[:, :2] * np.array([width, height], dtype=np.float32)

            world_source = result.hand_world_landmarks[hand_index]
            world = np.array(
                [(lm.x, lm.y, lm.z) for lm in world_source], dtype=np.float32
            )

            observations.append(
                HandObservation(
                    handedness=category.category_name if category else "Unknown",
                    score=float(category.score) if category else 0.0,
                    landmarks_px=pixels,
                    landmarks_norm=norm,
                    world_landmarks=world,
                )
            )
        return observations

    def select_hand(
        self, observations: list[HandObservation]
    ) -> HandObservation | None:
        """Pick the configured hand, requiring a minimum handedness score."""
        wanted = self._settings.target_handedness.lower()
        candidates = [
            obs
            for obs in observations
            if obs.handedness.lower() == wanted
            and obs.score >= self._settings.min_confidence_for_control
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda obs: obs.score)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            log.info("Hand landmarker closed")

    def __enter__(self) -> HandTracker:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
