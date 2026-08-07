"""Webcam capture with graceful failure and optional horizontal flip."""

from __future__ import annotations

import logging
from types import TracebackType

import cv2
import numpy as np

from src.config.settings import CameraSettings

log = logging.getLogger(__name__)

_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or produces no frames."""


class CameraManager:
    """Owns a ``cv2.VideoCapture`` and hands out BGR frames.

    Use as a context manager so the device is always released, including on
    exceptions -- a webcam left open stays locked against other applications.
    """

    def __init__(self, settings: CameraSettings) -> None:
        self._settings = settings
        self._capture: cv2.VideoCapture | None = None
        self._actual_size: tuple[int, int] = (0, 0)

    @property
    def actual_size(self) -> tuple[int, int]:
        """(width, height) the driver actually granted."""
        return self._actual_size

    def open(self) -> None:
        s = self._settings
        backend = _BACKENDS.get(s.backend.lower())
        if backend is None:
            raise CameraError(
                f"Unknown camera backend {s.backend!r}; expected one of {sorted(_BACKENDS)}"
            )

        capture = cv2.VideoCapture(s.index, backend)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera index {s.index} using the {s.backend} backend. "
                "Check that no other app is using the webcam and that camera "
                "access is enabled in Windows Privacy settings."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
        capture.set(cv2.CAP_PROP_FPS, s.target_fps)
        # Keep the driver queue shallow: a deep buffer adds latency because
        # read() would return stale frames.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise CameraError(
                f"Camera index {s.index} opened but returned no frames."
            )

        height, width = frame.shape[:2]
        self._actual_size = (width, height)
        self._capture = capture
        log.info(
            "Camera initialized: index=%d backend=%s %dx%d (requested %dx%d) reported_fps=%.1f",
            s.index,
            s.backend,
            width,
            height,
            s.width,
            s.height,
            capture.get(cv2.CAP_PROP_FPS),
        )

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or ``None`` if the grab failed."""
        if self._capture is None:
            raise CameraError("read() called before open()")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        if self._settings.flip_horizontal:
            # Mirror so the preview reads like a mirror and MediaPipe's
            # handedness labels refer to the real-world hand.
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            log.info("Camera released")

    def __enter__(self) -> CameraManager:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
