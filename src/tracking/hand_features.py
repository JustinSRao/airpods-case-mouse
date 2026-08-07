"""Geometry derived from the 21 hand landmarks.

Everything here works in *pixel* space and is then normalised by the measured
palm width. That combination makes downstream numbers invariant to both camera
resolution and how far the hand sits from the lens, which is what lets a single
sensitivity value keep working when the screen angle changes slightly.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Landmark(IntEnum):
    """MediaPipe hand landmark indices."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


MCP_LANDMARKS = (
    Landmark.INDEX_MCP,
    Landmark.MIDDLE_MCP,
    Landmark.RING_MCP,
    Landmark.PINKY_MCP,
)

# Smallest palm width (px) we will trust. Below this the hand is either barely
# in frame or badly mis-detected, and dividing by it would explode the deltas.
MIN_PALM_WIDTH_PX = 8.0


def palm_width(landmarks_px: np.ndarray) -> float:
    """Knuckle span from index MCP to pinky MCP, in pixels.

    This is the most stable scale reference on the hand: unlike the wrist-to-
    middle-finger length it barely changes when fingers curl or the hand
    rotates about the forearm axis.
    """
    span = landmarks_px[Landmark.INDEX_MCP] - landmarks_px[Landmark.PINKY_MCP]
    return float(np.linalg.norm(span))


def hand_scale(landmarks_px: np.ndarray) -> float:
    """Palm width clamped to a safe minimum, for use as a divisor."""
    return max(palm_width(landmarks_px), MIN_PALM_WIDTH_PX)


def wrist_anchor(landmarks_px: np.ndarray) -> np.ndarray:
    return landmarks_px[Landmark.WRIST].copy()


def mcp_centroid(landmarks_px: np.ndarray) -> np.ndarray:
    """Mean of the four finger knuckles."""
    return landmarks_px[list(MCP_LANDMARKS)].mean(axis=0)


def palm_centroid(landmarks_px: np.ndarray) -> np.ndarray:
    """Mean of the wrist and the four finger knuckles.

    Averaging five landmarks cancels a good deal of per-landmark jitter, and
    including the wrist pulls the point toward the centre of the palm -- close
    to where the AirPods case actually sits under the hand.
    """
    points = landmarks_px[[Landmark.WRIST, *MCP_LANDMARKS]]
    return points.mean(axis=0)


_ANCHOR_STRATEGIES = {
    "wrist": wrist_anchor,
    "mcp_centroid": mcp_centroid,
    "palm_centroid": palm_centroid,
}

ANCHOR_STRATEGY_NAMES = tuple(_ANCHOR_STRATEGIES)


def compute_anchor(landmarks_px: np.ndarray, strategy: str) -> np.ndarray:
    """Return the 2D tracking anchor for the configured strategy."""
    try:
        fn = _ANCHOR_STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"Unknown anchor strategy {strategy!r}; "
            f"expected one of {ANCHOR_STRATEGY_NAMES}"
        ) from None
    return fn(landmarks_px)


def fingertip_relative_to_palm(
    landmarks_px: np.ndarray, tip: Landmark
) -> np.ndarray:
    """Fingertip offset from the palm centroid, in palm-width units.

    This is the core quantity for telling a *click* apart from *moving the
    whole case*: translating the hand leaves this unchanged, while bending a
    finger does not.
    """
    offset = landmarks_px[tip] - palm_centroid(landmarks_px)
    return offset / hand_scale(landmarks_px)
