"""Debug overlay: hand skeleton, anchor, and a live metrics panel.

This view is the main tuning instrument for the project -- the numbers on
screen are how thresholds get chosen from real measurements instead of
guesses.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.mouse.motion_mapper import MotionResult
from src.tracking.hand_features import Landmark
from src.tracking.hand_tracker import HAND_CONNECTIONS, HandObservation

# BGR colours.
_WHITE = (255, 255, 255)
_GREY = (170, 170, 170)
_GREEN = (80, 220, 100)
_RED = (60, 60, 240)
_AMBER = (0, 190, 255)
_CYAN = (230, 220, 60)
_MAGENTA = (200, 80, 220)
_BONE = (200, 200, 200)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LINE_HEIGHT = 18
_PANEL_WIDTH = 250


@dataclass
class HudState:
    """Everything the overlay needs to render one frame."""

    fps: float
    mouse_control_enabled: bool
    hand: HandObservation | None
    anchor_px: np.ndarray | None
    anchor_strategy: str
    palm_width_px: float
    motion: MotionResult
    tracking_lost_for: float
    status_message: str = ""


def draw_landmarks(frame: np.ndarray, hand: HandObservation) -> None:
    """Draw the 21-point skeleton."""
    points = hand.landmarks_px.astype(np.int32)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, tuple(points[start]), tuple(points[end]), _BONE, 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        # Highlight the two landmarks that will become mouse buttons.
        if index == Landmark.INDEX_TIP:
            colour, radius = _GREEN, 7
        elif index == Landmark.MIDDLE_TIP:
            colour, radius = _AMBER, 7
        else:
            colour, radius = _CYAN, 3
        cv2.circle(frame, tuple(point), radius, colour, -1, cv2.LINE_AA)


def draw_anchor(frame: np.ndarray, anchor_px: np.ndarray, motion: MotionResult) -> None:
    """Draw the tracking anchor and an arrow showing the current motion."""
    centre = tuple(anchor_px.astype(np.int32))
    cv2.circle(frame, centre, 10, _MAGENTA, 2, cv2.LINE_AA)
    cv2.drawMarker(frame, centre, _MAGENTA, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

    # Scale the arrow down so a fast flick does not draw off-screen.
    tip = (
        int(centre[0] + motion.dx * 0.15),
        int(centre[1] + motion.dy * 0.15),
    )
    if abs(tip[0] - centre[0]) > 2 or abs(tip[1] - centre[1]) > 2:
        cv2.arrowedLine(frame, centre, tip, _MAGENTA, 2, cv2.LINE_AA, tipLength=0.3)


def draw_panel(frame: np.ndarray, state: HudState) -> None:
    """Draw the translucent metrics panel down the left edge."""
    lines: list[tuple[str, tuple[int, int, int]]] = []

    lines.append((f"FPS: {state.fps:5.1f}", _WHITE))
    if state.mouse_control_enabled:
        lines.append(("MOUSE CONTROL: ON", _GREEN))
    else:
        lines.append(("MOUSE CONTROL: OFF", _RED))
    lines.append(("", _WHITE))

    if state.hand is not None:
        lines.append((f"RIGHT HAND: TRACKING ({state.hand.score:.2f})", _GREEN))
        lines.append((f"anchor: {state.anchor_strategy}", _GREY))
        lines.append((f"palm width: {state.palm_width_px:5.1f} px", _GREY))
    else:
        lines.append(("RIGHT HAND: NOT DETECTED", _RED))
        if state.tracking_lost_for > 0:
            lines.append((f"lost for: {state.tracking_lost_for:.2f}s", _GREY))

    lines.append(("", _WHITE))
    lines.append(("CURSOR", _WHITE))
    motion = state.motion
    lines.append((f"  dx: {motion.dx:+7.2f}", _GREY))
    lines.append((f"  dy: {motion.dy:+7.2f}", _GREY))
    lines.append((f"  speed: {motion.hand_speed:5.2f} pw/s", _GREY))
    lines.append((f"  gain:  {motion.gain:5.2f}", _GREY))
    if motion.in_dead_zone:
        lines.append(("  [dead zone]", _AMBER))
    if motion.clamped:
        lines.append(("  [velocity clamped]", _AMBER))

    lines.append(("", _WHITE))
    lines.append(("INDEX:  (milestone 3)", _GREY))
    lines.append(("MIDDLE: (milestone 4)", _GREY))

    lines.append(("", _WHITE))
    lines.append(("ESC quit | PPPPP mouse | F9 anchor", _GREY))
    if state.status_message:
        lines.append((state.status_message, _AMBER))

    height = _LINE_HEIGHT * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (_PANEL_WIDTH, height), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 20
    for text, colour in lines:
        if text:
            cv2.putText(frame, text, (8, y), _FONT, 0.42, colour, 1, cv2.LINE_AA)
        y += _LINE_HEIGHT


def render(frame: np.ndarray, state: HudState, draw_skeleton: bool = True) -> np.ndarray:
    """Draw the full overlay onto ``frame`` in place and return it."""
    if state.hand is not None and draw_skeleton:
        draw_landmarks(frame, state.hand)
    if state.anchor_px is not None:
        draw_anchor(frame, state.anchor_px, state.motion)
    draw_panel(frame, state)
    return frame
