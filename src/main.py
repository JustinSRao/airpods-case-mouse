"""Entry point: webcam -> right-hand tracking -> Windows cursor.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe -m src.main

Hotkeys are global (they work even when the preview window is not focused):

    ESC      quit
    P x5     toggle mouse control on/off  (emergency disable)
             -- five taps of P within five seconds
    F5 / F6  invert the X / Y axis
    F7       cycle how the controlling hand is chosen
    F9       cycle the palm anchor strategy
    F10      re-centre / reset the motion filter
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.camera.camera_manager import CameraError, CameraManager
from src.config.settings import USER_CONFIG_PATH, AppSettings
from src.debug.hud import HudState, PressRow, render
from src.hotkeys import (
    VK_ESCAPE,
    VK_F5,
    VK_F6,
    VK_F7,
    VK_F9,
    VK_F10,
    VK_P,
    Hotkeys,
    MultiPressDetector,
)
from src.mouse.motion_mapper import MotionMapper, MotionResult
from src.gestures.finger_state import (
    PressDetector,
    PressEvent,
    PressState,
    PressThresholds,
    RateThresholds,
    TransientPressDetector,
)
from src.mouse.mouse_controller import (
    MouseButton,
    MouseController,
    enable_dpi_awareness,
    get_screen_size,
)
from src.tracking.hand_features import (
    ANCHOR_STRATEGY_NAMES,
    compute_anchor,
    hand_scale,
    palm_width,
    press_metric,
)
from src.tracking.hand_tracker import (
    SELECTION_MODES,
    HandTracker,
    HandTrackerError,
)

log = logging.getLogger("airpods_mouse")

_HOTKEY_BINDINGS = {
    "quit": VK_ESCAPE,
    "toggle_mouse": VK_P,
    "invert_x": VK_F5,
    "invert_y": VK_F6,
    "swap_hand": VK_F7,
    "cycle_anchor": VK_F9,
    "recenter": VK_F10,
}

# Tapping a plain letter key once would fire constantly during normal typing,
# so the toggle requires a deliberate burst.
TOGGLE_PRESS_COUNT = 5
TOGGLE_PRESS_WINDOW = 5.0


class FpsMeter:
    """Rolling average frame rate over a short window."""

    def __init__(self, window: int = 30) -> None:
        self._samples: deque[float] = deque(maxlen=window)
        self._last = time.perf_counter()

    def tick(self) -> float:
        """Record a frame; returns the seconds elapsed since the previous one."""
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            self._samples.append(dt)
        return dt

    @property
    def fps(self) -> float:
        if not self._samples:
            return 0.0
        return len(self._samples) / sum(self._samples)


class AirPodsMouseApp:
    """Owns the capture loop and all mutable runtime state."""

    def __init__(
        self,
        settings: AppSettings,
        mouse_enabled_at_start: bool,
        run_seconds: float | None = None,
    ) -> None:
        self._settings = settings
        self._mouse_enabled = mouse_enabled_at_start
        self._run_seconds = run_seconds
        self._anchor_index = self._initial_anchor_index()
        self._running = True
        self._status_message = ""
        self._status_expires = 0.0

        self._mouse = MouseController()
        self._mapper = MotionMapper(settings.cursor)
        self._hotkeys = Hotkeys(_HOTKEY_BINDINGS)
        self._toggle_taps = MultiPressDetector(
            count=TOGGLE_PRESS_COUNT, window=TOGGLE_PRESS_WINDOW
        )
        self._fps = FpsMeter()

        self._last_hand_time: float | None = None
        self._had_hand = False
        self._press_detectors = self._build_press_detectors()

    def _build_press_detectors(self) -> dict[str, tuple[PressDetector, MouseButton]]:
        """One detector per finger, or none at all if not yet calibrated."""
        gestures = self._settings.gestures
        if not gestures.enabled:
            log.warning(
                "Finger clicking DISABLED (not calibrated). Run: "
                ".\\.venv\\Scripts\\python.exe -m scripts.calibrate_press"
            )
            return {}

        detectors: dict[str, tuple[object, MouseButton]] = {}
        for finger, button in (
            ("index", MouseButton.LEFT),
            ("middle", MouseButton.RIGHT),
        ):
            try:
                if gestures.mode == "transient":
                    detector = TransientPressDetector(
                        finger,
                        RateThresholds(
                            press=getattr(gestures, f"{finger}_press_rate"),
                            release=getattr(gestures, f"{finger}_release_rate"),
                            min_state_duration=gestures.min_state_duration,
                        ),
                        signal_time_constant=gestures.rate_signal_time_constant,
                        rate_time_constant=gestures.rate_time_constant,
                    )
                elif gestures.mode == "level":
                    detector = PressDetector(
                        finger,
                        PressThresholds(
                            press=getattr(gestures, f"{finger}_press_threshold"),
                            release=getattr(gestures, f"{finger}_release_threshold"),
                            min_state_duration=gestures.min_state_duration,
                        ),
                        baseline_time_constant=gestures.baseline_time_constant,
                        signal_time_constant=gestures.signal_time_constant,
                    )
                else:
                    raise ValueError(
                        f"Unknown gesture mode {gestures.mode!r}; "
                        "expected 'transient' or 'level'"
                    )
                detectors[finger] = (detector, button)
            except ValueError as exc:
                # An uncalibrated or inverted pair would otherwise click wildly.
                log.error("%s -- %s clicking disabled", exc, finger)
        for finger, (detector, button) in detectors.items():
            log.info(
                "Clicking: %s -> %s (%s mode, metric=%s, press>%.3f release<%.3f)",
                finger,
                button.label,
                gestures.mode,
                getattr(gestures, f"{finger}_metric"),
                detector.thresholds.press,
                detector.thresholds.release,
            )
        return detectors

    def _initial_anchor_index(self) -> int:
        try:
            return ANCHOR_STRATEGY_NAMES.index(self._settings.anchor.strategy)
        except ValueError:
            log.warning(
                "Unknown anchor strategy %r, falling back to %r",
                self._settings.anchor.strategy,
                ANCHOR_STRATEGY_NAMES[0],
            )
            return 0

    @property
    def _anchor_strategy(self) -> str:
        return ANCHOR_STRATEGY_NAMES[self._anchor_index]

    def run(self) -> int:
        settings = self._settings
        enable_dpi_awareness()
        screen_w, screen_h = get_screen_size()
        log.info("Primary display: %dx%d", screen_w, screen_h)
        log.info(
            "Mouse control starts %s -- tap P five times within five seconds to toggle",
            "ENABLED" if self._mouse_enabled else "DISABLED",
        )

        # DirectShow takes a few seconds to enumerate devices. Say so, or the
        # silence looks like a hang and invites a Ctrl+C.
        log.info(
            "Opening camera %d (DirectShow takes a few seconds)...",
            settings.camera.index,
        )

        try:
            with (
                CameraManager(settings.camera) as camera,
                HandTracker(settings.tracking) as tracker,
            ):
                log.info("Ready. Preview window is open.")
                self._loop(camera, tracker)
        except (CameraError, HandTrackerError) as exc:
            log.error("%s", exc)
            return 1
        except KeyboardInterrupt:
            log.info("Interrupted by user (Ctrl+C)")
        finally:
            # The single most important line in the program: never leave a
            # synthetic mouse button held down.
            self._mouse.release_all()
            cv2.destroyAllWindows()
            log.info("Shutdown complete, all mouse buttons released")
        return 0

    def _loop(self, camera: CameraManager, tracker: HandTracker) -> None:
        settings = self._settings
        show_preview = settings.debug.show_preview
        start = time.perf_counter()
        consecutive_grab_failures = 0

        while self._running:
            if self._run_seconds is not None and (
                time.perf_counter() - start >= self._run_seconds
            ):
                self._request_quit(f"--run-seconds {self._run_seconds:g}")
                break

            dt = self._fps.tick()
            frame = camera.read()
            if frame is None:
                consecutive_grab_failures += 1
                if consecutive_grab_failures >= 30:
                    raise CameraError("Camera stopped delivering frames")
                continue
            consecutive_grab_failures = 0

            timestamp_ms = int((time.perf_counter() - start) * 1000)
            observations = tracker.process(frame, timestamp_ms)
            hand = tracker.select_hand(observations)

            motion = MotionResult()
            anchor = None
            width = 0.0

            if hand is not None:
                if not self._had_hand:
                    log.info("Right hand acquired")
                    # Fresh acquisition: start from this position, no delta,
                    # and re-learn each finger's resting baseline from scratch.
                    self._mapper.reset()
                    self._mouse.reset_residual()
                    for detector, _ in self._press_detectors.values():
                        detector.reset()
                self._had_hand = True
                self._last_hand_time = time.perf_counter()

                anchor = compute_anchor(hand.landmarks_px, self._anchor_strategy)
                width = palm_width(hand.landmarks_px)
                motion = self._mapper.update(
                    anchor, hand_scale(hand.landmarks_px), dt
                )

                if self._mouse_enabled and not motion.in_dead_zone:
                    self._mouse.move_relative(motion.dx, motion.dy)

                self._update_presses(hand, time.perf_counter())
            else:
                self._handle_tracking_loss()

            self._handle_hotkeys()

            if show_preview:
                state = HudState(
                    fps=self._fps.fps,
                    mouse_control_enabled=self._mouse_enabled,
                    hand=hand,
                    anchor_px=anchor,
                    anchor_strategy=self._anchor_strategy,
                    palm_width_px=width,
                    motion=motion,
                    tracking_lost_for=self._time_since_hand(),
                    status_message=self._active_status(),
                    observations=observations,
                    selection=settings.tracking.selection,
                    invert_x=settings.cursor.invert_x,
                    invert_y=settings.cursor.invert_y,
                    press_rows=[
                        PressRow(
                            label=finger,
                            state=detector.state,
                            metric=detector.metric,
                            press=detector.thresholds.press,
                            release=detector.thresholds.release,
                        )
                        for finger, (detector, _) in self._press_detectors.items()
                    ],
                )
                render(frame, state, draw_skeleton=settings.debug.draw_landmarks)
                cv2.imshow(settings.debug.window_name, frame)
                # 1 ms wait keeps the OpenCV window responsive; the global
                # hotkeys above are what actually drive control.
                if cv2.waitKey(1) & 0xFF == 27:
                    self._request_quit("ESC (preview window)")

    def _time_since_hand(self) -> float:
        if self._last_hand_time is None:
            return 0.0
        return time.perf_counter() - self._last_hand_time

    def _handle_tracking_loss(self) -> None:
        if self._had_hand:
            log.info("Tracking lost")
            self._had_hand = False
        self._mapper.reset()
        self._mouse.reset_residual()

        if (
            self._mouse.held_buttons
            and self._time_since_hand() >= self._settings.tracking.tracking_loss_timeout
        ):
            log.warning("Tracking lost beyond timeout - releasing all mouse buttons")
            self._release_all_presses(time.perf_counter())
            # Belt and braces: catches anything held that no detector owns.
            self._mouse.release_all()

    def _handle_hotkeys(self) -> None:
        self._hotkeys.poll()

        if self._hotkeys.just_pressed("quit"):
            self._request_quit("ESC")

        now = time.perf_counter()
        if self._hotkeys.just_pressed("toggle_mouse"):
            if self._toggle_taps.register(now):
                self._toggle_mouse_control()
            else:
                taps, needed = self._toggle_taps.progress
                self._set_status(f"P {taps}/{needed}...", duration=TOGGLE_PRESS_WINDOW)
        else:
            self._toggle_taps.expire(now)

        for axis in ("x", "y"):
            if self._hotkeys.just_pressed(f"invert_{axis}"):
                attribute = f"invert_{axis}"
                cursor = self._settings.cursor
                flipped = not getattr(cursor, attribute)
                setattr(cursor, attribute, flipped)
                # The mapper caches the settings object, so no rebuild is
                # needed; just drop history so the flip cannot emit one
                # doubled delta.
                self._mapper.reset()
                self._mouse.reset_residual()
                log.info("invert_%s -> %s", axis, flipped)
                self._set_status(f"invert {axis.upper()}: {'ON' if flipped else 'OFF'}")

        if self._hotkeys.just_pressed("swap_hand"):
            tracking = self._settings.tracking
            index = (SELECTION_MODES.index(tracking.selection) + 1) % len(
                SELECTION_MODES
            )
            tracking.selection = SELECTION_MODES[index]
            self._mapper.reset()
            self._mouse.reset_residual()
            log.info("Hand selection -> %s", tracking.selection)
            self._set_status(f"select: {tracking.selection}")

        if self._hotkeys.just_pressed("cycle_anchor"):
            self._anchor_index = (self._anchor_index + 1) % len(ANCHOR_STRATEGY_NAMES)
            self._mapper.reset()
            log.info("Anchor strategy -> %s", self._anchor_strategy)
            self._set_status(f"anchor: {self._anchor_strategy}")

        if self._hotkeys.just_pressed("recenter"):
            self._mapper.reset()
            self._mouse.reset_residual()
            self._set_status("filter reset")

    def _press_metric(self, hand, finger: str) -> float:
        metric = getattr(self._settings.gestures, f"{finger}_metric")
        return press_metric(
            hand.landmarks_px, hand.world_landmarks, finger, metric
        )

    def _update_presses(self, hand, now: float) -> None:
        """Feed each detector and translate its events into mouse buttons."""
        for finger, (detector, button) in self._press_detectors.items():
            event = detector.update(self._press_metric(hand, finger), now)
            if event is None:
                continue
            # Detector state advances regardless, so that releasing a finger
            # while control is off does not leave it stuck DOWN internally.
            if not self._mouse_enabled:
                continue
            if event is PressEvent.PRESSED:
                self._mouse.press(button)
            else:
                self._mouse.release(button)

    def _release_all_presses(self, now: float) -> None:
        """Force every finger UP and send the matching mouse-ups.

        Used on tracking loss and when control is switched off, so a button
        can never outlive the hand that pressed it.
        """
        for detector, button in self._press_detectors.values():
            if detector.force_release(now) is PressEvent.RELEASED:
                self._mouse.release(button)

    def _toggle_mouse_control(self) -> None:
        self._mouse_enabled = not self._mouse_enabled
        if not self._mouse_enabled:
            self._release_all_presses(time.perf_counter())
            self._mouse.release_all()
            self._mouse.reset_residual()
        self._mapper.reset()
        log.info("Mouse control %s", "ENABLED" if self._mouse_enabled else "DISABLED")
        self._set_status(f"mouse {'ON' if self._mouse_enabled else 'OFF'}")

    def _request_quit(self, reason: str) -> None:
        log.info("Quit requested via %s", reason)
        self._running = False

    def _set_status(self, message: str, duration: float = 1.5) -> None:
        self._status_message = message
        self._status_expires = time.perf_counter() + duration

    def _active_status(self) -> str:
        if time.perf_counter() < self._status_expires:
            return self._status_message
        return ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn an AirPods case into a webcam-tracked mouse."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=USER_CONFIG_PATH,
        help="Path to an optional user config JSON overlaid on the defaults.",
    )
    parser.add_argument(
        "--camera-index", type=int, default=None, help="Override the camera index."
    )
    parser.add_argument(
        "--enable-mouse",
        action="store_true",
        help="Start with mouse control already ON (default: OFF, tap P five times).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without the debug window (headless tuning / benchmarking).",
    )
    parser.add_argument(
        "--anchor",
        choices=ANCHOR_STRATEGY_NAMES,
        default=None,
        help="Override the palm anchor strategy.",
    )
    parser.add_argument(
        "--selection",
        choices=SELECTION_MODES,
        default=None,
        help="How to choose the controlling hand. Cycle live with F7.",
    )
    parser.add_argument(
        "--hand",
        choices=("Right", "Left"),
        default=None,
        help="Handedness label to follow (only used with --selection handedness).",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=None,
        help="How many hands to detect. Use 2 to see both labels while debugging.",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Exit automatically after N seconds (for automated smoke tests).",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ...")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = AppSettings.load(user_path=args.config)

    if args.camera_index is not None:
        settings.camera.index = args.camera_index
    if args.anchor is not None:
        settings.anchor.strategy = args.anchor
    if args.hand is not None:
        settings.tracking.target_handedness = args.hand
    if args.selection is not None:
        settings.tracking.selection = args.selection
    if args.num_hands is not None:
        settings.tracking.num_hands = args.num_hands
    if args.no_preview:
        settings.debug.show_preview = False
    if args.log_level is not None:
        settings.debug.log_level = args.log_level

    logging.basicConfig(
        level=getattr(logging, settings.debug.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    app = AirPodsMouseApp(
        settings,
        mouse_enabled_at_start=args.enable_mouse,
        run_seconds=args.run_seconds,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
