"""Typed application settings, loaded from JSON with layered overrides.

Two files are involved:

* ``config/default_config.json`` -- committed defaults, always loaded first.
* ``config/config.json``        -- optional, git-ignored, machine-specific
                                   overrides (calibration lives here later).

Any key absent from a file simply keeps the value from the layer below it, so
the user config only needs to contain the handful of values being tuned.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default_config.json"
USER_CONFIG_PATH = REPO_ROOT / "config" / "config.json"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "hand_landmarker.task"


@dataclass
class CameraSettings:
    """Webcam capture parameters.

    ``backend`` defaults to DirectShow: on this machine the Media Foundation
    backend throws a ``cv::Mat`` step assertion at 1280x720, while DirectShow
    is stable at every resolution tested.
    """

    index: int = 0
    # 720p rather than 480p. Measured: identical 28.5 FPS (the webcam, not the
    # code, is the ceiling) but palm width goes from ~64 px to ~107 px. Press
    # detection is limited by landmark noise, and more pixels across the hand
    # is the most direct way to reduce it.
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    backend: str = "dshow"  # "dshow" | "msmf" | "any"
    # Webcams show a non-mirrored view. Flipping gives the natural "selfie"
    # view AND makes MediaPipe's handedness label match the real-world hand.
    flip_horizontal: bool = True


@dataclass
class TrackingSettings:
    """MediaPipe HandLandmarker parameters."""

    model_path: str = str(DEFAULT_MODEL_PATH)
    # Two, so position-based selection can tell the hands apart. With only one
    # slot MediaPipe may lock onto whichever hand it saw first.
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # How to choose which detected hand drives the mouse:
    #   "rightmost"  - furthest right in the mirrored preview. Since the
    #                  preview is mirrored, that IS the user's right hand.
    #   "leftmost"   - mirror image of the above, for left-handed use.
    #   "handedness" - trust MediaPipe's Left/Right label.
    #
    # Position is the default because handedness proved unreliable in the
    # actual pose this app uses: a palm-down hand resting on a case, seen from
    # a steeply angled-down webcam, gets confidently mislabelled (~0.97). A
    # label that can flip when a finger bends would swap hands mid-click,
    # whereas the hand on the case never crosses to the other side of frame.
    selection: str = "rightmost"

    # Only consulted when selection == "handedness".
    target_handedness: str = "Right"
    # Seconds without the target hand before all mouse buttons are released.
    tracking_loss_timeout: float = 0.35
    # Below this handedness score the observation is treated as "no hand".
    min_confidence_for_control: float = 0.5


@dataclass
class AnchorSettings:
    """Which landmark(s) define the point we track for cursor motion."""

    # "wrist" | "mcp_centroid" | "palm_centroid"
    #   wrist         - landmark 0 only. Most affected by wrist rotation.
    #   mcp_centroid  - mean of the four finger MCP knuckles.
    #   palm_centroid - mean of wrist + the four MCP knuckles (default).
    strategy: str = "palm_centroid"


@dataclass
class CursorSettings:
    """Camera-space motion -> screen-pixel motion mapping."""

    # Screen pixels per one "palm width" of hand travel. Deltas are divided by
    # the measured palm width first, so sensitivity is independent of how far
    # the hand sits from the camera.
    sensitivity: float = 1600.0
    x_sensitivity: float = 1.0
    y_sensitivity: float = 1.0
    invert_x: bool = False
    # Inverted by default, measured on the reference setup: with the screen
    # tilted down at a case on the desk, sliding the hand toward the body must
    # move the cursor down. The raw camera-Y sign gives the opposite, because
    # the steep downward view flips how desk-plane motion projects into the
    # image. Toggle live with F6 if a different screen angle reverses it.
    invert_y: bool = True

    # Per-frame motion (in palm-width units) below this is treated as jitter.
    dead_zone: float = 0.004

    # "ema" | "one_euro" | "none". EMA is the simple default; One Euro is
    # available for Milestone 8 tuning.
    filter: str = "ema"
    # EMA weight of the newest sample: 1.0 = no smoothing, lower = smoother
    # but laggier.
    smoothing: float = 0.5
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.02
    one_euro_d_cutoff: float = 1.0

    # Gain curve: gain = 1 + acceleration * speed, where speed is in
    # palm-widths/second. 0.0 disables acceleration entirely.
    acceleration: float = 0.35
    # Hard ceiling on cursor speed, in screen pixels per second.
    max_velocity: float = 4000.0


@dataclass
class GestureSettings:
    """Finger-press detection.

    Disabled until calibrated, on purpose. Thresholds are meaningless before
    they have been measured against a real hand, and a wrong threshold means
    stray clicks on whatever is under the cursor. Run:

        .\\.venv\\Scripts\\python.exe -m scripts.calibrate_press

    which measures your resting and pressing values, writes thresholds into
    config/config.json, and sets ``enabled`` to true.
    """

    enabled: bool = False

    # Which scalar drives each finger's state machine. Chosen per finger by
    # the calibrator, which measures every candidate and keeps whichever
    # actually separates rest from press -- the winner is not the same for
    # every hand, finger or camera angle. See PRESS_METRICS in hand_features.
    index_metric: str = "total_flexion"
    middle_metric: str = "total_flexion"

    # Units depend on ``metric``; both are written by the calibrator.
    # press must exceed release -- the gap is the hysteresis band.
    index_press_threshold: float = 0.0
    index_release_threshold: float = 0.0
    middle_press_threshold: float = 0.0
    middle_release_threshold: float = 0.0

    # Minimum seconds between state changes, to debounce a single noisy frame.
    min_state_duration: float = 0.05

    # How fast the resting baseline follows the finger, in seconds. Thresholds
    # are measured against deviation from this, not the raw metric, because
    # resting posture drifts by more than a press changes anything.
    # Shorter = adapts to posture faster but starts ignoring slow presses;
    # longer = holds a press better but reacts to posture changes sluggishly.
    baseline_time_constant: float = 1.0

    # Fast smoothing applied to the metric before comparing against the
    # baseline, in seconds. Landmark jitter is per-frame white noise while a
    # press lasts ~1s, so averaging a handful of frames cuts the noise without
    # blunting the press. Must stay well below baseline_time_constant.
    signal_time_constant: float = 0.12


@dataclass
class DebugSettings:
    show_preview: bool = True
    window_name: str = "AirPods Mouse - Debug"
    draw_landmarks: bool = True
    log_level: str = "INFO"


@dataclass
class AppSettings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    anchor: AnchorSettings = field(default_factory=AnchorSettings)
    cursor: CursorSettings = field(default_factory=CursorSettings)
    gestures: GestureSettings = field(default_factory=GestureSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)

    @classmethod
    def load(
        cls,
        default_path: Path = DEFAULT_CONFIG_PATH,
        user_path: Path | None = USER_CONFIG_PATH,
    ) -> AppSettings:
        settings = cls()
        for path in (default_path, user_path):
            if path is None:
                continue
            if not path.is_file():
                # A missing user config is normal; a missing default is not.
                level = logging.DEBUG if path == user_path else logging.WARNING
                log.log(level, "Config not found, skipping: %s", path)
                continue
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            _merge_into_dataclass(settings, data, path.name)
            log.info("Loaded config: %s", path)
        return settings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")
        log.info("Saved config: %s", path)


def _merge_into_dataclass(target: Any, data: dict[str, Any], source: str) -> None:
    """Recursively apply ``data`` onto the dataclass instance ``target``.

    Unknown keys are warned about rather than silently dropped, so typos in a
    hand-edited config surface immediately.
    """
    for key, value in data.items():
        if not hasattr(target, key):
            log.warning("Unknown config key %r in %s (ignored)", key, source)
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into_dataclass(current, value, source)
        else:
            setattr(target, key, value)
