# Project notes for Claude

Webcam-tracked mouse: right hand rests on an AirPods case, hand motion drives
the Windows cursor, index/middle finger presses are the mouse buttons.
Milestones 1–4 work end to end. Next up: cursor-feel polish (8), data recorder
(6), personalised classifier (7).

## Press detection: what was actually learned

This took many rounds. Do not re-derive it.

- **An AirPods case does not depress.** A finger *pressed* on a rigid lid and
  one *resting* on it are the same pose. No metric of any kind separates them,
  and several rounds of better filtering, higher resolution and smarter
  threshold search all correctly found nothing.
- **The signal is the LIFT, not the push.** Detection works because the
  release lifts the finger clear of the case. Any protocol or interaction
  change must preserve that, or clicking stops working.
- **Compare against a rolling baseline, never an absolute level.** Resting
  posture drifts over seconds by more than a press changes anything.
  `BaselineTracker` follows slow drift and freezes while held.
- **Never discard the transition when recording.** An early calibration
  dropped 0.6 s after each cue as "settling" — exactly when the gesture
  happens — and so compared two identical resting states across four runs.
  Recording now keeps every frame; settling is applied at analysis time.
- **Judge thresholds by simulating the state machine**, not by percentile
  overlap. The overlap rule implicitly demanded d' > 3.29 and ignored
  hysteresis and debounce, rejecting settings a real detector handles cleanly
  (0/12 vs 12/12 on the same data).
- **But simulation alone overfits.** On recordings containing no press at all
  it found a "clean" threshold ~25% of the time. A minimum effect size
  (`MIN_DPRIME`) is required as well. Keep both gates.
- **MediaPipe world landmarks are a whole-hand fit**, so one finger moving
  perturbs another finger's joint angles. The `*_2d` metrics from image
  landmarks are independent; `self_check` asserts it.
- Measured separation is d' ~2.9 (index) and ~3.7 (middle) using `drop`.
  Workable, not generous.

**Always save recordings and re-tune offline** with
`calibrate_press --analyse data/<file>.json`. Live re-runs cost the user a
fresh performance of the gesture and no two are alike.

## Environment facts (verified, don't re-derive)

- Interpreter: **`.\.venv\Scripts\python.exe`** (Python 3.13). The system
  default is 3.14 — do not use it.
- Always run modules from the repo root: `python -m src.main`, not
  `python src/main.py`.
- **MediaPipe 1.0.0 removed `mediapipe.solutions.hands`.** Use the Tasks API
  (`mediapipe.tasks.python.vision.HandLandmarker`). Most online examples are
  stale.
- The model bundle `models/hand_landmarker.task` is git-ignored; fetch with
  `.\scripts\download_model.ps1`.
- Camera: **index 0 only**. Use the **DirectShow** backend — Media Foundation
  throws a `cv::Mat` step assertion at 1280×720. Real ceiling ≈ 30 FPS; the
  full pipeline measures ≈28 FPS at both 480p and 720p, so 720p is free and is
  the default (palm width ~107 px vs ~64 px, which matters for press noise).
- **Handedness labels are not trusted.** MediaPipe confidently mislabels which
  hand is which in this pose (~0.97 on the wrong answer). The controlling hand
  is chosen by position (`selection: "rightmost"`, mirrored preview), which
  cannot flip mid-gesture.
- `invert_y` is **true** by default. Measured, not derived — the raw camera-Y
  sign moved the cursor the wrong way on the reference setup.
- Do **not** install `opencv-python`; mediapipe pulls `opencv-contrib-python`
  and the two conflict.
- Display is 2560×1600. `enable_dpi_awareness()` must run before any pixel
  measurement or Windows reports a virtualised 1707×1067.

## Commands

```powershell
.\.venv\Scripts\python.exe -m src.main                      # run (mouse off; tap P x5 to enable)
.\.venv\Scripts\python.exe -m src.main --run-seconds 6 --no-preview   # smoke test
.\.venv\Scripts\python.exe -m scripts.self_check            # offline checks (~100)
.\.venv\Scripts\python.exe -m scripts.bench --seconds 10    # perf + jitter profile
.\.venv\Scripts\python.exe -m scripts.press_monitor         # live metric vs noise floor
.\.venv\Scripts\python.exe -m scripts.calibrate_press       # measure press thresholds
.\.venv\Scripts\python.exe -m scripts.calibrate_press --analyse data\<file>.json
```

Hotkeys: **P x5** toggles mouse control, ESC quits, F5/F6 invert X/Y,
F7 cycles hand selection, F9 cycles palm anchor, F10 resets the filter.

## Conventions

- **Measure, never guess thresholds.** Build HUD instrumentation first, read
  real numbers off the user's hand, then pick values. See the
  `gesture-instrument` skill.
- **All geometry is normalised by palm width** so it is invariant to camera
  distance and resolution. `scripts/self_check.py` asserts the scale- and
  translation-invariance properties — keep those passing.
- **Cursor motion is relative**, never an absolute webcam→screen mapping.
- **Never leave a mouse button held.** Any change to input, the capture loop,
  or shutdown gets the `mouse-safety-check` skill run against it.
- Tests must never send real mouse *button* events — stub
  `MouseController._send`. A synthetic click lands on whatever window is under
  the pointer.
- Defaults go in `config/default_config.json` (committed); machine-specific
  tuning goes in `config/config.json` (git-ignored).
- Never commit: the venv, the `.task` model, anything under `data/`, or
  recorded frames. Landmark data is personal biometric data.

## Skills

`tune-cursor` (feel: jitter/lag/drift), `gesture-instrument` (press detection),
`mouse-safety-check` (stuck-button audit).
