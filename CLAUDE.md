# Project notes for Claude

Webcam-tracked mouse: right hand rests on an AirPods case, hand motion drives
the Windows cursor. Milestones 1–2 done (tracking + cursor). Milestone 3
(index finger → left click) is next.

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
  full pipeline measures ≈28 FPS at 640×480.
- Do **not** install `opencv-python`; mediapipe pulls `opencv-contrib-python`
  and the two conflict.
- Display is 2560×1600. `enable_dpi_awareness()` must run before any pixel
  measurement or Windows reports a virtualised 1707×1067.

## Commands

```powershell
.\.venv\Scripts\python.exe -m src.main                      # run (mouse off; F8 enables)
.\.venv\Scripts\python.exe -m src.main --run-seconds 6 --no-preview   # smoke test
.\.venv\Scripts\python.exe -m scripts.self_check            # offline checks
.\.venv\Scripts\python.exe -m scripts.bench --seconds 10    # perf + jitter profile
```

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
