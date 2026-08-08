# AirPods Case Mouse

Turn an AirPods charging case into a vision-tracked mouse using nothing but a
laptop's built-in webcam.

Rest your right hand on the case, slide it around the desk, and the Windows
cursor follows. Eventually, pressing your index finger will be left-click and
your middle finger right-click — as real held button states, so dragging works.

No extra hardware. No markers. No cloud. Everything runs locally on the CPU.

> **Status: Milestones 1–4 implemented.** Hand tracking, relative cursor
> movement, and index/middle finger clicking. Clicking is **disabled until you
> run the calibrator**, which measures thresholds against your own hand.

---

## How it works

```
webcam ─► mirror frame ─► MediaPipe HandLandmarker ─► pick the right hand
                                                            │
                                                     21 landmarks
                                                            │
                                                   palm anchor (5-point centroid)
                                                            │
                                            smoothing filter (EMA / One Euro)
                                                            │
                                      delta ÷ palm width  → scale-free "hand units"
                                                            │
                                     dead zone → acceleration → sensitivity → clamp
                                                            │
                                              SendInput relative move ─► cursor
```

Two ideas do most of the work:

**Everything is divided by palm width.** Motion is measured in *palm widths*,
not pixels. Lean toward the camera and your hand covers more pixels, but the
ratio to your palm width is unchanged — so the cursor behaves identically. One
sensitivity value keeps working when the screen angle or your posture shifts.

**Movement is relative, never absolute.** Each frame contributes
`anchor_now − anchor_previous`. There is no fixed mapping from a webcam
position to a screen coordinate, so your hand can drift anywhere in frame and
the cursor stays usable — exactly like lifting and repositioning a real mouse.

The same normalisation is what will separate a *click* from *moving the case*:
a fingertip's position **relative to the palm** does not change when the whole
hand translates, but does change when the finger bends.

---

## Prerequisites

- Windows 10 or 11
- Python **3.13** or 3.12 (see the note below)
- A working built-in or USB webcam

### A note on the Python version

MediaPipe 1.0.0 ships ABI-independent wheels (`py3-none-win_amd64`), so it does
install on Python 3.14. This project standardises on **3.13** because NumPy,
OpenCV and the rest of the stack all have mature 3.13 wheels today.

If `py -3.13` is not available:

```powershell
winget install Python.Python.3.13
```

---

## Installation

From the repository root, in PowerShell:

```powershell
# 1. Create the virtual environment
py -3.13 -m venv .venv

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Download the MediaPipe hand model (~7.5 MB, not committed to git)
.\scripts\download_model.ps1
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

You can skip activation entirely by calling the venv interpreter directly,
which is what every command below does.

---

## Running

```powershell
# Debug preview, mouse control OFF until you tap P five times (recommended first run)
.\.venv\Scripts\python.exe -m src.main

# Start with mouse control already enabled
.\.venv\Scripts\python.exe -m src.main --enable-mouse
```

Mouse control starts **disabled on purpose**. Get your hand positioned and
confirm the HUD says `RIGHT HAND: TRACKING` before enabling it.

### Hotkeys

Hotkeys are **global** — they work even when the preview window is not focused.
That is deliberate: when the cursor runs away from you, window focus is exactly
what you have lost.

| Key | Action |
| --- | --- |
| `P` ×5 | Toggle mouse control ON/OFF (**emergency disable**) — five taps within five seconds |
| `ESC` | Quit immediately |
| `F5` / `F6` | Invert the X / Y axis |
| `F7` | Cycle how the controlling hand is chosen (rightmost → leftmost → handedness → any) |
| `F9` | Cycle the palm anchor strategy (wrist → MCP centroid → palm centroid) |
| `F10` | Reset the motion filter |

The toggle needs a burst of taps rather than a single press because `P` is an
ordinary letter key — one tap would fire constantly while typing. The HUD shows
`P 3/5...` as you tap, and the count decays after five seconds.

### Useful flags

| Flag | Purpose |
| --- | --- |
| `--enable-mouse` | Start with control on |
| `--no-preview` | Headless (no debug window) |
| `--anchor palm_centroid` | Override the anchor strategy |
| `--camera-index 1` | Use a different camera |
| `--run-seconds 10` | Auto-exit; for smoke tests |
| `--log-level DEBUG` | Verbose logging |

---

## Physical setup

This matters more than any config value.

1. Put the **AirPods case on the desk directly in front of the laptop**, just
   below the trackpad.
2. **Tilt the screen slightly forward/down** until the webcam sees the desk in
   front of the keyboard.
3. Rest your **right hand on top of the case**, index and middle fingers on the
   lid.
4. Launch with the preview window and check that:
   - the hand skeleton is drawn on your hand,
   - the HUD reads `RIGHT HAND: TRACKING`,
   - the magenta anchor circle sits in the middle of your palm and stays put
     when you hold still,
   - `palm width` reads roughly 60–110 px.
5. Tap **P** five times and move the case.

Good lighting on your hand helps a lot. Backlighting (a bright window behind
the desk) is the most common cause of unstable tracking.

---

## Calibration (required for clicking)

Clicking ships **disabled**. Press thresholds are meaningless until measured
against a real hand — a guessed one produces stray clicks on whatever happens
to be under the cursor.

```powershell
.\.venv\Scripts\python.exe -m scripts.calibrate_press
```

It never sends mouse input; it only records. For each finger it runs two
phases:

1. **REST** (5 s) — hand on the case, fingers relaxed. Your resting finger may
   already be quite curled, which is exactly why "fingertip is low = click"
   does not work.
2. **PRESS** (8 s) — press and release repeatedly, the way you actually would.

It then prints both distributions and places the thresholds in the gap between
them:

```
  index REST : min=12.30  p05=13.10  median=15.80  p95=18.40  max=19.90
  index PRESS: min=31.20  p05=33.60  median=41.70  p95=49.10  max=52.30
  separation: rest p95 = 18.40 -> press p05 = 33.60
  -> press > 26.76, release < 22.96 (gap 15.20)
```

If the two ranges **overlap**, it refuses to write a threshold and says so —
that means the metric cannot separate your rest from your press, and no
threshold would work. Try `--metric drop`, or improve lighting and hand
position so tracking is steadier.

Results are written to `config/config.json` (git-ignored) and clicking is
enabled only if every requested finger separated cleanly.

Useful flags: `--dry-run` (measure without writing), `--fingers index`,
`--metric drop`, `--rest-seconds` / `--press-seconds`.

### How a press is detected

The metric is **finger flexion**: the total bend of the PIP and DIP joints, in
degrees, computed from MediaPipe's 3D *world* landmarks rather than image
pixels. That matters — world landmarks are metric with their origin at the
hand's centre, so the value is unchanged when you slide the case around, and
largely immune to the foreshortening this steep camera angle produces.

That invariance is the whole trick. Sliding the case moves every landmark
together and leaves joint angles untouched; bending a finger changes them.
So *moving the mouse* and *clicking* are cleanly separable.

On top of the metric sits a two-state machine with **hysteresis** (press and
release thresholds differ, so noise in the gap cannot flip the state) and
**debounce** (a minimum dwell time, which absorbs a single noisy frame). It
emits `mouseDown` and `mouseUp` exactly once per transition, so click-and-hold
and dragging work as with a real mouse.

### Direction conventions

The preview is mirrored, so it reads like a mirror:

- Move the case **right** → cursor moves **right**
- Move the case **away from you** (toward the screen) → cursor moves **up**

`invert_y` ships **enabled**, measured rather than derived: on the reference
setup the raw camera-Y sign moved the cursor the opposite way to the hand, so
the sign is flipped by default.

If either axis feels backwards on your setup, press **F5** (X) or **F6** (Y) to
flip it live — the HUD shows the current state — then write the setting you
landed on into `config/config.json` to make it stick.

---

## Configuration

`config/default_config.json` holds committed defaults. To change anything,
create **`config/config.json`** with only the keys you want to override — it is
git-ignored, so machine-specific tuning never gets committed.

```jsonc
{
  "cursor": {
    "sensitivity": 2000.0,
    "dead_zone": 0.006
  }
}
```

### Tune these first, in this order

| Key | Default | What it does |
| --- | --- | --- |
| `cursor.sensitivity` | `1600.0` | Screen pixels per palm width of hand travel. **Start here.** Raise if crossing the screen is a stretch; lower if it feels twitchy. |
| `cursor.dead_zone` | `0.004` | Motion below this (in palm widths/frame) is ignored. Raise if the cursor creeps while your hand rests; lower if slow movement feels sticky. |
| `cursor.smoothing` | `0.5` | EMA weight of the newest sample. `1.0` = no smoothing, lower = smoother but laggier. |
| `cursor.acceleration` | `0.35` | `gain = 1 + acceleration × speed`. `0.0` gives a perfectly linear 1:1 feel. |
| `anchor.strategy` | `palm_centroid` | Try all three live with **F9** and keep whichever holds steadiest. |

### Everything else

<details>
<summary>Full configuration reference</summary>

**`camera`** — `index` (0), `width` (640), `height` (480), `target_fps` (30),
`backend` (`dshow`), `flip_horizontal` (true).

`backend` defaults to DirectShow because the Media Foundation backend threw a
`cv::Mat` step assertion at 1280×720 on the development machine. DirectShow was
stable at every resolution tested.

**`tracking`** — `num_hands` (1), the three MediaPipe confidence thresholds
(0.5), `target_handedness` (`Right`), `tracking_loss_timeout` (0.35 s),
`min_confidence_for_control` (0.5).

**`cursor`** — the tuning table above, plus `x_sensitivity` / `y_sensitivity`
(per-axis trim), `invert_x` / `invert_y`, `max_velocity` (4000 px/s hard
ceiling), `filter` (`ema` | `one_euro` | `none`) and the three One Euro
parameters.

**`gestures`** — press/release thresholds. Placeholders until Milestone 3.

**`debug`** — `show_preview`, `window_name`, `draw_landmarks`, `log_level`.

</details>

---

## Development tools

```powershell
# Offline checks: filters, motion mapping, mouse bookkeeping. No camera,
# no real cursor movement.
.\.venv\Scripts\python.exe -m scripts.self_check

# Real capture + real inference, headless. Reports where frame time goes and
# how much the anchor jitters (use this to pick a dead zone).
.\.venv\Scripts\python.exe -m scripts.bench --seconds 10
```

Measured on the development laptop at 640×480:

| Stage | Mean | p95 |
| --- | --- | --- |
| Capture | 19.0 ms | 33.3 ms |
| Inference | 14.2 ms | 22.5 ms |
| Full frame | 33.2 ms | 46.3 ms |

**≈28 FPS**, against a camera that caps at 30. Capture and inference are
currently serialised, so a capture thread would recover the last couple of
frames — but not more, because the sensor itself is the ceiling. That is a
Milestone 8 concern, not a reason to add threads now.

---

## Safety

The program controls your mouse, so it is built to fail safe. All virtual
buttons are released when:

- the right hand disappears for longer than `tracking_loss_timeout`,
- mouse control is toggled off with five taps of P,
- the camera stops delivering frames,
- an exception escapes the capture loop,
- the application exits for any reason (`try/finally`).

Held-button state is tracked in-process rather than queried from Windows, so
`release_all()` always knows exactly what to let go of. A stuck mouse button is
the worst failure this project can have, and the shutdown path is written
around preventing it.

The first frame after the hand is (re)acquired always produces **zero** cursor
motion, so a momentary false detection cannot fling the pointer across the
screen.

---

## Troubleshooting

**`Could not open camera index 0`**
Close Teams/Zoom/Camera, then check *Settings → Privacy & security → Camera*.
Try `--camera-index 1`.

**Camera opens but crashes at higher resolution**
Known Media Foundation bug — keep `backend` as `dshow`.

**The wrong hand is being tracked**
Press **F7** to cycle the selection mode. The HUD lists every detected hand with
a `>` beside the chosen one, so you can see the effect immediately.

Note that MediaPipe's `Left`/`Right` labels are **not trusted by default**, and
for good reason: with a palm-down hand resting on a case, seen from a steeply
angled-down webcam, it confidently mislabels which hand is which (~0.97
confidence on the wrong answer). Since a label that flips when a finger bends
would swap hands mid-click, the default `rightmost` mode picks by position
instead — and the hand on the case never crosses to the other side of frame.

Left-handed? Use `--selection leftmost`.

**Cursor jitters while your hand is still**
Raise `cursor.dead_zone`. Run `scripts.bench` to see your actual anchor jitter
in palm-width units and set the dead zone just above the p95 figure.

**Cursor feels laggy**
Lower smoothing strength by raising `cursor.smoothing` toward `1.0`, or switch
`cursor.filter` to `"one_euro"`, which smooths hard at rest and barely at all
during fast movement.

**Cursor drifts on its own**
Usually the anchor wandering as MediaPipe re-fits the hand. Try a different
anchor with F9; `palm_centroid` averages five landmarks and is normally the
steadiest.

**Movement feels doubly accelerated**
Windows applies its own pointer acceleration to relative input. Either set
`cursor.acceleration` to `0.0`, or turn off *Enhance pointer precision* in
*Settings → Bluetooth & devices → Mouse → Additional mouse settings → Pointer
Options*.

**`ModuleNotFoundError: No module named 'src'`**
Run from the repository root using `-m` (`python -m src.main`), not
`python src/main.py`.

---

## Project layout

```
airpods-case-mouse/
├── config/
│   ├── default_config.json    committed defaults
│   └── config.json            your overrides (git-ignored)
├── models/                    hand_landmarker.task (git-ignored)
├── scripts/
│   ├── download_model.ps1
│   ├── self_check.py          offline correctness checks
│   └── bench.py               headless performance profile
└── src/
    ├── main.py                capture loop, hotkeys, safety
    ├── camera/camera_manager.py
    ├── tracking/
    │   ├── hand_tracker.py    MediaPipe Tasks wrapper
    │   └── hand_features.py   anchors, palm width, relative geometry
    ├── mouse/
    │   ├── mouse_controller.py  SendInput via ctypes
    │   ├── cursor_filter.py     EMA / One Euro / none
    │   └── motion_mapper.py     anchor motion → cursor pixels
    ├── debug/hud.py
    ├── hotkeys.py             global keys via GetAsyncKeyState
    └── config/settings.py
```

### Implementation notes

**MediaPipe 1.0.0 removed `mediapipe.solutions.hands`.** Almost every tutorial
online still uses it. The supported path is now the Tasks API
(`mediapipe.tasks.python.vision.HandLandmarker`) with a downloaded `.task`
bundle, which is what this project uses.

**VIDEO mode, not LIVE_STREAM.** VIDEO is synchronous, so a frame and its
landmarks stay together and the loop keeps one obvious ordering.
LIVE_STREAM's async callback only pays off once inference is the bottleneck,
which it is not at 30 FPS.

**Raw `SendInput` over PyAutoGUI/pynput.** It produces genuine relative motion
through the same path as a real mouse, gives true independent button down/up
for dragging, and adds no dependency. PyAutoGUI implements `moveRel` as an
absolute `SetCursorPos` and carries a built-in pause per call.

**Sub-pixel motion is accumulated.** `SendInput` takes integers, so naively
rounding would discard every movement below one pixel per frame and destroy
fine control. Fractions carry over to the next frame instead.

---

## Roadmap

| Milestone | Status |
| --- | --- |
| 1 — Camera + right-hand tracking + debug HUD | ✅ Done |
| 2 — Relative palm-anchor cursor movement | ✅ Done |
| 3 — Index finger → left button (press/hold/release) | ✅ Done |
| 4 — Middle finger → right button | ✅ Done |
| 5 — Calibration workflow | ✅ Press calibration done; cursor calibration planned |
| 6 — Landmark dataset recorder | Planned |
| 7 — Personalised press classifier | Planned |
| 8 — Cursor feel: One Euro, threaded capture, adaptive dead zone | Planned |
| 9 — Optional tray app / GUI | Planned |

---

## Known limitations

- **Clicking is heuristic, not learned.** It uses one hand-relative geometric
  feature with hysteresis. A personalised classifier (Milestone 7) should do
  better, especially at rejecting false positives.
- **Press thresholds are per-setup.** Change your posture or screen angle
  enough and you may need to recalibrate.
- **~28 FPS ceiling**, set by the webcam, not the code. Every frame of latency
  is ~33 ms, which also bounds how fast a press can be detected.
- **Right hand only**, by design.
- **Lighting sensitive.** Backlighting degrades landmark stability badly.
- **Thresholds are placeholders.** The values in `gestures` were not measured
  from a real hand and mean nothing until Milestone 3 instruments them.
- **The anchor is the palm, not the case.** The AirPods case is not detected;
  the hand resting on it is the proxy. Fiducial-marker and object-detection
  options remain open if palm tracking proves insufficient.
- **Windows only.** `SendInput` and `GetAsyncKeyState` are Win32 APIs.

---

## Privacy

Camera frames are processed in memory and never written to disk. Nothing leaves
the machine. `data/` and all recording formats are git-ignored so future
landmark datasets — which are personal biometric data — cannot be committed by
accident.
