---
name: gesture-instrument
description: Build measurement instrumentation for a finger gesture BEFORE writing detection logic — for index/middle press detection, click thresholds, press/release hysteresis, or any new gesture. Use when work touches finger press detection, click chatter, false-positive clicks, or picking a press threshold.
---

# Instrument first, threshold second

## Read this before touching press detection

Hard-won on real data. Re-deriving any of it costs hours.

1. **A rigid object cannot be pressed *into*.** A finger held against an
   AirPods case sits exactly where a resting finger sits. The detectable event
   is the **lift**, not the push. Preserve that in any protocol or interaction
   change.
2. **Never discard the transition.** An early calibration dropped 0.6 s after
   each cue as "settling" — precisely when the gesture happened — and so
   compared two identical resting states. Four rounds of analysis found
   nothing, and d' *fell* as the analysis improved. Record every frame; apply
   settling at analysis time.
3. **Baseline-relative, never absolute.** Resting posture drifts more over a
   few seconds than a press changes anything.
4. **Judge by simulating the state machine**, not percentile overlap.
   Hysteresis and debounce change the answer near the boundary: the overlap
   rule scored 0/12 on data the real detector handled 12/12.
5. **But always pair simulation with a minimum effect size.** Alone, the
   threshold search declared success on ~25% of recordings containing *no
   press at all*.
6. **Save the recording.** `--analyse` re-runs offline. Never make the user
   perform the gesture again just to try a parameter.
7. **Check the instrument before believing it.** The first press monitor
   auto-scaled each bar to its own peak, so pure noise looked exactly like
   signal. Display deviations in units of the metric's own noise floor.

The project rule: **do not guess press thresholds.** Every number in
`config.gestures` must come from a recording of the user's actual hand doing
the actual motion. A guessed threshold produces false-positive clicks, and a
false click is worse than a missed one — it fires on whatever is under the
cursor.

## Order of work

1. Pick candidate features (below).
2. Add them to the debug HUD as **live numbers**.
3. Have the user rest, move the case, and press — and read off the ranges.
4. Only then choose thresholds, with hysteresis.
5. Verify against the false-positive checklist.

Never collapse these into one step.

## The core constraint

The whole hand moves while the fingers also move. So a press can **never** be
detected from absolute fingertip position.

```
BAD:   index_tip.y > threshold                 # fires whenever the case moves down
GOOD:  (index_tip - palm_centroid) / palm_width  # translation- and scale-invariant
```

`hand_features.fingertip_relative_to_palm()` already does this, and
`scripts/self_check.py` asserts both invariances — keep those assertions
passing.

The distinguishing test:

- Whole hand translates, fingertip-relative stays constant → **case movement**
- Palm still, fingertip-relative changes → **press**

`HandObservation.world_landmarks` is also worth trying: MediaPipe returns it in
metres with the origin at the hand's centre, so it is already
translation-invariant without any arithmetic.

## Candidate features

Per finger, over a short temporal window (4–10 frames ≈ 130–330 ms):

- fingertip position relative to palm centroid, in palm widths
- PIP and DIP joint angles
- fingertip-to-palm distance
- velocity and acceleration of the above
- curvature change along the finger chain
- **ratio of fingertip motion to whole-hand motion** — the single most
  discriminating signal

## Instrumentation pattern

Extend `HudState` and `draw_panel` in `src/debug/hud.py` to show the live
metric next to its threshold, matching the existing format:

```
INDEX:
  State: UP
  Press metric: 0.32
  Press threshold: 0.62
```

Watching that number while pressing is how the threshold gets chosen. Add a
rolling min/max readout so the user can see the full range of a real press
without freeze-framing.

## State machine, not per-frame classification

Never emit a click from a single frame's comparison. Use explicit states with
**hysteresis** (press and release thresholds must differ) and a debounce:

```
INDEX_UP --(metric > press_threshold)--> INDEX_DOWN   → send mouseDown ONCE
INDEX_DOWN --(metric < release_threshold)--> INDEX_UP → send mouseUp ONCE
```

Index and middle must be tracked independently, and all four combinations
(neither / index / middle / both) must be valid states.

## False-positive checklist

Before declaring a gesture working, confirm each of these produces **zero**
clicks:

- [ ] Moving the case fast in every direction
- [ ] Lifting the hand off the case and putting it back
- [ ] Hand entering and leaving the frame
- [ ] Resting the hand still for 30 s
- [ ] Pressing the *other* finger
- [ ] Tracking lost mid-press → button must be **released**, never stuck

That last one is non-negotiable — `MouseController.release_all()` exists for it.

## Then, and only then

Write measured values into `config/default_config.json` with a comment in the
commit message recording how they were obtained.
