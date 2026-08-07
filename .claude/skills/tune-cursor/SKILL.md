---
name: tune-cursor
description: Diagnose and fix how the cursor *feels* — jitter, lag, drift, wrong direction, too fast/slow, or overshoot. Use when the user says the pointer is jittery, laggy, drifty, sticky, twitchy, inverted, hard to control, or "doesn't feel like a real mouse". Measures real numbers from the webcam before changing any value.
---

# Tuning cursor feel

The rule for this project: **never guess a threshold that can be measured.**
Run the instrumentation, read the actual numbers off the user's hand, then
change one value at a time.

## Step 1 — Reproduce and classify

Ask which symptom it is, or infer from the wording. Each has a different cause
and a different knob. Do not change more than one at a time.

| Symptom | Cause | Knob |
| --- | --- | --- |
| Cursor creeps while hand is still | Anchor jitter exceeds the dead zone | `cursor.dead_zone` ↑ |
| Slow deliberate movement ignored | Dead zone too large | `cursor.dead_zone` ↓ |
| Cursor trails behind the hand | Over-smoothing | `cursor.smoothing` ↑ toward 1.0, or `filter` → `one_euro` |
| Smooth when slow, laggy when fast | EMA's fixed tradeoff | `filter` → `one_euro`, raise `one_euro_beta` |
| Wanders with hand stationary | Anchor re-fitting | Cycle anchor with **F9** |
| Overshoots constantly | Sensitivity or acceleration too high | `cursor.sensitivity` ↓, then `cursor.acceleration` ↓ |
| Can't cross the screen | Sensitivity too low | `cursor.sensitivity` ↑ |
| Feels doubly accelerated | Windows pointer accel stacking on ours | `cursor.acceleration` → 0.0, or disable *Enhance pointer precision* |
| Axis backwards | Setup geometry | `cursor.invert_x` / `invert_y` |

## Step 2 — Measure

```powershell
.\.venv\Scripts\python.exe -m scripts.bench --seconds 15
```

Have the user hold the hand **still on the case** for the whole run. The line
that matters:

```
Anchor step: median 0.00xxx pw   p95 0.00xxx pw   (current dead_zone = 0.004)
```

That is per-frame anchor jitter in palm widths. **Set `dead_zone` just above
the p95 value.** Below it the cursor creeps; far above it, slow movement dies.

Also check `palm width` — it should be roughly 60–110 px. Outside that, the
physical setup is wrong (hand too far/close) and no config value will fix it.

For anchor stability, run the app and cycle **F9** through all three
strategies, watching whether the magenta circle stays put. `palm_centroid`
averages five landmarks and usually wins, but `wrist` is sometimes steadier if
the fingers are moving a lot.

## Step 3 — Change one value

Edit **`config/config.json`** (git-ignored, overrides only) — never
`default_config.json`:

```jsonc
{ "cursor": { "dead_zone": 0.006 } }
```

Then re-run and have the user confirm. If a default turns out to be genuinely
wrong for everyone rather than just this machine, move it into
`default_config.json` and say so.

## Reference: what each parameter actually does

- `sensitivity` — screen pixels per palm width of travel. The master gain.
- `dead_zone` — palm widths/frame below which motion is discarded. Subtracted
  rather than thresholded, so motion ramps from zero instead of jumping.
- `smoothing` — EMA weight of the newest sample. 1.0 = off.
- `acceleration` — `gain = 1 + acceleration × speed`, speed in palm widths/s.
- `max_velocity` — hard ceiling in px/s; a safety net, not a feel knob.
- `one_euro_min_cutoff` — lower = less jitter at rest.
- `one_euro_beta` — higher = less lag when moving fast.

## Don't

- Don't stack changes; you lose track of what helped.
- Don't add a new filter before trying `one_euro`, which is already implemented.
- Don't tune around a physical setup problem — bad lighting and a hand that is
  half out of frame look exactly like bad parameters in the HUD.
