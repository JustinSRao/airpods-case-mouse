---
name: mouse-safety-check
description: Audit any change that touches mouse input, the capture loop, tracking state, or shutdown for the stuck-button failure mode. Use before committing changes to mouse_controller, main loop, gesture state machines, or error handling — and whenever the user reports a stuck/held mouse button or losing control of the cursor.
---

# Stuck-button audit

This program synthesises real mouse input. The worst thing it can do is leave a
button held down — the user then cannot click anything, including this
program's own window, to fix it. Every path that can hold a button must have a
path that releases it.

## The invariant

> If a virtual button is down, there exists a reachable code path that releases
> it, and that path runs on **every** exit — normal, error, or crash.

## Audit checklist

Walk the diff against every item. Anything unchecked blocks the commit.

- [ ] **Process exit** — `release_all()` in a `finally` that cannot be skipped.
      Verify it survives an exception thrown from inside the capture loop, not
      just a clean return.
- [ ] **Tracking loss** — buttons released after `tracking_loss_timeout`. Test
      by pulling the hand out of frame mid-press.
- [ ] **Control toggle (F8)** — disabling releases immediately, no timeout.
- [ ] **Camera failure** — a dead camera raises rather than spinning forever;
      the raise must unwind through the `finally`.
- [ ] **Low confidence** — dropping below `min_confidence_for_control` is
      treated as no-hand, not as a frozen last-known state.
- [ ] **Exceptions** — no bare `except:` that swallows an error while a button
      is held. Logging and re-raising is fine; silent continue is not.
- [ ] **Idempotence** — `release_all()` is safe to call twice; `press()` twice
      sends one event.
- [ ] **Both buttons** — index and middle held simultaneously release cleanly.
- [ ] **Reacquisition** — the first frame after regaining tracking emits zero
      motion, so a false detection cannot fling the cursor.

## Verify, don't assume

```powershell
.\.venv\Scripts\python.exe -m scripts.self_check
```

This covers button bookkeeping with the send path stubbed. **It never sends
real button events** — a synthetic click during a test would land on whatever
window is under the pointer. Keep it that way: if you add button tests, stub
`MouseController._send`.

For live testing, always start with control **off** (the default) and enable
with F8 only once the HUD confirms tracking.

## Reviewing new code

Any new early `return`, `break`, or `raise` inside the capture loop is a
potential leak. Ask: *if a button were held right now, would it be released?*

State that answers "no" belongs behind `MouseController`, which tracks held
buttons in-process precisely so `release_all()` always knows what to let go of.
Do not query Windows for button state — a desync there means a stuck button
nobody can find.

## If the user reports a stuck button right now

Tell them immediately: press **F8** to disable control, or **ESC** to quit. If
the program is already dead and the button is still held, a physical mouse
click of the same button clears it, since the OS button state is global.
