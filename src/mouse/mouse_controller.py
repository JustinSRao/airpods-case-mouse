"""Windows mouse control via ``SendInput``.

Why raw ``SendInput`` through ctypes rather than PyAutoGUI or pynput:

* It is the API Windows itself defines for synthesising input, so it produces
  genuine relative motion (``MOUSEEVENTF_MOVE`` without ``ABSOLUTE``) that
  travels the same code path as a real mouse -- including the user's pointer
  speed and "enhance pointer precision" settings.
* It gives true independent button down/up events, which is what dragging
  needs. PyAutoGUI's ``moveRel`` is implemented as an absolute
  ``SetCursorPos`` and carries a built-in pause, adding latency we cannot
  afford in a per-frame loop.
* It needs no third-party dependency.

The ``MouseController`` interface is deliberately small so a different backend
can be dropped in later without touching the motion pipeline.
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from enum import Enum

log = logging.getLogger(__name__)

_user32 = ctypes.WinDLL("user32", use_last_error=True)

# ULONG_PTR is pointer-sized; wintypes has no portable alias for it.
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
_user32.SendInput.restype = wintypes.UINT


class MouseButton(Enum):
    LEFT = ("left", MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP)
    RIGHT = ("right", MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP)
    MIDDLE = ("middle", MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP)

    def __init__(self, label: str, down_flag: int, up_flag: int) -> None:
        self.label = label
        self.down_flag = down_flag
        self.up_flag = up_flag


class MouseController:
    """Sends relative motion and button events, and tracks what is held.

    Held-button state is tracked here (not by querying Windows) so that
    ``release_all`` can guarantee we never leave a synthetic button stuck down
    -- the single most dangerous failure mode of this application.
    """

    def __init__(self) -> None:
        self._held: set[MouseButton] = set()
        # Motion below one pixel per frame would be truncated away by the
        # integer SendInput API, killing fine control. Carry the fraction over
        # to the next frame instead.
        self._residual_x = 0.0
        self._residual_y = 0.0

    @property
    def held_buttons(self) -> frozenset[MouseButton]:
        return frozenset(self._held)

    def is_held(self, button: MouseButton) -> bool:
        return button in self._held

    def move_relative(self, dx: float, dy: float) -> tuple[int, int]:
        """Move the cursor by a relative amount; returns the integer step sent."""
        self._residual_x += dx
        self._residual_y += dy
        step_x = int(self._residual_x)
        step_y = int(self._residual_y)
        self._residual_x -= step_x
        self._residual_y -= step_y

        if step_x == 0 and step_y == 0:
            return (0, 0)

        self._send(dx=step_x, dy=step_y, flags=MOUSEEVENTF_MOVE)
        return (step_x, step_y)

    def press(self, button: MouseButton) -> None:
        if button in self._held:
            return
        self._send(flags=button.down_flag)
        self._held.add(button)
        log.info("%s mouse DOWN", button.label.upper())

    def release(self, button: MouseButton) -> None:
        if button not in self._held:
            return
        self._send(flags=button.up_flag)
        self._held.discard(button)
        log.info("%s mouse UP", button.label.upper())

    def release_all(self) -> None:
        """Release every button we believe is held. Safe to call repeatedly."""
        for button in list(self._held):
            self.release(button)

    def reset_residual(self) -> None:
        """Drop accumulated sub-pixel motion (on tracking loss or disable)."""
        self._residual_x = 0.0
        self._residual_y = 0.0

    def _send(self, flags: int, dx: int = 0, dy: int = 0) -> None:
        event = _INPUT(
            type=INPUT_MOUSE,
            u=_INPUTUNION(
                mi=_MOUSEINPUT(
                    dx=dx, dy=dy, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0
                )
            ),
        )
        sent = _user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_INPUT))
        if sent != 1:
            error = ctypes.get_last_error()
            # Do not raise: a dropped event mid-loop must not tear down the
            # app while a button might be held. Log loudly instead.
            log.error("SendInput failed (flags=0x%04X, WinError %d)", flags, error)


def enable_dpi_awareness() -> None:
    """Opt out of DPI virtualisation so pixel counts are physical pixels.

    Without this, a process on a scaled display sees a shrunken virtual
    desktop (e.g. 1707x1067 instead of 2560x1600) and every pixel measurement
    is silently wrong. Relative motion happens to be unaffected, but the
    reported screen size and any future absolute mapping would not be.

    Best effort: the per-monitor-v2 context is Windows 10 1703+, so fall back
    to the older system-DPI call, and ignore failure on anything older.
    """
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == -4
        if _user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except AttributeError:
        pass
    try:
        _user32.SetProcessDPIAware()
    except AttributeError:
        log.debug("No DPI awareness API available; pixel sizes may be virtualised")


def get_screen_size() -> tuple[int, int]:
    """Primary display size in pixels (SM_CXSCREEN / SM_CYSCREEN)."""
    return (_user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1))
