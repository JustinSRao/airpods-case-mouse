"""Global hotkey polling via ``GetAsyncKeyState``.

Deliberately not using OpenCV's ``waitKey``: that only delivers keys while the
preview window has focus, which is useless for an emergency disable -- the
moment the cursor runs away, focus is exactly what you have lost. ``keyboard``
and similar libraries would work but install a system-wide hook (and often
want administrator rights); polling ``GetAsyncKeyState`` once per frame is
dependency-free, needs no elevation, and works regardless of focus.
"""

from __future__ import annotations

import ctypes

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
_user32.GetAsyncKeyState.restype = ctypes.c_short

# High bit of the return value means "currently down".
_KEY_DOWN_MASK = 0x8000

VK_ESCAPE = 0x1B
VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79


class Hotkeys:
    """Edge-triggered global key watcher.

    Call :meth:`poll` once per frame, then query :meth:`just_pressed`. Edges
    are computed against our own previous snapshot rather than
    ``GetAsyncKeyState``'s "pressed since last call" low bit, which is shared
    process-wide and unreliable if anything else queries the same key.
    """

    def __init__(self, bindings: dict[str, int]) -> None:
        self._bindings = dict(bindings)
        self._previous = {name: False for name in self._bindings}
        self._current = dict(self._previous)

    def poll(self) -> None:
        self._previous = self._current
        self._current = {
            name: bool(_user32.GetAsyncKeyState(vk) & _KEY_DOWN_MASK)
            for name, vk in self._bindings.items()
        }

    def just_pressed(self, name: str) -> bool:
        """True on the frame the key transitions from up to down."""
        return self._current.get(name, False) and not self._previous.get(name, False)

    def is_down(self, name: str) -> bool:
        return self._current.get(name, False)
