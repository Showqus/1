"""Работа с Windows: мышь, горячие клавиши, DPI, звук.

На других ОС используются заглушки (нужны только для тестов и разработки).
"""
from __future__ import annotations

import logging
import sys
import time

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

VK_CODES = {f"F{i}": 0x6F + i for i in range(1, 13)}
VK_CODES.update({"PAUSE": 0x13, "END": 0x23, "HOME": 0x24, "INSERT": 0x2D, "SCROLLLOCK": 0x91})
VK_SHIFT, VK_CONTROL, VK_MENU = 0x10, 0x11, 0x12


class InputDevice:
    """Интерфейс мыши, которым пользуется бот."""

    def get_cursor(self) -> tuple[int, int]:
        raise NotImplementedError

    def set_cursor(self, x: int, y: int) -> None:
        raise NotImplementedError

    def mouse_down(self) -> None:
        raise NotImplementedError

    def mouse_up(self) -> None:
        raise NotImplementedError

    def idle_seconds(self) -> float:
        """Сколько секунд пользователь не трогал мышь и клавиатуру."""
        return 1e9

    def window_title_at(self, x: int, y: int) -> str:
        return ""

    def window_at(self, x: int, y: int) -> int:
        """Идентификатор окна верхнего уровня в точке (0 — неизвестно)."""
        return 0


def set_dpi_awareness() -> str:
    """Включить DPI-awareness, чтобы координаты мыши и снимков совпадали при масштабе 125%/150%."""
    if not IS_WINDOWS:
        return "не Windows"
    import ctypes

    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per-monitor v2"
    except Exception:  # noqa: BLE001 - старые Windows
        pass
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return "per-monitor"
    except Exception:  # noqa: BLE001
        pass
    try:
        if ctypes.windll.user32.SetProcessDPIAware():
            return "system"
    except Exception:  # noqa: BLE001
        pass
    return "не удалось"


def beep(kind: str = "warning") -> None:
    if not IS_WINDOWS:
        return
    try:
        import winsound

        flags = {"warning": winsound.MB_ICONEXCLAMATION, "error": winsound.MB_ICONHAND, "info": winsound.MB_ICONASTERISK}
        winsound.MessageBeep(flags.get(kind, winsound.MB_OK))
    except Exception:  # noqa: BLE001
        log.debug("Не удалось подать звуковой сигнал", exc_info=True)


def key_pressed(name: str) -> bool:
    """Нажата ли сейчас клавиша (например "F8" или "SHIFT+F9")."""
    if not IS_WINDOWS:
        return False
    import ctypes

    parts = name.upper().replace(" ", "").split("+")
    get = ctypes.windll.user32.GetAsyncKeyState
    get.restype = ctypes.c_short
    mods = {"SHIFT": VK_SHIFT, "CTRL": VK_CONTROL, "ALT": VK_MENU}
    for p in parts[:-1]:
        if p in mods and not (get(mods[p]) & 0x8000):
            return False
    vk = VK_CODES.get(parts[-1])
    if vk is None:
        return False
    return bool(get(vk) & 0x8000)


def system_info() -> dict:
    info = {"platform": sys.platform, "python": sys.version.split()[0]}
    if not IS_WINDOWS:
        return info
    import ctypes
    import platform

    info["windows"] = platform.platform()
    try:
        user32 = ctypes.windll.user32
        info["virtual_screen"] = (
            user32.GetSystemMetrics(76), user32.GetSystemMetrics(77),
            user32.GetSystemMetrics(78), user32.GetSystemMetrics(79),
        )
        info["primary_screen"] = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
        info["monitors"] = user32.GetSystemMetrics(80)
        try:
            info["dpi_primary"] = user32.GetDpiForSystem()
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        info["error"] = repr(e)
    return info


if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.windll.user32
    _kernel32 = ctypes.windll.kernel32

    ULONG_PTR = ctypes.c_size_t

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    INPUT_MOUSE = 0
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004

    _user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    _user32.SendInput.restype = wintypes.UINT
    _user32.WindowFromPoint.argtypes = (wintypes.POINT,)
    _user32.WindowFromPoint.restype = wintypes.HWND
    _user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
    _user32.GetAncestor.restype = wintypes.HWND
    _user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)

    class WinInput(InputDevice):
        def get_cursor(self) -> tuple[int, int]:
            pt = wintypes.POINT()
            _user32.GetCursorPos(ctypes.byref(pt))
            return pt.x, pt.y

        def set_cursor(self, x: int, y: int) -> None:
            if not _user32.SetCursorPos(int(x), int(y)):
                log.warning("SetCursorPos(%s, %s) не сработал, код ошибки %s", x, y, _kernel32.GetLastError())

        def _send(self, flags: int) -> None:
            inp = INPUT(type=INPUT_MOUSE)
            inp.u.mi = MOUSEINPUT(0, 0, 0, flags, 0, 0)
            sent = _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
            if sent != 1:
                log.error(
                    "SendInput не отправил клик (код ошибки %s). Если браузер запущен от администратора — "
                    "запустите бота тоже от администратора.", _kernel32.GetLastError(),
                )

        def mouse_down(self) -> None:
            self._send(MOUSEEVENTF_LEFTDOWN)

        def mouse_up(self) -> None:
            self._send(MOUSEEVENTF_LEFTUP)

        def idle_seconds(self) -> float:
            lii = LASTINPUTINFO(cbSize=ctypes.sizeof(LASTINPUTINFO))
            if not _user32.GetLastInputInfo(ctypes.byref(lii)):
                return 1e9
            now = _kernel32.GetTickCount() & 0xFFFFFFFF
            return ((now - lii.dwTime) & 0xFFFFFFFF) / 1000.0

        def window_at(self, x: int, y: int) -> int:
            try:
                hwnd = _user32.WindowFromPoint(wintypes.POINT(int(x), int(y)))
                if not hwnd:
                    return 0
                return int(_user32.GetAncestor(hwnd, 2) or hwnd)  # GA_ROOT
            except Exception:  # noqa: BLE001
                log.debug("window_at failed", exc_info=True)
                return 0

        def window_title_at(self, x: int, y: int) -> str:
            try:
                root = self.window_at(x, y)
                if not root:
                    return ""
                buf = ctypes.create_unicode_buffer(512)
                _user32.GetWindowTextW(root, buf, 512)
                return buf.value
            except Exception:  # noqa: BLE001
                log.debug("window_title_at failed", exc_info=True)
                return ""


class DummyInput(InputDevice):
    """Заглушка для не-Windows: только запоминает позицию курсора."""

    def __init__(self):
        self.pos = (0, 0)
        self.clicks: list[tuple[int, int]] = []
        self._down = False
        self._last = time.monotonic()

    def get_cursor(self) -> tuple[int, int]:
        return self.pos

    def set_cursor(self, x: int, y: int) -> None:
        self.pos = (int(x), int(y))

    def mouse_down(self) -> None:
        self._down = True

    def mouse_up(self) -> None:
        if self._down:
            self.clicks.append(self.pos)
        self._down = False


def default_input() -> InputDevice:
    if IS_WINDOWS:
        return WinInput()
    log.warning("Не Windows: используется заглушка мыши, клики не выполняются")
    return DummyInput()
