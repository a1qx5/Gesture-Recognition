import winreg
import ctypes
from ctypes import wintypes

class DisplayScaleManager:
    DPI_TO_PERCENT = {
        96: 100,
        120: 125,
        144: 150,
        168: 175,
        192: 200,
    }
    PERCENT_TO_DPI = {v: k for k, v in DPI_TO_PERCENT.items()}

    @staticmethod
    def get_current_scale():
        """
        Get the current windows display scale %.

        Returns:
            int: Current scale %.
        """
        dpi_value = None  # Initialize to avoid reference errors

        try:
            # Open the registry with key (read only)
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Control Panel\Desktop\WindowMetrics",
                0,
                winreg.KEY_READ
            )
            dpi_value, _ = winreg.QueryValueEx(key, "AppliedDPI")
            winreg.CloseKey(key)
        except WindowsError:
            pass

        if dpi_value is None:
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Control Panel\Desktop",
                    0,
                    winreg.KEY_READ
                )
                dpi_value, _ = winreg.QueryValueEx(key, "LogPixels")
                winreg.CloseKey(key)
            except WindowsError:
                pass

        if dpi_value is None:
            return 100

        # Convert DPI to percentage using class attribute
        if dpi_value in DisplayScaleManager.DPI_TO_PERCENT.keys():
            return DisplayScaleManager.DPI_TO_PERCENT[dpi_value]
        else:
            # Find closest DPI value if exact match not found
            closest = min(DisplayScaleManager.DPI_TO_PERCENT.keys(), key=lambda k: abs(k - dpi_value))
            return DisplayScaleManager.DPI_TO_PERCENT[closest]

    @staticmethod
    def set_display_scale(percent: int):
        """
        Set the windows display scale percentage.
        Args:
            percent: Scale percentage (100, 125, 150, 175, or 200)

        Returns:
            tuple: (success: bool, message: str)
        """

        if percent not in DisplayScaleManager.PERCENT_TO_DPI.keys():
            return False, f"Invalid Scale: {percent}%"

        dpi = DisplayScaleManager.PERCENT_TO_DPI[percent]

        try:
            # Write to WindowMetrics\AppliedDPI
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Control Panel\Desktop\WindowMetrics",
                0,
                winreg.KEY_WRITE
            )
            winreg.SetValueEx(key, "AppliedDPI", 0, winreg.REG_DWORD, dpi)
            winreg.CloseKey(key)

            # Write to Desktop\LogPixels (fallback for compatibility)
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Control Panel\Desktop",
                0,
                winreg.KEY_WRITE
            )
            winreg.SetValueEx(key, "LogPixels", 0, winreg.REG_DWORD, dpi)
            winreg.CloseKey(key)
        except PermissionError:
            return False, "Permission denied. Run as administrator."
        except Exception as e:
            return False, f"Registry write failed: {str(e)}"

        HWND_BROADCAST = 0xFFFF
        WM_SETTINGCHANGE = 0x001A
        SMTO_ABORTIFHUNG = 0x0002

        try:
            result = ctypes.c_long()
            ctypes.windll.user32.SendMessageTimeoutW(
                HWND_BROADCAST,
                WM_SETTINGCHANGE,
                0,
                "WindowMetrics",
                SMTO_ABORTIFHUNG,
                5000,
                ctypes.byref(result)
            )
        except Exception as e:
            pass

        return True, f"Display scale set to {percent}%."

    @staticmethod
    def get_available_scales():
        """
        Get list of available display scale percentages.

        Returns:
            list: Available scale percentages [100, 125, 150, 175, 200]
        """
        return sorted(DisplayScaleManager.PERCENT_TO_DPI.keys())
