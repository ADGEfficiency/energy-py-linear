"""Handles conversion of power (MW) to energy (MWh) at different interval frequencies."""


class Freq:
    """Convert power and energy measurements for interval data frequencies."""

    def __init__(self, mins: int) -> None:
        """Initialize a Freq class."""
        self.mins = mins

    def mw_to_mwh(self, mw: float) -> float:
        """Convert power MW to energy MWh."""
        return mw * self.mins / 60

    def mwh_to_mw(self, mw: float) -> float:
        """Convert energy MWh to power MW."""
        return mw * 60 / self.mins

    def __repr__(self) -> str:
        """Control printing."""
        return f"Freq(mins={self.mins})"
