class Freq:
    def __init__(self, mins: int) -> None:
        self.mins = mins

    def mw_to_mwh(self, mw: float) -> float:
        return mw * self.mins / 60

    def mwh_to_mw(self, mw: float) -> float:
        return mw * 60 / self.mins

    def __repr__(self) -> str:
        return f"Freq(mins={self.mins})"
