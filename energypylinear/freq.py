class Freq:
    def __init__(self, mins: int):
        self.mins = mins

    def mw_to_mwh(self, mw: float):
        return mw * 60 / self.mins
