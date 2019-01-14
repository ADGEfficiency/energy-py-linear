import json

from make_logger import make_logger

logger = make_logger()

class Battery(object):
    """
    Electric battery operating in price arbitrage

    power      float [MW] same for charge & discharge
    capacity   float [MWh]
    efficiency float [%] round trip, applied to
    timestep   str   5min, 1hr etc
    """

    def __init__(
            self,
            power,
            capacity,
            efficiency=0.9,
            timestep='5min'

    ):
        self.power = float(power)
        self.capacity = float(capacity)
        self.efficiency = float(efficiency)

        args = {"args":
                {"power": self.power,
                 "capacity": self.capacity,
                 "efficiency": self.efficiency}}
        logger.info(json.dumps(args))

if __name__ == '__main__':

    model = Battery(power=2, capacity=4)

    def read_logs():
        with open('battery.log') as f:
            logs = f.read().splitlines()

        return [json.loads(log) for log in logs]

    logs = read_logs()



