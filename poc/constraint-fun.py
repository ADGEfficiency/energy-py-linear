efficiency = 0.9
datas = [
    {"charge": 1.0, "bin": 1, "losses": 0.1},
    {"charge": 0.5, "bin": 1, "losses": 0.05},
    {"charge": -1.0, "bin": 0},
    # {"charge": 1.0, "bin": 0, "valid": False},
    # {"charge": -1.0, "bin": 1, "valid": False},
]
M = 1000
for data in datas:
    # losses = data["losses"]
    losses = data.get("losses", 0)
    charge = data["charge"]
    bin = data["bin"]

    valid = data.get("valid", True)

    print(f"{valid=}")
    print(charge <= M * bin)
    print(charge >= -M * (1 - bin))
    """
y <= p * x * b
y >= p * x * b
y <= M * (1 - b)
    """

    print(losses <= efficiency * charge * bin)
    print(losses >= efficiency * charge * bin)
