import pytest

from energypylinear.battery import Battery, freq_to_timestep


@pytest.mark.parametrize(
    "power, capacity, initial_charge, freq",
    [
        (2, 4, 0, "60T"),
        (4, 2, 2, "60T"),
        (2, 4, 0, "30T"),
        (4, 2, 2, "30T"),
        (2, 4, 0, "5T"),
        (4, 2, 2, "5T"),
    ],
)
def test_cost_calculation(power, capacity, initial_charge, freq):
    prices = [10, 20, 30, -90, 50, 2000, -1]

    mdl = Battery(power=power, capacity=capacity, efficiency=1.0)
    results = mdl.optimize(prices=prices, initial_charge=initial_charge, freq=freq)
    dispatch = [res["Net [MW]"] for res in results if res["Net [MW]"] is not None]

    timestep = freq_to_timestep(freq)
    check_actual_costs = (
        sum([d * p for d, p in zip(dispatch[:-1], prices[:-1])]) / timestep
    )

    actual = f"Actual Cost [$/{freq}]"
    actual_costs = [res[actual] for res in results if res[actual] is not None]

    assert round(check_actual_costs, 6) == round(sum(actual_costs[:-1]), 6)
