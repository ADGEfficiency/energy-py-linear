import energypylinear as epl


def test_carbon_optimization():
    prices = [10, 50, 10, 50, 10]
    forecasts = [50, 10, 50, 10, 50]
    carbons = [0.1, 0.2, 0.3, 0.4, 0.5]

    model = epl.Battery(power=2, capacity=4, efficiency=1.0)

    price_opt = model.optimize(
        prices.copy(),
        forecasts=forecasts.copy(),
        carbon=carbons.copy(),
        freq="30T",
        objective="price",
    )
    price_cost = sum([d["Actual Cost [$/30T]"] for d in price_opt])

    forecast_opt = model.optimize(
        prices.copy(),
        forecasts=forecasts.copy(),
        carbon=carbons.copy(),
        freq="30T",
        objective="carbon",
    )
    forecast_cost = sum([d["Actual Cost [$/30T]"] for d in forecast_opt])

    carbon_opt = model.optimize(
        prices.copy(),
        forecasts=forecasts.copy(),
        carbon=carbons.copy(),
        freq="30T",
        objective="carbon",
    )
    carbon_cost = sum([d["Actual Cost [$/30T]"] for d in carbon_opt])

    assert price_cost < carbon_cost
    assert forecast_cost > price_cost
