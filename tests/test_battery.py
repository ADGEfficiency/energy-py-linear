


def test_battery1():
    """ should charge in first two periods and discharge in last """

    prices = [10, 20, 30]
    initial_charge = 0
    power_rating = 1
    capacity = 2

    model = energypylinear.make(
        'battery', power=power_rating, capacity=capacity
    )

    summary = model.optimize(
        prices=prices, initial_charge=initial_charge
    )

    info = pd.DataFrame().from_dict(model.info)
    dispatch = info.loc[:, 'Power [MW]'].values

    np.testing.assert_array_equal(
        dispatch, np.array([power_rating, power_rating, -power_rating])
    )



"""

capacity = 1
dispatch = [power_rating, 0, -power_rating]


prices = 10, 10, 10
initial = 0
-> zero profit or cost

prices = 10, 10, 10
initial = 5
-> 5 * 10 profit

prices = [10, 50, 10, 50, 10]
charge, discharge, charge, discharge

"""

