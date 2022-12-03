"""
Electric vehicle models

Both smart and on-demand charging

- On demand charging is modelled by reducing the charge event length,
- Smart charging / queuing is modelled by having a large charge event length.

In future do V2G

- V2G / bi-directional charging modelled by allowing charge_mwh to be positive and negative
- bi-directional charging would require tracking the capacity of the car, which would require an additional input of `Car` with a capacity

# Input Data

chargers

- power_mw

charge_event_mwh

- list of floats
- n charge events
- mwh of charge required for that charge event

charge_events

- 2D array of binary integers
- n charge events * m intervals
- 0 = charger disconnected, 1 = charger connected

cars

- for bi-directional only
- capacity_mwh
- leave to future

# Variables

charge_mwh

- actual charging in each interval
- 2D array of floats, same shape as charge_event_binary
- n charge events * n_chargers * m intervals
- should I try to make this 3D?

charge_binary

- same shape as charge_mwh
- n charge events * n_chargers * m intervals
- linked to the charge_mwh via a min and/or max constraint

Example - 2 chargers, 3 charge events, 4 intervals

charger_0, charge_event_0, [0, 0, 0, 0]
charger_0, charge_event_1, [0, 0, 0, 0]
charger_0, charge_event_2, [0, 0, 0, 0]
charger_1, charge_event_0, [0, 0, 0, 0]
charger_1, charge_event_1, [0, 0, 0, 0]
charger_1, charge_event_2, [0, 0, 0, 0]

# Constraints

constrain each charge event to get all the charge it needs
- energy balance constraint
- charge_mwh.sum(time) == charge_event_mwh

one charger can only be attached to one charge event at a time

one charge event can only have one attached charger at a time

only one charger can be used for one charge event across all times
- this prevents chargers switching during charge events

# Objective

minimum energy cost
- energy cost = charge_mwh * electricity_price

incentivize early charging
- want to reduce the cost to charge earlier on
- could just reduce prices earlier in the intervals???
- perhaps quantify as `value_of_time` ?
- interval_length * value_per_interval
- if charged 30 min earlier -> some benefit

could perhaps just add something like the below to each interval:
- [5, 10, 15, 20, 25, 30]
- and multiply it by the binary variable?
- 5 = `value_of_interval`

could maybe test by reversing the value_of_interval
"""
