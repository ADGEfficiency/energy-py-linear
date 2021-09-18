import pytest

from energypylinear.battery import freq_to_timestep, timestep_to_freq


def test_timestep_to_freq():
    assert timestep_to_freq(1) == "60T"
    assert timestep_to_freq(2) == "30T"
    assert timestep_to_freq(4) == "15T"
    assert timestep_to_freq(12) == "5T"


def test_freq_to_timestep():
    assert freq_to_timestep("60T") == 1
    assert freq_to_timestep("30T") == 2
    assert freq_to_timestep("15T") == 4
    assert freq_to_timestep("5T") == 12

    with pytest.raises(Exception):
        freq_to_timestep("5min")

    with pytest.raises(Exception):
        freq_to_timestep("60")

    with pytest.raises(Exception):
        freq_to_timestep("1H")


def test_freq_to_timestep_integration():
    #  MWh = MW / step
    power = 30
    assert power / freq_to_timestep("60T") == power
    assert power / freq_to_timestep("30T") == power / 2
    assert power / freq_to_timestep("15T") == power / 4
    assert power / freq_to_timestep("5T") == power / 12

    power = 10
    assert power / freq_to_timestep("60T") == power
    assert power / freq_to_timestep("30T") == power / 2
    assert power / freq_to_timestep("15T") == power / 4
    assert power / freq_to_timestep("5T") == power / 12
