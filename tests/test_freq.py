import energypylinear as epl


def test_freq():

    freq = epl.freq.Freq(60)
    assert 100 == freq.mw_to_mwh(100)
    assert 100 == freq.mwh_to_mw(100)

    freq = epl.freq.Freq(30)
    assert 50 == freq.mw_to_mwh(100)
    assert 200 == freq.mwh_to_mw(100)

    freq = epl.freq.Freq(5)
    assert 200 / 12 == freq.mw_to_mwh(200)
    assert 200 * 12 == freq.mwh_to_mw(200)
