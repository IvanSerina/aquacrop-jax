from datetime import date

from aquacrop_jax.climate import date_to_day_number, day_number_to_date


def test_aquacrop_day_number_epoch():
    assert date_to_day_number(date(2014, 5, 21)) == 41414
    assert day_number_to_date(41414) == date(2014, 5, 21)
