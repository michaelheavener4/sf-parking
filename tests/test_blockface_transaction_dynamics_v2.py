import math

import numpy as np


def conditional_survival(durations, ages, extra):
    durations = np.asarray(durations, float)
    return np.asarray([
        float(np.sum(durations > age + extra)) / max(int(np.sum(durations > age)), 1)
        for age in ages
    ])


def test_conditional_survival_is_monotone_in_extra_time():
    d = np.array([0.5, 1.0, 2.0, 4.0])
    a = np.array([0.0, 0.5])
    s1 = conditional_survival(d, a, 0.5)
    s2 = conditional_survival(d, a, 1.0)
    assert np.all(s2 <= s1)


def test_zero_capacity_is_not_divided():
    expected_active = 2.0
    capacity = 0
    availability = 1.0 - min(1.0, max(0.0, expected_active / max(capacity, 1)))
    assert math.isclose(availability, 0.0)
