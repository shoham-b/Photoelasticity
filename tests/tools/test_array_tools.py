import numpy as np

from photoelasticity.tools.array_tools import moving_average


def test_moving_average():
    avg = moving_average(np.array([1, 2, 3, 4, 5]), 2)
    assert np.allclose(avg, [1.5, 2.5, 3.5, 4.5])
