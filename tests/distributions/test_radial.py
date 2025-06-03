import pytest

# ------------------------------------------------------------------------


@pytest.mark.basic
def test_coordinateConversion():
    import numpy as np
    from distributions._radial import angular_coordinates
    from distributions._radial import cartesian_coordinates

    r = np.array([1, 2])
    arr = np.pi*np.array([[0.5, 1], [0.5, 1]])
    print(arr)

    print(angular_coordinates(np.array([2, 2])))

    x = cartesian_coordinates(r, arr)
    print(f"cart: {x}")

    ra, a = angular_coordinates(x)

    print(f"ang: {a}")

    assert r == pytest.approx(ra)
    assert arr == pytest.approx(a)