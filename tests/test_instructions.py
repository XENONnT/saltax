import numpy as np
from saltax.instructions.generator import *


RUNID = 53169


def test_generate_vertex():
    # Test that the function returns the expected number of vertices
    x, y, z = generate_vertex(size=10)
    assert len(x) == 10
    assert len(y) == 10
    assert len(z) == 10

    # Test that the function generates vertices within the specified ranges
    x, y, z = generate_vertex(size=1000)
    assert np.all(x >= -R_RANGE[1])
    assert np.all(x <= R_RANGE[1])
    assert np.all(y >= -R_RANGE[1])
    assert np.all(y <= R_RANGE[1])
    assert np.all(z >= Z_RANGE[0])
    assert np.all(z <= Z_RANGE[1])

    # Test that the function generates random vertices
    x1, y1, z1 = generate_vertex(size=1000)
    x2, y2, z2 = generate_vertex(size=1000)
    assert not np.all(x1 == x2)
    assert not np.all(y1 == y2)
    assert not np.all(z1 == z2)

def test_generate_times():
    # Test that the function generates event times within the specified range
    start_time = 0
    end_time = 1e9
    times = generate_times(start_time, end_time, size=1000)
    assert times.min() >= start_time
    assert times.max() <= end_time

    # Test that the function generates event times with the expected rate
    rate = 1e6
    times = generate_times(start_time, end_time, size=1000, rate=rate)
    assert len(times) == 1000
    assert abs(len(times) / (end_time - start_time) - rate) < 0.1

    # Test that the function generates event times in the expected mode
    times = generate_times(start_time, end_time, size=1000, time_mode='realistic')
    assert len(times) == 1000
    assert (np.diff(times) >= 0).all()

    times = generate_times(start_time, end_time, size=1000, time_mode='uniform')
    assert len(times) == 1000
    assert (np.diff(times) >= 0).all()

def test_get_run_start_end():
    # Test with a non-integer runid
    with pytest.raises(AssertionError):
        get_run_start_end('abc')

    # Test with a runid that does not exist in RunDB
    with pytest.raises(RuntimeError):
        get_run_start_end(999999)

    # Test with runid RUNID
    runid = RUNID
    start_time, end_time = get_run_start_end(runid)

    # Check that the start and end times are integers
    assert isinstance(start_time, int)
    assert isinstance(end_time, int)

    # Check that the start time is before the end time
    assert start_time < end_time
