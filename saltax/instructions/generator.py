import numpy as np
import nestpy
import wfsim
from packaging import version


SALT_TIME_INTERVAL = 1e7 # in unit of ns. The number should be way bigger then full drift time
Z_RANGE = (-148.15, 0) # in unit of cm
R_RANGE = (0, 66.4) # in unit of cm


def generate_vertex(r_range=R_RANGE, z_range=Z_RANGE, size=1):
    """
    Generate a random vertex in the TPC volume.
    :param r_range: (r_min, r_max) in cm
    :param z_range: (z_min, z_max) in cm
    :param size: number of vertices to generate
    :return: x, y, z coordinates of the vertex
    """
    phi = np.random.uniform(size=size) * 2 * np.pi
    r = r_range[1] * np.sqrt(
        np.random.uniform((r_range[0] / r_range[1]) ** 2, 1, size=size)
    )

    z = np.random.uniform(z_range[0], z_range[1], size=size)
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y, z

def generate_times(start_time, end_time, size=None, rate=1e9/SALT_TIME_INTERVAL, time_mode='uniform'):
    """
    Generate an array of event times in the given time range.
    :param start_time: start time in ns
    :param end_time: end time in ns
    :param size: rough number of events to generate
    :param rate: rate of events in Hz
    :param time_mode: 'uniform' or 'realistic'
    :return: array of event times in ns
    """
    total_time_ns = end_time - start_time
    estimated_size = int(total_time_ns * rate / 1e9)

    assert time_mode in ["realistic", "uniform"], "timemode must be either 'realistic' or 'uniform'"

    # Generating event times from exponential
    # This one doesn't work for salting!!!
    if time_mode == "realistic":
        dt = np.random.exponential(1 / rate, size=estimated_size-1)
        times = np.append([1.0], 1.0 + dt.cumsum()) * 1e9
        times = times.round().astype(np.int64)
        times += start_time

    # Generating event times from uniform
    elif time_mode == "uniform":
        dt = (1 / rate) * np.ones(estimated_size-1)
        times = np.append([1.0], 1.0 + dt.cumsum()) * 1e9
        times = times.round().astype(np.int64)
        times += start_time
    
    # Removing events that are too close to the start or end of the run
    times = times[times < end_time-1/rate]
    times = times[times > start_time+1/rate]

    if size is None:
        return times
    elif size >= 0:
        return times[:min(int(size), len(times))]
    
