import numpy as np
import nestpy
import wfsim
from packaging import version


def generate_vertex(r_range=(0, 66.4), z_range=(-148.15, 0), size=1):
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


def generate_times(rate, size, timemode):
    assert timemode in ["realistic", "uniform"], "timemode must be either 'realistic' or 'uniform'"
    # Generating event times from exponential
    if timemode == "realistic":
        dt = np.random.exponential(1 / rate, size=size - 1)
        times = np.append([1.0], 1.0 + dt.cumsum()) * 1e9
        times = times.round().astype(np.int64)
        return times
    # Generating event times from uniform
    elif timemode == "uniform":
        dt = (1 / rate) * np.ones(size - 1)
        times = np.append([1.0], 1.0 + dt.cumsum()) * 1e9
        times = times.round().astype(np.int64)
        return times
