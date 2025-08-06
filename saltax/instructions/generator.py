import os
import logging
import pytz
import tarfile
import tempfile
import numpy as np
import pandas as pd

import nestpy
import utilix
import straxen
from straxen import units
from fuse.plugins.detector_physics.csv_input import ChunkCsvInput

from saltax.utils import COLL
from saltax.plugins.csv_input import SALT_TIME_INTERVAL

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
log = logging.getLogger("saltax.instructions.generator")

DEFAULT_EN_RANGE = (0.2, 15.0)  # in unit of keV
Z_RANGE = (-straxen.tpc_z, 0)  # in unit of cm
R_RANGE = (0, straxen.tpc_r)  # in unit of cm
NC = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())
SE_INSTRUCTIONS_FILE = "se_instructions.csv.gz"
AMBE_INSTRUCTIONS_FILE = "minghao_aptinput.csv.gz"
YBE_INSTRUCTIONS_FILE = "ybe_wfsim_instructions_6806_events_time_modified.csv"
NEST_RNG = nestpy.RandomGen.rndm()


def load_csv_gz(instructions_file):
    """Load a CSV file from utilix storage, which can be gzipped or not.

    :param instructions_file: name of the file to load
    :return: instructions in numpy record array

    """
    downloader = utilix.mongo_storage.MongoDownloader()
    path = downloader.download_single(instructions_file)
    if instructions_file.endswith(".csv.gz"):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tar = tarfile.open(path, mode="r:gz")
            tar.extractall(path=tmpdirname)
            instructions = pd.read_csv(
                os.path.join(tmpdirname, instructions_file.replace("csv.gz", "csv"))
            ).to_records(index=False)
    else:
        instructions = pd.read_csv(path).to_records(index=False)
    return instructions


def generate_vertex(rng, r_range=R_RANGE, z_range=Z_RANGE, size=1):
    """Generate a random vertex in the TPC volume.

    :param rng: random number generator
    :param r_range: (r_min, r_max) in cm
    :param z_range: (z_min, z_max) in cm
    :param size: number of vertices to generate
    :return: x, y, z coordinates of the vertex

    """
    phi = rng.uniform(size=size) * 2 * np.pi
    r = r_range[1] * np.sqrt(rng.uniform((r_range[0] / r_range[1]) ** 2, 1, size=size))

    z = rng.uniform(z_range[0], z_range[1], size=size)
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y, z


def constrain_radius(xs, ys, r_max=R_RANGE[-1] - 0.001):
    """Push out of TPC radius instructions back into radius."""
    rs = np.sqrt(xs**2 + ys**2)
    xs_new = np.array(xs)
    ys_new = np.array(ys)
    xs_new[rs > r_max] = xs[rs > r_max] * r_max / rs[rs > r_max]
    ys_new[rs > r_max] = ys[rs > r_max] * r_max / rs[rs > r_max]

    return xs_new, ys_new


def generate_times(
    start_time, end_time, rng, size=None, rate=units.s / SALT_TIME_INTERVAL, time_mode="uniform"
):
    """Generate an array of event times in the given time range.

    :param start_time: start time in ns
    :param end_time: end time in ns
    :param rng: random number generator
    :param size: rough number of events to generate (default: None)
    :param rate: rate of events in Hz
    :param time_mode: 'uniform' or 'realistic'
    :return: array of event times in ns

    """
    total_time_ns = end_time - start_time
    estimated_size = int(total_time_ns * rate / units.s)

    assert time_mode in [
        "realistic",
        "uniform",
    ], "time_mode must be either \
        'realistic' or 'uniform'"

    # Generating event times from exponential
    # This one doesn't work for salting!!!
    if time_mode == "realistic":
        dt = rng.exponential(1 / rate, size=estimated_size - 1)
        times = np.append([0], dt.cumsum()) * units.s
        times = times.round().astype(np.int64)
        times += start_time

    # Generating event times from uniform
    elif time_mode == "uniform":
        dt = (1 / rate) * np.ones(estimated_size - 1)
        times = np.append([0], dt.cumsum()) * units.s
        times = times.round().astype(np.int64)
        times += start_time

    # Removing events that are too close to the start or end of the run
    times = times[times < (end_time - 1 / rate * units.s)]
    times = times[times > (start_time + 1 / rate * units.s)]

    if size is None:
        return times
    elif size >= 0:
        return times[: min(int(size), len(times))]


def get_run_start_end(run_id):
    """Get the start and end time of a run in unix time in ns, from RunDB.

    :param run_id: run number
    :return: start time, end time in unix time in ns

    """
    # Get the datetime of start and end time of the run from RunDB
    doc = COLL.find_one({"number": int(run_id)})
    if doc is None:
        raise RuntimeError(f"Cannot find run_id {run_id} in RunDB")
    dt_start = doc["start"].replace(tzinfo=pytz.UTC)
    dt_end = doc["end"].replace(tzinfo=pytz.UTC)

    # Transform the datetime to unix time in ns
    unix_time_start_ns = int(dt_start.timestamp() * units.s)
    unix_time_end_ns = int(dt_end.timestamp() * units.s)

    return unix_time_start_ns, unix_time_end_ns


def instr_file_name(
    run_id=None,
    recoil=8,
    generator_name="flat",
    mode="all",
    en_range=DEFAULT_EN_RANGE,
    rate=units.s / SALT_TIME_INTERVAL,
    output_folder=None,
    chunk_number=None,
):
    """Generate the instruction file name based on the run_id, recoil, generator_name, mode, and
    rate.

    :param generator_name: name of the generator (default: 'flat')
    :param recoil: NEST recoil type (default: 8)
    :param mode: 's1', 's2', or 'all' (default: 'all')
    :param run_id: run number (default: None)
    :param en_range: (en_min, en_max) in keV (default: DEFAULT_EN_RANGE)
    :param rate: rate of events in Hz
    :param output_folder: output directory to save the instruction file (default: None)
    :return: instruction file name

    """
    if en_range is None:
        raise RuntimeError("en_range must be specified, and it can even be placeholder (0, 0)")

    if run_id is None:
        raise RuntimeError(
            "run_id must be specified to generate instruction file name. "
            "It is usually the same as the run number of the strax context."
        )

    if output_folder is None:
        raise RuntimeError(
            "output_folder must be specified to generate instruction file name. "
            "It is usually the same as the output folder of the strax context."
        )

    run_id = str(run_id).zfill(6)
    en_range = str(en_range[0]) + "_" + str(en_range[1])
    rate = int(rate)
    filename = os.path.join(
        output_folder,
        "-".join([run_id, str(recoil), generator_name, en_range, mode, str(rate)]) + ".csv",
    )

    if chunk_number is not None:
        filename = filename.replace(".csv", f"_{chunk_number[0]}_{chunk_number[-1] + 1}.csv")

    return filename


def generator_se(
    run_id,
    n_tot=None,
    rate=units.s / SALT_TIME_INTERVAL,
    r_range=R_RANGE,
    z_range=Z_RANGE,
    time_mode="uniform",
):
    """Generate instructions for a run with single electron.

    :param run_id: run number
    :param n_tot: total number of events to generate (default: None)
    :param rate: rate of events in Hz (default: units.s / SALT_TIME_INTERVAL)
    :param r_range: (r_min, r_max) in cm (default: R_RANGE)
    :param z_range: (z_min, z_max) in cm (default: Z_RANGE)
    :param time_mode: 'uniform' or 'realistic' (default: 'uniform')
    :return: instructions in numpy array

    """
    rng = np.random.default_rng(seed=int(run_id))
    start_time, end_time = get_run_start_end(run_id)
    times = generate_times(
        start_time, end_time, rng=rng, size=n_tot, rate=rate, time_mode=time_mode
    )
    n_tot = len(times)

    instr = np.zeros(n_tot, dtype=ChunkCsvInput.needed_csv_input_fields())
    instr["eventid"] = instr["cluster_id"] = np.arange(n_tot)
    instr["t"] = times

    # Generating unoformely distributed events for give R and Z range
    instr["x"], instr["y"], instr["z"] = generate_vertex(
        rng=rng, r_range=r_range, z_range=z_range, size=n_tot
    )

    # And assigning quanta
    instr["photons"] = 0
    instr["electrons"] = 1
    instr["excitons"] = 0

    return instr


def generator_se_bootstrapped(
    run_id,
    se_instructions_file=SE_INSTRUCTIONS_FILE,
):
    """Generate instructions for a run with single electron.

    We will use XYT information from bootstrapped data single electrons to make the simulation more
    realistic
    :param run_id: run number
    :param se_instructions_file: file containing se instructions (default: SE_INSTRUCTIONS_FILE)
    :param xyt_files_at: directory to search for instructions of x, y, t information
    :return: instructions in numpy array

    """
    # load instructions
    run_id = str(run_id).zfill(6)
    se_instructions = load_csv_gz(se_instructions_file)
    se_instructions = se_instructions[se_instructions["run_id"] == int(run_id)]

    # stay in runtime range
    start_time, end_time = get_run_start_end(run_id)
    # empirical patch to stay in run
    mask_in_run = se_instructions["t"] < (end_time - 1 / 20 * units.s)
    mask_in_run &= se_instructions["t"] > (start_time + 1 / 20 * units.s)
    xs = se_instructions["x"][mask_in_run]
    ys = se_instructions["y"][mask_in_run]
    ts = se_instructions["t"][mask_in_run]

    # clean up nan
    mask_is_nan = np.isnan(xs) + np.isnan(ys) + np.isnan(ts)
    xs = xs[~mask_is_nan]
    ys = ys[~mask_is_nan]
    ts = ts[~mask_is_nan]

    # stay inside TPC radius
    xs, ys = constrain_radius(xs, ys)

    n_tot = len(ts)
    instr = np.zeros(n_tot, dtype=ChunkCsvInput.needed_csv_input_fields())
    instr["eventid"] = instr["cluster_id"] = np.arange(n_tot)
    instr["t"] = ts
    instr["x"] = xs
    instr["y"] = ys
    instr["z"] = -0.00001  # Just to avoid drift time

    # And assigning quanta
    instr["photons"] = 0
    instr["electrons"] = 1
    instr["excitons"] = 0

    return instr


def generator_neutron(
    run_id,
    efield_map,
    recoil=0,
    n_tot=None,
    rate=units.s / SALT_TIME_INTERVAL,
    time_mode="uniform",
    neutron_instructions_file=None,
):
    """Generate instructions for a run with AmBe source.

    AmBe instruction was first generated by full-chain simulation, and then passing the post-epix
    instruction to feed this function. Each event with a certain event_id in the fed instructions
    will be shifted in time based on the time_mode you specified.
    :param run_id: run number
    :param n_tot: total number of events to generate (default: None)
    :param rate: rate of events in Hz (default: units.s / SALT_TIME_INTERVAL)
    :param time_mode: 'uniform' or 'realistic' (default: 'uniform')
    :param neutron_instructions_file: file containing neutron instructions (default: None)
    :return: instructions in numpy array

    """
    # determine time offsets to shift neutron instructions
    rng = np.random.default_rng(seed=int(run_id))
    start_time, end_time = get_run_start_end(run_id)
    times_offset = generate_times(
        start_time, end_time, rng=rng, size=n_tot, rate=rate, time_mode=time_mode
    )
    n_tot = len(times_offset)

    neutron_instructions = load_csv_gz(neutron_instructions_file)

    # check recoil
    unique_recoil = np.unique(neutron_instructions["recoil"])
    if not np.all(unique_recoil == recoil):
        log.warning(
            f"Recoil in neutron instructions ({unique_recoil}) "
            f"does not match the requested recoil ({recoil})."
        )

    # bootstrap instructions
    if not np.all(np.diff(neutron_instructions["event_number"]) >= 0):
        raise RuntimeError(
            "Neutron instructions must be sorted by event_number in ascending order."
        )
    event_numbers, event_indices, event_counts = np.unique(
        neutron_instructions["event_number"],
        return_index=True,
        return_counts=True,
    )
    event_indices = np.append(event_indices, len(neutron_instructions))

    _indices = rng.choice(len(event_numbers), size=n_tot, replace=True)
    indices = np.hstack([np.arange(event_indices[i], event_indices[i + 1]) for i in _indices])

    # assign instructions
    instr = np.zeros(len(indices), dtype=ChunkCsvInput.needed_csv_input_fields())
    instr["eventid"] = np.repeat(np.arange(n_tot), event_counts[_indices])
    instr["cluster_id"] = np.arange(len(instr))

    instr["t"] = np.repeat(times_offset, event_counts[_indices])
    instr["t"] += neutron_instructions["time"][indices]

    instr["x"] = neutron_instructions["x"][indices]
    instr["y"] = neutron_instructions["y"][indices]
    instr["z"] = neutron_instructions["z"][indices]

    instr["nestid"] = neutron_instructions["recoil"][indices]
    instr["ed"] = neutron_instructions["e_dep"][indices]

    instr["photons"] = np.where(
        neutron_instructions["type"][indices] == 1,
        neutron_instructions["amp"][indices],
        0,
    )
    instr["electrons"] = np.where(
        neutron_instructions["type"][indices] == 2,
        neutron_instructions["amp"][indices],
        0,
    )
    instr["excitons"] = neutron_instructions["n_excitons"][indices]

    instr["e_field"] = efield_map(
        np.array(
            [
                np.sqrt(instr["x"] ** 2 + instr["y"] ** 2),
                instr["z"],
            ]
        ).T
    )

    ind = np.cumsum(event_counts[_indices])[:-1]
    if np.any(instr["t"][ind] - instr["t"][ind - 1] < 0):
        raise RuntimeError("Neutron instructions overlap with the next event.")

    return instr


def generator_ambe(
    run_id,
    efield_map,
    recoil=0,
    n_tot=None,
    rate=units.s / SALT_TIME_INTERVAL,
    time_mode="uniform",
    ambe_instructions_file=AMBE_INSTRUCTIONS_FILE,
):
    """Generate instructions for a run with AmBe source.

    AmBe instruction was first generated by full-chain simulation, and then passing the post-epix
    instruction to feed this function. Each event with a certain event_id in the fed instructions
    will be shifted in time based on the time_mode you specified.
    :param run_id: run number
    :param n_tot: total number of events to generate (default: None)
    :param rate: rate of events in Hz (default: units.s / SALT_TIME_INTERVAL)
    :param time_mode: 'uniform' or 'realistic' (default: 'uniform')
    :param ambe_instructions_file: file containing ambe instructions (default:
        AMBE_INSTRUCTIONS_FILE)
    :return: instructions in numpy array

    """
    return generator_neutron(
        run_id=run_id,
        efield_map=efield_map,
        n_tot=n_tot,
        rate=rate,
        time_mode=time_mode,
        neutron_instructions_file=ambe_instructions_file,
    )


def generator_ybe(
    run_id,
    efield_map,
    recoil=0,
    n_tot=None,
    rate=units.s / SALT_TIME_INTERVAL,
    time_mode="uniform",
    ybe_instructions_file=YBE_INSTRUCTIONS_FILE,
):
    """Generate instructions for a run with YBe source.

    YBe instruction was first generated by full-chain simulation, and then passing the post-epix
    instruction to feed this function. Each event with a certain event_id in the fed instructions
    will be shifted in time based on the time_mode you specified.
    :param run_id: run number
    :param n_tot: total number of events to generate (default: None)
    :param rate: rate of events in Hz (default: units.s / SALT_TIME_INTERVAL)
    :param time_mode: 'uniform' or 'realistic' (default: 'uniform')
    :param ybe_instructions_file: file containing ybe instructions (default: YBE_INSTRUCTIONS_FILE)
    :return: instructions in numpy array

    """
    return generator_neutron(
        run_id=run_id,
        efield_map=efield_map,
        recoil=recoil,
        n_tot=n_tot,
        rate=rate,
        time_mode=time_mode,
        neutron_instructions_file=ybe_instructions_file,
    )


def generator_flat(
    run_id,
    efield_map,
    en_range=DEFAULT_EN_RANGE,
    recoil=8,
    n_tot=None,
    rate=units.s / SALT_TIME_INTERVAL,
    nc=NC,
    r_range=R_RANGE,
    z_range=Z_RANGE,
    mode="all",
    time_mode="uniform",
):
    """Generate instructions for a run with flat energy spectrum.

    :param run_id: run number
    :param en_range: (en_min, en_max) in keV (default: (0.2, 15.0))
    :param recoil: NEST recoil type (default: 8)
    :param n_tot: total number of events to generate (default: None)
    :param rate: rate of events in Hz (default: units.s / SALT_TIME_INTERVAL)
    :param nc: NEST calculator (default: NC)
    :param r_range: (r_min, r_max) in cm (default: R_RANGE)
    :param z_range: (z_min, z_max) in cm (default: Z_RANGE)
    :param mode: 's1', 's2', or 'all' (default: 'all')
    :param time_mode: 'uniform' or 'realistic' (default: 'uniform')
    :return: instructions in numpy array

    """
    rng = np.random.default_rng(seed=int(run_id))
    start_time, end_time = get_run_start_end(run_id)
    times = generate_times(
        start_time, end_time, rng=rng, size=n_tot, rate=rate, time_mode=time_mode
    )
    n_tot = len(times)

    instr = np.zeros(n_tot, dtype=ChunkCsvInput.needed_csv_input_fields())
    instr["eventid"] = instr["cluster_id"] = np.arange(n_tot)
    instr["t"] = times

    # Generating unoformely distributed events for give R and Z range
    instr["x"], instr["y"], instr["z"] = generate_vertex(
        rng=rng, r_range=r_range, z_range=z_range, size=n_tot
    )

    # Making energy
    instr["ed"] = rng.uniform(en_range[0], en_range[1], size=n_tot)
    instr["nestid"] = recoil

    # Getting local field from field map
    instr["e_field"] = efield_map(
        np.array([np.sqrt(instr["x"] ** 2 + instr["y"] ** 2), instr["z"]]).T
    )

    # And generating quantas from nest
    NEST_RNG.set_seed(int(run_id))
    NEST_RNG.lock_seed()
    for i in range(n_tot):
        y = nc.GetYields(
            interaction=nestpy.INTERACTION_TYPE(instr["nestid"][i]),
            energy=instr["ed"][i],
            drift_field=instr["e_field"][i],
        )
        quantas = nc.GetQuanta(y)
        instr["photons"][i] = quantas.photons
        instr["electrons"][i] = quantas.electrons
        instr["excitons"][i] = quantas.excitons
    NEST_RNG.unlock_seed()

    # Selecting event types
    if mode == "s1":
        instr["electrons"] = 0
    elif mode == "s2":
        instr["photons"] = 0
    elif mode == "all":
        pass
    else:
        raise RuntimeError("Unknown mode: ", mode)

    return instr
