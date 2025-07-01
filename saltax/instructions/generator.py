import os
import pytz
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

import nestpy
import straxen
from straxen import units
from utilix import xent_collection
from fuse.plugins.detector_physics.csv_input import ChunkCsvInput

from saltax.plugins.csv_input import SALT_TIME_INTERVAL

DEFAULT_EN_RANGE = (0.2, 15.0)  # in unit of keV
Z_RANGE = (-148.15, 0)  # in unit of cm
R_RANGE = (0, 66.4)  # in unit of cm
DOWNLOADER = straxen.MongoDownloader()
NC = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())
FIELD_FILE = "fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz"
FIELD_MAP = straxen.InterpolatingMap(
    straxen.get_resource(DOWNLOADER.download_single(FIELD_FILE), fmt="json.gz"),
    method="RegularGridInterpolator",
)
SE_INSTRUCTIONS_DIR = "/project/lgrandi/yuanlq/salt/se_instructions"
AMBE_INSTRUCTIONS_FILE = "/project/lgrandi/yuanlq/salt/ambe_instructions/minghao_aptinput.csv"
YBE_INSTRUCTIONS_FILE = (
    "/project2/lgrandi/ghusheng/ybe_instrutions/"
    "ybe_wfsim_instructions_6806_events_time_modified.csv"
)
BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "generated"
)

coll = xent_collection()


def generate_vertex(r_range=R_RANGE, z_range=Z_RANGE, size=1):
    """Generate a random vertex in the TPC volume.

    :param r_range: (r_min, r_max) in cm
    :param z_range: (z_min, z_max) in cm
    :param size: number of vertices to generate
    :return: x, y, z coordinates of the vertex

    """
    phi = np.random.uniform(size=size) * 2 * np.pi
    r = r_range[1] * np.sqrt(np.random.uniform((r_range[0] / r_range[1]) ** 2, 1, size=size))

    z = np.random.uniform(z_range[0], z_range[1], size=size)
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
    start_time, end_time, size=None, rate=units.s / SALT_TIME_INTERVAL, time_mode="uniform"
):
    """Generate an array of event times in the given time range.

    :param start_time: start time in ns
    :param end_time: end time in ns
    :param size: rough number of events to generate, default: None i.e. generate events until
        end_time
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
        dt = np.random.exponential(1 / rate, size=estimated_size - 1)
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


def get_run_start_end(runid):
    """Get the start and end time of a run in unix time in ns, from RunDB.

    :param runid: run number
    :return: start time, end time in unix time in ns

    """
    # Get the datetime of start and end time of the run from RunDB
    doc = coll.find_one({"number": int(runid)})
    if doc is None:
        raise RuntimeError(f"Cannot find runid {runid} in RunDB")
    dt_start = doc["start"].replace(tzinfo=pytz.UTC)
    dt_end = doc["end"].replace(tzinfo=pytz.UTC)

    # Transform the datetime to unix time in ns
    unix_time_start_ns = int(dt_start.timestamp() * units.s)
    unix_time_end_ns = int(dt_end.timestamp() * units.s)

    return unix_time_start_ns, unix_time_end_ns


def instr_file_name(
    recoil,
    generator_name,
    mode,
    runid=None,
    en_range=DEFAULT_EN_RANGE,
    rate=units.s / SALT_TIME_INTERVAL,
    base_dir=BASE_DIR,
    **kwargs,
):
    """Generate the instruction file name based on the runid, recoil, generator_name, mode, and
    rate.

    :param recoil: NEST recoil type
    :param generator_name: name of the generator
    :param mode: 's1', 's2', or 'all'
    :param runid: run number, default: None, which means we
        are loading data and instruction doesn't matter (strax lineage
        unaffected)
    :param en_range: (en_min, en_max) in keV, default: DEFAULT_EN_RANGE as a placeholder
    :param rate: rate of events in Hz
    :param base_dir: base directory to save the instruction file,
        default: BASE_DIR
    :return: instruction file name

    """
    if en_range is not None:
        en_range = str(en_range[0]) + "_" + str(en_range[1])
    else:
        raise RuntimeError("en_range must be specified, and it can even be placeholder (0,0)")
    # FIXME: this will shoot errors if we are on OSG rather than midway
    if runid is None:
        return "Data-loading only, no instruction file needed."
    else:
        rate = int(rate)
        runid = str(runid).zfill(6)
        filename = os.path.join(
            base_dir,
            "-".join([runid, str(recoil), generator_name, en_range, mode, str(rate)]) + ".csv",
        )

        return filename


def generator_se(
    runid,
    n_tot=None,
    rate=units.s / SALT_TIME_INTERVAL,
    r_range=R_RANGE,
    z_range=Z_RANGE,
    time_mode="uniform",
    **kwargs,
):
    """Generate instructions for a run with single electron.

    :param runid: run number
    :param n_tot: total number of events to generate, default: None i.e. generate events until
        end_time
    :param rate: rate of events in Hz, default: units.s / SALT_TIME_INTERVAL
    :param r_range: (r_min, r_max) in cm, default: R_RANGE, defined above
    :param z_range: (z_min, z_max) in cm, default: Z_RANGE, defined above
    :param time_mode: 'uniform' or 'realistic', default: 'uniform'
    :return: instructions in numpy array

    """
    start_time, end_time = get_run_start_end(runid)
    times = generate_times(start_time, end_time, size=n_tot, rate=rate, time_mode=time_mode)
    n_tot = len(times)

    instr = np.zeros(n_tot, dtype=ChunkCsvInput.needed_csv_input_fields())
    instr["eventid"] = instr["cluster_id"] = np.arange(1, n_tot + 1)
    instr["t"] = times

    # Generating unoformely distributed events for give R and Z range
    instr["x"], instr["y"], instr["z"] = generate_vertex(
        r_range=r_range, z_range=z_range, size=n_tot
    )

    # And assigning quanta
    instr["photons"] = 0
    instr["electrons"] = 1
    instr["excitons"] = 0

    return instr


def generator_se_bootstrapped(runid, xyt_files_at=SE_INSTRUCTIONS_DIR, **kwargs):
    """Generate instructions for a run with single electron.

    We will use XYT information from bootstrapped data single electrons to make the simulation more
    realistic
    :param runid: run number
    :param xyt_files_at: directory to search for instructions of x,y,t information

    """
    # load instructions
    runid = str(runid).zfill(6)
    with open(os.path.join(xyt_files_at, "se_xs_dict.pkl"), "rb") as f:
        xs = pickle.load(f)[runid]
    with open(os.path.join(xyt_files_at, "se_ys_dict.pkl"), "rb") as f:
        ys = pickle.load(f)[runid]
    with open(os.path.join(xyt_files_at, "se_ts_dict.pkl"), "rb") as f:
        ts = pickle.load(f)[runid]

    # stay in runtime range
    start_time, end_time = get_run_start_end(runid)
    mask_in_run = ts < (end_time - 1 / 20 * units.s)  # empirical patch to stay in run
    mask_in_run &= ts > (start_time + 1 / 20 * units.s)  # empirical patch to stay in run
    xs = xs[mask_in_run]
    ys = ys[mask_in_run]
    ts = ts[mask_in_run]

    # clean up nan
    mask_is_nan = np.isnan(xs) + np.isnan(ys) + np.isnan(ts)
    xs = xs[~mask_is_nan]
    ys = ys[~mask_is_nan]
    ts = ts[~mask_is_nan]

    # stay inside TPC radius
    xs, ys = constrain_radius(xs, ys)

    n_tot = len(ts)
    instr = np.zeros(n_tot, dtype=ChunkCsvInput.needed_csv_input_fields())
    instr["eventid"] = instr["cluster_id"] = np.arange(1, n_tot + 1)
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
    runid,
    n_tot=None,
    rate=units.s / SALT_TIME_INTERVAL,
    time_mode="uniform",
    neutron_instructions_file=None,
    fmap=FIELD_MAP,
    **kwargs,
):
    """Generate instructions for a run with AmBe source.

    AmBe instruction was first generated by full-chain simulation, and then passing the post-epix
    instruction to feed this function. Each event with a certain event_id in the fed instructions
    will be shifted in time based on the time_mode you specified.
    :param runid: run number
    :param n_tot: total number of events to generate, default: None i.e. generate events until
        end_time
    :param rate: rate of events in Hz, default: units.s / SALT_TIME_INTERVAL
    :param time_mode: 'uniform' or 'realistic', default: 'uniform'
    :param neutron_instructions_file: file containing neutron instructions, default: None
    :param fmap: field map, default: FIELD_MAP, defined above
    :return: instructions in numpy array

    """
    # determine time offsets to shift neutron instructions
    start_time, end_time = get_run_start_end(runid)
    times_offset = generate_times(start_time, end_time, size=n_tot, rate=rate, time_mode=time_mode)
    n_tot = len(times_offset)

    # bootstrap instructions
    neutron_instructions = pd.read_csv(neutron_instructions_file)
    neutron_event_numbers = np.random.choice(
        np.unique(neutron_instructions.event_number), n_tot, replace=True
    )

    # assign instructions
    instr = np.zeros(0, dtype=ChunkCsvInput.needed_csv_input_fields())
    for i in tqdm(range(n_tot)):
        # bootstrapped neutron instruction
        selected_neutron = neutron_instructions[
            neutron_instructions["event_number"] == neutron_event_numbers[i]
        ]
        # instruction for i-th event
        instr_i = np.zeros(len(selected_neutron), dtype=ChunkCsvInput.needed_csv_input_fields())
        instr_i["t"] = times_offset[i] + selected_neutron["time"]
        instr_i["eventid"] = i + 1
        instr_i["cluster_id"] = len(instr) + np.arange(len(selected_neutron))
        instr_i["x"] = selected_neutron["x"]
        instr_i["y"] = selected_neutron["y"]
        instr_i["z"] = selected_neutron["z"]
        instr_i["nestid"] = selected_neutron["recoil"]
        instr_i["ed"] = selected_neutron["e_dep"]
        instr_i["photons"][selected_neutron["type"] == 1] = selected_neutron["amp"][
            selected_neutron["type"] == 1
        ]
        instr_i["electrons"][selected_neutron["type"] == 2] = selected_neutron["amp"][
            selected_neutron["type"] == 2
        ]
        instr_i["excitons"] = selected_neutron["n_excitons"]
        instr_i["e_field"] = fmap(
            np.array(
                [
                    np.sqrt(selected_neutron["x"] ** 2 + selected_neutron["y"] ** 2),
                    selected_neutron["z"],
                ]
            ).T
        )

        # concatenate instr
        instr = np.concatenate((instr, instr_i))

    return instr


def generator_ambe(
    runid,
    n_tot=None,
    rate=units.s / SALT_TIME_INTERVAL,
    time_mode="uniform",
    ambe_instructions_file=AMBE_INSTRUCTIONS_FILE,
    fmap=FIELD_MAP,
    **kwargs,
):
    """Generate instructions for a run with AmBe source.

    AmBe instruction was first generated by full-chain simulation, and
    then passing the post-epix instruction to feed this function. Each
    event with a certain event_id in the fed instructions will be
    shifted in time based on the time_mode you specified.
    :param runid: run number
    :param n_tot: total number of events to generate, default: None i.e.
        generate events until end_time
    :param rate: rate of events in Hz, default: units.s / SALT_TIME_INTERVAL
    :param time_mode: 'uniform' or 'realistic', default: 'uniform'
    :param ambe_instructions_file: file containing ambe instructions,
        default: AMBE_INSTRUCTIONS_FILE
    :param fmap: field map, default: FIELD_MAP, defined above
    :return: instructions in numpy array

    """
    return generator_neutron(
        runid=runid,
        n_tot=n_tot,
        rate=rate,
        time_mode=time_mode,
        neutron_instructions_file=ambe_instructions_file,
        fmap=fmap,
        **kwargs,
    )


def generator_ybe(
    runid,
    n_tot=None,
    rate=units.s / SALT_TIME_INTERVAL,
    time_mode="uniform",
    ybe_instructions_file=YBE_INSTRUCTIONS_FILE,
    fmap=FIELD_MAP,
    **kwargs,
):
    """Generate instructions for a run with YBe source.

    YBe instruction was first generated by full-chain simulation, and
    then passing the post-epix instruction to feed this function. Each
    event with a certain event_id in the fed instructions will be
    shifted in time based on the time_mode you specified.
    :param runid: run number
    :param n_tot: total number of events to generate, default: None i.e.
        generate events until end_time
    :param rate: rate of events in Hz, default: units.s / SALT_TIME_INTERVAL
    :param time_mode: 'uniform' or 'realistic', default: 'uniform'
    :param ybe_instructions_file: file containing ybe instructions,
        default: YBE_INSTRUCTIONS_FILE
    :param fmap: field map, default: FIELD_MAP, defined above
    :return: instructions in numpy array

    """
    return generator_neutron(
        runid=runid,
        n_tot=n_tot,
        rate=rate,
        time_mode=time_mode,
        neutron_instructions_file=ybe_instructions_file,
        fmap=fmap,
        **kwargs,
    )


def generator_flat(
    runid,
    en_range=DEFAULT_EN_RANGE,
    recoil=8,
    n_tot=None,
    rate=units.s / SALT_TIME_INTERVAL,
    fmap=FIELD_MAP,
    nc=NC,
    r_range=R_RANGE,
    z_range=Z_RANGE,
    mode="all",
    time_mode="uniform",
    **kwargs,
):
    """Generate instructions for a run with flat energy spectrum.

    :param runid: run number
    :param en_range: (en_min, en_max) in keV, default: (0.2, 15.0)
    :param recoil: NEST recoil type, default: 8 (beta ER)
    :param n_tot: total number of events to generate, default: None i.e. generate events until
        end_time
    :param rate: rate of events in Hz, default: units.s / SALT_TIME_INTERVAL
    :param fmap: field map, default: FIELD_MAP, defined above
    :param nc: NEST calculator, default: NC, defined above
    :param r_range: (r_min, r_max) in cm, default: R_RANGE, defined above
    :param z_range: (z_min, z_max) in cm, default: Z_RANGE, defined above
    :param mode: 's1', 's2', or 'all', default: 'all'
    :param time_mode: 'uniform' or 'realistic', default: 'uniform'
    :return: instructions in numpy array

    """
    start_time, end_time = get_run_start_end(runid)
    times = generate_times(start_time, end_time, size=n_tot, rate=rate, time_mode=time_mode)
    n_tot = len(times)

    instr = np.zeros(n_tot, dtype=ChunkCsvInput.needed_csv_input_fields())
    instr["eventid"] = instr["cluster_id"] = np.arange(1, n_tot + 1)
    instr["t"] = times

    # Generating unoformely distributed events for give R and Z range
    instr["x"], instr["y"], instr["z"] = generate_vertex(
        r_range=r_range, z_range=z_range, size=n_tot
    )

    # Making energy
    instr["ed"] = np.random.uniform(en_range[0], en_range[1], size=n_tot)
    instr["nestid"] = recoil

    # Getting local field from field map
    instr["e_field"] = fmap(np.array([np.sqrt(instr["x"] ** 2 + instr["y"] ** 2), instr["z"]]).T)

    # And generating quantas from nest
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
