import numpy as np
import pandas as pd
import nestpy
import wfsim
import pytz
import straxen
from zoneinfo import ZoneInfo
from utilix import xent_collection
import datetime
import os


SALT_TIME_INTERVAL = 5e7 # in unit of ns. The number should be way bigger then full drift time
Z_RANGE = (-148.15, 0) # in unit of cm
R_RANGE = (0, 66.4) # in unit of cm
DOWNLOADER = straxen.MongoDownloader()
NC = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())
FIELD_FILE = "fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz"
FIELD_MAP = straxen.InterpolatingMap(
    straxen.get_resource(DOWNLOADER.download_single(FIELD_FILE), fmt="json.gz"),
    method="RegularGridInterpolator",
)
#BASE_DIR = "/project2/lgrandi/yuanlq/shared/saltax_instr/"
BASE_DIR = os.path.abspath(__file__)[:-12] + '../../generated/'


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

def generate_times(start_time, end_time, size=None, 
                   rate=1e9/SALT_TIME_INTERVAL, time_mode='uniform'):
    """
    Generate an array of event times in the given time range.
    :param start_time: start time in ns
    :param end_time: end time in ns
    :param size: rough number of events to generate, default: None i.e. generate events until end_time
    :param rate: rate of events in Hz
    :param time_mode: 'uniform' or 'realistic'
    :return: array of event times in ns
    """
    total_time_ns = end_time - start_time
    estimated_size = int(total_time_ns * rate / 1e9)

    assert time_mode in ["realistic", "uniform"], "time_mode must be either \
        'realistic' or 'uniform'"

    # Generating event times from exponential
    # This one doesn't work for salting!!!
    if time_mode == "realistic":
        dt = np.random.exponential(1 / rate, size=estimated_size-1)
        times = np.append([0], dt.cumsum()) * 1e9
        times = times.round().astype(np.int64)
        times += start_time

    # Generating event times from uniform
    elif time_mode == "uniform":
        dt = (1 / rate) * np.ones(estimated_size-1)
        times = np.append([0], dt.cumsum()) * 1e9
        times = times.round().astype(np.int64)
        times += start_time
    
    # Removing events that are too close to the start or end of the run
    times = times[times < (end_time-1/rate*1e9)]
    times = times[times > (start_time+1/rate*1e9)]

    if size is None:
        return times
    elif size >= 0:
        return times[:min(int(size), len(times))]
    
def get_run_start_end(runid):
    """
    Get the start and end time of a run in unix time in ns, from RunDB.
    :param runid: run number in integer
    :return: start time, end time in unix time in ns
    """
    # Get the datetime of start and end time of the run from RunDB
    assert type(runid)==int, "runid must be an integer"
    try:
        doc = xent_collection().find_one({'number':runid})
    except:
        raise RuntimeError("Cannot find runid %d in RunDB"%(runid))
    if doc is None:
        raise RuntimeError("Cannot find runid %d in RunDB"%(runid))
    dt_start, dt_end = doc['start'], doc['end']

    # Get timezones
    chicago_tz = ZoneInfo('America/Chicago')
    utc_tz = pytz.utc

    # Transform the datetime to Chicago time
    dt_start_year, dt_end_year = dt_start.year, dt_end.year
    dt_start_month, dt_end_month = dt_start.month, dt_end.month
    dt_start_day, dt_end_day = dt_start.day, dt_end.day
    dt_start_hour, dt_end_hour = dt_start.hour, dt_end.hour
    dt_start_minute, dt_end_minute = dt_start.minute, dt_end.minute
    dt_start_second, dt_end_second = dt_start.second, dt_end.second
    dt_start_ms, dt_end_ms = dt_start.microsecond, dt_end.microsecond
    dt_start_transformed = datetime.datetime(dt_start_year, dt_start_month, 
                                             dt_start_day, dt_start_hour, 
                                             dt_start_minute, dt_start_second, dt_start_ms,
                                             tzinfo=utc_tz).astimezone(chicago_tz)
    dt_end_transformed = datetime.datetime(dt_end_year, dt_end_month, 
                                           dt_end_day, dt_end_hour,
                                           dt_end_minute, dt_end_second, dt_end_ms,
                                           tzinfo=utc_tz).astimezone(chicago_tz)
    
    # Transform the datetime to unix time in ns
    unix_time_start_ns = int(dt_start_transformed.timestamp() * 1e9 + 
                             dt_start_transformed.microsecond * 1000)
    unix_time_end_ns = int(dt_end_transformed.timestamp() * 1e9 +
                           dt_end_transformed.microsecond * 1000)
    
    return unix_time_start_ns, unix_time_end_ns

def instr_file_name(runid, instr, recoil, generator_name, mode, rate=1e9/SALT_TIME_INTERVAL,
                    base_dir=BASE_DIR):
    """
    Generate the instruction file name and then save the csv instructions.
    :param runid: run number in integer
    :param instr: instructions in numpy array
    :param recoil: NEST recoil type
    :param generator_name: name of the generator
    :param mode: 's1', 's2', or 'all'
    :param rate: rate of events in Hz
    :param base_dir: base directory to save the instruction file, default: BASE_DIR
    :return: instruction file name
    """
    # FIXME: this will shoot errors if we are on OSG rather than midway
    if base_dir[-1] != '/':
        base_dir += '/'

    rate = int(rate)
    runid = str(runid).zfill(6)
    filename = BASE_DIR + runid + "-" + str(recoil) + "-" + \
        generator_name + "-" + mode + "-" + str(rate) + ".csv"
    
    # if the file already exists, we don't want to overwrite it
    if not os.path.exists(filename):
        pd.DataFrame(instr).to_csv(filename, index=False)
    else:
        print("Instruction file already exists at: %s" % (filename))
        
    print("Instruction file at: %s" % (filename))

    return filename

def generator_se(runid, 
                 n_tot=None, rate=1e9/SALT_TIME_INTERVAL, 
                 r_range=R_RANGE, z_range=Z_RANGE, 
                 time_mode="uniform", *args):
    """
    Generate instructions for a run with single electron.
    :param runid: run number in integer
    :param n_tot: total number of events to generate, default: None i.e. generate events until end_time
    :param rate: rate of events in Hz, default: 1e9/SALT_TIME_INTERVAL
    :param r_range: (r_min, r_max) in cm, default: R_RANGE, defined above
    :param z_range: (z_min, z_max) in cm, default: Z_RANGE, defined above
    :param time_mode: 'uniform' or 'realistic', default: 'uniform'
    :return: instructions in numpy array
    """
    start_time, end_time = get_run_start_end(runid)
    times = generate_times(start_time, end_time, size=n_tot, 
                           rate=rate, time_mode=time_mode)
    n_tot = len(times)
    
    instr = np.zeros(n_tot, dtype=wfsim.instruction_dtype)
    instr["event_number"] = np.arange(1, n_tot + 1)
    instr["type"][:] = 2
    instr["time"][:] = times

    # Generating unoformely distributed events for give R and Z range
    x, y, z = generate_vertex(r_range=r_range, z_range=z_range, size=n_tot)
    instr["x"][:] = x
    instr["y"][:] = y
    instr["z"][:] = z

    # And generating quantas from nest
    for i in range(0, n_tot):
        instr["amp"][i] = 1
        instr["n_excitons"][i] = 0
        
    return instr
    
def generator_se_bootstrapped(runid, 
                              xyt_files_at='/project/lgrandi/yuanlq/salt/se_instructions/'):
    """
    Generate instructions for a run with single electron. We will use XYT information from
    bootstrapped data single electrons to make the simulation more realistic
    :param runid: run number in integer
    :param xyt_files_at: directory to search for instructions of x,y,t information
    """
    # load instructions
    runid_str = str(runid).zfill(6)
    with open(xyt_files_at+"se_xs_dict.pkl", 'rb') as f:
        xs = pickle.load(f)[runid_str]
    with open(xyt_files_at+"se_ys_dict.pkl", 'rb') as f:
        ys = pickle.load(f)[runid_str]
    with open(xyt_files_at+"se_ts_dict.pkl", 'rb') as f:
        ts = pickle.load(f)[runid_str]

    # stay in runtime range
    start_time, end_time = get_run_start_end(runid)
    mask_in_run = ts[ts < (end_time - 1/20*1e9)]    # empirical patch to stay in run
    mask_in_run = ts[ts > (start_time + 1/20*1e9)]  # empirical patch to stay in run
    xs = xs[mask_in_run]
    ys = ys[mask_in_run]
    ts = ts[mask_in_run]

    n_tot = len(ts)
    instr["event_number"] = np.arange(1, n_tot + 1)
    instr["type"][:] = 2
    instr["time"][:] = times
    instr["x"][:] = xs
    instr["y"][:] = ys
    instr["z"][:] = -0.00001 # Just to avoid drift time

    # And generating quantas from nest
    for i in range(0, n_tot):
        instr["amp"][i] = 1
        instr["n_excitons"][i] = 0
    
    return instr

def generator_flat(runid, en_range=(0.2, 15.0), recoil=8,
                   n_tot=None, rate=1e9/SALT_TIME_INTERVAL, 
                   fmap=FIELD_MAP, nc=NC, 
                   r_range=R_RANGE, z_range=Z_RANGE, 
                   mode="all", time_mode="uniform",):
    """
    Generate instructions for a run with flat energy spectrum.
    :param runid: run number in integer
    :param en_range: (en_min, en_max) in keV, default: (0, 30.0)
    :param recoil: NEST recoil type, default: 7 (beta ER)
    :param n_tot: total number of events to generate, default: None i.e. generate events until end_time
    :param rate: rate of events in Hz, default: 1e9/SALT_TIME_INTERVAL
    :param fmap: field map, default: FIELD_MAP, defined above
    :param nc: NEST calculator, default: NC, defined above
    :param r_range: (r_min, r_max) in cm, default: R_RANGE, defined above
    :param z_range: (z_min, z_max) in cm, default: Z_RANGE, defined above
    :param mode: 's1', 's2', or 'all', default: 'all'
    :param time_mode: 'uniform' or 'realistic', default: 'uniform'
    :return: instructions in numpy array
    """
    start_time, end_time = get_run_start_end(runid)
    times = generate_times(start_time, end_time, size=n_tot, 
                           rate=rate, time_mode=time_mode)
    n_tot = len(times)
    
    instr = np.zeros(2 * n_tot, dtype=wfsim.instruction_dtype)
    instr["event_number"] = np.arange(1, n_tot + 1).repeat(2)
    instr["type"][:] = np.tile([1, 2], n_tot)
    instr["time"][:] = times.repeat(2)

    # Generating unoformely distributed events for give R and Z range
    x, y, z = generate_vertex(r_range=r_range, z_range=z_range, size=n_tot)
    instr["x"][:] = x.repeat(2)
    instr["y"][:] = y.repeat(2)
    instr["z"][:] = z.repeat(2)

    # Making energy
    ens = np.random.uniform(en_range[0], en_range[1], size=n_tot)
    instr["recoil"][:] = recoil
    instr["e_dep"][:] = ens.repeat(2)

    # Getting local field from field map
    instr["local_field"] = fmap(np.array([np.sqrt(x**2 + y**2), z]).T).repeat(2)

    # And generating quantas from nest
    for i in range(0, n_tot):
        y = nc.GetYields(interaction=nestpy.INTERACTION_TYPE(instr["recoil"][2 * i]),
                         energy=instr["e_dep"][2 * i],
                         drift_field=instr["local_field"][2 * i],)
        quantas = nc.GetQuanta(y)
        instr["amp"][2 * i] = quantas.photons
        instr["amp"][2 * i + 1] = quantas.electrons
        instr["n_excitons"][2 * i : 2 * (i + 1)] = quantas.excitons
    
    # Selecting event types
    if mode == "s1":
        instr = instr[instr["type"] == 1]
    elif mode == "s2":
        instr = instr[instr["type"] == 2]
    elif mode == "all":
        pass
    else:
        raise RuntimeError("Unknown mode: ", mode)
    
    # Filter out 0 amplitudes
    instr = instr[instr["amp"] > 0]
    
    return instr
