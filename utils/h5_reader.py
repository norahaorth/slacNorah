from datetime import datetime
import h5py
import numpy as np

from quench_config import (
    SIGNAL_TIME_MAP,
    LABELS,
    CHECKED,
    NOTE,
    CHECKED_AT,
    NEEDS_SPECIALIST,
    LOADED_Q_CHANGE_FOR_QUENCH,
    FREQUENCY_KEYS,
    SAVED_Q_LOADED_KEYS,
)
import re 


# TODO: look at this function again later for combination options
def find_event_groups(hdf5_file, cm=None, cav=None, year=None):
    """This function is for finding each event groups/identifiers (decay ref, forward power, fault waveform) for each cm/cav/date, used for plotting """
    events = []
    def collect_from_cavity_group(cav_group, cav_path):
        for name, obj in cav_group.items():
            if not isinstance(obj, h5py.Group):
                continue
            if year and not name.startswith(year):
                continue
            if has_signal(obj):
                events.append(f"{cav_path}/{name}")
 
    if cm and cav:
        if cm in hdf5_file and cav in hdf5_file[cm]:
            collect_from_cavity_group(hdf5_file[cm][cav], f"{cm}/{cav}")
    elif cm:
        if cm in hdf5_file:
            for cav_name in list_cavities(hdf5_file, cm):
                collect_from_cavity_group(hdf5_file[cm][cav_name], f"{cm}/{cav_name}")
    else:

        def visitor(name, obj):
            """ This function is for exploring the h5 file structure"""
            if isinstance(obj, h5py.Group) and has_signal(obj):
                if year and not name.split("/")[-1].startswith(year):
                    return
                #keys = set(obj.keys())
                #if keys & set(SIGNAL_TIME_MAP.keys()):
                events.append(name)

        hdf5_file.visititems(visitor)
    return sorted(events)

# ** Load waveform data for the selected event **
#TODO look at older version of this
def load_signal_data(group):
    """Load signals data (decay_ref, fault_waveform, forward_power, reverse_power)."""
    signal_data = {}

    for signal_name, time in SIGNAL_TIME_MAP.items():
        if signal_name not in group:  # if an event is missing some signals like the very first ones in 2022 (they are missing the decay reference), skip the missing items 
            continue

        y = np.array(group[signal_name])    # load the signals into array 
        x = None

        if time in group:
            t = np.array(group[time])  # load time data into array 
            # only assign t to x if its shape matches that of y 
            if t.shape[0] == y.shape[0]:
                x = t

        if x is None:
            x = np.arange(y.shape[0])   # if no time is available, create an x-axis starts from 0 to the length of y 

        signal_data[signal_name] = (x, y)   # store the loaded signal data 

    return signal_data

def list_cryomodules(h5_file):
    "This function is used for the events filter in the interface"
    return sorted(k for k in h5_file.keys() if re.fullmatch(r"CM\d+", k))


def list_cavities(h5_file, cm):
    "This function is used for the events filter in the interface"
    if cm not in h5_file:
        return []
    return sorted(k for k in h5_file[cm].keys() if re.fullmatch(r"CAV\d+", k))

def list_years(h5_file, cm, cav):
    "This function is used for the events filter in the interface"
    years = set()
    if cm in h5_file and cav in h5_file[cm]:
        for name in h5_file[cm][cav].keys():
            match = re.match (r"(\d{4})\d{4}_\d{6}", name)
            if match:
                years.add(match.group(1))
    return sorted(years)

def has_signal(group):
    return bool(set(group.keys()) & set(SIGNAL_TIME_MAP.keys()))


#TODO: should we remove this?
"""
def suggest_classification(time_data, fault_data, frequency, saved_q_loaded):
    
    This function is responsible for suggesting whether an event is real or false, this is how it works:
    - Trim the waveform to start at t = 0
    - Trim the tail once the amplitude is < 0.002
    - Fit ln(A0/A(t)) vs. time to a line; the slope gives the decay rate and loaded Q = (pi * frequency)/ slope
    - Suggest Real if the loaded Q is < (LOADED_Q_CHANGE_FOR_QUENCH * saved_q_loaded), otherwise suggest False 
   
    https://education.molssi.org/python-data-analysis/03-data-fitting/index.html

    
    other = None

    time_data = np.asarray(time_data, dtype=float)
    fault_data = np.asarray(fault_data, dtype=float)

    time_0 = 0
    # Look for time 0 (quench). These waveforms capture data beforehand
    for time_0, timestamp in enumerate(time_data):
        if timestamp >= 0:
            break

    fault_data = fault_data[time_0:]
    time_data = time_data[time_0:]
    end_decay = len(fault_data) - 1

    # Find where the amplitude decays to "zero"
    for end_decay, amp in enumerate(fault_data):
        if amp < 0.002:
            break

    if end_decay <= 1:
        other = "end_decay_not_found"
        pre_quench_amp = fault_data[0]
    else:
        fault_data = fault_data[:end_decay]
        time_data = time_data[:end_decay]
        pre_quench_amp = fault_data[0]

    try:
        with np.errstate(divide="raise", invalid="raise"):
            log_ratio = np.log(pre_quench_amp / fault_data)
            exponential_term = np.polyfit(time_data, log_ratio, 1)[0]
            loaded_q = (np.pi * frequency) / exponential_term
    except (FloatingPointError, ZeroDivisionError):
        return {"is_real": None, "loaded_q": np.nan, "other_issue": "divide_by_zero_or_invalid_value"}

    thresh_for_quench = LOADED_Q_CHANGE_FOR_QUENCH * saved_q_loaded
    is_real = bool(loaded_q < thresh_for_quench)
    return {"is_real": is_real, "loaded_q": loaded_q, "other_issue": other}

"""

# ** Classification suggestion **
#TODO: look at for reducing file access
def load_event_data_for_classification(path, event_path):
    """load frequency and saved_Q for computing the classsification suggestion"""
    with h5py.File(path, "r") as f:
        group = f[event_path]
        signal_data = load_signal_data(group)
        frequency = get_scalar(group, FREQUENCY_KEYS)   #read cavity frequency 
        saved_q_loaded = get_scalar(group, SAVED_Q_LOADED_KEYS) #read the saved loaded Q
    return signal_data, frequency, saved_q_loaded


def get_scalar(group, keys):
    """
    This function is used for extracting a single scalar value (frequency, saved Q) 
    from the h5 file to compute the REAL/ FALSE classification suggestion
    """
    
    for key in keys:
        # Case1: the key is a dataset and it is stored inside the group 
        if key in group:
            try:
                # If it's an array, take the first element using flat[0]
                arr = np.asarray(group[key]) 
                return float(arr.flat[0]) if arr.shape else float(arr)
            except Exception:
                continue
        # Case2: the key is stored as an attribute 
        if key in group.attrs:
            try:
                val = group.attrs[key]
                # decode to a string first if the attribute is stored as bytes 
                if isinstance(val, bytes):
                    val = val.decode()
                return float(val)
            except Exception:
                continue
    return None

def read_event_status(group):
    """Read one event's labeling attrs (checked, label, note, checked_at,
    needs_specialist) into a plain dict. Shared by app.py (to build the
    dropdown/status table) and merge_labels.py (to compare labelers)."""
    attrs = group.attrs
    return {
        "checked": bool(attrs.get(CHECKED, False)),
        "label": attrs.get(LABELS, None),
        "note": attrs.get(NOTE, None),
        "checked_at": attrs.get(CHECKED_AT, None),
        "needs_specialist": bool(attrs.get(NEEDS_SPECIALIST, False)),
    }

def write_label(file_path, event_path, label, srf_note, needs_specialist):
    """ writing the label, note and checked status to the hdf5 file for each event (cm/cav/date) """
    note = srf_note.strip() if srf_note and srf_note.strip() else (
        f"This event has been already checked and the waveform was labeled as {label}"
    )

    with h5py.File(file_path, "a") as f:
        group = f[event_path]
        group.attrs[LABELS] = label
        group.attrs[CHECKED] = True
        group.attrs[NOTE] = note
        group.attrs[CHECKED_AT] = datetime.now().strftime("%Y-%m-%d")
        group.attrs[NEEDS_SPECIALIST] = bool(needs_specialist)


