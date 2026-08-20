import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h5py
from utils.label_helpers import parse_h5_event_path
from utils.quench_data_summary import load_csv
from utils.h5_reader import find_event_groups
from utils.label_helpers import norm_cm, norm_cav
from pathlib import Path


def add_multipacting_flags(file_path, multipacting_file, flag_attr="Multipacting"):
    """
    Add a multipacting flag to each events in the h5 file
    - The function matches the event name in the h5 file with the data in the csv file
    - Each event in the h5 file gets a boolean attribute so either true or false
    - Match only after ignoring the time (HHMMSS) in the h5 file since the events in the csv file are saved with no time records
    """
    dataframe = load_csv(multipacting_file)  # Load the csv file into a dataframe
    multipacting_keys = build_multipacting_keys(dataframe)

    matched = 0  # flagged multipacting events
    total = 0  # valid processed events

    # read the h5 file
    with h5py.File(file_path, "a") as f:
        event_paths = find_event_groups(f)

        # Loop over each path
        for path in event_paths:
            is_valid, is_multipacting = flag_event(
                f, path, multipacting_keys, flag_attr
            )
            if is_valid:
                total += 1
                if is_multipacting:
                    matched += 1  # Increment the multipacting events counter

    print(
        f"Flagged {matched} multipacting events out of {total} valid events in the h5 file."
    )
    return matched, total


# build a set of keys to identify each multipacting event
def build_multipacting_keys(dataframe):
    multipacting_keys = set()
    for _, row in dataframe.iterrows():
        key = (
            norm_cm(row["cm"]),  # cryomodule as a string
            norm_cav(row["cav"]),  # cavity as a string
            f"{int(row['year']):04d}",  # year : 4 digits, e.g. 2025
            f"{int(row['month']):02d}",  # month : 2 didgits, e.g. 07
            f"{int(row['day']):02d}",  # day : 2 digits, e.g. 19
        )
        multipacting_keys.add(key)

    return multipacting_keys


def flag_event(f, path, multipacting_keys, flag_attr):
    parsed = parse_h5_event_path(path)
    if not parsed:
        return False, False

    cm, cav, date_str, _time_str = parsed
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]

    event_key = (norm_cm(cm), norm_cav(cav), year, month, day)

    is_multipacting = (
        event_key in multipacting_keys
    )  # Check if the event is in the multipcating set
    f[path].attrs[flag_attr] = bool(
        is_multipacting
    )  # Write the boolean result back to the h5 file

    return True, is_multipacting


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    csv_file = ROOT / "config" / "all_mp_dates.csv"  # The mp file

    # Change this field:
    h5_file_path = (
        "/Users/username/directory/data/quench_data_L0.h5"  # Your local h5 file path
    )

    add_multipacting_flags(file_path=h5_file_path, multipacting_file=str(csv_file))
