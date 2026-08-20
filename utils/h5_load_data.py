import os
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from numpy.typing import NDArray
from typing import Iterator, Tuple, Dict
from dataclasses import fields
from pathlib import Path
from utils.config import DATA_DIR, H5_GLOB, DataBundle, QuenchData


# Iterates through all HDF5 events to compile metadata and format a DataBundle with physically sorted pandas DataFrames for plotting.
def build_plotter_bundle() -> DataBundle:
    records = []

    for event_id, filename, quench_data in load_quench_events(
        H5_GLOB, load_waveforms=False
    ):
        cm, cav, timestamp_str = event_id.split("/")

        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

        records.append(
            {
                "cm": cm.upper(),
                "cav": cav,
                "year": str(dt.year),
                "month": str(dt.month),
                "day": str(dt.day),
                "classification": quench_data.quench_classification,
                "is_real": quench_data.quench_classification == "real",
                "is_mp": bool(quench_data.is_mp),
            }
        )

    events = pd.DataFrame(records)
    print(f"Loaded {len(events)} quench events from {H5_GLOB}")

    CM_ORDER = ["CM01", "CM02", "CM03", "CMH1", "CMH2"] + [
        f"CM{n:02d}" for n in range(4, 36)
    ]
    present = [cm for cm in CM_ORDER if cm in set(events["cm"])]
    events["cm"] = pd.Categorical(events["cm"], categories=present, ordered=True)

    events_no_hl = events[~events["cm"].isin(["CMH1", "CMH2"])]
    real_events = events_no_hl[events_no_hl["classification"] == "real"]
    nomp_nohl_real_all = real_events[~real_events["is_mp"]]

    return DataBundle(
        all_events=events,
        events_no_hl=events_no_hl,
        real_events=real_events,
        nomp_nohl_real_all=nomp_nohl_real_all,
    )


# Extracts datasets from a single HDF5 group into a QuenchData dataclass
def extract_quench_data(group: h5py.Group, load_waveforms: bool = True) -> QuenchData:
    data_dict = {}

    for field in fields(QuenchData):
        if field.name in group:
            if not load_waveforms:
                continue

            item = group[field.name]
            if isinstance(item, h5py.Dataset):
                data_dict[field.name] = item[()]
        elif field.name in group.attrs:
            val = group.attrs[field.name]

            if isinstance(val, bytes):
                val = val.decode("utf-8")

            data_dict[field.name] = val

    if "frequency" not in data_dict:
        if "FREQ" in group.attrs:
            data_dict["frequency"] = float(group.attrs["FREQ"])  # type: ignore

    if "saved_q_loaded" not in data_dict:
        if "QLOADED" in group.attrs:
            data_dict["saved_q_loaded"] = float(group.attrs["QLOADED"])  # type: ignore

    return QuenchData(**data_dict)


# Loads and formats waveform data from an HDF5 file for easy UI plotting
def get_ui_waveform_signals(
    file_path: str, event_path: str
) -> Tuple[Dict[str, Tuple[NDArray, NDArray]], float, float]:
    with h5py.File(file_path, "r") as f:
        group = f[event_path]
        quench_data = extract_quench_data(group, load_waveforms=True)  # type: ignore

    signal_data = {}
    signal_time_map = {
        "forward_power": "forward_time",
        "reverse_power": "reverse_time",
        "fault_waveform": "fault_time",
        "decay_reference": "forward_time",
    }

    for signal_name, time_name in signal_time_map.items():
        y_data = getattr(quench_data, signal_name, None)
        if y_data is None:
            continue

        y = np.array(y_data)
        x_data = getattr(quench_data, time_name, None)
        x = None

        if x_data is not None:
            t = np.array(x_data)
            if t.shape[0] == y.shape[0]:
                x = t

        if x is None:
            x = np.arange(y.shape[0])

        signal_data[signal_name] = (x, y)

    return (
        signal_data,
        getattr(quench_data, "frequency", 1.3e9),
        getattr(quench_data, "saved_q_loaded", 4e7),
    )


# Traverses HDF5 files to extract and yield quench event datasets
def load_quench_events(
    file_pattern: str = "*.h5", load_waveforms: bool = True
) -> Iterator[Tuple[str, str, QuenchData]]:
    folder = Path(DATA_DIR)

    safe_pattern = os.path.basename(file_pattern)

    for h5_file in folder.glob(safe_pattern):
        with h5py.File(h5_file, "r") as f:
            for cm_name, cm_group in f.items():
                if not isinstance(cm_group, h5py.Group):
                    continue

                for cav_name, cav_group in cm_group.items():
                    if not isinstance(cav_group, h5py.Group):
                        continue

                    for timestamp, event_group in cav_group.items():
                        if not isinstance(event_group, h5py.Group):
                            continue

                        event_id = f"{cm_name}/{cav_name}/{timestamp}"

                        quench_data = extract_quench_data(event_group, load_waveforms)
                        yield (event_id, h5_file.name, quench_data)
