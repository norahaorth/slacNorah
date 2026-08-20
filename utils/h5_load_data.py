import os
import glob
import h5py
import numpy as np
import pandas as pd
import re
from datetime import datetime
from numpy.typing import NDArray
from typing import Iterator, Tuple, Dict
from dataclasses import fields
from pathlib import Path
from utils.config import DATA_DIR, H5_GLOB, DataBundle, QuenchData
from interface.quench_config import SIGNAL_TIME_MAP

EVENT_COLS = ["source_file", "cm", "cav", "date", "year", "month", "day", "is_real"]
CMHLs = ["CMH1", "CMH2"]
config_dir: str = "config"

MP_SMARTSHEET_PATH = os.path.join(
    os.path.dirname(__file__), "..", config_dir, "MPdates_smartsheet.csv"
)
ALL_MP_PATH = os.path.join(
    os.path.dirname(__file__), "..", config_dir, "all_mp_dates.csv"
)


# Extracts datasets from a single HDF5 group into a QuenchData dataclass
def extract_quench_data(group: h5py.Group, load_waveforms: bool = True) -> QuenchData:
    data_dict = {}

    for field in fields(QuenchData):
        if field.name in group:
            if not load_waveforms:
                data_dict[field.name] = np.array([])
                continue

            item = group[field.name]
            if isinstance(item, h5py.Dataset):
                data_dict[field.name] = item[()]
        elif field.name in group.attrs:
            val = group.attrs[field.name]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            data_dict[field.name] = val

    raw_label = group.attrs.get("quench_labels", "unknown")
    if isinstance(raw_label, np.ndarray) and raw_label.size > 0:
        raw_label = raw_label[0]
    if isinstance(raw_label, bytes):
        raw_label = raw_label.decode("utf-8")

    clean_label = str(raw_label).strip("[]\"' ").lower()

    data_dict["quench_classification"] = clean_label

    if "frequency" not in data_dict and "FREQ" in group.attrs:
        data_dict["frequency"] = float(group.attrs["FREQ"])  # type: ignore

    if "saved_q_loaded" not in data_dict and "QLOADED" in group.attrs:
        data_dict["saved_q_loaded"] = float(group.attrs["QLOADED"])  # type: ignore

    return QuenchData(**data_dict)


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


def filter_events(
    events, classification=None, exclude_hl=False, exclude_mp=True, mp_source="all"
):
    """Return a subset of events by classification, HL, and MP membership."""
    sub = events
    if exclude_hl:
        sub = sub[~sub["cm"].isin(CMHLs)]

    if classification == "real":
        sub = sub[sub["is_real"].astype(bool)]
    elif classification == "false":
        sub = sub[~sub["is_real"].astype(bool)]
    elif classification is not None:
        raise ValueError(
            f"classification must be None, 'real', or 'false' (got {classification!r})"
        )

    if exclude_mp:
        sub = mp_events(sub, keep=False, source=mp_source)
    return sub.reset_index(drop=True)


def _mp_keys(source="all"):
    if source == "all":
        df = pd.read_csv(ALL_MP_PATH, dtype=str)
        date = (
            df["year"].str.zfill(4) + df["month"].str.zfill(2) + df["day"].str.zfill(2)
        )
        return set(zip(df["cm"], df["cav"], date))

    if source == "smartsheet":
        MP = pd.read_csv(MP_SMARTSHEET_PATH)
        cm = "CM" + MP["CM"].astype(int).astype(str).str.zfill(2)
        cav = "CAV" + MP["CAV"].astype(int).astype(str)
        date = pd.to_datetime(MP["date"], format="%m/%d/%y").dt.strftime("%Y%m%d")
        return set(zip(cm, cav, date))
    raise ValueError(f"source must be 'smartsheet' or 'all' (got {source!r})")


def mp_events(events, keep=False, source="all"):
    if events.empty:
        return events

    keys = _mp_keys(source=source)
    day = events["date"].str[:8]
    in_mp = [k in keys for k in zip(events["cm"], events["cav"], day)]
    mask = in_mp if keep else [not x for x in in_mp]
    return events[mask].reset_index(drop=True)


# Builds the main plot ready DataBundle by loading and filtering events
def build_plotter_bundle(
    source_glob: str = H5_GLOB, load_waveforms: bool = False
) -> DataBundle:
    records = []

    for event_id, filename, quench_data in load_quench_events(
        source_glob, load_waveforms=load_waveforms
    ):
        cm, cav, timestamp_str = event_id.split("/")
        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        records.append(
            {
                "cm": cm.upper(),
                "cav": cav,
                "date": timestamp_str,
                "year": str(dt.year),
                "month": str(dt.month),
                "day": str(dt.day),
                "classification": quench_data.quench_classification,
                "is_real": quench_data.quench_classification == "real",
                "is_mp": bool(getattr(quench_data, "is_mp", False)),
            }
        )

    events = pd.DataFrame(records)
    print(f"Loaded {len(events)} quench events from {source_glob}")

    CM_ORDER = ["CM01", "CM02", "CM03", "CMH1", "CMH2"] + [
        f"CM{n:02d}" for n in range(4, 36)
    ]
    present = [cm for cm in CM_ORDER if cm in set(events["cm"])]
    events["cm"] = pd.Categorical(events["cm"], categories=present, ordered=True)

    events_no_hl = filter_events(events, exclude_hl=True)
    real_events = filter_events(events, classification="real", exclude_hl=True)
    nomp_nohl_real_all = filter_events(
        events, classification="real", exclude_hl=True, exclude_mp=True
    )

    return DataBundle(
        all_events=events,
        events_no_hl=events_no_hl,
        real_events=real_events,
        nomp_nohl_real_all=nomp_nohl_real_all,
    )


def get_ui_waveform_signals(
    file_path: str, event_path: str
) -> Tuple[Dict[str, Tuple[NDArray, NDArray]], float, float]:
    """Loads and formats waveform data from an HDF5 file for easy UI plotting."""
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


# Return raw dictionaries of waveforms + attrs for the quenches in events
def load_quench_waveforms(events, source):
    if isinstance(events, pd.Series):
        events = events.to_frame().T
    wanted = set(zip(events["cm"], events["cav"], events["date"]))

    out = {}
    for path in _resolve_paths(source):
        with h5py.File(path, "r") as f:
            for cm in f:
                if cm not in {w[0] for w in wanted}:
                    continue
                for cav in f[cm]:  # type: ignore
                    if (cm, cav) not in {(w[0], w[1]) for w in wanted}:
                        continue
                    for ts in f[cm][cav]:  # type: ignore
                        if (cm, cav, ts) not in wanted:
                            continue
                        g = f[cm][cav][ts]  # type: ignore
                        out[f"{cm}/{cav}/{ts}"] = {
                            "datasets": {k: g[k][...] for k in g.keys()},  # type: ignore
                            "attrs": {k: g.attrs[k] for k in g.attrs.keys()},
                        }
    return out


def _resolve_paths(source):
    if isinstance(source, str) and any(c in source for c in "*?["):
        paths = sorted(glob.glob(source))
        if not paths:
            raise FileNotFoundError(f"No files matched glob: {source}")
        return paths
    if isinstance(source, (str, os.PathLike)):
        return [source]
    return list(source)


def load_csv(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".txt"):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, delim_whitespace=True)  # type: ignore
    raise ValueError(f"Error with file type: {path}")


def list_cryomodules(h5_file):
    return sorted(k for k in h5_file.keys() if re.fullmatch(r"CM\d+", k))


def list_cavities(h5_file, cm):
    return (
        sorted(k for k in h5_file[cm].keys() if re.fullmatch(r"CAV\d+", k))
        if cm in h5_file
        else []
    )


def list_years(h5_file, cm, cav):
    years = set()
    if cm in h5_file and cav in h5_file[cm]:
        for name in h5_file[cm][cav].keys():
            if match := re.match(r"(\d{4})\d{4}_\d{6}", name):
                years.add(match.group(1))
    return sorted(years)


def has_signal(group):
    return bool(set(group.keys()) & set(SIGNAL_TIME_MAP.keys()))


def peak_quench_day_per_cavity(events, top_n=3, real_only=True, save_path=None):
    df = events[events["is_real"]] if real_only else events
    daily = (
        df.groupby(["cm", "cav", "year", "month", "day"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values(
            ["cm", "cav", "count", "year", "month", "day"],
            ascending=[True, True, False, True, True, True],
        )
    )
    peak = daily.groupby(["cm", "cav"], observed=True).head(top_n).copy()
    peak["rank"] = peak.groupby(["cm", "cav"], observed=True).cumcount() + 1
    if save_path:
        out = (
            save_path
            if os.path.isabs(save_path) or os.path.dirname(save_path)
            else os.path.join(os.path.dirname(__file__), "data", save_path)
        )
        peak.to_csv(out, index=False)
    return peak.reset_index(drop=True)


def print_peak_quench_day_summary(events, top_n=3, real_only=True):
    peak = peak_quench_day_per_cavity(events, top_n=top_n, real_only=real_only)
    label = "real" if real_only else "all"
    print(f"\nTop {top_n} quench days per cavity ({label} quenches):")
    header = f"  {'CM':<5} {'CAV':<5} {'rank':<5} {'date':<10} {'count':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in peak.iterrows():
        print(
            f"  {row['cm']:<5} {row['cav']:<5} {int(row['rank']):<5} {int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d} {int(row['count']):>6}"
        )
    return peak


def peak_days_not_in_mp(peak, mp_events_df):
    keys = set(
        zip(
            mp_events_df["cm"],
            mp_events_df["cav"],
            mp_events_df["year"],
            mp_events_df["month"],
            mp_events_df["day"],
        )
    )
    mask = [
        (cm, cav, y, m, d) not in keys
        for cm, cav, y, m, d in zip(
            peak["cm"], peak["cav"], peak["year"], peak["month"], peak["day"]
        )
    ]
    return peak[mask].reset_index(drop=True)
