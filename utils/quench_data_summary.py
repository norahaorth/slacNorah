import glob
import os

import h5py  # type: ignore[import-untyped]
import pandas as pd
import re
from interface.quench_config import SIGNAL_TIME_MAP


EVENT_COLS = ["source_file", "cm", "cav", "date", "year", "month", "day", "is_real"]

CMHLs = ["CMH1", "CMH2"]


def filter_events(
    events, classification=None, exclude_hl=False, exclude_mp=True, mp_source="all"
):
    """Return a subset of ``events`` by classification, HL, and MP membership.

    `classification`:
      - None     -> keep all rows (real + false)
      - "real"   -> keep rows where ``is_real`` is True
      - "false"  -> keep rows where ``is_real`` is False
    `exclude_hl`: if True, drop rows whose ``cm`` is in ``CMHLs``
    (CMH1, CMH2). Default keeps them.
    `exclude_mp`: if True, drop rows that match an entry in the chosen
    MP table via ``mp_events(..., keep=False, source=mp_source)``.
    `mp_source`: "all" (merged ``data/all_mp_dates.csv``, default) or
    "sebastian" (raw input).
    """
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


config_dir: str = "config"
MP = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", config_dir, "MPdates_smartsheet.csv")
)
ALL_MP_PATH = os.path.join(
    os.path.dirname(__file__), "..", config_dir, "all_mp_dates.csv"
)


def _mp_keys(source="all"):
    """Return the set of (cm, cav, yyyymmdd) tuples that were MP-processed.

    `source`:
      - "all" (default): read from data/all_mp_dates.csv, the merged set
        built by ``build_all_mp_dates.py``.
      - "smartsheet": read from MPdates_smartsheet.csv (raw input).
    """
    if source == "all":
        df = pd.read_csv(ALL_MP_PATH, dtype=str)
        date = (
            df["year"].str.zfill(4) + df["month"].str.zfill(2) + df["day"].str.zfill(2)
        )
        return set(zip(df["cm"], df["cav"], date))
    if source == "smartsheet":
        cm = "CM" + MP["CM"].astype(int).astype(str).str.zfill(2)
        cav = "CAV" + MP["CAV"].astype(int).astype(str)
        # Dates in MPdates_smartsheet.csv look like "10/10/24"; Convert to normal YYYYMMDD format.
        date = pd.to_datetime(MP["date"], format="%m/%d/%y").dt.strftime("%Y%m%d")
        return set(zip(cm, cav, date))
    raise ValueError(f"source must be 'smartsheet' or 'all' (got {source!r})")


def mp_events(events, keep=False, source="all"):
    """Filter events by MP-processing membership.

    `keep`:
      - False (default): drop rows that match an MP entry.
      - True: keep only rows that match an MP entry.
    `source`: which MP table to consult — "all" (merged
    ``data/all_mp_dates.csv``, default) or "smartsheet" (raw input).

    Match key is ``(cm, cav, YYYYMMDD)``. Returns a new DataFrame.
    """
    keys = _mp_keys(source=source)
    day = events["date"].str[:8]
    in_mp = [k in keys for k in zip(events["cm"], events["cav"], day)]
    mask = in_mp if keep else [not x for x in in_mp]
    return events[mask].reset_index(drop=True)


def peak_quench_day_per_cavity(events, top_n=3, real_only=True, save_path=None):
    """For each (cm, cav), return the ``top_n`` days with the most quenches.

    Returns a DataFrame with columns
    ``["cm", "cav", "year", "month", "day", "count", "rank"]`` sorted by
    cm, cav, then rank. ``rank`` is 1 for the busiest day, 2 for the
    next busiest, etc. Ties are broken by the earliest date. Cavities
    with fewer than ``top_n`` distinct quench days return all available
    days. If ``save_path`` is given (filename or full path), the result
    is also written to CSV. Bare filenames are placed in ``data/``.
    """
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
    print(f"  cavities: {peak.groupby(['cm', 'cav'], observed=True).ngroups}")
    header = f"  {'CM':<5} {'CAV':<5} {'rank':<5} {'date':<10} {'count':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in peak.iterrows():
        date = f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d}"
        print(
            f"  {row['cm']:<5} {row['cav']:<5} {int(row['rank']):<5} "
            f"{date:<10} {int(row['count']):>6}"
        )
    return peak


def peak_days_not_in_mp(peak, mp_events_df):
    """Return rows of ``peak`` whose (cm, cav, year, month, day) does not
    appear in ``mp_events_df``.

    Useful for filtering peak-quench-day results down to days that were
    not MP-processed.
    """
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


def _resolve_paths(source):
    if isinstance(source, str) and any(c in source for c in "*?["):
        paths = sorted(glob.glob(source))
        if not paths:
            raise FileNotFoundError(f"No files matched glob: {source}")
        return paths
    if isinstance(source, (str, os.PathLike)):
        return [source]
    return list(source)


def load_quench_events(source):
    """Return a flat events DataFrame, one row per quench event with EVENT_COLS.

    Source can be one of the following:
          - a path to a single ``quench_data_L*.h5`` file
          - a list/tuple of such paths
          - a glob string like ``"quench_data_L*.h5"``
    """
    rows = []
    for path in _resolve_paths(source):
        with h5py.File(path, "r") as f:
            for cm in f:  # "CM01"
                for cav in f[cm]:  # type: ignore # "CAV1"
                    for ts in f[cm][cav]:  # type: ignore # "YYYYMMDD_HHMMSS"
                        attrs = f[cm][cav][ts].attrs  # type: ignore
                        rows.append(
                            (
                                f"{path}::{cm}/{cav}/{ts}",
                                cm,
                                cav,
                                ts,
                                ts[:4],
                                ts[4:6],
                                ts[6:8],
                                bool(attrs.get("quench_classification", False)),
                            )
                        )
    return pd.DataFrame(rows, columns=EVENT_COLS)


def load_quench_waveforms(events, source):
    """Return waveforms + attrs for the quenches in `events`.

    Parameters: events : pd.DataFrame or pd.Series
        A row, Series, or sub-DataFrame from ``load_quench_events``. Only
        the ``cm``, ``cav``, ``date`` columns are used.
    source : same value passed to ``load_quench_events`` to build `events`.

    Returns: dict keyed by ``"CM01/CAV1/YYYYMMDD_HHMMSS"``.
    """
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


def load_csv(path):
    """
    load the csv or txt files into pandas dataframe

    - csv files are read direclty
    - txt files are comma-seperated first, if that fails it fall back to whitespace-seperated
    - Otherwise, it throws an error
    """
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".txt"):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, delim_whitespace=True)  # type: ignore
    else:
        raise ValueError(f"Error with file type: {path}")


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
            match = re.match(r"(\d{4})\d{4}_\d{6}", name)
            if match:
                years.add(match.group(1))
    return sorted(years)


def has_signal(group):
    return bool(set(group.keys()) & set(SIGNAL_TIME_MAP.keys()))


# ----------------------------------------------------------------------- #
# Find days that *look* like MP (a cavity's busiest quench days) but are
# not recorded in MPdates.csv. These are candidate missing MP entries.
# ----------------------------------------------------------------------- #
# PEAK_TOP_N = 10          # consider each cavity's top-N busiest days
# PEAK_MIN_COUNT = 10     # only flag days with more than this many quenches
# cm33 = events[(events["cm"] == "CM33") & (~events["is_real"].astype(bool))]
# peakdf = peak_quench_day_per_cavity(cm33, top_n=PEAK_TOP_N, real_only=False)
# # candidates = peak_days_not_in_mp(peakdf, onlympevents)
# # candidates = candidates[candidates["count"] > PEAK_MIN_COUNT].reset_index(drop=True)
# # candidates = candidates[~candidates["cm"].isin(["CM34", "CM35"])].reset_index(drop=True)

# print(f"\nCandidate missing MP days "
#       f"(top-{PEAK_TOP_N} per cavity, count > {PEAK_MIN_COUNT}):")
# print(peakdf.to_string(index=False))
# peakdf.to_csv(os.path.join(HERE, "data", "non_mp_peak_quench_days_cm33.csv"),
#                   index=False)
