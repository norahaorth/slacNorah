"""Cross-check per-cryomodule quench counts.

Compare the counts printed by ``save_data_h5.py`` (captured in a log file)
against the counts derived from the HDF5 files via
``quench_data_summary.load_quench_events``.

Log lines looked for (one per cavity)::

    Data for CM01 CAV1 has 135 quench events.
"""

import os
import re
from collections import defaultdict

import pandas as pd

from utils.h5_load_data import load_quench_events


HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "data", "20260510_error_log.txt")
H5_GLOB = os.path.join(HERE, "data", "quench_data_L*.h5")

LINE_RE = re.compile(r"Data for (CM\w+)\s+CAV\w+ has (\d+) quench events")


def counts_from_log(path):
    counts = defaultdict(int)
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                counts[m.group(1)] += int(m.group(2))
    return pd.Series(counts, name="log").sort_index()


def counts_from_h5(glob_pattern):
    events = load_quench_events(glob_pattern)
    return events.groupby("cm").size().rename("h5").sort_index()


def main():
    log = counts_from_log(LOG_PATH)
    h5 = counts_from_h5(H5_GLOB)
    diff = pd.concat([log, h5], axis=1).fillna(0).astype(int)
    diff["delta"] = diff["h5"] - diff["log"]

    print(diff.to_string())
    print(
        f"\nTotals: log={int(diff['log'].sum())} h5={int(diff['h5'].sum())} "
        f"delta={int(diff['delta'].sum())}"
    )

    mismatched = diff[diff["delta"] != 0]
    if mismatched.empty:
        print("\nOK: every cryomodule matches.")
        return 0
    print(f"\nMISMATCH on {len(mismatched)} cryomodule(s):")
    print(mismatched.to_string())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
