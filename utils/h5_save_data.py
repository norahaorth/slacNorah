import glob
import pandas as pd
import h5py  # type: ignore[import-untyped]
import os
import time
from utils.srf_waveforms import (
    load_fault_file,
    grab_common_data,
    validate_quench_lisa,
    label_to_values,
)

start_time = time.time()

local = False  # True
if local:
    DATA_DIR = r"/Users/nneveu/Google Drive/My Drive/srf/q/"
    SAVE_DIR = "."
else:
    DATA_DIR = r"/mccfs2/u1/lcls/physics/rf_lcls2/fault_data/"
    SAVE_DIR = "/sdf/group/ad/org/lfd/sclp/data/"


def _savefile_for_lx(lx):
    return os.path.join(SAVE_DIR, f"quench_data_L{lx}.h5")


def _get_lx_dir(lx):
    """Get accelerating section, L0, L1, L2, or L3 directory."""
    lnum = f"{lx:1d}"
    return os.path.join(DATA_DIR, f"ACCL_L{lnum}B_*")


def _get_quench_filenames(lx):
    """Get sorted quench files for Lx section."""
    lx_dir = _get_lx_dir(lx)
    quench_files = glob.glob(os.path.join(lx_dir, "**", "*QUENCH.txt"), recursive=True)
    return sorted(quench_files)


def load_quench_data(quench_files) -> pd.DataFrame:
    """Load quench files and return DataFrame of all quench waveforms."""
    quench_data = []
    for filename in quench_files:
        df = load_fault_file(filename)
        quench_data.append(grab_common_data(df))
    return pd.concat(quench_data, ignore_index=True) if quench_data else pd.DataFrame()


def save_filenames_to_txt(quench_files, output_txt):
    """Save list of quench filenames to a text file."""
    with open(output_txt, "w") as f:
        for file in quench_files:
            f.write(f"{os.path.basename(file)}\n")


all_data = []
for lx in range(4):
    quench_files = _get_quench_filenames(lx)
    # save_filenames_to_txt(quench_files, output_txt)
    print(f"L{lx}: found {len(quench_files)} quench files")
    all_data.append(load_quench_data(quench_files))

all_data = pd.concat(all_data, ignore_index=True)

for lx in range(4):
    lx_tag = f"ACCL_L{lx}B_"
    lx_data = all_data[all_data["source_file"].str.contains(lx_tag, na=False)]
    if lx_data.empty:
        print(f"L{lx}: no data, skipping")
        continue

    savefile = _savefile_for_lx(lx)
    print(f"\nWriting L{lx} -> {savefile} ({len(lx_data)} rows)")

    with h5py.File(savefile, "w") as h5file:
        for (cm, cav), cav_data in lx_data.groupby(
            ["cryomodule", "cavity"], dropna=False
        ):  # type: ignore
            print(f"Processing CM{cm} CAV{cav}...")
            cm_group = h5file.require_group(f"CM{cm}")
            cav_group = cm_group.require_group(f"CAV{cav}")
            quench_files = cav_data["source_file"].unique()

            print(f"Data for CM{cm} CAV{cav} has {len(quench_files)} quench events.")
            for filename in quench_files:
                quench_data = cav_data[cav_data["source_file"] == filename]
                timestamp = quench_data["file_date"].iloc[0]

                labeled_values = label_to_values(quench_data)
                quench_group = cav_group.create_group(timestamp)
                quench_result = validate_quench_lisa(quench_data)
                quench_group.attrs["quench_classification"] = quench_result[
                    "is_real"
                ]  # boolean
                quench_group.attrs["calculated_q_loaded"] = quench_result["loaded_q"]
                quench_group.attrs["other_issue"] = str(quench_result["other_issue"])

                for label, values in labeled_values.items():
                    if len(values) > 1:  # don't save freq and q again
                        quench_group.create_dataset(label, data=values)
                    elif label in ["frequency", "saved_q_loaded"]:
                        quench_group.attrs[label] = values[0]

print(f"Total runtime: {time.time() - start_time:.2f} seconds")
