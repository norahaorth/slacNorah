"""
Script to fix missing quench data.

This script automatically scans through all .h5 files in the local `data/`
directory, identifies cavity events missing the `decay_reference` dataset,
calculates the exponential decay curve, and permanently saves it.
Events that already contain a `decay_reference` are automatically skipped.

Usage:
    Run this script from the root project directory using:
    $ python -m utils.add_missing_decay_reference
"""

import h5py
import glob
import numpy as np
from pathlib import Path
from classification.logic import QuenchData, find_quench_time
from numpy.typing import NDArray


# Calculates the expected exponential decay curve after a quench event
def calculate_decay_reference(qd: QuenchData) -> np.ndarray:
    idx_0 = find_quench_time(qd)
    decay_ref = np.zeros_like(qd.fault_time)

    decay_ref[:idx_0] = qd.fault_waveform[:idx_0]

    A = qd.fault_waveform[idx_0]
    f = qd.frequency
    Ql = qd.saved_q_loaded
    t_after_fault = qd.fault_time[idx_0:]

    exponent = -(np.pi * f * t_after_fault) / Ql
    decay_ref[idx_0:] = A * np.exp(exponent)

    return decay_ref


# Scans HDF5 files and adds the missing decay_reference dataset to cavity events that are missing it
def process_h5_files(search_pattern: str) -> None:
    h5_files = glob.glob(search_pattern, recursive=True)
    files_fixed_count = 0

    for file_path in h5_files:
        try:
            with h5py.File(file_path, "r+") as h5_file:
                event_paths = []

                def collect_events(name, node):
                    if isinstance(node, h5py.Group) and "fault_waveform" in node:
                        event_paths.append(name)

                h5_file.visititems(collect_events)

                file_was_patched = False

                for event_id in event_paths:
                    event_group = h5_file[event_id]

                    if "decay_reference" in event_group:  # type: ignore
                        continue

                    fault_time: NDArray[np.float64] = np.array(
                        event_group["fault_time"][:],  # type: ignore
                        dtype=np.float64,  # type: ignore
                    )
                    fault_waveform: NDArray[np.float64] = np.array(
                        event_group["fault_waveform"][:],  # type: ignore
                        dtype=np.float64,  # type: ignore
                    )

                    frequency: float = (
                        float(event_group["frequency"][()])  # type: ignore
                        if "frequency" in event_group  # type: ignore
                        else 1300000000.0
                    )

                    if "saved_q_loaded" not in event_group:  # type: ignore
                        print(
                            f"Warning: 'saved_q_loaded' missing for {event_id}. Skipping."
                        )
                        continue

                    saved_q_loaded: float = float(event_group["saved_q_loaded"][()])  # type: ignore

                    temp_qd = QuenchData(
                        fault_time=fault_time,
                        fault_waveform=fault_waveform,
                        forward_power=np.array([], dtype=np.float64),
                        forward_time=np.array([], dtype=np.float64),
                        reverse_power=np.array([], dtype=np.float64),
                        reverse_time=np.array([], dtype=np.float64),
                        frequency=frequency,
                        saved_q_loaded=saved_q_loaded,
                    )

                    new_decay_ref = calculate_decay_reference(temp_qd)
                    event_group.create_dataset("decay_reference", data=new_decay_ref)  # type: ignore

                    file_was_patched = True

            if file_was_patched:
                print(f"Fixed: {Path(file_path).name}")
                files_fixed_count += 1

        except Exception as e:
            print(f"Failed to fix {Path(file_path).name}: {e}")

    if files_fixed_count == 0:
        print("All files already have decay_reference.")


if __name__ == "__main__":
    pattern = (
        "./data/**/quench_data_L[0-9].h5"  # File path of data to add decay reference
    )
    process_h5_files(pattern)
