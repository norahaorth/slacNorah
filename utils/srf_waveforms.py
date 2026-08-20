import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict
from utils.config import IMG_DIR


def validate_quench_lisa(quench_data):
    """
    Evaluates real vs false quench based on decay rate.
    Expects a QuenchData dataclass from the HDF5 loader.
    """
    LOADED_Q_CHANGE_FOR_QUENCH = 0.6
    other = None

    freq_val = getattr(quench_data, "frequency", None)
    if freq_val is None or (isinstance(freq_val, np.ndarray) and freq_val.size == 0):
        print("Warning: Missing frequency, using default 1.3e9")
        frequency = 1300000000.0
    else:
        frequency = float(np.asarray(freq_val).flat[0])

    time_data = np.array(quench_data.fault_time)
    fault_data = np.array(quench_data.fault_waveform)

    if len(time_data) == 0 or len(fault_data) == 0:
        print("Warning: Empty waveforms provided to validate_quench_lisa.")
        return {"is_real": False, "loaded_q": np.nan, "other_issue": "empty_waveforms"}

    time_0 = 0
    for i, timestamp in enumerate(time_data):
        if timestamp >= 0:
            time_0 = i
            break

    fault_data = fault_data[time_0:]
    time_data = time_data[time_0:]
    end_decay = len(fault_data) - 1

    for i, amp in enumerate(fault_data):
        if amp < 0.002:
            end_decay = i
            break

    if end_decay <= 1:
        print("Warning: End of decay not found, using all data points.")
        other = "end_decay_not_found"
        pre_quench_amp = fault_data[0]
    else:
        fault_data = fault_data[:end_decay]
        time_data = time_data[:end_decay]
        pre_quench_amp = fault_data[0]

    saved_q_val = getattr(quench_data, "saved_q_loaded", None)
    if saved_q_val is None or (
        isinstance(saved_q_val, np.ndarray) and saved_q_val.size == 0
    ):
        print("Warning: Missing saved_q_loaded value.")
        return {
            "is_real": False,
            "loaded_q": np.nan,
            "other_issue": "missing_saved_q_loaded",
        }

    saved_loaded_q = float(np.asarray(saved_q_val).flat[0])

    try:
        with np.errstate(divide="raise", invalid="raise"):
            log_ratio = np.log(pre_quench_amp / fault_data)
            exponential_term = np.polyfit(time_data, log_ratio, 1)[0]
            loaded_q = (np.pi * frequency) / exponential_term
    except (FloatingPointError, ZeroDivisionError, TypeError) as e:
        print(f"Warning: divide-by-zero / invalid value encountered: {e}")
        return {
            "is_real": False,
            "loaded_q": np.nan,
            "other_issue": "divide_by_zero_or_invalid_value",
        }

    thresh_for_quench = LOADED_Q_CHANGE_FOR_QUENCH * saved_loaded_q
    is_real = loaded_q < thresh_for_quench

    return {"is_real": is_real, "loaded_q": loaded_q, "other_issue": other}


def plot_quench_waveforms(single_quench_data, save_name="quench_plot.png"):
    line_styles = {
        "fault_waveform": {"color": "indigo", "linestyle": "-", "linewidth": 3},
        "forward_power": {"color": "green", "marker": "o"},
        "reverse_power": {"color": "orange", "marker": "x"},
        "decay_reference": {"color": "darkcyan", "linestyle": "--", "linewidth": 3},
    }

    plt.figure(figsize=(6.5, 4))
    plt.tick_params(axis="both", which="major", labelsize=14)

    if hasattr(single_quench_data, "__dataclass_fields__"):
        data_dict = asdict(single_quench_data)
    else:
        data_dict = single_quench_data

    time_axis = data_dict.get("fault_time")

    for key, data in data_dict.items():
        if key in line_styles and data is not None and len(data) > 0:
            style = line_styles.get(key, {})
            plt.plot(time_axis, data, label=key.replace("_", " ").title(), **style)  # type: ignore

    plt.xlabel("Time (s)", size=14)
    plt.ylabel("Amplitude (MV)", size=14)
    plt.title("Cavity Quench Waveforms", size=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.xlim(-0.02, 0.06)

    save_path = os.path.join(IMG_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    print("Plot saved successfully")
    plt.close()


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.h5_load_data import load_quench_events
    from utils.config import H5_GLOB

    events_iterator = load_quench_events(H5_GLOB, load_waveforms=True)

    try:
        event_id, filename, quench_data = next(events_iterator)
        print(f"Testing event: {event_id} from {filename}")

        result = validate_quench_lisa(quench_data)
        print(f"Is Real?   {result['is_real']}")

        plot_filename = f"plot_{event_id.replace('/', '_')}.png"
        plot_quench_waveforms(quench_data, save_name=plot_filename)

    except StopIteration:
        print("Test failed:")
    except Exception:
        print("Test failed")
