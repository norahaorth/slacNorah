import numpy as np
from enum import Enum
from utils.h5_load_data import QuenchData
import streamlit as st


class QuenchStatus(Enum):
    real = "real"
    false = "false"
    other = "other"
    cavity_off = "cavity_off"


# Locates the array index corresponding to the onset of the quench event at time zero
def find_quench_time(quench_event_data: QuenchData) -> int:
    return int(np.searchsorted(quench_event_data.fault_time, 0.0))


# Verifies that the overall average of the entire fault waveform is greater than the threshold
def is_overall_average_sufficient(quench_event_data: QuenchData) -> bool:
    return bool(np.mean(quench_event_data.fault_waveform) > 0.1)


# Evaluates the pre-quench window to confirm the cavity is on
def pre_quench_amplitude(quench_event_data: QuenchData, time_0: int) -> bool:
    avg_waveform = np.mean(quench_event_data.fault_waveform[:time_0])
    avg_fwd_power = np.mean(quench_event_data.forward_power[:time_0])

    total_avg = (avg_waveform + avg_fwd_power) / 2.0

    return bool((avg_waveform >= 0.1) and (avg_fwd_power > 0.01) and (total_avg > 0.2))


# Calculates measured decay time and expected theoretical decay constant
def calculate_decay_metrics(
    quench_event_data: QuenchData, time_0: int
) -> tuple[float, float]:
    decay_waveform = quench_event_data.fault_waveform[time_0:]

    if len(decay_waveform) == 0:
        return -1.0, 1.0

    target_1 = decay_waveform[0] / np.e
    idx_1 = np.searchsorted(-decay_waveform, -target_1)

    if idx_1 >= len(decay_waveform):
        return -1.0, 1.0

    # Calculate t1 using absolute indices rather than creating a new decay_time slice
    t1 = (
        quench_event_data.fault_time[time_0 + idx_1]
        - quench_event_data.fault_time[time_0]
    )

    expected_tau = quench_event_data.saved_q_loaded / (
        np.pi * quench_event_data.frequency
    )

    return float(t1), float(expected_tau)


# Determines the operational status of the quench event
def classify(event_data: QuenchData) -> QuenchStatus:
    if not is_overall_average_sufficient(event_data):
        return QuenchStatus.cavity_off

    time_0 = find_quench_time(event_data)

    if not pre_quench_amplitude(event_data, time_0):
        return QuenchStatus.cavity_off

    t1, expected_tau = calculate_decay_metrics(event_data, time_0)

    if t1 < 0:
        return QuenchStatus.other

    if t1 < 0.60 * expected_tau:
        return QuenchStatus.real

    if t1 >= 0.60 * expected_tau:
        return QuenchStatus.false

    return QuenchStatus.other


def compute_suggestion(signal_data, frequency, saved_q_loaded):
    """Compute the classification suggestion using the classify system written by Norah"""

    # If there is no fault_waveform, we are unable to classify
    if "fault_waveform" not in signal_data:
        return None

    x_fault, y_fault = signal_data[
        "fault_waveform"
    ]  # Split the fault_waveform (time, amplitude) tuple into two separate arrays

    # If the forward_power is missing, we can't run the classifier
    if "forward_power" not in signal_data:
        return None
    x_fwd, y_fwd = signal_data[
        "forward_power"
    ]  # Split the forward_power (time, amplitude) tuple into two separate arrays

    # reverse_power may or may not exist, if missing assign none to the time and amplitude
    x_rev, y_rev = signal_data.get("reverse_power", (None, None))

    try:
        # Build the QuenchData object
        # Convert every array into float for safer math calculations
        quench_event = QuenchData(
            fault_time=np.asarray(x_fault, dtype=float),
            fault_waveform=np.asarray(y_fault, dtype=float),
            forward_power=np.asarray(y_fwd, dtype=float),
            forward_time=np.asarray(x_fwd, dtype=float),
            reverse_power=np.asarray(y_rev, dtype=float)
            if y_rev is not None
            else np.array(
                []
            ),  # Reverse power amplitude if available, else an empty array
            reverse_time=np.asarray(x_rev, dtype=float)
            if x_rev is not None
            else np.array([]),  # Reverse time if available, else an empty array
        )

        if frequency is not None:
            # Convert frequency into numpy no matter what type of data it came in
            quench_event.frequency = float(np.asarray(frequency).flat[0])
        if saved_q_loaded is not None:
            # Convert saved_q_loaded into numpy no matter what type of data it came in
            quench_event.saved_q_loaded = float(np.asarray(saved_q_loaded).flat[0])

        return classify(
            quench_event
        )  # Calls classify function which returns a QuenchStatus [real, false, other or cavoty off]
    except Exception as e:
        st.error(f"Classification suggestion has failed: {e}")
        return None
