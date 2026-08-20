import os
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

H5_GLOB = os.path.join(ROOT, "data", "quench_data_L*.h5")
DATA_DIR: str = os.path.join(ROOT, "data")
IMG_DIR = os.path.join(ROOT, "images")

os.makedirs(IMG_DIR, exist_ok=True)


@dataclass
class QuenchData:
    fault_time: NDArray[np.float64]
    fault_waveform: NDArray[np.float64]
    forward_power: NDArray[np.float64]
    forward_time: NDArray[np.float64]
    reverse_power: NDArray[np.float64]
    reverse_time: NDArray[np.float64]
    decay_reference: Optional[NDArray[np.float64]] = None
    frequency: float = 1300000000.0
    saved_q_loaded: float = 40000000.0
    quench_classification: str = "real"
    is_mp: bool = False


@dataclass
class DataBundle:
    all_events: pd.DataFrame
    events_no_hl: pd.DataFrame
    real_events: pd.DataFrame
    nomp_nohl_real_all: pd.DataFrame
