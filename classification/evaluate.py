"""
Evaluates the accuracy of the quench classification algorithm.

This script automatically loads all data from the local `data/`
directory, runs the classification logic, and compares the predictions against
the ground-truth labels in `quench_data_L0_labeled.h5`.

Usage:
    Run this script from the root project directory using:
    $ python -m classification.evaluate
"""

import h5py
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any
from utils.config import DATA_DIR
from .data_loader import QuenchData, load_quench_events
from .logic import classify


# Runs the classification logic on all provided events
def run_classification(
    events_iterator: Iterator[Tuple[str, str, QuenchData]],
) -> Dict[str, Tuple[Any, str]]:
    classification_results = {}

    for event_id, filename, event_data in events_iterator:
        label = classify(event_data)
        classification_results[event_id] = (label, filename)

    return classification_results


# Compares your predicted labels against the true labels and prints any mismatches
def compare_classification(
    predictions: Dict[str, Tuple[Any, str]], ground_truth_file: Path
) -> None:
    correct = 0
    total = 0

    with h5py.File(ground_truth_file, "r") as f:
        for event_id, (predicted_enum, source_file) in predictions.items():
            if event_id in f:
                event_group = f[event_id]
                true_label = event_group.attrs.get("quench_labels")

                if true_label is None:
                    continue

                if isinstance(true_label, bytes):
                    true_label = true_label.decode("utf-8")

                if str(true_label).strip().lower() == "not_sure":
                    continue

                total += 1

                predicted_val = (
                    predicted_enum.value
                    if hasattr(predicted_enum, "value")
                    else str(predicted_enum)
                )

                if predicted_val.strip().upper() == str(true_label).strip().upper():
                    correct += 1
                else:
                    print(
                        f"Mismatch in {source_file:22} | Event: {event_id:32} | Predicted: {predicted_val.lower():5} | Actual: {str(true_label).lower():5}"
                    )

    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\nResults: {correct}/{total} correct ({accuracy:.2f}%)")
    else:
        print("\nWarning: No matching labeled events found.")


# Loads data, runs classification, and checks the accuracy
def main() -> None:

    target_files = "quench_data_L[0-9].h5"
    events_iterator = load_quench_events(target_files)
    prediction_results = run_classification(events_iterator)
    labeled_file_path = (
        Path(DATA_DIR) / "quench_data_L0.h5"
    )  # File path of labeled data to be used for comparison
    compare_classification(prediction_results, labeled_file_path)


if __name__ == "__main__":
    main()
