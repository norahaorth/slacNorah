"""
h5_combine.py

Merge several copies of the same h5 file (each labeled by a
different person) into one h5 file with all labels combined. 

Usage:
    python h5_combine.py output.h5 data/Labeling_party/quench_data_L1_KvettaQ.h5 data/Labeling_party/quench_data_L1_Noor.h5 ...
    python h5_combine.py output.h5 data/Labeling_party/quench_data_L1_KvettaQ.h5 data/Labeling_party/quench_data_L1_Noor.h5 ...

Conflict resolution options:
    error (default) stop and report conflicts, write nothing
    first / last    keep whichever input file was listed first/last
    newest          keep whichever labeler's checked_at is most recent
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import shutil
from pathlib import Path

import h5py

from interface.quench_config import LABELS, CHECKED, NOTE, CHECKED_AT, NEEDS_SPECIALIST
from h5_reader import find_event_groups, read_event_status as read_status


def pick_winner(candidates, on_conflict):
    """Pick a winner from a list of candidates based on the conflict resolution strategy."""
    if len({status["label"] for _, status in candidates}) == 1:
        return candidates[0], False  # no conflict, all labels are the same
    """
    If we reach here, there is a conflict. Handle according to the specified strategy.
    Throw an error and ask the user to resolve manually if on_conflict is "error"
    Example on how the strategey works:
    Position	        File	        Label	    checked_at (when they labeled it)
    1st	                kvetta.h5	    false 	                2026-01-10
    2nd	                noor.h5	        real 	                2026-03-25
    3rd	                norah.h5	    false 	                2026-02-15

    If the user specifies --on-conflict first, the winner will be kvetta.h5 (the first file listed)
    If the user specifies --on-conflict last, the winner will be norah.h5 (the last file listed)
    If the user specifies --on-conflict newest, the winner will be noor.h5 (the file with the most recent checked_at date)
    """
    if on_conflict == "error":
        raise ValueError(", ".join(f"{src}={s['label']}" for src, s in candidates))
    # Take the first listed file, or the last listed file, or the newest based on checked_at
    if on_conflict == "first":
        return candidates[0], True
    if on_conflict == "last":
        return candidates[-1], True
    if on_conflict == "newest":
        return max(candidates, key=lambda item: item[1]["checked_at"] or ""), True
    
    raise ValueError(f"unknown --on-conflict value: {on_conflict!r}") 


def plan_merge(input_paths, on_conflict):
    """ 
    plan_merge:
    1. Open all files (read-only)
    2. For each event, gather everyone's opinion
    3. Pick a winner or flag conflicts based on the previous specified strategy
    4. Keep the specilaist flag anyway, regardless of who won
    5. Return a dict of decisions and a list of conflicts
    6. Close all files
    """
    readers = [h5py.File(p, "r") for p in input_paths]
    try:
        decisions, conflicts = {}, []

        for event_path in find_event_groups(readers[0]):
            candidates = [
                (p.name, read_status(r[event_path]))
                for p, r in zip(input_paths, readers)
                if event_path in r and read_status(r[event_path])["checked"]
            ]
            if not candidates:
                continue

            try:
                (winner_src, winner), had_conflict = pick_winner(candidates, on_conflict)
            except ValueError:
                conflicts.append((event_path, candidates))
                continue

            if had_conflict:
                conflicts.append((event_path, candidates))

            # Keep "needs specialist" if ANY labeler flagged it, regardless of who won.
            needs_specialist = any(s["needs_specialist"] for _, s in candidates)
            decisions[event_path] = {**winner, "needs_specialist": needs_specialist}

        return decisions, conflicts
    finally:
        for r in readers:
            r.close()


def apply_merge(output_path, base_input, decisions):
    """
    Write the merged labels to a new h5 file. Here is how it works:
    1. Copy the first input file to the output path (this preserves all the data and structure)
    2. Open the output file in append mode
    3. For each event in the decisions dict, write the label, note, checked status, checked_at, and needs_specialist attributes to the corresponding event group
    4. Close the output file
    """
    shutil.copyfile(base_input, output_path)
    with h5py.File(output_path, "a") as out_f:
        for event_path, status in decisions.items():
            group = out_f[event_path]
            group.attrs[CHECKED] = True
            group.attrs[LABELS] = status["label"]
            group.attrs[NOTE] = status["note"] or ""
            group.attrs[CHECKED_AT] = status["checked_at"] or ""
            group.attrs[NEEDS_SPECIALIST] = status["needs_specialist"]


def print_conflicts(conflicts):
    """
    Print a summary of conflicts. This is called in two cases:
    1. When the merge is aborted in error mode 
    2. After a successful merge that used first/last/newest 
    """
    print(f"\n{len(conflicts)} event(s) had conflicting labels:")
    for event_path, candidates in conflicts:
        print(f"  {event_path}: " + ", ".join(f"{src}={s['label']}" for src, s in candidates))


def main():
    # Creates an argument parser
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defining the arguments for the script
    parser.add_argument("output", help="Path to write the merged h5 file to")
    parser.add_argument("inputs", nargs="+", help="Input h5 files to merge (2 or more)")
    parser.add_argument("--on-conflict", choices=["first", "last", "newest", "error"], default="error")
    # read what the user typed and store it in args
    args = parser.parse_args()

    # Merge need at least 2 files, otherwise there's nothing to merge
    if len(args.inputs) < 2:
        parser.error("Provide at least 2 input files to merge")

    # Does the file exist? If not, print an error message and exit
    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.is_file():
            parser.error(f"Input file not found: {p}")

    # Do the planning and merging, and handle conflicts according to the specified strategy
    decisions, conflicts = plan_merge(input_paths, args.on_conflict)

    # if there are conflicts and the user specified error mode, abort the merge, print the conflicts and tell the user how to fix it 
    if conflicts and args.on_conflict == "error":
        print("Merge aborted - conflicting labels found (nothing was written).")
        print_conflicts(conflicts)
        print("\nRe-run with --on-conflict first|last|newest to resolve automatically.")
        sys.exit(1)

    # Proceed with the merge after the conflicts were resolved 
    apply_merge(args.output, input_paths[0], decisions)
    print(f"Merged {len(input_paths)} files into {args.output}") # Success message, how many files were merged and where the output file is located
    print(f"{len(decisions)} labeled event(s) written") # Success message, how many events were labeled and written to the output file
    # If there were conflicts and the user specified first/last/newest, print a summary of the conflicts that were resolved automatically
    if conflicts:
        print_conflicts(conflicts)


if __name__ == "__main__":
    main()