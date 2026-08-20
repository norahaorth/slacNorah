import re
from interface.quench_config import LABEL_DISPLAY_TO_STORED
from datetime import datetime
from pathlib import Path


def get_cm_cav_num_from_pv(name):
    """Get cryomodule and cavity number from h5 file or PV."""
    if "/" in name or "\\" in name:
        try:
            parts = Path(name).parts
            cm_num = parts[-3].replace("CM", "")
            cav_num = parts[-2].replace("CAV", "")
            return cm_num, cav_num
        except Exception as e:
            print(f"Error parsing h5 file path: {name}. Exception: {e}")
            return None, None

    pv_parts = name.split(":")
    try:
        cav_num = pv_parts[2][2]
        cm_num = pv_parts[2][:2]
        return cm_num, cav_num
    except Exception as e:
        print(f"Error parsing PV name: {name}. Exception: {e}")
        return None, None


def parse_h5_event_path(raw_name):
    try:
        path = Path(raw_name).with_suffix("")
        basefile = path.name
        date_str, time_str = basefile.split("_")
        cm, cav = get_cm_cav_num_from_pv(raw_name)
        return cm, cav, date_str, time_str
    except (IndexError, ValueError):
        return None


def convert_pv_name_plot_string(raw_name):
    """Make a cryomodule and cavity name for plots"""
    parsed = parse_h5_event_path(raw_name)
    if parsed:
        cm, cav, date_str, time_str = parsed
        try:
            date = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            formatted_date = date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            formatted_date = f"{date_str}_{time_str}"
        return f"Cryomodule {cm}, Cavity {cav} |  {formatted_date}"

    cm, cav = get_cm_cav_num_from_pv(raw_name)

    if cm and cav:
        return f"cryomodule {cm}, cavity {cav}"

    print(f"Warning: Could not parse PV name: {raw_name}")
    return raw_name


def to_string(value):
    """Decode byte attributes to string."""
    return value.decode() if isinstance(value, bytes) else value


def normalize_label(label):
    """Normalize any label to a canonical form so case, spaces, and
    underscores never matter. 'NOT SURE', 'not_sure', 'Not Sure' all match."""
    label = to_string(label)

    if not label:
        return ""
    return str(label).strip().lower().replace(" ", "_")


def norm_cm(cm):
    digits = re.sub(r"\D", "", str(cm))
    return digits


def norm_cav(cav):
    digits = re.sub(r"\D", "", str(cav))
    return digits


def display_label(status, unlabeled="Unlabeled"):
    """Uppercased, human readable label, or unlabeled if nothing is set."""
    label = normalize_label(status["label"])
    return label.upper() if label else unlabeled


# A helper function that is responsible for the status of the file in the dropdown
def checked_status(event_path, event_status):
    """Format an event's dropdown label: name | checked | label."""
    status = event_status[event_path]
    event_name = convert_pv_name_plot_string(event_path)
    if status["checked"]:
        label = normalize_label(status["label"])
        label = label.upper() if label else "UNLABELED"
        return f"{event_name}   |   Checked: Yes    |   Label: {label}"
    return f"{event_name}   |   Checked: No |   Label: unlabeled"


def build_summary_table(display_name, checked, label, note, flag, when):
    """Build a markdown table for each event."""
    return f"""
    | **Event name**                        | {display_name}                      |
    |---------------------------------------|-------------------------------------|
    | **Checked**                           | {checked}                           |
    | **Label**                             | {label}                             |
    | **Note**                              | {note}                              |
    | **Need a specialist to inspect the cavity** | {flag}                        |
    | **Last updated**                      | {when}                              |
    """


# A function to format the event status(checked or unchecked), label(Real or False or Other) and note
def format_event_status(event_path, status):
    """Render an event's status as a markdown table."""
    checked = "Yes" if status["checked"] else "No"
    label = normalize_label(status["label"])
    label = label.upper() if label else "Unlabeled"

    note = status["note"]
    if isinstance(note, bytes):
        note = note.decode()
    note = note if note else "None"
    when = status["checked_at"] if status["checked_at"] else ""
    if isinstance(when, bytes):
        when = when.decode()

    flag = "Yes" if status["needs_specialist"] else "No"

    display_name = convert_pv_name_plot_string(event_path).replace("|", "\\|")

    return build_summary_table(display_name, checked, label, note, flag, when)


def event_matches_label(event_path, event_status, target_label):
    """This function is used for the label filter"""
    if target_label == "All":
        return True

    status = event_status[event_path]
    label = normalize_label(status["label"])

    if target_label == "Unlabeled":
        return (not status["checked"]) or (not label)

    target = LABEL_DISPLAY_TO_STORED.get(target_label.upper(), target_label)
    target = normalize_label(target)

    return label == target
