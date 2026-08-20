import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h5py

import streamlit as st

from utils.h5_reader import (
    write_label,
    # parse_event_path,
    find_event_groups,
    load_event_data_for_classification,
)

from utils.quench_data_summary import list_cryomodules, list_cavities, list_years
from plotting import build_figure
from quench_config import (
    LABEL_OPTIONS,
    LABEL_BUTTONS,
    LABELS,
    CHECKED,
    NOTE,
    CHECKED_AT,
    NEEDS_SPECIALIST,
)

# from utils.srf_waveforms import calculate_loaded_q, validate_quench_lisa
from utils.srf_waveforms import convert_pv_name_plot_string
from classification.logic import QuenchStatus, compute_suggestion
from utils.label_helpers import checked_status, format_event_status, event_matches_label


@st.cache_data(show_spinner=False)
def cached_cryomodules(path, mtime):
    with h5py.File(path, "r") as f:
        return list_cryomodules(f)


@st.cache_data(show_spinner=False)
def cached_cavities(path, mtime, cm):
    with h5py.File(path, "r") as f:
        return list_cavities(f, cm)


@st.cache_data(show_spinner=False)
def cached_years(path, mtime, cm, cav):
    with h5py.File(path, "r") as f:
        return list_years(f, cm, cav)


@st.cache_data(show_spinner=False)
def cached_events(path, mtime, cm, cav, year):
    with h5py.File(path, "r") as f:
        # TODO: Revisit once this function is merged/combined with the other one
        # not sure yet whether we'll still need this find_event_groups call or if the merged version will handle it differently.
        events = find_event_groups(f, cm=cm, cav=cav, year=year)
        event_status = {}
        for event in events:
            attrs = f[event].attrs
            event_status[event] = {
                "checked": bool(attrs.get(CHECKED, False)),
                "label": attrs.get(LABELS, None),
                "note": attrs.get(NOTE, None),
                "checked_at": attrs.get(CHECKED_AT, None),
                "needs_specialist": bool(attrs.get(NEEDS_SPECIALIST, False)),
            }
    return events, event_status


# ** File Selection **
def get_file_path():
    """Ask for the h5 file path and validate it."""
    selected_path = st.text_input(
        "Enter the full HDF5 File Path", value=""
    )  # Getting the HDF5 file path from the user

    # if nothing was entered, it will reask you to enter a path
    if not selected_path:
        st.info("Enter the full path to a HDF5 file above.")
        st.stop()

    # if entered something else otherthan a file path, it will give you an error message
    if not os.path.isfile(selected_path):
        st.error(f"File not found: {selected_path}")
        st.stop()
    return selected_path


def render_filters(selected_path, file_mtime):
    """Render the CM/CAV/year/label filters."""
    try:
        cm_options = ["All"] + cached_cryomodules(selected_path, file_mtime)
    except Exception as e:
        st.error(f"Could not open the file: {e}")
        st.stop()

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        selected_cm = st.selectbox("Cryomodule", cm_options, key="filter_cm")

    cav_options = ["All"]

    if selected_cm != "All":
        cav_options += cached_cavities(selected_path, file_mtime, selected_cm)

    with filter_col2:
        selected_cav = st.selectbox(
            "Cavity", cav_options, key="filter_cav", disabled=(selected_cm == "All")
        )

    year_options = ["All"]
    if selected_cm != "All" and selected_cav != "All":
        year_options += cached_years(
            selected_path, file_mtime, selected_cm, selected_cav
        )

    with filter_col3:
        selected_year = st.selectbox(
            "Year",
            year_options,
            key="filter_year",
            disabled=(selected_cm == "All" or selected_cav == "All"),
        )

    with filter_col4:
        selected_option = st.selectbox("Label", LABEL_OPTIONS, key="filter_label")

    cm_filter = selected_cm if selected_cm != "All" else None
    cav_filter = selected_cav if selected_cav != "All" else None
    year_filter = selected_year if selected_year != "All" else None

    if cm_filter is None:
        st.caption(
            "Narrow down by cryomodule, cavity, year and label for a faster and more focused set of events"
        )

    return cm_filter, cav_filter, year_filter, selected_option


def get_events(selected_path, file_mtime, cm_filter, cav_filter, year_filter):
    """Load events for the chosen filters."""
    try:
        events, event_status = cached_events(
            selected_path, file_mtime, cm_filter, cav_filter, year_filter
        )

    except Exception as e:
        st.error(f"Could not read events from the h5 file: {e}")
        st.stop()

    if not events:
        st.warning("No recognizable quench events found")
        st.stop()

    return events, event_status


def filter_events_by_label(events, event_status, label):
    """This function is used to filter events by label"""
    if label == "All":
        return events

    filtered = [e for e in events if event_matches_label(e, event_status, label)]

    if not filtered:
        st.warning("No events found with this label")
        st.stop()
    return filtered


# ** Event selection **
def select_waveform_event(events, event_status, filter_key):
    """Select events from the dropdown."""
    if (
        "selected_event" in st.session_state
        and st.session_state["selected_event"] in events
    ):
        default_index = events.index(st.session_state["selected_event"])
    else:
        default_index = 0

    event_path = st.selectbox(
        f"Select event ({len(events)} found)",
        events,
        index=default_index,
        format_func=lambda p: checked_status(p, event_status),
        key=f"event_selectbox_{filter_key}",
    )
    st.session_state["selected_event"] = event_path
    return event_path


def show_event_status(event_path, current_status):
    """Show the status tabel and a specilist warning if needed."""

    st.markdown(format_event_status(event_path, current_status), unsafe_allow_html=True)

    if current_status["needs_specialist"]:
        st.warning("A specialist needs to inspect the cavity")


# ** Plot **
def render_plot(signal_data, event_path):
    """Draw the plot for the given event."""
    fig = build_figure(signal_data, title=convert_pv_name_plot_string(event_path))

    st.plotly_chart(
        fig,
        use_container_width=False,
        key=f"main_chart_{event_path}",
    )


# Printing the classification
def render_suggestion(suggestion):
    """Show the classification suggestion based on QuenchStatus."""
    if suggestion is None:
        st.info("Data is unavailable")
        return

    if suggestion == QuenchStatus.cavity_off:
        st.info("The system suggests the **cavity was OFF** during this event.")
    elif suggestion == QuenchStatus.real:
        st.info("The system suggests that the given waveform is **REAL**.")
    elif suggestion == QuenchStatus.false:
        st.info("The system suggests that the given waveform is **FALSE**.")
    else:
        st.info("The system suggests that the given waveform is **OTHER**.")


def render_labeling_options(current_status, event_path):
    """
    Show the note field, specialist checkbox and the labeling buttons.
    Returns the clicked label or set as unlabled, the note and the status of the specialist checkbox.
    """

    SRF_note = st.text_area(
        "Add a note (optional), If you decide to leave it blank, a generated note will be used.",
        value="",
        key=f"note_{event_path}",
    )

    # A checkbox if there is a need for a specialist to check the cvaity in person
    needs_specialist = st.checkbox(
        "Needs specialist to inpect the cavity in person",
        value=current_status["needs_specialist"],
        key=f"specilist_{event_path}",
    )

    clicked_option = None
    # 3 colums fo the 3 options (real, false , other)
    columns = st.columns(len(LABEL_BUTTONS))

    for col, (display_text, stored_value) in zip(columns, LABEL_BUTTONS):
        with col:
            if st.button(display_text, use_container_width=True):
                clicked_option = stored_value

    return clicked_option, SRF_note, needs_specialist


def save_label(selected_path, event_path, clicked_option, SRF_note, needs_specialist):
    """Save the label back to the h5 file using write_label function."""
    try:
        write_label(
            selected_path, event_path, clicked_option, SRF_note, needs_specialist
        )
        st.success(
            f"Saved: '{event_path}' marked as **{clicked_option.upper()}** and checked."
        )
        cached_events.clear()  # clear cache so the new label shows up
        st.rerun()
    except Exception as e:
        st.error(f"Could not write label to file: {e}")


def main():
    st.set_page_config(page_title="Quench Labeler", layout="wide")
    st.title("Quench Labeling Interface")

    # ** File Selection **
    selected_path = get_file_path()
    file_mtime = os.path.getmtime(selected_path)

    # ** Filters **
    cm_filter, cav_filter, year_filter, label_option = render_filters(
        selected_path, file_mtime
    )

    # ** Load events for the chosen filters **
    events, event_status = get_events(
        selected_path, file_mtime, cm_filter, cav_filter, year_filter
    )

    # ** Apply the label filter **
    events = filter_events_by_label(events, event_status, label_option)

    # ** Event selection + status display **
    filter_key = f"{cm_filter}_{cav_filter}_{year_filter}_{label_option}"
    event_path = select_waveform_event(events, event_status, filter_key)
    current_status = event_status[event_path]
    show_event_status(event_path, current_status)

    # ** Compute Classification Suggestion **
    signal_data, frequency, saved_q_loaded = load_event_data_for_classification(
        selected_path, event_path
    )
    suggestion = compute_suggestion(signal_data, frequency, saved_q_loaded)

    # ** Plot + magnifier **
    render_plot(signal_data, event_path)

    # ** Labeling the waveform **
    st.subheader("Label this waveform")
    render_suggestion(suggestion)

    clicked_option, SRF_note, needs_specialist = render_labeling_options(
        current_status, event_path
    )

    if clicked_option:
        save_label(
            selected_path, event_path, clicked_option, SRF_note, needs_specialist
        )


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# TODO: Magnifier tool — commented out
# ---------------------------------------------------------------------------

# def render_plot(signal_data, event_path):
#     """Draw the original plot next to a magnifier preview that is used to show the specific part selected to be zoomed into."""

#     fig = build_figure(signal_data, title=convert_pv_name_plot_string(event_path))

#     st.caption("Drag a box on the plot to preview a zoomed-in view on the right side of the screen")

#     col_main, col_zoom = st.columns([2, 1])

#     with col_main:
#         select_event = st.plotly_chart(
#             fig,
#             use_container_width=False,
#             on_select="rerun",
#             selection_mode=("box",),
#             key=f"main_chart_{event_path}",
#         )

#     with col_zoom:
#         st.caption("🔍 Magnifier preview")

#         box_list = select_event.selection.get("box", []) if select_event else []
#         x_range, y_range = (None, None)
#         if box_list:
#             x_range, y_range = extract_box_range(box_list[0])

#         if x_range and y_range:
#             render_zoom_figure(fig, x_range, y_range, event_path)
#         else:
#             st.info("No selection yet. Drag a box on the plot to preview it here")


# def render_zoom_figure(fig, x_range, y_range, event_path):
#     """Draw the zommed-in part next to the original plot."""
#     zoom_fig = go.Figure(fig)
#     zoom_fig.update_layout(
#         xaxis=dict(range=x_range, title=None),
#         yaxis=dict(range=y_range, title=None),
#         margin=dict(l=10, r=10, t=10, b=10),
#         width=260,
#         height=300,
#         showlegend=False,
#         title=None,
#         dragmode=False,
#         )
#     st.plotly_chart(
#         zoom_fig,
#         use_container_width=False,
#         config={"staticPlot": True},
#         key=f"zoom_chart_{event_path}",
#     )
