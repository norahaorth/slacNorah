import os

from plotter.quench_plots import (
    box_plot_quenches_per_cavity,
    bar_quenches_per_cryo,
    bar_real_vs_false_stacked,
    bar_real_vs_false_grouped,
    pie_real_vs_false,
    scatter_total_real_false,
    bar_quenches_per_year,
    line_quenches_all_years,
    bar_quenches_per_cavity,
    bar_quenches_per_month,
    line_quenches_by_section_over_time,
)
from utils.config import IMG_DIR
from utils.config import DataBundle


def plot_data(PLOTS: dict[str, bool], data_bundle: DataBundle):
    events = data_bundle.all_events
    events_no_hl = data_bundle.events_no_hl
    real_events = data_bundle.real_events
    nomp_nohl_real_all = data_bundle.nomp_nohl_real_all

    # Box plot: real quenches per cavity, slice of cryomodules
    if PLOTS["box_real_slice_cm"]:
        cm_slice = (35, 37)
        box_plot_quenches_per_cavity(
            events,
            classification="real",
            cm_slice=cm_slice,
            save_path=os.path.join(
                IMG_DIR,
                f"real_quench_distributions_per_cryo_{cm_slice[0]}-{cm_slice[1] - 1}.png",
            ),
        )

    if PLOTS["box_all"]:
        box_plot_quenches_per_cavity(
            events_no_hl,
            classification=None,
            log=True,
            annotate_totals=True,
            compact_label=True,
            section_dividers=True,
            font_size=20,
            figsize=(22, 7),
            title="All quench distributions per cryomodule (2022-2025)",
            save_path=os.path.join(
                IMG_DIR, "box_all_quench_distributions_per_cryo_no_hl.png"
            ),
        )

    if PLOTS["box_real_all"]:
        box_plot_quenches_per_cavity(
            nomp_nohl_real_all,
            classification="real",
            # events2022, classification="real",
            log=True,
            annotate_totals=True,
            compact_label=True,
            section_dividers=True,
            font_size=20,
            figsize=(22, 7),
            title="All real quench distributions per cryomodule (2022-2025)",
            save_path=os.path.join(
                IMG_DIR, "box_all_real_quench_distributions_per_cryo_nohl_nomp.png"
            ),
        )

    # All quenches per cryomodule
    if PLOTS["bar_all_per_cryo"]:
        bar_quenches_per_cryo(
            nomp_nohl_real_all,
            section_colors=True,
            title="Number of quenches per cryomodule (2022-2025)",
            save_path=os.path.join(IMG_DIR, "bar_all_quench_counts_per_cryo_nomp.png"),
        )

    # Real and false stacked
    if PLOTS["bar_real_vs_false_stk"]:
        bar_real_vs_false_stacked(
            events_no_hl,
            title="Real vs false quenches per cryomodule (2022-2025)",
            save_path=os.path.join(IMG_DIR, "real_vs_false_quenches_stacked_no_hl.png"),
        )

    # Real and false grouped, log scale, subset 5-10
    if PLOTS["bar_real_vs_false_grp"]:
        bar_real_vs_false_grouped(
            events,  # cm_slice=(7, 12),
            log=True,
            title="Real vs false quenches per cryomodule on log scale (2022-2025)",
            save_path=os.path.join(IMG_DIR, "real_vs_false_quenches_log_scale.png"),
        )

    # Real-only bar
    if PLOTS["bar_real_per_cryo"]:
        bar_quenches_per_cryo(
            events,
            classification="real",
            section_colors=True,
            title="Real quenches per cryomodule (2022-2025)",
            save_path=os.path.join(IMG_DIR, "real_quenches_per_cryo.png"),
        )

    # False-only bar
    if PLOTS["bar_false_per_cryo"]:
        bar_quenches_per_cryo(
            events,
            classification="false",
            section_colors=True,
            title="False quenches per cryomodule (2022-2025)",
            save_path=os.path.join(IMG_DIR, "false_quenches_per_cryo.png"),
        )

    # Pie chart
    if PLOTS["pie_real_vs_false"]:
        pie_real_vs_false(
            events,
            title="Overall quench classification CM01-CM35 (2022-2025)",
            save_path=os.path.join(IMG_DIR, "real_vs_false_pie.png"),
        )

    # Scatter: total / real / false per cryomodule
    if PLOTS["scatter_totals"]:
        scatter_total_real_false(
            events,
            log=True,
            section_dividers=True,
            font_size=17,
            figsize=(18, 7),
            title="Total / real / false quenches per cryomodule (2022-2025)",
            save_path=os.path.join(IMG_DIR, "scatter_total_real_false.png"),
        )

    # One bar chart per year
    if PLOTS["bar_per_year"]:
        for year in sorted(events["year"].unique()):
            bar_quenches_per_year(
                events,
                year,
                save_path=os.path.join(IMG_DIR, f"quenches_{year}_by_cryo.png"),
            )

    # All-years line plot
    if PLOTS["line_all_years"]:
        line_quenches_all_years(
            events,
            log=True,
            font_size=17,
            figsize=(22, 7),
            save_path=os.path.join(IMG_DIR, "quenches_per_cryo_all_years.png"),
        )

    # Per-cavity bar for each cryomodule
    events2025 = real_events[real_events["year"] == "2025"]
    cm34 = events2025[events2025["cm"] == "CM34"]
    cm35 = events2025[events2025["cm"] == "CM35"]
    cavity_events = cm34
    if PLOTS["bar_per_cavity"]:
        if cavity_events.empty:
            print("Skipping bar_per_cavity: No events found for this filter.")
        else:
            years = sorted(cavity_events["year"].unique())
            yr_label = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"
            for cm in cavity_events["cm"].unique():
                bar_quenches_per_cavity(
                    cavity_events,
                    cm,
                    title=f"{cm} ({yr_label})",
                    figsize=(7, 6),
                    save_path=os.path.join(
                        IMG_DIR, f"quenches_per_cavity_{cm}_{yr_label}.png"
                    ),
                )

    # Monthly bar for one (cm, cav, year)
    if PLOTS["bar_per_month"]:
        cm, cav, year = "CM20", "CAV4", "2022"
        bar_quenches_per_month(
            real_events,
            cm=cm,
            cav=cav,
            year=year,
            save_path=os.path.join(
                IMG_DIR, f"quenches_per_month_{cm}_{cav}_{year}.png"
            ),
        )

    # Per-section line over time (L0/L1/L2/L3, HL excluded)
    if PLOTS["line_section_time"]:
        line_quenches_by_section_over_time(
            nomp_nohl_real_all,
            sections=("L0", "L1", "L2", "L3"),
            title="Real quenches per linac section over time (2022-2025)",
            save_path=os.path.join(
                IMG_DIR, "real_quenches_by_section_over_time_nohl_nomp.png"
            ),
        )
