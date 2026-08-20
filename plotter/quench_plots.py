import matplotlib.pyplot as plt
import numpy as np

"""
Questions answered with these plots:
    (1) Which cryomodule quenched the most?
    (2) How many real quenches per cryomodule?
    (3) How many false quenches per cryomodule?
    (4) How many quenches per year?
    (5) Which cavity quenched the most?
"""

DEFAULT_FONT = 20
DEFAULT_FIGSIZE = (14, 6)

# Common axis labels reused across most plots. Pass an explicit string to
# `_style` to override; pass None to fall back to these defaults.
XLABEL_CM = "Cryomodule number"
YLABEL_COUNT = "Number of quenches"

REAL_COLOR = "#009E73"  # Okabe-Ito green
FALSE_COLOR = "#8C1515"  # Stanford cardinal red
BAR_COLOR = "#009E73"  # "#0072B2"  # Okabe-Ito blue (generic single-color bars)

# Linac sections (used for grouping/coloring CMs in plots).
SECTIONS = [
    ("L0", ["CM01"], "#0072B2"),
    ("L1", ["CM02", "CM03"], "#009E73"),
    ("HL", ["CMH1", "CMH2"], "#D55E00"),
    ("L2", [f"CM{n:02d}" for n in range(4, 16)], "#AA4499"),
    ("L3", [f"CM{n:02d}" for n in range(16, 36)], "#E69F00"),
]


def _section_for(cm):
    for name, members, color in SECTIONS:
        if cm in members:
            return name, color
    return None, "#888888"


def section_colors_for(cms):
    """Per-CM color list using the SECTIONS table."""
    return [_section_for(cm)[1] for cm in cms]


def add_section_decorations(ax, cms, font_size=DEFAULT_FONT, x_offset=1):
    """Draw faint vertical separators between linac sections plus a legend.

    `x_offset` is the x-coordinate of the first item: 1 for boxplot
    (1-based positions), 0 for bar/line plots (0-based).
    """
    secs = [_section_for(cm)[0] for cm in cms]
    for i in range(1, len(secs)):
        if secs[i] != secs[i - 1]:
            ax.axvline(i + x_offset - 0.5, color="gray", linewidth=0.8, alpha=0.45)
    seen = []
    for sec, cm in zip(secs, cms):
        if sec and sec not in [s for s, _ in seen]:
            seen.append((sec, _section_for(cm)[1]))
    if seen:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=c, edgecolor="#222222", alpha=0.65, label=s)
            for s, c in seen
        ]
        existing = ax.get_legend()
        section_legend = ax.legend(
            handles=handles,
            title="Linac section",
            loc="upper left",
            fontsize=font_size - 2,
            title_fontsize=font_size - 2,
            framealpha=0.9,
        )
        # ax.legend() displaces any prior legend; restore it as a separate artist.
        if existing is not None:
            ax.add_artist(section_legend)
            ax.add_artist(existing)


def add_section_dividers(ax, cms, font_size=DEFAULT_FONT, x_offset=1):
    """Faint vertical separators between linac sections plus a label
    centered above each section, at the top of the axes.

    Use this when you do not want boxes/bars themselves colored by section.
    `x_offset` is the x-coordinate of the first item: 1 for boxplot, 0 for bar/line.
    """
    secs = [_section_for(cm)[0] for cm in cms]
    boundaries = [0]
    for i in range(1, len(secs)):
        if secs[i] != secs[i - 1]:
            ax.axvline(i + x_offset - 0.5, color="gray", linewidth=2.0, alpha=0.6)
            boundaries.append(i)
    boundaries.append(len(secs))
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        name = secs[start]
        if not name:
            continue
        center = (start + stop - 1) / 2 + x_offset
        # Place labels just inside the top of the plot (axes-fraction y=0.97).
        ax.text(
            center,
            0.97,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=font_size,
            fontweight="bold",
            color="#333333",
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _filter_class(events, classification):
    """classification: 'real' or 'false'. Pass the events frame directly
    if no filtering is wanted."""
    if classification == "real":
        return events[events["is_real"].astype(bool)]
    if classification == "false":
        return events[~events["is_real"].astype(bool)]
    raise ValueError(
        f"classification must be 'real' or 'false' (got {classification!r})"
    )


_CLASS_LABEL = {None: "", "real": "Real ", "false": "False "}
_CLASS_COLOR = {None: BAR_COLOR, "real": REAL_COLOR, "false": FALSE_COLOR}


def _annotate_bars(ax, bars, offset=10, fontsize=8):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            str(int(h)),
            ha="center",
            fontsize=fontsize,
        )


def _real_false_by_cm(events):
    """Return (cms, real_counts, false_counts) aligned to a stable CM order."""
    real = events[events["is_real"].astype(bool)].groupby("cm", observed=True).size()
    false = events[~events["is_real"].astype(bool)].groupby("cm", observed=True).size()
    cms = list(real.index.union(false.index, sort=False))
    r = real.reindex(cms, fill_value=0).values
    f = false.reindex(cms, fill_value=0).values
    return cms, r, f


def _clean_grid(ax, font_size):
    """Horizontal-only grid plus consistent tick sizing."""
    ax.grid(True, axis="y", which="both", alpha=0.35)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", labelsize=font_size - 2)


def _style(
    ax, xlabel, ylabel, title, font_size, xticks=None, xticklabels=None, rotation=90
):
    ax.set_xlabel(xlabel if xlabel is not None else XLABEL_CM, fontsize=font_size)
    ax.set_ylabel(ylabel if ylabel is not None else YLABEL_COUNT, fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        # Anchor rotated labels at their right end so the end of the
        # label sits under the tick mark.
        if rotation and rotation % 180 != 0:
            ax.set_xticklabels(
                xticklabels, rotation=rotation, ha="right", rotation_mode="anchor"
            )
        else:
            ax.set_xticklabels(xticklabels, rotation=rotation)
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #


def box_plot_quenches_per_cavity(
    events,
    classification=None,
    cm_slice=None,
    ylim=None,
    title=None,
    annotate_totals=False,
    log=False,
    section_dividers=False,
    compact_label=False,
    font_size=DEFAULT_FONT,
    figsize=(8, 6),
    save_path=None,
    show=False,
):
    """Box plot of per-cavity quench counts, one box per cryomodule.

    `cm_slice` is a (start, stop) tuple to plot a subset of cryomodules
    in the order of ``events['cm']`` (e.g. (3, 8)). None = all.
    `classification` is 'real', 'false', or None for no filtering.
    `ylim` may be None for autoscale based on the data extent (with a
    small margin); pass an explicit (lo, hi) tuple to override.
    `annotate_totals` adds the per-CM total to each x-tick label.
    `compact_label` puts the total on the same line as the CM name.
    `log` sets the y-axis to log scale.
    `section_dividers` draws faint vertical lines between linac sections
    (L0/L1/HL/L2/L3) and labels each section at the top of the plot.
    """
    sub = _filter_class(events, classification) if classification else events
    counts = sub.groupby(["cm", "cav"], observed=True).size().unstack(fill_value=0)
    cms = counts.index.tolist()  # preserves Categorical order
    if cm_slice is not None:
        cms = cms[cm_slice[0] : cm_slice[1]]
    data = [counts.loc[cm].values for cm in cms]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.65,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="#444444", linewidth=1.0),
        capprops=dict(color="#444444", linewidth=1.0),
        flierprops=dict(
            marker="o",
            markersize=3.5,
            markerfacecolor="#444444",
            markeredgecolor="none",
            alpha=0.5,
        ),
    )

    fill = "#8C1515"  # cardinal red
    for patch in bp["boxes"]:
        patch.set_facecolor(fill)
        patch.set_edgecolor("#222222")
        patch.set_alpha(0.30)
    if section_dividers:
        add_section_dividers(ax, cms, font_size=font_size, x_offset=1)

    if log:
        ax.set_yscale("log")
    elif ylim is not None:
        ax.set_ylim(*ylim)
    else:
        # Auto-fit to the data with ~5% headroom and a small negative
        # margin so boxes near zero aren't clipped at the axis.
        data_max = max((arr.max() for arr in data if len(arr)), default=0)
        pad = max(data_max * 0.05, 1)
        ax.set_ylim(-pad, data_max + pad)

    if annotate_totals:
        sep = " " if compact_label else "\n"
        labels = [f"{cm}{sep}(n={int(sum(d))})" for cm, d in zip(cms, data)]
    else:
        labels = cms

    label = _CLASS_LABEL[classification]
    _style(
        ax,
        None,
        f"Number of {label}quenches",
        title or f"{label}quench distributions per cryomodule",
        font_size,
        xticks=np.arange(1, len(cms) + 1),
        xticklabels=labels,
        rotation=45,
    )
    _clean_grid(ax, font_size)
    _finish(fig, save_path, show)


def bar_quenches_per_cryo(
    events,
    classification=None,
    title=None,
    section_colors=False,
    font_size=DEFAULT_FONT,
    figsize=DEFAULT_FIGSIZE,
    save_path=None,
    show=False,
):
    """Bar chart: total quenches per cryomodule.

    `classification` is 'real', 'false', or None for no filtering.
    `section_colors` colors bars by linac section and adds separators/legend.
    """
    sub = _filter_class(events, classification) if classification else events
    counts = sub.groupby("cm", observed=True).size().sort_index()
    cms = counts.index.tolist()
    bar_color = (
        section_colors_for(cms) if section_colors else _CLASS_COLOR[classification]
    )
    label = _CLASS_LABEL[classification]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(cms, counts.values, color=bar_color)
    _annotate_bars(ax, bars, offset=max(counts.values) * 0.01 + 1)
    if section_colors:
        add_section_decorations(ax, cms, font_size=font_size, x_offset=0)
    _style(
        ax,
        None,
        f"Number of {label}Quenches",
        title or f"{label}quenches per cryomodule",
        font_size,
        xticks=np.arange(len(counts)),
        xticklabels=cms,
    )
    _finish(fig, save_path, show)


def bar_real_vs_false_stacked(
    events,
    title="Real vs False Quenches per Cryomodule",
    font_size=DEFAULT_FONT,
    figsize=(20, 8),
    save_path=None,
    show=False,
):
    """Stacked bar: real (bottom) + false (top) per cryomodule."""
    cms, r, f = _real_false_by_cm(events)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        cms, r, label="Real quenches", color=REAL_COLOR, edgecolor="grey", linewidth=0.6
    )
    ax.bar(
        cms,
        f,
        bottom=r,
        label="False quenches",
        color=FALSE_COLOR,
        hatch="//",
        edgecolor="grey",
        linewidth=0.6,
    )
    _style(
        ax, None, None, title, font_size, xticks=np.arange(len(cms)), xticklabels=cms
    )
    ax.set_xlim(-0.6, len(cms) - 0.4)
    ax.tick_params(axis="both", labelsize=14)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(14)
    ax.legend(fontsize=20)
    _finish(fig, save_path, show)


def scatter_total_real_false(
    events,
    title="Total / Real / False Quenches per Cryomodule",
    log=False,
    section_dividers=False,
    font_size=DEFAULT_FONT,
    figsize=DEFAULT_FIGSIZE,
    save_path=None,
    show=False,
):
    """Scatter with three overlaid series (total, real, false) per CM.

    `log` sets the y-axis to log scale.
    `section_dividers` adds linac-section vertical lines + top labels.
    """
    cms, r, f = _real_false_by_cm(events)
    total = r + f
    x = np.arange(len(cms))

    # Colors match the pie chart (Okabe-Ito green, Stanford cardinal red);
    # distinct shapes + white edges keep series readable when overlapping.
    series = [
        ("Total", total, "#000000", "o", 140),  # circle
        ("Real", r, "#009E73", "D", 110),  # diamond, green
        ("False", f, "#8C1515", "^", 120),  # triangle, cardinal red
    ]
    fig, ax = plt.subplots(figsize=figsize)
    for label, y, color, marker, size in series:
        # Connecting line makes the per-series trend easy to follow.
        ax.plot(x, y, color=color, linewidth=1.8, alpha=0.5, zorder=2)
        ax.scatter(
            x,
            y,
            label=label,
            color=color,
            marker=marker,
            s=size,
            edgecolor="white",
            linewidth=1.4,
            zorder=3,
        )
    if log:
        ax.set_yscale("log")
    ax.legend(
        loc="center right",
        fontsize=font_size - 2,
        title_fontsize=font_size - 2,
        framealpha=0.9,
    )
    if section_dividers:
        add_section_dividers(ax, cms, font_size=font_size, x_offset=0)
    _style(ax, None, None, title, font_size, xticks=x, xticklabels=cms, rotation=45)
    _clean_grid(ax, font_size)
    _finish(fig, save_path, show)


def bar_real_vs_false_grouped(
    events,
    cm_slice=None,
    log=False,
    title=None,
    font_size=DEFAULT_FONT,
    figsize=(15, 7),
    save_path=None,
    show=False,
):
    """Side-by-side bars of real and false quenches per cryomodule."""
    cms, r, f = _real_false_by_cm(events)
    if cm_slice is not None:
        cms = cms[cm_slice[0] : cm_slice[1]]
        r = r[cm_slice[0] : cm_slice[1]]
        f = f[cm_slice[0] : cm_slice[1]]

    x = np.arange(len(cms))
    w = 0.4
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w / 2, r, width=w, label="Real Quenches", color=REAL_COLOR)
    ax.bar(x + w / 2, f, width=w, label="False Quenches", color=FALSE_COLOR)
    if log:
        ax.set_yscale("log")
    suffix = " (Log Scale)" if log else ""
    _style(
        ax,
        None,
        None,
        title or f"Real vs False Quenches per Cryomodule{suffix}",
        font_size,
        xticks=x,
        xticklabels=cms,
    )
    ax.legend()
    _finish(fig, save_path, show)


def pie_real_vs_false(
    events,
    title="Overall Quench Classification",
    font_size=DEFAULT_FONT,
    figsize=(6, 6),
    save_path=None,
    show=False,
):
    """Pie chart of real vs false quenches across the whole input."""
    real = int(events["is_real"].astype(bool).sum())
    false = int((~events["is_real"].astype(bool)).sum())
    print(f"Real quenches: {real}, False quenches: {false}, Total: {real + false}")
    names = ["Real", "False"]
    fig, ax = plt.subplots(figsize=figsize)
    # autopct receives the slice's percentage; pair it with the matching name.
    name_iter = iter(names)

    def _label(pct):
        return f"{pct:.1f}%\n{next(name_iter)}"

    wedges, texts, autotexts = ax.pie(  # type: ignore
        [real, false],
        labels=None,
        # Pie-specific palette: CB-safe green (Okabe-Ito) for real,
        # Stanford cardinal red for false.
        colors=["#009E73", "#8C1515"],
        autopct=_label,
        startangle=90,
        textprops={"fontsize": font_size + 4, "color": "white"},
    )
    # Hatch only the false wedge so the two are still distinguishable in B&W.
    wedges[1].set_hatch("//")
    wedges[0].set_edgecolor("white")
    wedges[0].set_linewidth(1.5)
    wedges[1].set_edgecolor("grey")
    wedges[1].set_linewidth(1.5)
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_ha("center")  # type: ignore
    # Two-line title rendered close to the pie.
    if "\n" not in title:
        words = title.split()
        mid = len(words) // 2 + 1
        title = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    ax.set_title(title, fontsize=font_size, pad=6)
    ax.axis("equal")
    _finish(fig, save_path, show)


def bar_quenches_per_year(
    events,
    year,
    title=None,
    font_size=DEFAULT_FONT,
    figsize=DEFAULT_FIGSIZE,
    save_path=None,
    show=False,
):
    """Bar chart of quench counts per cryomodule for a single year."""
    sub = events[events["year"] == str(year)]
    counts = sub.groupby("cm", observed=True).size().sort_index()
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index, counts.values, color=BAR_COLOR)
    _annotate_bars(ax, bars, offset=max(counts.values) * 0.01 + 1)
    _style(
        ax,
        None,
        None,
        title or f"Number of quenches in {year} by cryomodule",
        font_size,
        xticks=np.arange(len(counts)),
        xticklabels=counts.index.tolist(),
    )
    _finish(fig, save_path, show)


def line_quenches_all_years(
    events,
    title="Number of Quenches per Cryomodule (All Years)",
    font_size=DEFAULT_FONT,
    figsize=DEFAULT_FIGSIZE,
    ylim=None,
    log=False,
    save_path=None,
    show=False,
):
    """One line per year showing quench counts vs cryomodule."""
    pivot = (
        events.groupby(["cm", "year"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    cms = pivot.index.tolist()
    years = sorted(pivot.columns)
    colors = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#984ea3",
        "#a65628",
        "#999999",
    ]

    fig, ax = plt.subplots(figsize=figsize)
    for i, yr in enumerate(years):
        ax.plot(
            cms,
            pivot[yr].values,
            label=str(yr),
            color=colors[i % len(colors)],
            marker="o",
            markersize=6,
            linewidth=2,
            alpha=0.85,
        )
    if log:
        ax.set_yscale("log")
    elif ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(
        title="Year",
        loc="upper left",
        fontsize=font_size - 2,
        title_fontsize=font_size - 2,
        framealpha=0.9,
    )
    _style(
        ax,
        None,
        None,
        title,
        font_size,
        xticks=np.arange(len(cms)),
        xticklabels=cms,
        rotation=45,
    )
    _clean_grid(ax, font_size)
    _finish(fig, save_path, show)


def bar_quenches_per_cavity(
    events,
    cm,
    title=None,
    font_size=DEFAULT_FONT,
    figsize=DEFAULT_FIGSIZE,
    save_path=None,
    show=False,
):
    """Bar chart of quench counts per cavity for a single cryomodule."""
    sub = events[events["cm"] == cm]
    counts = sub.groupby("cav", observed=True).size().sort_index()
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index, counts.values, color=BAR_COLOR)
    _annotate_bars(ax, bars, offset=max(counts.values) * 0.01 + 1, fontsize=16)
    _style(
        ax,
        "Cavity number",
        None,
        title or f"Number of quenches per cavity in {cm}",
        font_size,
        xticks=np.arange(len(counts)),
        xticklabels=counts.index.tolist(),
    )
    _finish(fig, save_path, show)


_MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def bar_quenches_per_month(
    events,
    cm="CM20",
    cav="CAV1",
    year="2022",
    title=None,
    font_size=DEFAULT_FONT,
    figsize=DEFAULT_FIGSIZE,
    save_path=None,
    show=False,
):
    """Bar chart of monthly quench counts for a single cavity in a given year."""
    sub = events[
        (events["cm"] == cm) & (events["cav"] == cav) & (events["year"] == str(year))
    ]
    counts = (
        sub.groupby("month", observed=True)
        .size()
        .reindex([f"{m:02d}" for m in range(1, 13)], fill_value=0)
    )
    x = np.arange(1, 13)
    y = counts.values

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, y, color=BAR_COLOR, edgecolor="white", linewidth=0.8)
    if y.max() > 0:
        _annotate_bars(ax, bars, offset=max(y) * 0.01 + 0.5, fontsize=font_size - 4)
    _style(
        ax,
        "Month",
        None,
        title or f"Quenches per Month in {cm} {cav} ({year})",
        font_size,
        xticks=x,
        xticklabels=_MONTHS,
        rotation=0,
    )
    _clean_grid(ax, font_size)
    _finish(fig, save_path, show)


def line_quenches_by_section_over_time(
    events,
    sections=("L0", "L1", "L2", "L3"),
    classification="real",
    title="Quenches per linac section over time",
    log=False,
    font_size=DEFAULT_FONT,
    figsize=(9, 6),
    save_path=None,
    show=False,
):
    """Time series of quench counts per linac section.

    One line per entry in ``sections`` (default L0/L1/L2/L3 — HL excluded).
    X-axis is year (one point per year). `log=True` uses a log y-axis.
    `classification` is 'real', 'false', or None for no filtering.
    """
    sub = _filter_class(events, classification) if classification else events
    # Map each row to its section name; drop rows whose CM is in an
    # unrequested section (e.g. HL when only L0/L1/L2/L3 are wanted).
    section_of = {cm: name for name, members, _ in SECTIONS for cm in members}
    color_of = {name: c for name, _, c in SECTIONS}
    sub = sub.assign(section=sub["cm"].map(section_of))
    sub = sub[sub["section"].isin(sections)]

    counts = (
        sub.groupby([sub["year"].astype(int), "section"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    counts.index.name = "year"
    # Reindex to the full year range so gaps show as zeros, not jumps.
    full_idx = range(int(counts.index.min()), int(counts.index.max()) + 1)
    counts = counts.reindex(full_idx, fill_value=0)
    x = list(counts.index)

    section_markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    section_marker = {
        name: section_markers[i % len(section_markers)]
        for i, name in enumerate(sections)
    }

    fig, ax = plt.subplots(figsize=figsize)
    for name in sections:
        if name not in counts.columns:
            continue
        y = counts[name].values
        color = color_of[name]
        ax.plot(
            x,
            y,
            label=name,
            color=color,
            marker=section_marker[name],
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.0,
            linewidth=2.2,
            alpha=0.9,
            zorder=2,
        )
    if log:
        ax.set_yscale("log")
    ax.legend(
        loc="upper right",
        fontsize=font_size - 2,
        title_fontsize=font_size - 2,
        framealpha=0.9,
    )
    _style(
        ax,
        "Year",
        None,
        title,
        font_size,
        xticks=x,
        xticklabels=[str(y) for y in x],
        rotation=0,
    )
    _clean_grid(ax, font_size)
    _finish(fig, save_path, show)
