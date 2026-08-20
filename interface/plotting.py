import plotly.graph_objects as go
from quench_config import POWER_SIGNALS, HEADROOM, STYLES, LINE_STYLES, MARKERS


def plot_style(signal_name, x, y):
    """This function is responsible for plotting using markers, dash, line ...etc"""
    style = STYLES.get(signal_name, {})
    color = style.get("color")
    dash = LINE_STYLES.get(style.get("linestyle", "-"), "solid")  # type: ignore
    width = style.get("linewidth", 2)
    opacity = style.get("alpha", 1.0)
    marker_symbol = MARKERS.get(style.get("marker"))  # type: ignore
    markersize = style.get("markersize", 6)

    is_power = signal_name in POWER_SIGNALS
    if is_power:
        # y= y * MW_TO_KW
        yaxis = "y2"
    else:
        yaxis = "y1"

    traces = []
    if marker_symbol:
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=signal_name,
                legendgroup=signal_name,
                showlegend=False,
                opacity=opacity,
                line=dict(color=color, dash=dash, width=width),
                yaxis=yaxis,
            )
        )
        markevery = max(style.get("markevery", 1), 1)  # type: ignore
        traces.append(
            go.Scatter(
                x=x[::markevery],
                y=y[::markevery],
                mode="markers",
                name=signal_name,
                legendgroup=signal_name,
                showlegend=False,
                opacity=opacity,
                marker=dict(
                    symbol=marker_symbol,
                    size=markersize,
                    color=color,
                    line=dict(width=1, color=color),
                ),
                yaxis=yaxis,
            )
        )
        traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines+markers",
                name=signal_name,
                legendgroup=signal_name,
                showlegend=True,
                line=dict(color=color, dash=dash, width=width),
                marker=dict(
                    symbol=marker_symbol,
                    size=markersize,
                    color=color,
                    line=dict(width=1, color=color),
                ),
                yaxis=yaxis,
            )
        )
    else:
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=signal_name,
                legendgroup=signal_name,
                showlegend=True,
                opacity=opacity,
                line=dict(color=color, dash=dash, width=width),
                yaxis=yaxis,
            )
        )

    return traces


def build_figure(signal_data, title):
    """Build the full figure from the signals data."""
    fig = go.Figure()

    ordered = sorted(
        signal_data.items(), key=lambda kv: 0 if kv[0] in POWER_SIGNALS else 1
    )

    mv_max = 0
    kw_max = 0
    # finding the max for each y_axis so we can add some headroom to the plot
    for name, (x, y) in signal_data.items():
        if name in POWER_SIGNALS:
            kw_max = max(kw_max, float(max(y)))
        else:
            mv_max = max(mv_max, float(max(y)))

    for signal_name, (x, y) in ordered:
        for trace in plot_style(signal_name, x, y):
            fig.add_trace(trace)

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Time (s)",
            anchor="y",
            range=[-0.2, 0.2],
            fixedrange=False,
        ),
        yaxis=dict(
            title="Amplitude (MV)",
            side="left",
            rangemode="tozero",
            range=[0, mv_max * HEADROOM],
        ),
        yaxis2=dict(
            title="Powers (KW)",
            side="right",
            overlaying="y",
            anchor="x",
            showgrid=False,  # so it doesn't show grid twice
            rangemode="tozero",
            range=[0, kw_max * HEADROOM],
        ),
        template="plotly_white",
        width=900,
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        # dragmode="select",
        uirevision=title,
    )
    return fig


def extract_box_range(box_item):
    """Pull an (x_range, y_range) pair out of a selected part of the event"""
    if "x0" in box_item and "x1" in box_item:
        return sorted([box_item["x0"], box_item["x1"]]), sorted(
            [box_item["y0"], box_item["y1"]]
        )
    if "range" in box_item:
        r = box_item["range"]
        return sorted(r["x"]), sorted(r["y"])
    if "x" in box_item and "y" in box_item:
        return sorted(box_item["x"]), sorted(box_item["y"])
    return None, None
