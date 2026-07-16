# Import packages
import matplotlib.pyplot as plt
from shiny import ui
import numpy as np

from helpers.filtering_functions import get_active_channel_types, get_channel_groups, prepare_filtered_data

# Format elapsed seconds as hours, minutes, and seconds
def format_time(x, pos):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = int(x % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# Round numeric filter inputs to two decimals
def round_inputs(data_org, channel_mode, channels, input, ui, app_channel_types=None):
    if data_org is None:
        return
    for ch_type in get_active_channel_types(data_org, channel_mode, channels, app_channel_types):
        for suffix in ["low", "high", "scaling"]:
            input_id = f"{suffix}_{ch_type}"
            val = input[input_id]()
            if val is None:
                continue
            if isinstance(val, (int, float)):
                val_rounded = round(val, 2)
                if val != val_rounded:
                    ui.update_numeric(input_id, value=val_rounded)

# Move the displayed time window forward or backward
def move_window(data, size, current_w_st, direction):
    if data is None:
        return
    max_time_st = max(0, data.times[-1] - size)
    if direction == 'forward':
        w_st = min(max_time_st, current_w_st + size)
        w_st = max(w_st, 0)
    elif direction == 'previous':
        w_st = max(0, current_w_st - size)
        w_st = min(w_st, max_time_st)
    return w_st

# Create filter and scaling controls for active channel types
def channel_type_filters_scaling_function_ui(data, channel_mode, channels, saved_settings=None, app_channel_types=None):
    if data is None:
        return None
    channel_types = get_active_channel_types(
        data,
        channel_mode,
        channels,
        app_channel_types,
    )
    boxes = []
    for ch_type in channel_types:
        ch_settings = (saved_settings or {}).get(ch_type, {})
        selected_notch = str(ch_settings.get("notch", "50"))
        label = ch_type.replace("_", " ").upper()
        boxes.append(
            ui.card(
                ui.h5(f"{label} filters and scaling:"),
                ui.input_numeric(f"low_{ch_type}", f"Low cut-off (high pass) frequency (Hz):", value=ch_settings.get("low", 1 if ch_type == "eeg" else 0.3), min=0, step=0.1),
                ui.input_numeric(f"high_{ch_type}", f"High cut-off (low pass) frequency (Hz):", value=ch_settings.get("high", 100 if ch_type == "eeg" else 200), min=0, step=0.1),
                ui.input_select(f"notch_{ch_type}", f"Notch filter frequency (Hz):", choices=["None", "50", "60"], selected=selected_notch),
                ui.input_numeric(f"scaling_{ch_type}", f"Scaling factor (μV):", value=ch_settings.get("scaling", 200000 if ch_type == "body_position" else 40 if ch_type == "eeg" else 400 if ch_type == "ecg" else 100), min=0.01, step=0.01),
            )
        )
    return ui.div(*boxes)

# Restore previously selected filter settings
def restore_filter_settings(settings, ui):
    if not settings:
        return
    for ch_type, vals in settings.items():
        if "low" in vals:
            ui.update_numeric(f"low_{ch_type}", value=vals["low"])
        if "high" in vals:
            ui.update_numeric(f"high_{ch_type}", value=vals["high"])
        if "scaling" in vals:
            ui.update_numeric(f"scaling_{ch_type}", value=vals["scaling"])
        if "notch" in vals:
            ui.update_select(f"notch_{ch_type}", selected=str(vals["notch"]))

# Plot stacked time series for all displayed channels
def plot_layout(data_plot, w_start, input, show_bad_annotations=False, app_channel_types=None):
    if data_plot is None:
        return None
    channel_names = data_plot.ch_names
    ch_types = get_channel_groups(data_plot, app_channel_types)
    n_channels = len(channel_names)
    order = sorted(range(n_channels), key=lambda i: ch_types[i] != "eeg")[::-1]
    ordered_names = [channel_names[i] for i in order]
    unique_types = list(dict.fromkeys(ch_types))
    cmap = plt.get_cmap("tab10")
    type_to_color = {"eeg": "black"}
    next_color = 0
    for ch_type in unique_types:
        if ch_type == "eeg":
            continue
        type_to_color[ch_type] = cmap(next_color)
        next_color += 1
    spacing = 80
    offset = 0
    data, times = data_plot[:]
    times = times + w_start

    fig, ax = plt.subplots()
    yticks = []
    for i in order:
        ch_type = ch_types[i]
        scale = getattr(input, f"scaling_{ch_type}")()
        signal = data[i] * 1e6
        signal = signal - signal.mean()
        signal = (signal / scale) * spacing
        ax.plot(times, signal + offset, color=type_to_color[ch_type], linewidth=0.8)
        yticks.append(offset)
        offset += spacing
    ax.set_xlim(times[0], times[-1])
    if show_bad_annotations:
        window_start = times[0]
        window_end = times[-1]
        first_time = getattr(data_plot, "first_time", 0) or 0
        for ann in data_plot.annotations:
            if not str(ann["description"]).upper().startswith("BAD"):
                continue
            ann_start = ann["onset"]
            if first_time > 0 and first_time <= ann_start <= first_time + data_plot.times[-1]:
                ann_start = ann_start - first_time + w_start
            ann_end = ann["onset"] + ann["duration"]
            if first_time > 0 and first_time <= ann_end <= first_time + data_plot.times[-1]:
                ann_end = ann_end - first_time + w_start
            span_start = max(window_start, ann_start)
            span_end = min(window_end, ann_end)
            if span_end > span_start:
                ax.axvspan(span_start, span_end, color="red", alpha=0.08, linewidth=0, zorder=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    ax.set_xlabel("Time since recording start (HH:MM:SS)")
    ax.margins(x=0, y=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ordered_names)
    bad_channel_names = set(data_plot.info.get("bads", []))
    for label in ax.get_yticklabels():
        if label.get_text() in bad_channel_names:
            label.set_color("red")
    ax.set_ylim(-20, (n_channels - 1) * spacing + 20)
    return fig

# Create a plot containing an error message
def error_plot(message):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, color="crimson")
    return fig

# Calculate the required time-series plot height
def height_ts_plot(data, ch_mode, channels):
    if data is None:
        return 200
    if ch_mode == "Custom set":
        n_channels = len(channels or [])
    else:
        n_channels = len(data.ch_names)
    return max(200, n_channels * 15)

# Prepare data and build the time-series plot
def show_ts_plot(input, step, data, w_start, w_size, channel_mode, channels, reference, single_channel, avg_channels, filter_settings, show_bad_annotations=False, app_channel_types=None):
    if data is None:
        return None   
    w_end = w_start + w_size
    pad_start = max(0, w_start - 5)
    pad_end = min(data.times[-1], w_end + 5)
    try:
        data_to_plot = prepare_filtered_data(step, data, pad_start, pad_end, channel_mode, channels, reference, single_channel, avg_channels, filter_settings, app_channel_types=app_channel_types)
    except ValueError as e:
        return error_plot(str(e))
    data_to_plot.crop(tmin=w_start - pad_start, tmax=w_end - pad_start)
    return plot_layout(data_to_plot, w_start, input, show_bad_annotations, app_channel_types)

# Create window controls for bad-channel review
def bad_channel_window_controls(ui):
    return ui.div(
        ui.input_action_button("prev_window_bad_chn", "←", class_="btn-secondary"),
        ui.input_select(
            "window_size_bad_chn",
            "Window",
            choices={"10": "10s", "30": "30s", "60": "60s", "90": "90s", "120": "120s", "240": "240s", "360": "360s", "600": "600s", "1800": "1800s"},
            selected="10",
        ),
        ui.input_action_button("next_window_bad_chn", "→", class_="btn-secondary"),
        style="display: flex; justify-content: space-between; align-items: center; width: 100%;",
    )

