import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yasa
from shiny import req
from scipy.signal import spectrogram as scipy_spectrogram, welch
import base64
import html
import io

# Let user choose which channels to use for sleep staging and which metrics to include
def choosing_sleep_staging(data, channel_names, channel_types, ui):
    req(data)
    channels_by_type = {
        ch_type: [ch for ch, channel_type in zip(channel_names, channel_types) if channel_type == ch_type]
        for ch_type in sorted(set(channel_types), key=lambda x: (x != "eeg", x))
    }
    boxes = [ui.input_selectize("sleep_eeg_channels", "EEG channels:", choices=channels_by_type.get("eeg", []), selected=channels_by_type.get("eeg", [])[:1], multiple=True)]
    if channels_by_type.get("eog"):
        boxes.append(ui.input_select("sleep_eog_channel", "EOG channel:", choices=channels_by_type["eog"], selected=channels_by_type["eog"][0], multiple=False))
    if channels_by_type.get("emg"):
        boxes.append(ui.input_select("sleep_emg_channel", "EMG channel", choices=channels_by_type["emg"], selected=channels_by_type["emg"][0], multiple=False))
    boxes.append(ui.input_selectize("sleep_metrics", "Add calculations:", choices={"spindles": "Spindles", "slow_waves": "Slow waves"}, selected=["spindles", "slow_waves"], multiple=True))
    boxes.append(ui.input_selectize("sleep_visual_channels", "Visualise non-EEG channels:", choices=[channel_name for channel_name, channel_type in zip(channel_names, channel_types) if channel_type != "eeg"], selected=[], multiple=True))
    return ui.div(
        *boxes,
        ui.div(
            ui.input_action_button("run_sleep_staging", "Run sleep staging", class_="btn-primary"),
            ui.input_action_button("back_sleep_staging_main_panel", "Back to main panel", class_="btn-secondary"),
            style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;",
        ),
        style="width: 700px; margin-left: 150px; margin-top: 20px;"
    )

# Compute sleep staging and related metrics
def run_sleep_staging_analysis(data, eeg_channels, eog_channel=None, emg_channel=None, metrics=None, visual_channels=None):
    req(data)
    eeg_channels, metrics, visual_channels = [
        [] if chn is None or chn == "" else [chn] if isinstance(chn, str) else list(chn)
        for chn in (eeg_channels, metrics, visual_channels)
    ]
    eog_channel = eog_channel or None
    emg_channel = emg_channel or None
    metrics = set(metrics)
    if not eeg_channels:
        raise ValueError("Select at least one EEG channel for sleep staging.")
    analysis_data = data.copy()
    staging = {}
    for eeg_channel in eeg_channels:
        sls = yasa.SleepStaging(analysis_data, eeg_name=eeg_channel, eog_name=eog_channel, emg_name=emg_channel)
        hypnogram = sls.predict()
        proba = hypnogram.proba
        staging[eeg_channel] = {
            "prediction": hypnogram,
            "prediction_probability": proba,
            "sleep_statistics": hypnogram.sleep_statistics()
        }
    event_channels = [ch for ch in eeg_channels if ch in analysis_data.ch_names]
    event_data = analysis_data.copy().pick(event_channels)
    first_hypnogram = next(iter(staging.values()))["prediction"]
    event_hypnogram = average_sleep_hypnogram({"staging": staging}) if len(staging) > 1 else first_hypnogram
    spindles = None
    if "spindles" in metrics and hypnogram_has_stages(event_hypnogram, ["N1", "N2", "N3"]):
        spindles = yasa.spindles_detect(event_data, hypno=event_hypnogram, include=["N1", "N2", "N3"])
    slow_waves = None
    if "slow_waves" in metrics and hypnogram_has_stages(event_hypnogram, ["N2", "N3"]):
        slow_waves = yasa.sw_detect(event_data, hypno=event_hypnogram, include=["N2", "N3"])
    return {
        "staging": staging,
        "spindles": spindles,
        "spindles_summary": None if spindles is None else spindles.summary(),
        "slow_waves": slow_waves,
        "slow_waves_summary": None if slow_waves is None else slow_waves.summary(),
        "visual_channels": visual_channels,
    }

# Create the available choices and default
def sleep_display_choices(result):
    eeg_channels = list(result["staging"].keys())
    if len(eeg_channels) > 1:
        return ["Average"] + eeg_channels, "Average"
    return eeg_channels, eeg_channels[0]

# Validate and return the user’s current selection
def current_sleep_display_channel(result, input):
    req(result)
    choices, default = sleep_display_choices(result)
    try:
        selected = input.sleep_view_channel()
    except Exception:
        selected = default
    return selected if selected in choices else default

# Average sleep-stage probabilities across EEG channels
def average_sleep_hypnogram(result):
    probas = [item["prediction_probability"] for item in result["staging"].values()]
    avg_proba = sum(probas) / len(probas)
    return yasa.Hypnogram(avg_proba.idxmax(axis=1), freq="30s", proba=avg_proba)

# Check whether a hypnogram contains any requested stages
def hypnogram_has_stages(hypnogram, stages):
    values = hypnogram.hypno if hasattr(hypnogram, "hypno") else hypnogram
    return pd.Series(values).isin(stages).any()

# Get the signal and sampling rate for the selected channel view
def _plot_data_for_channel(data, result, selected_channel):
    channels = list(result["staging"].keys())
    picks = channels if selected_channel == "Average" else [selected_channel]
    plot_data = data.copy().pick(picks).get_data() * 1e6
    return plot_data.mean(axis=0), data.info["sfreq"]

# Get sleep-stage probabilities for one channel or their average
def sleep_stage_probabilities(result, selected_channel):
    if selected_channel == "Average":
        probas = [item["prediction_probability"] for item in result["staging"].values()]
        return sum(probas) / len(probas)
    return result["staging"][selected_channel]["prediction_probability"]

# Put sleep stages in a consistent plotting order
def _stage_plot_order(stages):
    stage_order = [stage for stage in ["W", "WAKE", "R", "REM", "N1", "N2", "N3"] if stage in stages]
    stage_order += [stage for stage in stages if stage not in stage_order]
    return stage_order

# Draw the hypnogram on an existing axis
def _plot_hypnogram_on_axis(ax, proba, selected_channel):
    stages = proba.idxmax(axis=1).to_numpy()
    ordered_stages = _stage_plot_order(list(pd.unique(stages)))
    stage_to_y = {stage: len(ordered_stages) - idx - 1 for idx, stage in enumerate(ordered_stages)}
    y = np.array([stage_to_y[stage] for stage in stages])
    x = np.arange(len(stages)) / 2 + 0.25
    ax.step(x, y, where="post", color="#202124", linewidth=1.5)
    ax.set_yticks(list(stage_to_y.values()))
    ax.set_yticklabels(list(stage_to_y.keys()))
    ax.set_ylabel("Stage")
    ax.set_title("Hypnogram" if selected_channel != "Average" else "Average hypnogram")
    ax.grid(axis="x", alpha=0.2)

# Draw sleep-stage probabilities on an existing axis
def _plot_probabilities_on_axis(ax, proba, selected_channel):
    stage_order = [stage for stage in ["N1", "N2", "N3", "R", "REM", "W", "WAKE"] if stage in proba.columns]
    stage_order += [stage for stage in proba.columns if stage not in stage_order]
    colors = {
        "N1": "#9dd9e8",
        "N2": "#35b5df",
        "N3": "#3f6f98",
        "R": "#9b3f8f",
        "REM": "#9b3f8f",
        "W": "#ffd34d",
        "WAKE": "#ffd34d",
    }
    x = np.arange(len(proba)) / 2 + 0.25
    y = [proba[stage].to_numpy() for stage in stage_order]
    ax.stackplot(x, y, labels=stage_order, colors=[colors.get(stage, None) for stage in stage_order], alpha=0.95)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Stage probabilities" if selected_channel != "Average" else "Average stage probabilities")
    ax.legend(loc="upper left", fontsize=8, ncol=min(len(stage_order), 6))
    ax.grid(axis="x", alpha=0.2)

# Get event results for the selected channel view
def _event_summary_for_channel(result, event_key, selected_channel):
    table = result.get(f"{event_key}_summary")
    if table is None:
        return pd.DataFrame()
    table = table.copy()
    channel_col = "Channel" if "Channel" in table.columns else "Chan" if "Chan" in table.columns else None
    if selected_channel != "Average" and channel_col:
        table = table[table[channel_col] == selected_channel]
    return table

# Check whether the selected view contains detected events
def _has_display_events(result, event_key, selected_channel):
    return not _event_summary_for_channel(result, event_key, selected_channel).empty

# Calculate event presence across short time bins
def _event_presence_values(result, event_key, selected_channel, duration_minutes):
    table = _event_summary_for_channel(result, event_key, selected_channel)
    bin_width_minutes = 1 / 60
    edges = np.arange(0, duration_minutes + bin_width_minutes, bin_width_minutes)
    if len(edges) < 2 or table.empty or "Start" not in table.columns or "Duration" not in table.columns:
        return edges, np.zeros(max(0, len(edges) - 1))

    channel_col = "Channel" if "Channel" in table.columns else "Chan" if "Chan" in table.columns else None
    if selected_channel == "Average" and channel_col:
        channels = list(result["staging"].keys())
    else:
        channels = [selected_channel]
    channel_to_idx = {channel: idx for idx, channel in enumerate(channels)}
    presence = np.zeros((len(channels), len(edges) - 1), dtype=bool)

    for _, event in table.iterrows():
        channel = event[channel_col] if channel_col else selected_channel
        if channel not in channel_to_idx:
            continue
        start = float(event["Start"]) / 60
        end = start + float(event["Duration"]) / 60
        first = np.searchsorted(edges, start, side="right") - 1
        last = np.searchsorted(edges, end, side="left")
        first = max(0, first)
        last = min(presence.shape[1], last)
        if first < last:
            presence[channel_to_idx[channel], first:last] = True

    return edges, presence.mean(axis=0)

# Draw event presence on an existing axis
def _plot_event_presence_on_axis(ax, result, event_key, selected_channel, label, duration_minutes):
    edges, values = _event_presence_values(result, event_key, selected_channel, duration_minutes)
    ax.set_facecolor("white")
    if len(values):
        ax.pcolormesh(edges, [0, 1], values[np.newaxis, :], cmap="gray_r", vmin=0, vmax=1, shading="flat")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels([label])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.2)

# Convert bad annotations to elapsed-minute ranges
def _bad_annotation_segments_minutes(data, duration_minutes):
    duration_seconds = duration_minutes * 60
    first_time = getattr(data, "first_time", 0) or 0
    segments = []
    for ann in data.annotations:
        if not str(ann["description"]).upper().startswith("BAD"):
            continue
        onset = float(ann["onset"])
        if first_time > 0 and first_time <= onset <= first_time + duration_seconds:
            onset -= first_time
        start = max(0, onset)
        end = min(duration_seconds, onset + float(ann["duration"]))
        if end > start:
            segments.append((start / 60, end / 60))
    return segments

# Draw bad-annotation ranges on an existing axis
def _plot_bad_annotations_on_axis(ax, data, duration_minutes):
    segments = _bad_annotation_segments_minutes(data, duration_minutes)
    ax.set_facecolor("white")
    for start, end in segments:
        ax.broken_barh([(start, end - start)], (0, 1), facecolors="#d62728", alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels(["Bad annotations"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.2)

# Draw a signal spectrogram on an existing axis
def _plot_spectrogram_on_axis(ax, signal, sfreq):
    nperseg = min(len(signal), int(30 * sfreq))
    noverlap = min(nperseg - 1, int(nperseg * 0.5))
    freqs, times, power = scipy_spectrogram(signal, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    keep = freqs <= min(25, sfreq / 2 - 0.1)
    power_db = 10 * np.log10(power[keep] + np.finfo(float).eps)
    ax.pcolormesh(times / 60, freqs[keep], power_db, shading="auto", cmap="magma")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (HH:MM)")
    ax.set_title("Time-frequency")
    ax.set_ylim(freqs[keep].min(), freqs[keep].max())
    ax.set_autoscale_on(False)
    ax.add_patch(Rectangle((0.012, 0.86), 0.25, 0.11, transform=ax.transAxes, facecolor="white", alpha=0.5, edgecolor="#cccccc", linewidth=0.8, zorder=5))
    gradient = np.linspace(0, 1, 128).reshape(1, -1)
    ax.imshow(gradient, extent=(0.03, 0.22, 0.925, 0.955), transform=ax.transAxes, cmap="magma", aspect="auto", zorder=6)
    vmin, vmax = np.nanmin(power_db), np.nanmax(power_db)
    ax.text(0.03, 0.895, f"{vmin:.0f}", transform=ax.transAxes, fontsize=8, va="center", zorder=7)
    ax.text(0.22, 0.895, f"{vmax:.0f}", transform=ax.transAxes, fontsize=8, ha="right", va="center", zorder=7)
    ax.text(0.245, 0.94, "dB", transform=ax.transAxes, fontsize=8, va="center", zorder=7)

# Format elapsed minutes as hours and minutes
def _format_elapsed_minutes(minutes, _pos):
    total_minutes = max(0, int(round(minutes)))
    hours, mins = divmod(total_minutes, 60)
    return f"{hours:02d}:{mins:02d}"

# Build the main sleep-staging summary figure
def plot_sleep_staging_summary(data, result, selected_channel):
    proba = sleep_stage_probabilities(result, selected_channel)
    signal, sfreq = _plot_data_for_channel(data, result, selected_channel)
    duration_minutes = len(signal) / sfreq / 60
    rows = ["hypnogram", "probabilities"]
    if result.get("spindles") is not None and _has_display_events(result, "spindles", selected_channel):
        rows.append("spindles")
    if result.get("slow_waves") is not None and _has_display_events(result, "slow_waves", selected_channel):
        rows.append("slow_waves")
    if _bad_annotation_segments_minutes(data, duration_minutes):
        rows.extend(["bad_annotation_gap", "bad_annotations"])
    rows.append("time_frequency")
    height_ratios = [
        1 if row == "hypnogram" else
        1.2 if row == "probabilities" else
        0.3 if row in ["spindles", "slow_waves", "bad_annotations"] else
        0.09 if row == "bad_annotation_gap" else
        0.7 if row.startswith("channel_type_") else
        2.2
        for row in rows
    ]
    fig, axes = plt.subplots(
        len(rows),
        1,
        sharex=True,
        figsize=(12, 9 + 0.35 * (len(rows) - 3)),
        height_ratios=height_ratios,
    )
    axes_by_row = dict(zip(rows, axes))
    _plot_hypnogram_on_axis(axes_by_row["hypnogram"], proba, selected_channel)
    _plot_probabilities_on_axis(axes_by_row["probabilities"], proba, selected_channel)
    if "spindles" in axes_by_row:
        _plot_event_presence_on_axis(axes_by_row["spindles"], result, "spindles", selected_channel, "Spindles", duration_minutes)
    if "slow_waves" in axes_by_row:
        _plot_event_presence_on_axis(axes_by_row["slow_waves"], result, "slow_waves", selected_channel, "Slow waves", duration_minutes)
    if "bad_annotation_gap" in axes_by_row:
        axes_by_row["bad_annotation_gap"].axis("off")
    if "bad_annotations" in axes_by_row:
        _plot_bad_annotations_on_axis(axes_by_row["bad_annotations"], data, duration_minutes)
    _plot_spectrogram_on_axis(axes_by_row["time_frequency"], signal, sfreq)
    axes_by_row["time_frequency"].set_xlim(0, duration_minutes)
    tick_count = min(8, max(2, int(np.ceil(duration_minutes)) + 1))
    axes_by_row["time_frequency"].set_xticks(np.linspace(0, duration_minutes, tick_count))
    axes_by_row["time_frequency"].xaxis.set_major_formatter(FuncFormatter(_format_elapsed_minutes))
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    fig.tight_layout()
    return fig

# Plot selected non-EEG sensors for each sleep stage
def plot_selected_sensors_by_stage(data, result, selected_channel, app_channel_types=None, filter_settings=None):
    visual_channels = [channel for channel in result.get("visual_channels", []) if channel in data.ch_names]
    if not visual_channels:
        return None
    proba = sleep_stage_probabilities(result, selected_channel)
    stages = proba.idxmax(axis=1)
    stage_order = _stage_plot_order(list(pd.unique(stages)))
    first_epoch_by_stage = {stage: int(np.where(stages.to_numpy() == stage)[0][0]) for stage in stage_order}
    n_rows = len(visual_channels) + 1
    n_cols = len(stage_order) + 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(8, 2.1 * n_cols), max(2.5, 1.1 * n_rows)),
        gridspec_kw={"width_ratios": [1] + [3] * (n_cols - 1)},
        squeeze=False,
    )
    axes[0, 0].text(0.5, 0.5, "Sensor", ha="center", va="center", fontweight="bold")
    for col_idx, stage in enumerate(stage_order, start=1):
        axes[0, col_idx].text(0.5, 0.5, str(stage), ha="center", va="center", fontweight="bold")
    sfreq = data.info["sfreq"]
    duration_samples = int(round(30 * sfreq))
    app_channel_types = app_channel_types or {}
    filter_settings = filter_settings or {}
    mne_types = dict(zip(data.ch_names, data.get_channel_types()))

    for row_idx, channel in enumerate(visual_channels, start=1):
        axes[row_idx, 0].text(0.5, 0.5, channel, ha="center", va="center")
        channel_group = app_channel_types.get(channel, mne_types.get(channel, "misc"))
        scaling = filter_settings.get(channel_group, {}).get("scaling")
        if scaling is None:
            scaling = 200000 if channel_group == "body_position" else 40 if channel_group == "eeg" else 400 if channel_group == "ecg" else 100
        signal = data.copy().pick([channel]).get_data()[0] * 1e6
        for col_idx, stage in enumerate(stage_order, start=1):
            epoch_idx = first_epoch_by_stage[stage]
            start = int(round(epoch_idx * 30 * sfreq))
            end = min(start + duration_samples, signal.size)
            segment = signal[start:end]
            times = np.arange(segment.size) / sfreq
            ax = axes[row_idx, col_idx]
            if segment.size:
                segment = segment - np.nanmean(segment)
                ax.plot(times, segment / scaling, color="#202124", linewidth=0.7)
                ax.axhline(0, color="#cccccc", linewidth=0.5)
            ax.set_xlim(0, 30)
            ax.set_ylim(-2, 2)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(0.6)

    fig.tight_layout()
    return fig

# Get sleep statistics for one channel or their average
def selected_sleep_statistics(result, selected_channel):
    if result is None:
        return None
    if selected_channel != "Average":
        stats = result["staging"][selected_channel]["sleep_statistics"]
    else:
        stats = average_sleep_hypnogram(result).sleep_statistics()
    if isinstance(stats, dict):
        stats = pd.DataFrame([stats])
    return stats

# Convert sleep statistics into a plain dictionary
def sleep_statistics_values(stats):
    if stats is None:
        return {}
    if isinstance(stats, pd.Series):
        return stats.to_dict()
    if isinstance(stats, pd.DataFrame):
        if stats.empty:
            return {}
        return stats.iloc[0].to_dict()
    return dict(stats)

# Build the reduced sleep-statistics display table
def sleep_statistics_table(result, selected_channel):
    return reduced_sleep_statistics_table(selected_sleep_statistics(result, selected_channel))

# Select and format the sleep statistics shown to users
def reduced_sleep_statistics_table(stats):
    if stats is None:
        return None
    values = sleep_statistics_values(stats)
    if not values and isinstance(stats, pd.DataFrame):
        return stats

    # Return the first available statistic for the given names
    def value_for(*keys):
        for key in keys:
            if key in values:
                return values[key]
        return None

    # Format one statistic with its display unit
    def format_value(value, unit):
        if value is None or pd.isna(value):
            return ""
        if unit == "%":
            return f"{value:.1f}%"
        if unit == "min":
            return f"{value:.1f} minutes"
        return str(value)

    wake = value_for("WAKE", "W")
    n1 = value_for("N1")
    n2 = value_for("N2")
    n3 = value_for("N3")
    rem = value_for("REM", "R")
    stage_total = sum(v for v in [wake, n1, n2, n3, rem] if v is not None)
    wake_percentage = None if wake is None or not stage_total else wake / stage_total * 100

    rows = [
        ("Sleep onset latency", value_for("SOL"), "min"),
        ("Sleep onset latency to 5 minutes of sleep", value_for("SOL_5MIN"), "min"),
        ("Wake after sleep onset", value_for("WASO"), "min"),
        ("Time asleep", value_for("SE"), "%"),
        ("Wake", wake_percentage, "%"),
        ("Stage N1 sleep", value_for("%N1"), "%"),
        ("Stage N2 sleep", value_for("%N2"), "%"),
        ("Stage N3 sleep", value_for("%N3"), "%"),
        ("REM sleep", value_for("%REM"), "%"),
    ]
    return pd.DataFrame([
        {"Metric": metric, "Value": format_value(value, unit)}
        for metric, value, unit in rows
    ])

# Build the epoch-by-epoch sleep probability table
def sleep_epoch_probability_table(result, selected_channel):
    proba = sleep_stage_probabilities(result, selected_channel).copy()
    proba = proba.rename(columns={"W": "WAKE", "R": "REM"})
    if proba.columns.duplicated().any():
        proba = proba.T.groupby(level=0).sum().T
    stage_order = ["WAKE", "N1", "N2", "N3", "REM"]
    for stage in stage_order:
        if stage not in proba.columns:
            proba[stage] = 0.0
    table = proba[stage_order].copy()
    table.insert(0, "Sleep stage", table.idxmax(axis=1))
    return table

# Format and filter an event table for the selected channel
def _event_table(table, selected_channel):
    if table is None:
        return None
    table = table.copy()
    if table.index.names and any(name is not None for name in table.index.names):
        table = table.reset_index()
    if "Stage" in table.columns:
        table["Stage"] = table["Stage"].replace({0: "WAKE", 1: "N1", 2: "N2", 3: "N3", 4: "REM"})
    channel_col = "Channel" if "Channel" in table.columns else "Chan" if "Chan" in table.columns else None
    if selected_channel != "Average" and channel_col:
        return table[table[channel_col] == selected_channel]
    return table

# Build the spindle results table
def spindles_table(result, selected_channel):
    return _event_table(result["spindles_summary"], selected_channel)

# Build the slow-wave results table
def slow_waves_table(result, selected_channel):
    return _event_table(result["slow_waves_summary"], selected_channel)

# Format bandpower results by sleep stage
def _format_bandpower_stage_table(table):
    if table is None:
        return None
    table = table.copy()
    if table.index.names and any(name is not None for name in table.index.names):
        table = table.reset_index()
    if "Stage" in table.columns:
        table["Stage"] = table["Stage"].replace({0: "WAKE", 1: "N1", 2: "N2", 3: "N3", 4: "REM", "W": "WAKE", "R": "REM"})
        table = table[table["Stage"].notna()]
    numeric_cols = [col for col in table.select_dtypes(include="number").columns if col not in ["FreqRes", "Relative"]]
    if "Stage" not in table.columns or not numeric_cols:
        return None
    table = table.groupby("Stage", dropna=False)[numeric_cols].mean().reset_index()
    stage_order = ["Overall", "WAKE", "N1", "N2", "N3", "REM"]
    table["Stage"] = pd.Categorical(table["Stage"], categories=stage_order, ordered=True)
    table = table.sort_values("Stage").reset_index(drop=True)
    table["Stage"] = table["Stage"].astype(str)
    return table[["Stage"] + numeric_cols]

# Calculate overall and stage-specific bandpower
def bandpower_table(data, result, selected_channel):
    if data is None or result is None:
        return None
    channels = list(result["staging"].keys()) if selected_channel == "Average" else [selected_channel]
    channels = [channel for channel in channels if channel in data.ch_names]
    if not channels:
        return None

    analysis_data = data.copy().pick(channels)
    overall = yasa.bandpower(analysis_data)
    if overall is not None:
        overall = overall.copy()
        if overall.index.names and any(name is not None for name in overall.index.names):
            overall = overall.reset_index()
        numeric_cols = [col for col in overall.select_dtypes(include="number").columns if col not in ["FreqRes", "Relative"]]
        overall = pd.DataFrame([overall[numeric_cols].mean()])
        overall.insert(0, "Stage", "Overall")

    hypnogram = _selected_hypnogram(result, selected_channel)
    hypno = hypnogram.upsample_to_data(analysis_data.get_data()[0], sf=analysis_data.info["sfreq"])
    by_stage = yasa.bandpower(analysis_data, hypno=hypno, include=[0, 1, 2, 3, 4])
    by_stage = _format_bandpower_stage_table(by_stage)
    if overall is not None and by_stage is not None:
        return _format_bandpower_stage_table(pd.concat([overall, by_stage], ignore_index=True))
    return _format_bandpower_stage_table(overall if overall is not None else by_stage)

# Get the hypnogram for the selected channel view
def _selected_hypnogram(result, selected_channel):
    if selected_channel == "Average":
        return average_sleep_hypnogram(result)
    return result["staging"][selected_channel]["prediction"]

# Plot power spectral density by sleep stage
def plot_psd_by_stage(data, result, selected_channel):
    signal, sfreq = _plot_data_for_channel(data, result, selected_channel)
    hypnogram = _selected_hypnogram(result, selected_channel)
    hypno = hypnogram.upsample_to_data(signal, sf=sfreq)
    stage_map = {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    colors = {"WAKE": "#99d7f1", "N1": "#009ddc", "N2": "#0a437a", "N3": "#720058", "REM": "#ffc512"}
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    nperseg = min(len(signal), int(4 * sfreq))
    if len(signal) >= max(8, int(2 * sfreq)):
        freqs, psd = welch(signal, fs=sfreq, nperseg=min(len(signal), nperseg))
        keep = freqs <= min(40, sfreq / 2)
        ax.semilogy(freqs[keep], psd[keep], label="Overall", color="#202124", linewidth=2)
        plotted = True
    for stage, stage_value in stage_map.items():
        stage_signal = signal[hypno == stage_value]
        if len(stage_signal) < max(8, int(2 * sfreq)):
            continue
        freqs, psd = welch(stage_signal, fs=sfreq, nperseg=min(len(stage_signal), nperseg))
        keep = freqs <= min(40, sfreq / 2)
        ax.semilogy(freqs[keep], psd[keep], label=stage, color=colors.get(stage))
        plotted = True
    if not plotted:
        return None
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (uV^2/Hz)")
    ax.set_title("PSD by sleep stage")
    ax.legend(title="Stage", fontsize=8)
    fig.tight_layout()
    return fig

# Plot an average detected event around its center
def _plot_average_event(detector, title, center, time_before, time_after, mask=None, show_std=True, figsize=(8, 3.2), symmetric_xlim=True):
    try:
        summary = detector.summary()
    except (AssertionError, ValueError):
        summary = pd.DataFrame()
    if center in summary.columns:
        valid_mask = summary[center] >= time_before
        if mask is not None:
            valid_mask = valid_mask & pd.Series(mask, index=summary.index)
        mask = valid_mask
        if not mask.any():
            return None
    try:
        df_sync = detector.get_sync_events(center=center, time_before=time_before, time_after=time_after, mask=mask)
    except (AssertionError, ValueError):
        return None
    if df_sync.empty:
        return None
    average = df_sync.groupby("Time")["Amplitude"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=figsize)
    line_color = "#4C72B0"
    ax.plot(average["Time"], average["mean"], color=line_color, linewidth=1.8)
    if show_std:
        lower = average["mean"] - average["std"].fillna(0)
        upper = average["mean"] + average["std"].fillna(0)
        ax.fill_between(average["Time"], lower, upper, color=line_color, alpha=0.18, linewidth=0)
    if symmetric_xlim:
        max_abs_time = max(abs(average["Time"].min()), abs(average["Time"].max()))
        if np.isfinite(max_abs_time) and max_abs_time > 0:
            ax.set_xlim(-max_abs_time, max_abs_time)
    ax.set_title(title)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude (uV)")
    return fig

# Create an empty plot containing a status message
def empty_sleep_plot(message):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, color="#666666")
    return fig

# Plot the average detected spindle waveform
def plot_spindle_average(result, selected_channel):
    if result["spindles"] is None:
        return None
    if selected_channel == "Average":
        return _plot_average_event(result["spindles"], "Average spindle", "Peak", 1, 1, show_std=False)
    mask = None
    if selected_channel != "Average":
        summary = result["spindles"].summary()
        if "Channel" in summary.columns:
            mask = summary["Channel"] == selected_channel
            if not mask.any():
                return None
    return _plot_average_event(result["spindles"], "Average spindle", "Peak", 1, 1, mask=mask, show_std=True)

# Plot the average detected slow-wave waveform
def plot_slow_wave_average(result, selected_channel):
    if result["slow_waves"] is None:
        return None
    if selected_channel == "Average":
        return _plot_average_event(result["slow_waves"], "Average SW", "NegPeak", 0.4, 0.8, show_std=False, symmetric_xlim=False)
    mask = None
    if selected_channel != "Average":
        summary = result["slow_waves"].summary()
        if "Channel" in summary.columns:
            mask = summary["Channel"] == selected_channel
            if not mask.any():
                return None
    return _plot_average_event(result["slow_waves"], "Average SW", "NegPeak", 0.4, 0.8, mask=mask, show_std=True, symmetric_xlim=False)

# Round numeric columns for display and export
def round_display_table(table, decimals=3):
    if table is None:
        return None
    rounded = table.copy()
    numeric_columns = rounded.select_dtypes(include="number").columns
    rounded[numeric_columns] = rounded[numeric_columns].round(decimals)
    return rounded

# Move the channel column to the start of a table
def channel_first_table(table):
    if table is None:
        return None
    if "Channel" in table.columns:
        channel_column = "Channel"
    elif "Chan" in table.columns:
        channel_column = "Chan"
    else:
        return table
    columns = [
        channel_column,
        *[
            column
            for column in table.columns
            if column != channel_column
        ],
    ]
    return table[columns]

# Convert a Matplotlib figure to an embedded HTML image
def fig_to_html(fig):
    if fig is None:
        return ""
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=140,
        bbox_inches="tight",
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{encoded}" '
        f'alt="" />'
    )

# Convert a table to report-ready HTML
def table_to_html(table):
    if table is None:
        return ""
    return table.to_html(
        index=False,
        border=0,
        classes="report-table",
    )

# Convert a table to CSV text
def table_to_csv(table):
    if table is None:
        return ""
    return table.to_csv(index=False)

# Create a safe folder name for ZIP exports
def safe_zip_folder_name(channel):
    folder = (
        str(channel)
        .replace("/", "*")
        .replace("\\", "*")
        .strip()
    )
    if folder in {"", ".", ".."}:
        return "Channel"
    return folder

# Build one HTML report section
def report_section(title, body):
    if not body:
        return ""
    return (
        f"<section>"
        f"<h2>{html.escape(title)}</h2>"
        f"{body}"
        f"</section>"
    )

# Build the complete downloadable sleep report
def build_sleep_report_html(data, result, selected_channel, channel_types, filter_settings):
    summary_figure = plot_sleep_staging_summary(
        data,
        result,
        selected_channel,
    )
    visual_figure = plot_selected_sensors_by_stage(
        data,
        result,
        selected_channel,
        channel_types,
        filter_settings,
    )
    statistics = round_display_table(
        sleep_statistics_table(
            result,
            selected_channel,
        )
    )
    bandpower = bandpower_table(
        data,
        result,
        selected_channel,
    )
    if bandpower is not None:
        bandpower = round_display_table(
            bandpower.drop(
                columns=["FreqRes", "Relative"],
                errors="ignore",
            )
        )
    bandpower_figure = plot_psd_by_stage(
        data,
        result,
        selected_channel,
    )
    sections = [
        report_section(
            "Sleep staging summary",
            fig_to_html(summary_figure),
        ),
        report_section(
            "Selected non-EEG sensors by sleep stage",
            fig_to_html(visual_figure),
        ),
        report_section(
            "Sleep statistics",
            table_to_html(statistics),
        ),
        report_section(
            "Bandpower",
            table_to_html(bandpower)
            + fig_to_html(bandpower_figure),
        ),
    ]

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sleep report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #202124; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 28px; border-bottom: 1px solid #dddddd; padding-bottom: 6px; }}
    img {{ max-width: 100%; display: block; margin: 12px 0 20px; }}
    .meta {{ color: #666666; margin-bottom: 24px; }}
    .report-table {{ border-collapse: collapse; margin: 12px 0 20px; width: 100%; font-size: 14px; }}
    .report-table th, .report-table td {{ border: 1px solid #dddddd; padding: 6px 8px; text-align: left; }}
    .report-table th {{ background: #f2f4f5; }}
  </style>
</head>
<body>
  <h1>Sleep report</h1>
  <div class="meta">
    Displayed EEG result: {html.escape(str(selected_channel))}
  </div>
  {''.join(sections)}
</body>
</html>"""
