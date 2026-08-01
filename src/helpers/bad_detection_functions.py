# Import packages
from pyprep.find_noisy_channels import NoisyChannels
import mne
import numpy as np
import pandas as pd
from shiny import ui

from helpers.data_management import has_montage

# Preparing data
def prepare_data_for_detection(data, resample_sfreq=None):
    if data is None:
        return None
    prepared = data.copy()
    prepared.load_data()
    if resample_sfreq is not None and resample_sfreq < prepared.info["sfreq"]:
        prepared.resample(resample_sfreq)
    return prepared

# Finding bad channels using PyPREP
def find_bad_channels(data, resample_sfreq=None):
    if data is None:
        return None
    data = prepare_data_for_detection(data, resample_sfreq)
    eeg_data = data.copy().pick("eeg")
    if not eeg_data.ch_names:
        raise ValueError("Bad channel detection requires at least one EEG channel.")
    criteria = ["bad_by_correlation", "bad_by_deviation", "bad_by_flat", "bad_by_nan"]
    nc = NoisyChannels(eeg_data, random_state=42)
    nc.find_all_bads(ransac=False)
    bad_channels = sorted(set().union(*(getattr(nc, c) for c in criteria)))
    return bad_channels, nc

# Build the bad-channel review panel
def bad_channel_panel_content_ui():
    return ui.div(
        ui.br(),
        ui.p("Bad-channel detection uses a subset of the methods available in the PyPREP toolbox to improve computational efficiency. Additional detection methods can be enabled by modifying the code."),
        ui.p("Interpolating bad channels requires at least 4 correctly positioned EEG channels; channels sitting low and toward the front of the head may be excluded from this check even if their coordinates are valid, so interpolation can fail even with a montage applied."),
        ui.p(ui.strong("REFERENCE: "), "Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A. (2015). The PREP pipeline: standardized preprocessing for large-scale EEG analysis. Frontiers in Neuroinformatics, 9, 16. doi: 10.3389/fninf.2015.00016"),
        ui.br(),
        ui.output_text("bad_channel_list"),
        ui.br(),
        ui.output_ui("bad_channel_plot_ui"),
        ui.output_ui("bad_channel_window_controls_ui"),
        ui.br(),
        ui.output_data_frame("bad_channel_info_table"),
        ui.br(),
        ui.output_ui("bad_channel_handling_ui"),
        ui.div(ui.output_ui("bad_channel_apply_ui")),
        ui.div(ui.output_ui("bad_channel_back_ui"), style="margin-top: 10px;"),
        style="margin-left: 15px; margin-right: 15px;",
    )

# Build action controls for each bad channel
def bad_channel_handling_display(bad_chn):
    if not bad_chn:
        return None
    return ui.div(
    ui.h5("Choose action per bad channel:"),
    ui.p("Interpolation uses MNE’s default spherical-spline interpolation based on neighboring sensor positions. A valid montage is required, which should be set in the main panel."),
    ui.p("Marking a channel as bad will retain it in the data but label it as bad. This option is recommended if you wish to keep all the channels but avoid ICA algorithm to use the channels marked as bad to estimate the components."),
    ui.p("Deleting a channel will remove it from the data entirely. Keeping the channel will retain it unchanged in the data."),
    *[
        ui.div(
            ui.span(ch, style="width: 100px; display: inline-block;"),
            ui.input_select(
                f"bad_action_{ch}",
                label=None,
                choices={
                    "keep": "Keep",
                    "mark_bad": "Mark as bad",
                    "interpolate": "Interpolate",
                    "delete": "Delete",
                },
                selected="interpolate",
            ),
            style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;",
        )
        for ch in bad_chn
    ],
)

# Delete, interpolate or keep bad channels
def apply_bad_channel_actions(data, actions, channel_types, selected_channels):
    to_interpolate = [ch for ch, action in actions.items() if action == "interpolate"]
    to_mark_bad = [ch for ch, action in actions.items() if action == "mark_bad"]
    to_drop = [ch for ch, action in actions.items() if action == "delete"]
    if to_interpolate and not has_montage(data):
        raise ValueError(
            "Please apply a montage in the main panel before "
            "interpolating bad channels."
        )
    if not any((to_interpolate, to_mark_bad, to_drop)):
        return {
            "data": data,
            "data_changed": False,
            "deleted_channels": [],
            "channel_types": channel_types,
            "selected_channels": selected_channels,
        }
    processed = data.copy().load_data()
    existing_bads = list(processed.info.get("bads", []))
    if to_interpolate:
        try:
            processed.info["bads"] = to_interpolate
            processed.interpolate_bads(reset_bads=True)
        except Exception as error:
            raise ValueError(
                "Interpolation failed. Ensure a valid montage is applied. "
                f"Details: {error}"
            ) from error
    remaining_bads = [
        channel
        for channel in existing_bads
        if channel not in to_interpolate and channel not in to_drop
    ]
    processed.info["bads"] = sorted(set(remaining_bads + to_mark_bad))
    if to_drop:
        processed.drop_channels(to_drop)
    remaining_channels = processed.ch_names
    if to_drop:
        channel_types = {
            channel: group
            for channel, group in channel_types.items()
            if channel in remaining_channels
        }
        selected_channels = [
            channel
            for channel in (selected_channels or [])
            if channel in remaining_channels
        ] or remaining_channels
    return {
        "data": processed,
        "data_changed": True,
        "deleted_channels": to_drop,
        "channel_types": channel_types,
        "selected_channels": selected_channels,
    }

# Detect abrupt changes, extreme amplitudes, and missing samples
def annotate_bad_segments_combined(data, peak_threshold=100e-6, absolute_threshold=150e-6, min_duration=0.005):
    sfreq = data.info["sfreq"]
    min_samples = max(1, int(round(min_duration * sfreq)))
    values = data.get_data(picks="data")
    peak_by_channel = np.abs(np.diff(values, axis=1)) >= peak_threshold
    for ch_idx, channel_mask in enumerate(peak_by_channel):
        annotations = []
        start = None
        for idx, is_bad in enumerate(channel_mask):
            if is_bad and start is None:
                start = idx
            elif not is_bad and start is not None:
                if idx - start < min_samples:
                    annotations.append((start, idx))
                start = None
        if start is not None and len(channel_mask) - start < min_samples:
            annotations.append((start, len(channel_mask)))
        for start, stop in annotations:
            peak_by_channel[ch_idx, start:stop] = False
    peak_bad = np.zeros(values.shape[1], dtype=bool)
    peak_bad[:-1] = peak_by_channel.any(axis=0)
    absolute_bad = np.any(np.abs(values) >= absolute_threshold, axis=0)
    nan_bad = np.any(np.isnan(values), axis=0)
    bad_mask = peak_bad | absolute_bad | nan_bad
    bad_int = bad_mask.astype(int)
    changes = np.diff(np.r_[0, bad_int, 0])
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]
    durations_samples = stops - starts
    keep = durations_samples >= min_samples
    starts = starts[keep]
    durations_samples = durations_samples[keep]
    annotations = list(zip(data.times[starts], durations_samples / sfreq))
    return mne.Annotations(
        onset=[onset for onset, _ in annotations],
        duration=[duration for _, duration in annotations],
        description=["BAD_segment"] * len(annotations),
        orig_time=data.info.get("meas_date"),
    )

# Convert an annotation onset to loaded-window time
def annotation_onset_in_loaded_window(data, onset):
    first_time = getattr(data, "first_time", 0) or 0
    if first_time <= 0:
        return onset
    last_time = first_time + data.times[-1]
    if first_time <= onset <= last_time:
        return onset - first_time
    return onset

# Convert a window-relative onset to raw-data time
def annotation_onset_for_raw(data, onset):
    first_time = getattr(data, "first_time", 0) or 0
    if first_time <= 0:
        return onset
    if 0 <= onset <= data.times[-1]:
        return onset + first_time
    return onset

# Shift a loaded-window onset into raw-data time
def annotation_onset_from_loaded_window(data, onset):
    first_time = getattr(data, "first_time", 0) or 0
    if first_time <= 0:
        return onset
    return onset + first_time

# Convert annotation onsets for assignment to raw data
def annotations_for_raw(data, annotations):
    return mne.Annotations(
        onset=[annotation_onset_for_raw(data, onset) for onset in annotations.onset],
        duration=annotations.duration,
        description=annotations.description,
        orig_time=annotations.orig_time,
    )

# Detect and attach bad-segment annotations
def annotate_bad_segments(data, amplitude_peak=100e-6, amplitude_min_duration=0.005):
    if data is None:
        return None
    annotations = annotate_bad_segments_combined(
        data,
        peak_threshold=amplitude_peak,
        absolute_threshold=150e-6,
        min_duration=amplitude_min_duration,
    )
    preserved_annotations = keep_non_bad_annotations(data)
    detected_annotations = annotations_for_raw(data, annotations)
    data.set_annotations(preserved_annotations + detected_annotations)
    return data

# Build consecutive ten-second windows across the data
def ten_second_windows(data):
    return [
        (start, min(start + 10, data.times[-1]))
        for start in np.arange(0, data.times[-1], 10)
    ]

# Get a list of all bad segment windows
def get_bad_segment_windows(data, bad_windows):
    all_windows = ten_second_windows(data)
    if not all_windows:
        return []
    bad_indices = set()
    max_idx = len(all_windows) - 1
    for idx, (bad_start, bad_end) in enumerate(bad_windows):
        first_window = max(0, int(bad_start // 10))
        last_window = min(max_idx, int((bad_end - np.finfo(float).eps) // 10))
        for i in range(first_window, last_window + 1):
            bad_indices.add(i)
    return [all_windows[i] for i in sorted(bad_indices)]

# Get detected windows and neighboring display windows
def get_list_bad_segments(data):
    bad_ann = get_bad_annotation_segments(data)
    all_windows = ten_second_windows(data)
    bad_indices = set()
    bad_indices_with_neighbors = set()
    for i, (start, end) in enumerate(all_windows):
        for bad_start, bad_end in bad_ann:
            overlap = min(end, bad_end) - max(start, bad_start)
            if overlap > 0:
                bad_indices.add(i)
                bad_indices_with_neighbors.add(i)
                if i > 0:
                    bad_indices_with_neighbors.add(i - 1)
                if i < len(all_windows) - 1:
                    bad_indices_with_neighbors.add(i + 1)
                break
    bad_windows = [all_windows[i] for i in sorted(bad_indices)]
    bad_windows_with_neighbors = [all_windows[i] for i in sorted(bad_indices_with_neighbors)]
    return bad_windows, bad_windows_with_neighbors

# Extract time ranges from BAD annotations
def get_bad_annotation_segments(data):
    annotations = data.annotations
    return [
        (
            annotation_onset_in_loaded_window(data, ann['onset']),
            annotation_onset_in_loaded_window(data, ann['onset']) + ann['duration'],
        )
        for ann in annotations
        if str(ann["description"]).upper().startswith("BAD")
    ]

# UI for bad segment panel
def bad_segment_panel_ui(result):
    if result is None:
        return None
    _, bad_windows_display = get_list_bad_segments(result)
    if not bad_windows_display:
        return ui.div(
            ui.p(
                "No bad segments to display.",
                style="color: green; font-size: 18px;",
            ),
            ui.div(
                ui.input_action_button(
                    "back_bad_segments_main_panel",
                    "Back to main panel",
                    class_="btn-secondary",
                ),
                style="margin-top: 10px;",
            ),
            style="margin-left: 15px; margin-right: 15px;",
        )
    return ui.div(
        ui.p(
            "Bad segments are the exact time intervals automatically identified "
            "as containing artifacts. Bad windows are complete 10-second windows "
            "that overlap one or more detected bad segments. The previous and next "
            "windows are also displayed for context, but they are not automatically "
            "marked as bad. Running detection removes existing annotations whose "
            "descriptions start with 'BAD' and replaces them with newly detected bad "
            "segments; this panel displays the 10-second windows overlapping those new "
            "segments. You can manually select ranges by clicking 'Save selected "
            "range as bad segment'. After reviewing all windows, click 'Saved "
            "selections' to apply the changes."
        ),
        ui.output_ui("bad_segment_plot_ui"),
        ui.div(
            ui.input_action_button(
                "prev_bad_window",
                "←",
                class_="btn-secondary",
            ),
            ui.input_slider(
                "bad_keep_range",
                "Segment from this window to exclude from further analysis",
                min=0.0,
                max=10.0,
                value=(0.0, 10.0),
                step=0.1,
            ),
            ui.input_action_button(
                "apply_bad_segment_handling",
                "Save selected range as bad segment",
                class_="btn-primary",
            ),
            ui.input_action_button(
                "next_bad_window",
                "→",
                class_="btn-secondary",
            ),
            style=(
                "display: flex; justify-content: space-between; "
                "align-items: center; width: 100%;"
            ),
        ),
        bad_segment_action_group(
            description=(
                "Annotating segments as bad is recommended if you wish to keep "
                "all data but prevent ICA from using it to estimate components. "
                "Bad annotations are also displayed in the sleep report."
            ),
            buttons=[
                ("keep_detected_bad_segments", "Detected bad segments"),
                ("keep_displayed_bad_windows", "Detected bad windows"),
                ("confirm_bad_segments", "Saved selections"),
            ],
        ),
        bad_segment_action_group(
            description=(
                "Deleting data is recommended if you wish to omit the bad "
                "segments when running sleep staging."
            ),
            buttons=[
                ("delete_detected_bad_segments", "Detected bad segments"),
                ("delete_displayed_bad_windows", "Detected bad windows"),
                ("delete_confirm_bad_segments", "Saved selections"),
            ],
        ),
        ui.div(
            ui.input_action_button(
                "back_bad_segments_main_panel",
                "Back to main panel",
                class_="btn-secondary",
            ),
            style="margin-top: 10px;",
        ),
        style="margin-left: 15px; margin-right: 15px;",
    )

# Helper from function above
def bad_segment_action_group(description, buttons):
    return ui.div(
        ui.p(description),
        ui.div(
            *[
                ui.input_action_button(
                    button_id,
                    label,
                    class_="btn-primary",
                )
                for button_id, label in buttons
            ],
            style=(
                "display: flex; gap: 10px; align-items: center; "
                "flex-wrap: wrap;"
            ),
        ),
        style="margin-top: 10px;",
    )

# Labels for bad segment delete buttons
bad_segment_delete_button_labels = {
    "delete_detected_bad_segments": "Detected bad segments",
    "delete_displayed_bad_windows": "Detected bad windows",
    "delete_confirm_bad_segments": "Saved selections",
}

class BadSegmentReview:
    """Operations on one server session's bad-segment review state."""

    # Store reactive state for one bad-segment review
    def __init__(self, working_data, result, window_idx, saved_segments):
        self.working_data = working_data
        self.result = result
        self.window_idx = window_idx
        self.saved_segments = saved_segments

    # Clear temporary review state
    def reset(self):
        self.result.set(None)
        self.window_idx.set(0)
        self.saved_segments.set({})

    # Clear the review and return to the main panel
    def return_to_main(self, message):
        self.reset()
        ui.notification_show(message, type="message", duration=3000)
        ui.update_navset("main_nav", selected="panel-2")

    # Restore a delete button after its task finishes
    def restore_delete_button(self, button_id):
        if button_id:
            ui.update_action_button(
                button_id,
                label=bad_segment_delete_button_labels[button_id],
                disabled=False,
            )

    # Display an exception stored by an extended task
    @staticmethod
    def show_task_error(task):
        try:
            task.result()
        except Exception as error:
            ui.notification_show(str(error), type="error", duration=8000)

    # Copy reviewed annotations into the working data
    def commit_annotations(self, message):
        review_data = self.result.get()
        current_data = self.working_data.get()
        if review_data is None or current_data is None:
            return False

        updated_data = current_data.copy()
        updated_data.set_annotations(review_data.annotations)
        self.working_data.set(updated_data)
        self.return_to_main(message)
        return True

# Remove selected time ranges from the data
def delete_segments_from_data(data, segments):
    valid_segments = sorted((start, end) for start, end in segments if end > start)
    if not valid_segments:
        return data
    data_duration = data.times[-1]
    merged_segments = []
    for start, end in valid_segments:
        clipped_start = max(0, start)
        clipped_end = min(data_duration, end)
        if clipped_end <= clipped_start:
            continue
        if merged_segments and clipped_start <= merged_segments[-1][1]:
            merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], clipped_end))
        else:
            merged_segments.append((clipped_start, clipped_end))
    keep_segments = []
    keep_start = 0
    for delete_start, delete_end in merged_segments:
        if delete_start > keep_start:
            keep_segments.append((keep_start, delete_start))
        keep_start = max(keep_start, delete_end)
    if keep_start < data_duration:
        keep_segments.append((keep_start, data_duration))
    if not keep_segments:
        raise ValueError("Cannot delete all data. Select shorter bad segments.")
    kept_parts = [
        data.copy().crop(tmin=start, tmax=end, include_tmax=False)
        for start, end in keep_segments
        if end > start
    ]
    if len(kept_parts) == 1:
        return kept_parts[0]
    return mne.concatenate_raws(kept_parts)

# Save the selected range for one review window
def save_bad_segment_selection(ann_list, bad_window_idx, slider_range, saved_segments):
    idx = min(bad_window_idx, len(ann_list) - 1)
    w_start, w_end = ann_list[idx]
    rel_start, rel_end = slider_range
    updated_segments = dict(saved_segments)
    window_key = (w_start, w_end)
    is_last_window = idx == len(ann_list) - 1
    if rel_end <= rel_start:
        updated_segments.pop(window_key, None)
        return updated_segments, False, is_last_window
    new_start = max(w_start, w_start + rel_start)
    new_end = min(w_end, w_start + rel_end)
    if new_end <= new_start:
        updated_segments.pop(window_key, None)
        return updated_segments, False, is_last_window
    updated_segments[window_key] = (new_start, new_end)
    return updated_segments, True, is_last_window

# Build a table showing why each channel was marked bad
def bad_channel_criteria_table(nc, bad_chn):
    if not nc or not bad_chn:
        return None
    rows = []
    for ch in bad_chn:
        rows.append({
            "Channel": ch,
            "Lack of correlation with other channels": ch in nc.bad_by_correlation,
            "Standard deviation": ch in nc.bad_by_deviation,
            "Flat": ch in nc.bad_by_flat,
            "NaNs (if present)": ch in nc.bad_by_nan,  # Maybe add more criteria
        })
    return pd.DataFrame(rows)

# Return annotations that are not marked BAD
def keep_non_bad_annotations(data):
    old_ann = data.annotations
    keep_idx = [i for i, desc in enumerate(old_ann.description) if not desc.upper().startswith("BAD")]
    return old_ann[keep_idx]

# Return only annotations marked BAD
def keep_bad_annotations(data):
    old_ann = data.annotations
    keep_idx = [i for i, desc in enumerate(old_ann.description) if desc.upper().startswith("BAD")]
    return old_ann[keep_idx]

# Replace BAD annotations with the selected segments
def replace_bad_annotations(data, segs):
    new_annotations = keep_non_bad_annotations(data)
    valid_segments = [(start, end) for start, end in segs if end > start]
    if valid_segments:
        bad_window_annotations = mne.Annotations(
            onset=[annotation_onset_from_loaded_window(data, start) for start, _ in valid_segments],
            duration=[end - start for start, end in valid_segments],
            description=["BAD_bad_window"] * len(valid_segments),
            orig_time=data.info.get("meas_date"),
        )
        new_annotations = new_annotations + bad_window_annotations
    data.set_annotations(new_annotations)
    return data
