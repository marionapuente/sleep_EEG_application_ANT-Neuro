# Import packages
import os
import re
import tempfile
import mne
import numpy as np

CHANNEL_TYPE_CONFIG = {
    "load_emg_channels": ("emg", "emg", "EMG:"),
    "load_eog_channels": ("eog", "eog", "EOG:"),
    "load_ecg_channels": ("ecg", "ecg", "ECG:"),
    "load_respiration_channels": ("resp", "resp", "Respiration:"),
    "load_thermal_flow_channels": ("thermal_flow", "misc", "Thermal flow:"),
    "load_snore_channels": ("snore", "misc", "Snore:"),
    "load_body_position_channels": ("body_position", "misc", "Body position:"),
    "load_other_non_eeg_channels": ("other_non_eeg", "misc", "Other (not EEG):"),
}

# Normalize selected channels into a list
def selected_channels(value):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    return list(value)

# Function load data status
def status_and_load(path, window, resample_sfreq=None):
    correct_path = check_path(path)
    window_tuple = check_window(window)
    file_ext = os.path.splitext(correct_path)[1].lower()
    try:
        if file_ext == ".cnt":
            raw = mne.io.read_raw_ant(correct_path, preload=False)
        elif file_ext == ".edf":
            raw = mne.io.read_raw_edf(correct_path, preload=False)
        else:
            raise ValueError("Invalid file format. Please provide a .cnt or .edf file.")
    except Exception as e:
        raise ValueError(f"Could not read {file_ext.upper()} file: {e}")
    raw.set_channel_types({channel: "eeg" for channel in raw.ch_names})
    original_sfreq = raw.info["sfreq"]
    channel_names = raw.ch_names
    if window_tuple is not None:
        start, end = window_tuple
        raw.crop(tmin=start, tmax=end)
    if window_tuple is None:
        msg = f"Data loaded successfully from {correct_path} with full time window."
    else:
        msg = f"Data loaded successfully from {correct_path} with time window {window_tuple} seconds."
    target_sfreq = None
    if resample_sfreq is not None:
        resample_sfreq = float(resample_sfreq)
        if resample_sfreq <= 0:
            raise ValueError("Resampling frequency must be greater than 0.")
        if resample_sfreq > original_sfreq:
            raise ValueError(f"Resampling frequency ({resample_sfreq} Hz) cannot be higher than the original sampling frequency ({original_sfreq} Hz).")
        if resample_sfreq < original_sfreq:
            target_sfreq = resample_sfreq
            msg = f"{msg} Data will be resampled from {original_sfreq} Hz to {resample_sfreq} Hz when preprocessing is applied."
    return msg, raw, channel_names, target_sfreq

# Function to check path
def check_path(path):
    correct_path = str(path or "").strip().strip('"').strip("'")
    if not correct_path:
        raise ValueError("No file path provided. Please provide a valid file path.")
    if not correct_path.lower().endswith((".cnt", ".edf")):
        raise ValueError("Invalid file format. Please provide a .cnt or .edf file.")
    if not os.path.exists(correct_path):
        raise FileNotFoundError("File does not exist.")
    return correct_path

# Function to check window
def check_window(window):
    window = window.strip().strip('"').strip("'").lower()
    if window == "full" or window == "":
        return None
    try:
        start, end = window.split("-")
        start = float(start)
        end = float(end)
        if start < 0 or end <= start:
            raise ValueError
        return start, end
    except ValueError:
        raise ValueError("Invalid format. Use 'full' or 'start-end' in seconds separated by '-' (e.g., '0-3600'). End time must be after start time.")

# Set non-EEG channels from user selections
def set_and_describe_channels(data, input):
    if data is None:
        return None
    selections = {
        input_id: selected_channels(input[input_id]())
        for input_id in CHANNEL_TYPE_CONFIG
    }
    channel_groups = {
        channel: CHANNEL_TYPE_CONFIG[input_id][0]
        for input_id, channels in selections.items()
        for channel in channels
        if channel in data.ch_names
    }
    data.set_channel_types({channel: "eeg" for channel in data.ch_names})
    mapping = {
        channel: CHANNEL_TYPE_CONFIG[input_id][1]
        for input_id, channels in selections.items()
        for channel in channels
        if channel in data.ch_names
    }
    if mapping:
        data.set_channel_types(mapping)
    return channel_groups, data

# Update channel type choices
def update_channel_type_choices(data, input, ui):
    if data is None:
        return
    chn_type_ids = list(CHANNEL_TYPE_CONFIG)
    if any(input_id not in input for input_id in chn_type_ids):
        return
    selections = {
        input_id: selected_channels(input[input_id]())
        for input_id in chn_type_ids
    }
    for input_id, selected in selections.items():
        selected_elsewhere = {
            channel
            for other_input_id, channels in selections.items()
            if other_input_id != input_id
            for channel in channels
        }
        choices = [
            channel
            for channel in data.ch_names
            if channel not in selected_elsewhere
        ]
        ui.update_selectize(input_id, choices=choices, selected=selected)

# Channel type selection UI
def channel_type_selection_ui(ui, message, channel_names):
    selectors = [
        ui.input_selectize(
            input_id,
            config[2],
            choices=channel_names,
            selected=[],
            multiple=True,
        )
        for input_id, config in CHANNEL_TYPE_CONFIG.items()
    ]
    return ui.div(
        ui.p(message),
        ui.div(
            ui.h4("Indicate the channel types:"),
            ui.p(
                "Channel types are assigned according to channel names "
                "and not channel index."
            ),
            *selectors,
            style=(
                "color: black; margin-top: 18px; "
                "margin-bottom: 18px; max-width: 700px;"
            ),
        ),
        ui.input_action_button(
            "show_ts",
            "Display time series and preprocess data",
            style=(
                "background-color: #4CAF50; color: white; "
                "border: none;"
            ),
        ),
        style="color: green; font-size: 18px;",
    )

# Reset data and restore the related UI controls
def finish_reset(data, message, working_data, rereferencing_applied, ui):
    working_data.set(data)
    ui.update_selectize(
        "channels",
        choices=data.ch_names,
        selected=data.ch_names,
    )
    rereferencing_applied.set(False)
    ui.update_select(
        "reference",
        choices=[
            "Recorded",
            "Average",
            "Cz",
            "Another single channel:",
            "Average of these channels:",
        ],
        selected="Recorded",
    )
    ui.notification_show(message, type="message", duration=3000)

# Update the label and disabled state of a workflow button
def update_active_workflow_button(ui, disabled, active_id, active_label=None):
    workflow_button_labels = {
        "apply_montage": "Keep only channels displayed and apply filters selected to the whole data",
        "apply_main_montage": "Apply montage",
        "detect_bad_channels": "Detect bad channels",
        "apply_bad_channel_handling": "Apply handling",
        "detect_bad_segments": "Detect bad segments",
        "rereferencing": "Apply selected re-reference to whole data",
        "run_ica": "Run ICA",
        "reset": "Reset data to original",
        "reset_keep_bad_annotations": "Reset data keeping bad annotations",
        "proceed_sleep_staging": "Proceed to sleep staging",
        "show_insomnia_umap": "Generate insomnia UMAP",
    }
    label = workflow_button_labels.get(active_id)
    if label is not None:
        ui.update_action_button(
            active_id,
            label=active_label or label,
            disabled=disabled,
        )

# Read ANT-Neuro electrode positions from an ELC file
def read_ant_elc(elc_path):
    ch_pos = {}
    with open(elc_path, "r", encoding="utf-8", errors="ignore") as file:
        read_positions = False
        for line in file:
            line = line.strip()
            if line == "Positions":
                read_positions = True
                continue
            if not read_positions or ":" not in line:
                continue
            name, coords = line.split(":", maxsplit=1)
            xyz = coords.split()
            if len(xyz) != 3:
                continue
            ch_pos[name.strip()] = np.array([float(value) for value in xyz]) / 1000

    ch_pos.pop("EOG", None)
    if not ch_pos:
        raise ValueError("Could not read electrode positions from the .elc file.")
    ch_pos = {
        channel: np.array([-position[1], position[0], position[2]])
        for channel, position in ch_pos.items()
    }
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")

# Apply a standard or custom montage to a data copy
def apply_montage_selection(data, montage, custom_file, ui):
    if data is None:
        return None
    processed = data.copy()
    if montage == "custom":
        if not custom_file or not custom_file[0]["name"].lower().endswith(".elc"):
            ui.notification_show("Please upload a valid .elc file.", type="error", duration=5000)
            return None
        try:
            montage_to_use = read_ant_elc(custom_file[0]["datapath"])
            processed.set_montage(montage_to_use, match_case=False, on_missing="warn")
        except Exception as error:
            ui.notification_show(f"Could not apply the .elc montage: {error}", type="error", duration=8000)
            return None
        return processed
    if montage in ("standard_1020", "standard_1005"):
        processed.set_montage(montage, match_case=False, on_missing="warn")
        return processed
    return None

# Check whether data has a valid montage
def has_montage(data):
    if data is None:
        return False
    montage = data.get_montage()
    return montage is not None and bool(getattr(montage, "ch_names", []))

# Return EEG channel names that lack a valid, unique position
def channels_missing_position(data):
    if data is None:
        return []
    names, positions = [], []
    for ch, ch_type in zip(data.info["chs"], data.get_channel_types()):
        if ch_type != "eeg":
            continue
        names.append(ch["ch_name"])
        positions.append(ch["loc"][:3])
    if not names:
        return []
    positions = np.array(positions)
    names = np.array(names)
    nan_mask = np.any(np.isnan(positions), axis=1)
    missing = set(names[nan_mask].tolist())
    valid_positions = positions[~nan_mask]
    valid_names = names[~nan_mask]
    if len(valid_positions) > 0:
        _, inverse, counts = np.unique(valid_positions, axis=0, return_inverse=True, return_counts=True)
        missing.update(valid_names[counts[inverse] > 1].tolist())
    return sorted(missing)

# Export data to EDF format
def export_data(data, resample_sfreq=None):
    if data is None:
        return None
    tmp_path = None
    try:
        export_raw = data.copy().load_data()
        if resample_sfreq is not None and resample_sfreq < export_raw.info["sfreq"]:
            export_raw.resample(resample_sfreq)
        subject_info = export_raw.info.get("subject_info")
        if subject_info is not None:
            subject_info = dict(subject_info)
            for field in ("his_id", "first_name", "middle_name", "last_name"):
                value = subject_info.get(field)
                if value:
                    subject_info[field] = re.sub(r"\s+", "_", str(value).strip())
            export_raw.info["subject_info"] = subject_info
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp_path = tmp.name
        mne.export.export_raw(tmp_path, export_raw, fmt="edf", overwrite=True)
        with open(tmp_path, "rb") as edf_file:
            return edf_file.read()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
