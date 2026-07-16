# Get current app channel types
def channel_group_for_name(data, channel_name, app_channel_types=None):
    if app_channel_types and channel_name in app_channel_types:
        return app_channel_types[channel_name]
    if channel_name not in data.ch_names:
        return None
    return dict(zip(data.ch_names, data.get_channel_types()))[channel_name]

# Return the channel type for every channel in the dataset
def get_channel_groups(data, app_channel_types=None):
    mne_channel_types = dict(zip(data.ch_names, data.get_channel_types()))
    return [
        app_channel_types[channel_name]
        if app_channel_types and channel_name in app_channel_types
        else mne_channel_types[channel_name]
        for channel_name in data.ch_names
    ]

# Sort channel types with EEG first
def sort_channel_groups(channel_groups):
    return sorted(set(channel_groups), key=lambda x: (x != "eeg", x))

# Keep requested channel names that exist in the data
def existing_channels(data, channels):
    if not channels:
        return []
    available_channels = set(data.ch_names)
    return [str(channel) for channel in channels if str(channel) in available_channels]

# Select valid requested channels from the data
def pick_existing_channels(data, channels):
    if not channels:
        return
    valid_channels = existing_channels(data, channels)
    if not valid_channels:
        raise ValueError("None of the selected channels exist in the data.")
    data.pick(valid_channels)

# Convert channel input into a list of names
def parse_channel_names(channels):
    if isinstance(channels, str):
        return [ch.strip() for ch in channels.split(",") if ch.strip()]
    return channels or []

# Return the channel types currently selected for display
def get_active_channel_types(data, channel_mode, channels, app_channel_types=None):
    data = data.copy()
    if channel_mode == "Custom set" and channels:
        pick_existing_channels(data, channels)
    return sort_channel_groups(get_channel_groups(data, app_channel_types))

# Store settings values for different types of channels
def channel_type_filters_scaling_function(data, input, channel_mode, channels, app_channel_types=None):
    if data is None:
        return None
    settings = {}
    for ch_type in get_active_channel_types(data, channel_mode, channels, app_channel_types):
        low = input[f"low_{ch_type}"]()
        high = input[f"high_{ch_type}"]()
        notch = input[f"notch_{ch_type}"]()
        scaling = input[f"scaling_{ch_type}"]()
        settings[ch_type] = {
            "low": round(low, 2) if low is not None else None,
            "high": round(high, 2) if high is not None else None,
            "notch": str(notch),
            "scaling": round(scaling, 2) if scaling is not None else None,
        }
    return settings

# Check filter frequencies against the sampling rate
def validate_filter_settings(filter_settings, sfreq):
    if not filter_settings:
        return
    nyquist = sfreq / 2
    for ch_type, settings in filter_settings.items():
        low = settings.get("low")
        high = settings.get("high")
        notch = settings.get("notch")
        l_freq = low if low is not None and low > 0 else None
        h_freq = high if high is not None and high > 0 else None

        try:
            notch_freq = float(str(notch).replace(",", "."))
        except (TypeError, ValueError):
            notch_freq = None

        if l_freq is not None and l_freq >= nyquist:
            raise ValueError(
                f"{ch_type.upper()} high-pass frequency ({l_freq:g} Hz) must be lower than Nyquist "
                f"({nyquist:g} Hz). Choose a value below {nyquist:g} Hz or increase the resampling frequency."
            )
        if h_freq is not None and h_freq >= nyquist:
            raise ValueError(
                f"{ch_type.upper()} low-pass frequency ({h_freq:g} Hz) must be lower than Nyquist "
                f"({nyquist:g} Hz). Choose a value below {nyquist:g} Hz or increase the resampling frequency."
            )
        if l_freq is not None and h_freq is not None and l_freq >= h_freq:
            raise ValueError(
                f"{ch_type.upper()} high-pass frequency ({l_freq:g} Hz) must be lower than the low-pass "
                f"frequency ({h_freq:g} Hz)."
            )
        if notch_freq in (50, 60) and notch_freq >= nyquist:
            raise ValueError(
                f"{ch_type.upper()} notch frequency ({notch_freq:g} Hz) must be lower than Nyquist "
                f"({nyquist:g} Hz). Choose no notch, a lower notch frequency, or increase the resampling frequency."
            )

# Apply the selected EEG reference
def apply_rereference(processed_data, reference, single_channel, avg_channels):
    if reference == "Average":
        processed_data.set_eeg_reference("average")
    elif reference == "Cz":
        if "Cz" not in processed_data.ch_names:
            raise ValueError("Cz channel is not in the data.")
        processed_data.set_eeg_reference(["Cz"])
    elif reference == "Another single channel:":
        selected_channel = single_channel
        if isinstance(selected_channel, list):
            selected_channel = (
                selected_channel[0] if selected_channel else None
            )
        if selected_channel not in processed_data.ch_names:
            raise ValueError(
                f"{selected_channel} channel is not in the data."
            )
        processed_data.set_eeg_reference([selected_channel])
    elif reference == "Average of these channels:":
        selected_channels = parse_channel_names(avg_channels)
        valid_channels = [
            channel
            for channel in selected_channels
            if channel in processed_data.ch_names
        ]
        if not valid_channels:
            raise ValueError(
                "None of the specified channels are in the data."
            )
        processed_data.set_eeg_reference(valid_channels)
    return processed_data

# Select, reference, resample, and filter the requested data
def prepare_filtered_data(step, data, w_start, w_end, channel_mode, channels, reference, single_channel, avg_channels, filter_settings, resample_sfreq=None, app_channel_types=None):
    if data is None:
        return None

    if data is not None:

        if step == "apply_montage":
            data.load_data()
        data_to_process = data.copy()
        if w_start is not None and w_end is not None:
            data_to_process.crop(tmin=w_start, tmax=w_end)
        if step is None or step == 'apply_montage':
            if channel_mode == "Custom set" and channels:
                pick_existing_channels(data_to_process, channels)
        data_to_process.load_data()
        if resample_sfreq is not None and resample_sfreq < data_to_process.info["sfreq"]:
            data_to_process.resample(resample_sfreq)
        
        if step is None or step == 'rereference':
            apply_rereference(
                data_to_process,
                reference,
                single_channel,
                avg_channels,
            )

        if step is None or step == 'apply_montage':
            validate_filter_settings(filter_settings, data_to_process.info["sfreq"])
            ch_names = data_to_process.ch_names
            ch_types = get_channel_groups(data_to_process, app_channel_types)
            for ch_type, settings in filter_settings.items():
                picks = [ch for ch, typ in zip(ch_names, ch_types) if typ == ch_type]
                if not picks:
                    continue
                low = settings.get("low")
                high = settings.get("high")
                notch = settings.get("notch")
                l_freq = low if low is not None and low > 0 else None
                h_freq = high if high is not None and high > 0 else None
                if l_freq is not None or h_freq is not None:
                    data_to_process.filter(l_freq=l_freq, h_freq=h_freq, picks=picks)
                try:
                    notch = float(str(notch).replace(",", "."))
                except (TypeError, ValueError):
                    notch = None
                if notch in (50, 60):
                    data_to_process.notch_filter(freqs=notch, picks=picks)
    
    return data_to_process