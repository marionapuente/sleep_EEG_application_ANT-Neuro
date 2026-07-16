# Import packages
import math
import re
import mne
from mne_icalabel import label_components
import matplotlib.pyplot as plt

# Determine the height of the ICA plot
def height_ica_plot(ica, n_cols=6):
    n_rows = math.ceil(ica.n_components_ / n_cols)
    return round(max(460, n_rows * 220) * 0.82)

# Prepare data and fit ICA
def prepare_data_and_fit_ica(data):
    eeg_data = data.copy().pick("eeg").load_data()
    if eeg_data.info["sfreq"] > 250:
        eeg_data.resample(250)
    n_components = min(30, len(eeg_data.ch_names))
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter="auto", method="fastica")
    ica.fit(eeg_data, reject_by_annotation=True)
    return ica, eeg_data

# Label ICA components using ICLabel
def labelling_ica(eeg_data, ica):
    eeg_data.load_data()
    return ica, label_components(eeg_data, ica, method="iclabel")

# Get components to remove based on ICLabel results
def get_components_to_remove(ica_labels, threshold=0.80):
    if not ica_labels:
        return []
    labels = ica_labels.get("labels", [])
    probs = ica_labels.get("y_pred_proba", [])
    artifact_labels = {"eye", "eye blink", "muscle", "muscle artifact", "heart", "heart beat"}
    components = []
    for idx, label in enumerate(labels):
        confidence = probs[idx].max() if idx < len(probs) else 0
        if str(label).lower() in artifact_labels and confidence >= threshold:
            components.append(idx)
    return components

# Check user input for ICA removal
def parse_component_list(text):
    if not text or not text.strip():
        return []
    clean_text = text.strip()
    number_pattern = r"\d+"
    ic_pattern = r"(?i:ic)\s*\d+"
    if re.fullmatch(rf"{number_pattern}(?:,\s*{number_pattern})*", clean_text):
        return [int(value) for value in re.findall(r"\d+", clean_text)]
    if re.fullmatch(rf"{ic_pattern}(?:,\s*{ic_pattern})*", clean_text):
        return [int(value) for value in re.findall(r"\d+", clean_text)]
    raise ValueError("Wrong input. Use numbers separated by commas, such as 2, 3, or IC labels, such as IC3, IC4.")

# Validate components from user's input
def parse_and_validate_components(text, n_components):
    components = parse_component_list(text)
    if len(components) != len(set(components)):
        raise ValueError("Each ICA component can only be listed once.")
    invalid = [
        component
        for component in components
        if not 0 <= component < n_components
    ]
    if invalid:
        raise ValueError(
            f"ICA components must be between 0 and {n_components - 1}."
        )
    return components

# Plotting ICA components with labels and confidence scores
def plot_ica(ica, ica_labels=None):
    n_components = ica.n_components_
    ica_labels = ica_labels or {}
    labels = ica_labels.get("labels", ["unlabeled"] * n_components)
    probs = ica_labels.get("y_pred_proba", None)
    n_cols = 6
    n_rows = math.ceil(n_components / n_cols)
    fig = plt.figure(figsize=(4.0 * n_cols, 2.2 * n_rows * 0.82))
    outer = fig.add_gridspec(n_rows, n_cols, wspace=0.35, hspace=0.8)
    for i in range(n_components):
        row = i // n_cols
        col = i % n_cols
        label = labels[i] if i < len(labels) else "unknown"
        confidence = ""
        if probs is not None and i < len(probs):
            confidence = f" ({probs[i].max() * 100:.0f}%)"
        topo_ax = fig.add_subplot(outer[row, col])
        ica.plot_components(picks=i, axes=topo_ax, show=False)
        topo_ax.set_title(f"IC {i} - {label}{confidence}", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.03, wspace=0.35, hspace=0.8)
    return fig

# Show ICA using function above
def show_ica(eeg_data):
    ica, ica_labels = labelling_ica(eeg_data)
    return plot_ica(ica, ica_labels)
