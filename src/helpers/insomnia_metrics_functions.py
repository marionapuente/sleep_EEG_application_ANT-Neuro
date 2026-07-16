# Import packages
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.sleep_staging_functions import bandpower_table, selected_sleep_statistics, sleep_statistics_values

DFA_FREQUENCY_RANGE = (1.0, 45.0)
DFA_WINDOW_SECONDS = 300.0
DFA_RUNTIME = "c"
DFA_FALLBACK_RUNTIME = "python"
TOP20_INSOMNIA_FEATURES = [
    "SE",
    "WASO",
    "BANDPOWER_OVERALL_BETA",
    "BANDPOWER_N3_TOTALABSPOW",
    "BANDPOWER_REM_SIGMA",
    "SOL",
    "BANDPOWER_REM_TOTALABSPOW",
    "BANDPOWER_OVERALL_THETA",
    "BANDPOWER_WAKE_THETA",
    "BANDPOWER_N3_ALPHA",
    "BANDPOWER_N3_BETA",
    "BANDPOWER_OVERALL_SIGMA",
    "BANDPOWER_N1_SIGMA",
    "BANDPOWER_N3_DELTA",
    "BANDPOWER_N2_TOTALABSPOW",
    "BANDPOWER_N3_SIGMA",
    "BANDPOWER_REM_THETA",
    "BANDPOWER_N1_BETA",
    "BANDPOWER_N1_THETA",
    "BANDPOWER_N1_TOTALABSPOW",
]

# Load the optional crosci DFA functions
def _crosci_functions():
    try:
        from crosci.biomarkers import compute_spectrum_biomarkers, get_frequency_bins
    except ImportError as error:
        print(f"Could not import crosci DFA functions: {error}")
        return None, None
    return compute_spectrum_biomarkers, get_frequency_bins

# Extract a centered window of the requested duration
def _middle_window(data, duration_seconds):
    sfreq = data.info["sfreq"]
    total_seconds = data.n_times / sfreq
    if total_seconds <= duration_seconds:
        return data.copy()
    tmin = (total_seconds - duration_seconds) / 2
    tmax = tmin + duration_seconds
    return data.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)

# Limit the DFA frequency range to valid sampled frequencies
def _dfa_frequency_range(sfreq):
    lower, upper = DFA_FREQUENCY_RANGE
    upper = min(upper, sfreq / 2 - 0.1)
    if upper <= lower:
        return None
    return [lower, upper]

# Convert the sampling frequency to the integer crosci requires
def _crosci_sampling_frequency(sfreq):
    rounded = int(round(sfreq))
    if not np.isclose(sfreq, rounded):
        raise ValueError(f"crosci requires an integer sampling frequency, got {sfreq} Hz.")
    return rounded

# Create a column label for a DFA frequency band
def _dfa_band_label(frequency_range):
    labels = []
    for frequency in frequency_range:
        rounded = f"{frequency:.2f}".rstrip("0").rstrip(".")
        labels.append(rounded.replace(".", "_"))
    return f"{labels[0]}_{labels[1]}HZ"

# Calculate DFA values for one channel
def dfa_table(data, result, selected_channel):
    if data is None or result is None:
        print("Skipping DFA: data or sleep staging result is missing.")
        return None
    compute_spectrum_biomarkers, get_frequency_bins = _crosci_functions()
    if compute_spectrum_biomarkers is None:
        print("Skipping DFA: crosci is not available.")
        return None
    channels = [selected_channel]
    channels = [channel for channel in channels if channel in data.ch_names]
    if not channels:
        print(f"Skipping DFA for {selected_channel}: channel is not present in the data.")
        return None
    analysis_data = _middle_window(data.copy().pick(channels), DFA_WINDOW_SECONDS)
    sfreq = analysis_data.info["sfreq"]
    try:
        crosci_sfreq = _crosci_sampling_frequency(sfreq)
    except ValueError as error:
        print(f"Skipping DFA for {selected_channel}: {error}")
        return None
    frequency_range = _dfa_frequency_range(sfreq)
    if frequency_range is None:
        print(f"Skipping DFA for {selected_channel}: sampling frequency {sfreq} Hz is too low for range {DFA_FREQUENCY_RANGE}.")
        return None
    signal_matrix = analysis_data.get_data()
    if signal_matrix.size == 0:
        print(f"Skipping DFA for {selected_channel}: selected signal is empty.")
        return None

    biomarkers = None
    for runtime in [DFA_RUNTIME, DFA_FALLBACK_RUNTIME]:
        try:
            print(f"Computing DFA for {selected_channel} using crosci runtime='{runtime}' over middle {DFA_WINDOW_SECONDS:.0f}s.")
            biomarkers = compute_spectrum_biomarkers(
                signal_matrix,
                crosci_sfreq,
                frequency_range,
                runtime=runtime,
                biomarkers_to_compute=["DFA"],
            )
            break
        except Exception as error:
            print(f"DFA failed for {selected_channel} with crosci runtime='{runtime}': {error}")
            continue
    if biomarkers is None:
        print(f"Skipping DFA for {selected_channel}: all crosci runtimes failed.")
        return None
    dfa_matrix = np.asarray(biomarkers.get("DFA"), dtype=float)
    if dfa_matrix.ndim != 2 or dfa_matrix.size == 0:
        print(f"Skipping DFA for {selected_channel}: crosci returned an empty or invalid DFA matrix.")
        return None
    frequency_bins = get_frequency_bins(frequency_range)
    if len(frequency_bins) != dfa_matrix.shape[1]:
        print(f"DFA bin count mismatch for {selected_channel}: {len(frequency_bins)} bins, matrix has {dfa_matrix.shape[1]} columns.")
        frequency_bins = frequency_bins[:dfa_matrix.shape[1]]
    row = {"Stage": "Overall"}
    for idx, frequency_bin in enumerate(frequency_bins):
        values = dfa_matrix[:, idx]
        row[_dfa_band_label(frequency_bin)] = np.nanmean(values) if np.isfinite(values).any() else np.nan

    return pd.DataFrame([row])

# Combine sleep, bandpower, and DFA metrics for one channel
def insomnia_metrics_table(data, result, selected_channel):
    table = bandpower_table(data, result, selected_channel)
    dfa = dfa_table(data, result, selected_channel)
    stats_values = sleep_statistics_values(selected_sleep_statistics(result, selected_channel))
    metrics = {
        "SOL": stats_values.get("SOL"),
        "SOL_5MIN": stats_values.get("SOL_5MIN"),
        "WASO": stats_values.get("WASO"),
        "SE": stats_values.get("SE"),
    }
    if table is not None:
        table = table.copy()
        if table.index.names and any(name is not None for name in table.index.names):
            table = table.reset_index()
        if "Stage" in table.columns:
            table["Stage"] = table["Stage"].replace({0: "WAKE", 1: "N1", 2: "N2", 3: "N3", 4: "REM"})
        skip_cols = {"Chan", "Channel", "Stage", "FreqRes", "Relative"}
        band_cols = [col for col in table.select_dtypes(include="number").columns if col not in skip_cols]
        if "Stage" in table.columns:
            for _, row in table.iterrows():
                stage = str(row["Stage"]).upper()
                for band in band_cols:
                    metrics[f"BANDPOWER_{stage}_{str(band).upper()}"] = row[band]
    if dfa is not None:
        dfa = dfa.copy()
        if dfa.index.names and any(name is not None for name in dfa.index.names):
            dfa = dfa.reset_index()
        if "Stage" in dfa.columns:
            for _, row in dfa.iterrows():
                stage = str(row["Stage"]).upper()
                for band in [col for col in dfa.columns if col != "Stage"]:
                    if band in row:
                        metrics[f"DFA_{stage}_{band}"] = row[band]
    return pd.DataFrame([metrics])

# Build insomnia metrics for all staged channels and their average
def make_insomnia_metrics_table(data, result):
    if data is None or result is None:
        print("Skipping insomnia metrics export: data or sleep staging result is missing.")
        return None
    channels = [channel for channel in result["staging"].keys() if channel in data.ch_names]
    if not channels:
        print("Skipping insomnia metrics export: no sleep-staged channels are present in the data.")
        return None
    rows = []
    for channel in channels:
        table = insomnia_metrics_table(data, result, channel)
        if table is None or table.empty:
            continue
        row = table.iloc[0].to_dict()
        row = {"Channel": channel, **row}
        rows.append(row)
    if not rows:
        return None
    table = pd.DataFrame(rows)
    numeric_cols = table.select_dtypes(include="number").columns
    average_row = {column: np.nan for column in table.columns}
    average_row["Channel"] = "Average"
    for column in numeric_cols:
        average_row[column] = table[column].mean(skipna=True)
    return pd.concat([pd.DataFrame([average_row]), table], ignore_index=True)

# Select the 20 features used by the insomnia UMAP
def top20_insomnia_metrics_table(table):
    if table is None:
        return None
    table = table.copy()
    if "Channel" in table.columns and (table["Channel"] == "Average").any():
        table = table[table["Channel"] == "Average"].head(1)
    else:
        table = table.head(1)
    for feature in TOP20_INSOMNIA_FEATURES:
        if feature not in table.columns:
            table[feature] = pd.NA
    return table[["Channel", *TOP20_INSOMNIA_FEATURES]]

# List unavailable values among the 20 UMAP features
def missing_insomnia_features(table):
    if table is None or table.empty:
        return list(TOP20_INSOMNIA_FEATURES)
    row = table.iloc[0]
    return [
        feature
        for feature in TOP20_INSOMNIA_FEATURES
        if feature not in table.columns
        or pd.isna(pd.to_numeric(row[feature], errors="coerce"))
    ]

# Combine reference and user metrics for CSV download
def insomnia_umap_metrics_download_table(user_top20_table, user_label="Current data"):
    if user_top20_table is None or user_top20_table.empty:
        raise ValueError("No insomnia metrics are available for the current data.")
    user_label = str(user_label or "").strip() or "Current data"
    reference_path = Path(__file__).with_name("top20_svm_umap_input_features.csv")
    if not reference_path.exists():
        raise ValueError(f"Reference UMAP feature file not found: {reference_path}")
    reference_df = pd.read_csv(reference_path)
    if "Dataset" not in reference_df.columns:
        raise ValueError("Reference UMAP feature file must include a Dataset column.")
    reference_features = [f"Average_{feature}" for feature in TOP20_INSOMNIA_FEATURES]
    missing_reference = [feature for feature in reference_features if feature not in reference_df.columns]
    if missing_reference:
        raise ValueError("Reference UMAP feature file is missing: " + ", ".join(missing_reference))
    reference_out = pd.DataFrame()
    reference_out["Dataset"] = np.where(
        reference_df["Dataset"].astype(str).str.lower().str.contains("hlth_"),
        "Healthy",
        "Insomnia",
    )
    for feature, reference_feature in zip(TOP20_INSOMNIA_FEATURES, reference_features):
        reference_out[feature] = pd.to_numeric(reference_df[reference_feature], errors="coerce")
    user_row = user_top20_table.iloc[[0]].copy()
    missing_user = [feature for feature in TOP20_INSOMNIA_FEATURES if feature not in user_row.columns]
    if missing_user:
        raise ValueError("Current insomnia metrics are missing: " + ", ".join(missing_user))
    user_out = pd.DataFrame([{"Dataset": user_label}])
    for feature in TOP20_INSOMNIA_FEATURES:
        user_out[feature] = pd.to_numeric(user_row[feature], errors="coerce").iloc[0]

    return pd.concat([reference_out, user_out], ignore_index=True)

# Place the current recording in the reference UMAP space
def plot_insomnia_umap(user_top20_table, user_label="Current data"):
    try:
        import umap
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ValueError(f"Could not import UMAP dependencies: {error}")

    if user_top20_table is None or user_top20_table.empty:
        raise ValueError("No insomnia metrics are available for the current data.")
    user_label = str(user_label or "").strip() or "Current data"
    reference_path = Path(__file__).with_name("top20_svm_umap_input_features.csv")
    if not reference_path.exists():
        raise ValueError(f"Reference UMAP feature file not found: {reference_path}")
    reference_df = pd.read_csv(reference_path)
    if "Dataset" not in reference_df.columns:
        raise ValueError("Reference UMAP feature file must include a Dataset column.")
    reference_features = [f"Average_{feature}" for feature in TOP20_INSOMNIA_FEATURES]
    missing_reference = [feature for feature in reference_features if feature not in reference_df.columns]
    if missing_reference:
        raise ValueError("Reference UMAP feature file is missing: " + ", ".join(missing_reference))
    user_row = user_top20_table.iloc[[0]].copy()
    missing_user = [feature for feature in TOP20_INSOMNIA_FEATURES if feature not in user_row.columns]
    if missing_user:
        raise ValueError("Current insomnia metrics are missing: " + ", ".join(missing_user))
    reference_x = reference_df[reference_features].apply(pd.to_numeric, errors="coerce")
    user_x = user_row[TOP20_INSOMNIA_FEATURES].apply(pd.to_numeric, errors="coerce")
    user_x.columns = reference_features

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    reference_scaled = scaler.fit_transform(imputer.fit_transform(reference_x))
    user_scaled = scaler.transform(imputer.transform(user_x))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(5, len(reference_df) - 1),
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    reference_embedding = reducer.fit_transform(reference_scaled)
    user_embedding = reducer.transform(user_scaled)
    plot_df = pd.DataFrame(
        reference_embedding,
        columns=["UMAP1", "UMAP2"],
    )
    plot_df["Dataset"] = reference_df["Dataset"].astype(str)
    plot_df["Category"] = np.where(
        plot_df["Dataset"].str.lower().str.contains("hlth_"),
        "Healthy",
        "Insomnia",
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    colors = {"Healthy": "#2ca02c", "Insomnia": "#ff7f0e"}
    for category, color in colors.items():
        group = plot_df[plot_df["Category"] == category]
        ax.scatter(
            group["UMAP1"],
            group["UMAP2"],
            color=color,
            label=f"{category} reference (n={len(group)})",
            s=58,
            alpha=0.8,
        )
    ax.scatter(
        user_embedding[0, 0],
        user_embedding[0, 1],
        color="#1f77b4",
        edgecolor="#202124",
        marker="*",
        s=220,
        label=user_label,
        zorder=5,
    )
    ax.set_title("Current data in reference insomnia UMAP space")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig
