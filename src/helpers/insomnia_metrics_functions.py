# Import packages
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Extract the selected insomnia features for one staged channel
def _insomnia_features_for_channel(result, selected_channel):
    stats_values = result["staging"][selected_channel]["sleep_statistics"]
    if isinstance(stats_values, pd.Series):
        stats_values = stats_values.to_dict()
    elif isinstance(stats_values, pd.DataFrame):
        stats_values = {} if stats_values.empty else stats_values.iloc[0].to_dict()
    elif not isinstance(stats_values, dict):
        stats_values = dict(stats_values)

    metrics = {
        "SOL": stats_values.get("SOL"),
        "WASO": stats_values.get("WASO"),
        "SE": stats_values.get("SE"),
    }
    table = result.get("bandpower_tables", {}).get(selected_channel)
    if table is not None:
        table = table.copy()
        if table.index.names and any(name is not None for name in table.index.names):
            table = table.reset_index()
        if "Stage" in table.columns:
            table["Stage"] = table["Stage"].replace({0: "WAKE", 1: "N1", 2: "N2", 3: "N3", 4: "REM"})
            skip_cols = {"Chan", "Channel", "Stage", "FreqRes", "Relative"}
            band_cols = [col for col in table.select_dtypes(include="number").columns if col not in skip_cols]
            for _, row in table.iterrows():
                stage = str(row["Stage"]).upper()
                for band in band_cols:
                    feature = (f"BANDPOWER_{stage}_{str(band).upper()}")
                    if feature in TOP20_INSOMNIA_FEATURES:
                        metrics[feature] = row[band]
    return {
        feature: metrics.get(feature)
        for feature in TOP20_INSOMNIA_FEATURES
    }

# Build the averaged top-20 feature table used by the insomnia visualization
def build_top20_insomnia_table(result):
    if result is None:
        print("Skipping insomnia metrics: sleep staging result is missing.")
        return None
    channels = list(result.get("staging", {}))
    if not channels:
        print("Skipping insomnia metrics: no sleep-staged channels are available.")
        return None

    channel_table = pd.DataFrame([
        _insomnia_features_for_channel(result, channel)
        for channel in channels
    ])
    average_row = {"Channel": "Average"}
    for feature in TOP20_INSOMNIA_FEATURES:
        average_row[feature] = pd.to_numeric(
            channel_table[feature],
            errors="coerce",
        ).mean(skipna=True)
    return pd.DataFrame([average_row], columns=["Channel", *TOP20_INSOMNIA_FEATURES])

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
