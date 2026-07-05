import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_PATH = r"C:\LULC"

OUTPUT_BASE = (
    r"C:\Users\bhuvan.NRSCADMIN\Desktop"
    r"\GeoSpatial Analysis _Analysis part"
    r"\rastertransitions_output"
)

os.makedirs(OUTPUT_BASE, exist_ok=True)

CSV_PATTERNS = {
    "2005-06 to 2011-12": "_0506_to_1112_decoded.csv",
    "2011-12 to 2015-16": "_1112_to_1516_decoded.csv",
}

focus_classes = [
    "Agriculture>Cropland",
    "Agriculture>Plantation",
    "Agriculture>Fallow",
    "Buildup>Urban",
    "Buildup>Rural",
    "Buildup>Mining",
    "Forest>Forest Plantation",
    "Wetlands>River/Stream/Canals",
    "Wetlands>Waterbodies"
]

focus_l1 = [
    "Agriculture",
    "Barren/Unculturable/Wastelands",
    "Builtup",
    "Forest",
    "Grass/Grazing",
    "Snow&Glacier",
    "Wetlands"
]

def save_plot(out_dir, fname):
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

state_folders = [
    folder for folder in os.listdir(BASE_PATH)
    if os.path.isdir(os.path.join(BASE_PATH, folder))
]

for state in state_folders:
    state_path = os.path.join(BASE_PATH, state)
    output_dir = os.path.join(OUTPUT_BASE, state)
    os.makedirs(output_dir, exist_ok=True)
    all_frames = []
    for period, suffix in CSV_PATTERNS.items():
        matching_files = [
            f for f in os.listdir(state_path)
            if f.endswith(suffix)
        ]
        if len(matching_files) == 0:
            continue
        file_name = matching_files[0]
        csv_path = os.path.join(state_path, file_name)
        df = pd.read_csv(csv_path)
        df["Period"] = period
        all_frames.append(df)

    if len(all_frames) == 0:
        continue

    ALL_DF = pd.concat(all_frames, ignore_index=True)
    change_df = ALL_DF[ (ALL_DF["From category"].isin(focus_classes)) | ( ALL_DF["To category"].isin(focus_classes)) ].copy()
    change_df = change_df[
        change_df["From category"] != change_df["To category"]
    ]

    change_df["Transition"] = (
        change_df["From category"]
        + "  to  "
        + change_df["To category"]
    )

    transition_summary = (
        change_df.groupby(
            ["Period", "Transition"]
        )["deg2"]
        .sum()
        .reset_index()
    )

    top_transitions = (
        transition_summary.groupby("Transition")["deg2"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index
    )

    plot_df = transition_summary[
        transition_summary["Transition"].isin(top_transitions)
    ]

    pivot_df = plot_df.pivot_table(
        index="Transition",
        columns="Period",
        values="deg2",
        aggfunc="sum",
        fill_value=0
    )

    pivot_df["Total"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values(
        by="Total",
        ascending=False
    )

    pivot_df = pivot_df.drop(columns="Total")
    periods = list(pivot_df.columns)
    percent_df = pivot_df.copy()
    if len(periods) >= 2:
        old_period = periods[0]
        new_period = periods[1]

        percent_df["Percent Change"] = (
            (percent_df[new_period]- percent_df[old_period] ) / percent_df[old_period].replace(0, pd.NA) ) * 100
        percent_df["Percent Change"] = (  percent_df["Percent Change"] .fillna(0))
    fig, axes = plt.subplots( 2, 1,figsize=(20, 16))
    pivot_df.plot(kind="bar",ax=axes[0], width=0.8)
    axes[0].set_ylabel("Area Changed (deg²)")
    axes[0].set_title( f"{state} : Major LULC Transitions")
    axes[0].tick_params( axis="x", rotation=75)
    axes[0].legend( title="Time Period")
    percent_df = percent_df.sort_values( by="Percent Change", ascending=False)
    percent_df["Percent Change"].plot(kind="bar",ax=axes[1] )
    axes[1].set_ylabel("Percent Change (%)" )
    axes[1].set_title(f"{state} : Percentage Change in LULC Transitions" )
    axes[1].tick_params(axis="x", rotation=75)
    plt.tight_layout()
    save_plot( output_dir,f"{state}_Major_LULC_Transitions.png")


    ALL_DF["From_L1"] = ( ALL_DF["From category"] .str.split(">") .str[0])
    ALL_DF["To_L1"] = (
        ALL_DF["To category"]
        .str.split(">")
        .str[0]
    )
    l1_df = ALL_DF[
        (ALL_DF["From_L1"].isin(focus_l1)) &
        (ALL_DF["To_L1"].isin(focus_l1))
    ].copy()

    l1_df = l1_df[ l1_df["From_L1"] != l1_df["To_L1"]]
    l1_df["Transition"] = (l1_df["From_L1"]+ "  to  " + l1_df["To_L1"])
    transition_summary = (
        l1_df.groupby(
            ["Period", "Transition"]
        )["deg2"]
        .sum()
        .reset_index()
    )
    pivot_df = transition_summary.pivot_table(
        index="Transition",
        columns="Period",
        values="deg2",
        aggfunc="sum",
        fill_value=0
    )
    pivot_df["Total"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values(
        by="Total",
        ascending=False
    )

    pivot_df = pivot_df.drop(columns="Total")
    ax = pivot_df.plot(
        kind="bar",
        figsize=(14, 7),
        width=0.8
    )
    plt.ylabel("Area Changed (deg²)")
    plt.title( f"{state} : Major Land Transitions")
    plt.xticks( rotation=45, ha="right")
    plt.legend(title="Time Period")
    plt.tight_layout()
    save_plot(output_dir, f"{state}_Major_L1_Transitions.png" )
