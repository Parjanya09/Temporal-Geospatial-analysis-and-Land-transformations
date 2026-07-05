import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_PATH = r"C:\LULC"

CSV_FILES = {
    "2005-06_to_2011-12": "RasterChanges_MP_0506_to_1112.csv",
    "2011-12_to_2015-16": "RasterChanges_MP_1112_to_1516.csv",
    "2005-06_to_2015-16": "RasterChanges_MP_0506_to_1516.csv",
}

LU_DICT = {
    1: "Buildup>Urban",
    2: "Buildup>Rural",
    3: "Buildup>Mining",
    4: "Agriculture>Cropland",
    5: "Agriculture>Plantation",
    6: "Agriculture>Fallow",
    8: "Forest>Evergreen/Semi Evergreen",
    9: "Forest>Decidious",
    10: "Forest>Forest Plantation",
    11: "Forest>Scrub Forest",
    12: "Forest>Swamp/Mangroove Forest",
    13: "Grass-Grazing>Grass-Grazing",
    14: "Barren/Unculturable/Wastelands>Salt affected land",
    15: "Barren/Unculturable/Wastelands>Gullied",
    16: "Barren/Unculturable/Wastelands>Scrubland",
    17: "Barren/Unculturable/Wastelands>Sandyland",
    18: "Barren/Unculturable/Wastelands>Barren Rocks",
    20: "Wetlands/Waterbodies>Inland Wetland",
    22: "Wetlands/Waterbodies>River/Stream/Canals",
    23: "Wetlands/Waterbodies>Waterbodies",
}

VALID_CODES = [
    1, 2, 3, 4, 5, 6,8,
    9, 10, 11,12, 13,14,
    15, 16, 17, 18,20,
    22, 23
]

all_frames = []


def extract_classes(val):

    val = str(int(val))

    codes = sorted(
        VALID_CODES,
        key=lambda x: len(str(x)),
        reverse=True
    )

    for from_code in codes:
        from_str = str(from_code)
        if val.startswith(from_str):
            remaining = val[len(from_str):]
            if remaining != "":
                to_code = int(remaining)
                if to_code in VALID_CODES:
                    return from_code, to_code

    return np.nan, np.nan


for period, file_name in CSV_FILES.items():

    path = os.path.join(BASE_PATH, file_name)

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower()

    value_col = [c for c in df.columns if "value" in c][0]

    df["value_int"] = df[value_col].astype(int)

    df[["L1", "L2"]] = df["value_int"].apply(
        lambda x: pd.Series(extract_classes(x))
    )

    df = df.dropna(subset=["L1", "L2"])

    df["L1"] = df["L1"].astype(int)
    df["L2"] = df["L2"].astype(int)

    df["FROM"] = df["L1"].map(LU_DICT)
    df["TO"] = df["L2"].map(LU_DICT)

    df["FROM_MAIN"] = df["FROM"].str.split(">").str[0]
    df["TO_MAIN"] = df["TO"].str.split(">").str[0]

    df["Period"] = period

    all_frames.append(df)

ALL_DF = pd.concat(all_frames, ignore_index=True)

change_df = ALL_DF[
    ALL_DF["L1"] != ALL_DF["L2"]
].copy()


net_records = []

for period in change_df["Period"].unique():

    sub = change_df[
        change_df["Period"] == period
    ]
    outgoing = (
        sub.groupby("FROM_MAIN")["deg2"]
        .sum()
    )
    incoming = (
        sub.groupby("TO_MAIN")["deg2"]
        .sum()
    )
    all_classes = sorted(
        set(outgoing.index).union(
            set(incoming.index)
        )
    )
    for cls in all_classes:
        out_val = outgoing.get(cls, 0)
        in_val = incoming.get(cls, 0)
        net_change = in_val - out_val
        net_records.append([
            period,
            cls,
            net_change
        ])
net_df = pd.DataFrame(
    net_records,
    columns=[
        "Period",
        "Class",
        "NetChange"
    ]
)
l1_totals = (
    net_df.pivot(
        index="Class",
        columns="Period",
        values="NetChange"
    )
    .fillna(0)
)

x = np.arange(len(l1_totals.index))
width = 0.25
fig, ax = plt.subplots(figsize=(14, 7))

for i, period in enumerate(l1_totals.columns):
    vals = l1_totals[period].values
    colors = [
        "green" if v >= 0 else "red"
        for v in vals
    ]
    ax.bar(
        x + i * width,
        vals,
        width=width,
        label=period,
        color=colors
    )
ax.axhline(0, color="black")
ax.set_xticks(x + width)
ax.set_xticklabels(
    l1_totals.index,
    rotation=20,
    ha="right"
)
ax.set_ylabel("Net Area Change (deg²)")
ax.set_title("L1 Category Net Change Comparison")
ax.legend()
plt.tight_layout()
plt.show()
