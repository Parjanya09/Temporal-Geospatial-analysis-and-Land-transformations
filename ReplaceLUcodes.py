import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

BASE_PATH = r"C:\LULC"

states = {
    "HR": "Haryana",
    "MP": "Madhya Pradesh",
    "UP": "Uttar Pradesh"
}

years = [
    "0506_to_1112",
    "1112_to_1516",
    "0506_to_1516"
]

LU_DICT = {
    1: "Builtup>Urban",
    2: "Builtup>Rural",
    3: "Builtup>Mining",
    4: "Agriculture>Cropland",
    5: "Agriculture>Plantation",
    6: "Agriculture>Fallow",
    8: "Forest>Evergreen/Semi Evergreen",
    9: "Forest>Decidious",
    10: "Forest>Forest Plantation",
    11: "Forest>Scrub Forest",
    12: "Forest>Swamp/Mangroove Forest",
    13: "Grass-Grazing>Grass-Grazing",
    14: "Barren/Unculturable/Wastelands>Salt Affected land",
    15: "Barren/Unculturable/Wastelands>Gullied/Ravinous Land ",
    16: "Barren/Unculturable/Wastelands>Scrubland",
    17: "Barren/Unculturable/Wastelands>Sandy Area",
    18: "Barren/Unculturable/Wastelands>Barren Rocky",
    20: "Wetlands/Waterbodies>Inland Wetland",
    22: "Wetlands/Waterbodies>River/Stream/Canals",
    23: "Wetlands/Waterbodies>Waterbodies",
}

for short_name, state_name in states.items():

    state_path = os.path.join(BASE_PATH, state_name)
    for year in years:
        input_file = os.path.join(
            state_path,
           "RasterChanges_" f"{short_name}_" f"{year}.csv"
        )
        if not os.path.exists(input_file):
            print(f"Missing file: {input_file}")
            continue
        df = pd.read_csv(input_file)
        df.columns = df.columns.str.strip().str.lower()
        value_col = [c for c in df.columns if "value" in c][0]
        df["From L1>L2_Code"] = df[value_col].astype(int) // 100
        df["To L1>L2_Code"] = df[value_col].astype(int) % 100
        df["From category"] = df["From L1>L2_Code"].map(LU_DICT)
        df["To category"] = df["To L1>L2_Code"].map(LU_DICT)

        df["Conversion"] = (
            df["From category"] +
            " to " +
            df["To category"]
        )
        cols = [
            value_col,
            "From L1>L2_Code",
            "From category",
            "To L1>L2_Code",
            "To category",
            "Conversion"
        ]
        remaining_cols = [
            c for c in df.columns
            if c not in cols
        ]

        final_df = df[cols + remaining_cols]
        output_file = os.path.join(state_path,f"RasterChanges_{short_name}_{year}_decoded.csv")
        final_df.to_csv(output_file, index=False)
