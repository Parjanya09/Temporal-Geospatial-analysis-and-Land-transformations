import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_PATH = r"C:\Users\bhuvan.NRSCADMIN\Desktop\GeoSpatial Analysis _Analysis part"
FILE_NAME = "GSDP-current-all.csv"

path = os.path.join(BASE_PATH, FILE_NAME)
df = pd.read_csv(path)
df.columns = df.columns.str.strip()
mp_df = df[
    df["State"]
    .str.lower()
    .str.contains("madhya")
].copy()
gdp_0506 = mp_df["2005-06"].values[0]
gdp_1112 = mp_df["2011-12"].values[0]
gdp_1516 = mp_df["2015-16"].values[0]
change_1 = gdp_1112 - gdp_0506
change_2 = gdp_1516 - gdp_1112
periods = [
    "2005-06 → 2011-12",
    "2011-12 → 2015-16"
]
changes = [
    change_1,
    change_2
]
colors = [
    "green" if v >= 0 else "red"
    for v in changes
]
plt.figure(figsize=(8, 6))
bars = plt.bar(
    periods,
    changes,
    color=colors
)
for bar, val in zip(bars, changes):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        val,
        f"{val:,.0f}",
        ha="center",
        va="bottom"
    )

plt.axhline(0, color="black")
plt.ylabel("GSDP Change")
plt.title("Madhya Pradesh GSDP Change")
plt.tight_layout()
plt.show()