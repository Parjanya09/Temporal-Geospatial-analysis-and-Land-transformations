import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "refined data"

YEAR_FOLDERS = {
    "2005-06": os.path.join(BASE_DIR, "csv of 2005-2006"),
    "2011-12": os.path.join(BASE_DIR, "csv of 2011-2012"),
    "2015-16": os.path.join(BASE_DIR, "csv of 2015-2016"),
}
STATE_NAME = "Sikkim"   
OUTPUT_DIR = os.path.join( "approach_1_analysis_output",STATE_NAME.replace(" ", "_"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_state(state: str, year_folders: dict) -> dict[str, pd.DataFrame]:
    
    data = {}
    for label, folder in year_folders.items():
        path = os.path.join(folder, f"{state}.csv")
        if not os.path.exists(path):
            print(f"[WARN] Missing: {path}")
            continue

        raw = pd.read_csv(path, header=0)

        l1_col, l2_col = raw.columns[0], raw.columns[1]

        raw[l1_col] = raw[l1_col].ffill()
        raw["_label"] = raw[l1_col].str.strip() + " › " + raw[l2_col].str.strip()
        raw = raw.set_index("_label")
        raw = raw.drop(columns=[l1_col, l2_col])

        raw = raw.loc[:, ~raw.columns.str.contains("Grand Total", case=False)]

        raw = raw.apply(pd.to_numeric, errors="coerce")

        raw = raw.dropna(how="all")

        data[label] = raw

    return data


print("Loading data …")
DATA = load_state(STATE_NAME, YEAR_FOLDERS)
YEAR_LABELS = list(DATA.keys())
print(f"Years loaded : {list(DATA.keys())}")
for yr, df in DATA.items():
    print(f"  {yr} has  {df.shape[0]} categories  ×  {df.shape[1]} districts")

ALL_CATS  = list(DATA[YEAR_LABELS[0]].index)
ALL_DISTS = list(DATA[YEAR_LABELS[0]].columns)


def _save(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved in {path}")


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    return df.div(df.sum(axis=0), axis=1).fillna(0) * 100

print(f"\nProcessing: {STATE_NAME}")

OUTPUT_DIR = os.path.join(
        "approach_2_analysis_output",
        STATE_NAME.replace(" ", "_")
    )
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data …")
DATA = load_state(STATE_NAME, YEAR_FOLDERS)

print("\n[Plot 1] Raw heatmaps …")
fig, axes = plt.subplots(1, len(DATA), figsize=(9 * len(DATA), max(8, len(ALL_CATS) * 0.45)),sharey=True)
if len(DATA) == 1:
        axes = [axes]

for ax, (yr, df) in zip(axes, DATA.items()):
        sns.heatmap(
        df.fillna(0), ax=ax,
        cmap="YlOrRd", linewidths=0.25, linecolor="white",
        cbar_kws={"label": "Area (sq km)", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
        )
        ax.set_title(f"{yr}", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("District", fontsize=9)
        ax.set_ylabel("Land Use Sub-Category", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)

fig.suptitle(f"{STATE_NAME.replace('_',' ')} — Raw Land Use (sq km)", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
_save("01_raw_heatmaps.png")

print("[Plot 2] Normalised heatmaps …")
fig, axes = plt.subplots( 1, len(DATA),figsize=(9 * len(DATA), max(8, len(ALL_CATS) * 0.45)),sharey=True)
if len(DATA) == 1:
       axes = [axes]

for ax, (yr, df) in zip(axes, DATA.items()):
        sns.heatmap(
        _norm(df), ax=ax,
        cmap="Blues", vmin=0, vmax=70,
        linewidths=0.25, linecolor="white",
        cbar_kws={"label": "% of district total", "shrink": 0.6},
        )
        ax.set_title(f"{yr}", fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("District", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)

fig.suptitle(f"{STATE_NAME.replace('_',' ')} — Proportional Composition (% per district)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
_save("02_normalised_heatmaps.png")

print("[Plot 3] Change heatmap …")
y_first, y_last = YEAR_LABELS[0], YEAR_LABELS[-1]
df_first = DATA[y_first]
df_last  = DATA[y_last]
common_cats  = df_first.index.intersection(df_last.index)
common_dists = df_first.columns.intersection(df_last.columns)
delta = df_last.loc[common_cats, common_dists] - df_first.loc[common_cats, common_dists]

fig, ax = plt.subplots(figsize=(max(14, len(common_dists)*0.65), max(8, len(common_cats)*0.45)))
lim = np.nanpercentile(np.abs(delta.values), 95)
sns.heatmap(
    delta, ax=ax,
    cmap="RdBu_r", center=0, vmin=-lim, vmax=lim,
    linewidths=0.25, linecolor="white",
    cbar_kws={"label": f"Δ sq km  ({y_first} to {y_last})", "shrink": 0.6},
   )
ax.set_title(f"{STATE_NAME.replace('_',' ')} — Change in Land Use  ({y_first} to {y_last})\n" f"Red = loss   Blue = gain", fontsize=13, fontweight="bold")
ax.set_xlabel("District", fontsize=9)
ax.tick_params(axis="x", rotation=45, labelsize=6)
ax.tick_params(axis="y", rotation=0,  labelsize=7)
plt.tight_layout()
_save("03_change_heatmap.png")

print("[Plot 4] Percent-change heatmap …")

pct_change = (
    delta /
    df_first.loc[common_cats, common_dists].replace(0, np.nan)
    ) * 100

valid_vals = pct_change.values[np.isfinite(pct_change.values)]

if valid_vals.size == 0:
        print(f"  Skipping percent-change plot for {STATE_NAME} (all values NaN/inf)")
else:
        fig, ax = plt.subplots(
        figsize=(
            max(14, len(common_dists) * 0.65),
            max(8, len(common_cats) * 0.45)
        )
    )

        lim = np.nanpercentile(np.abs(valid_vals), 90)

        if lim == 0 or np.isnan(lim):
            lim = 1

        sns.heatmap(
        pct_change.clip(-lim, lim),
        ax=ax,
        cmap="PiYG",
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={
            "label": f"% change  ({y_first} to {y_last})",
            "shrink": 0.6
        },
    )

        ax.set_title(
        f"Percent Change in Land Use  ({y_first} to {y_last})\n"
        f"(clipped at ±{lim:.0f}%)",
        fontsize=13,fontweight="bold" )

        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0, labelsize=7)

        plt.tight_layout()

        ax.set_title(f"Percent Change in Land Use  ({y_first} to {y_last})\n(clipped at ±{lim:.0f}%)", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)
        plt.tight_layout()
        _save("04_pct_change_heatmap.png")


print("[Plot 5] Category–category correlation …")
fig, axes = plt.subplots(1, len(DATA), figsize=(8*len(DATA), max(8, len(ALL_CATS)*0.6)), sharey=True)
if len(DATA) == 1:
        axes = [axes]

for ax, (yr, df) in zip(axes, DATA.items()):
       corr = df.fillna(0).T.corr(method="pearson")
       mask = np.triu(np.ones_like(corr, dtype=bool))
       sns.heatmap(
        corr, ax=ax, mask=mask,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.4,
        annot=(len(corr) <= 12), fmt=".2f", annot_kws={"size": 6},
        cbar_kws={"label": "Pearson r", "shrink": 0.6},
         )
ax.set_title(f"{yr}", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=6)
ax.tick_params(axis="y", rotation=0,  labelsize=6)

fig.suptitle(f"{STATE_NAME.replace('_',' ')} — Category × Category Correlation\n" "(how land-use types co-occur across districts)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
_save("05_category_correlation.png")


print("[Plot 6] District–district correlation …")
for yr, df in DATA.items():
        corr = df.fillna(0).corr(method="pearson")
        n    = len(corr)
        fig, ax = plt.subplots(figsize=(max(12, n*0.55), max(10, n*0.55)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
        corr, ax=ax, mask=mask,
        cmap="PuOr", center=0, vmin=-1, vmax=1,
        linewidths=0.3,
        cbar_kws={"label": "Pearson r", "shrink": 0.6},
        )
ax.set_title(f"District × District Correlation — {yr}\n" "(districts with similar land-use profiles cluster together)", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=6)
ax.tick_params(axis="y", rotation=0,  labelsize=6)
plt.tight_layout()
_save(f"06_district_correlation_{yr.replace('-','_')}.png")

print("[Plot 7] Clustermaps …")

for yr, df in DATA.items():

        norm_df = _norm(df)

        norm_df = norm_df.replace([np.inf, -np.inf], np.nan)
        norm_df = norm_df.dropna(how="all", axis=0)
        norm_df = norm_df.dropna(how="all", axis=1)
        norm_df = norm_df.fillna(0)

        if norm_df.shape[0] < 2 or norm_df.shape[1] < 2:
            print(f"  Skipping clustermap for {yr} (insufficient data)")
            continue

        try:
            g = sns.clustermap(
                norm_df,
            cmap="YlGnBu",
            figsize=(
                max(16, norm_df.shape[1] * 0.65),
                max(10, norm_df.shape[0] * 0.5)),
            linewidths=0.2,
            linecolor="white",
            method="ward",
            metric="euclidean",
            cbar_kws={"label": "% share"},
            dendrogram_ratio=(0.12, 0.12),
            xticklabels=True,
            yticklabels=True,
           )

            g.ax_heatmap.tick_params(axis="x", labelsize=5, rotation=45)
            g.ax_heatmap.tick_params(axis="y", labelsize=6, rotation=0)

            g.fig.suptitle(
            f"Ward-Clustered Land Use Profiles — {yr}",
            fontsize=12, fontweight="bold",y=1.01)

            path = os.path.join(
            OUTPUT_DIR,
            f"07_clustermap_{yr.replace('-', '_')}.png"
        )

            g.fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"  Saved in {path}")

        except Exception as e:
            print(f"  Skipping clustermap for {yr}: {e}")


print("[Plot 8] Stacked bars …")
for yr, df in DATA.items():
        norm_df = _norm(df)
        ax = norm_df.T.plot(
        kind="bar", stacked=True,
        figsize=(max(14, df.shape[1]*0.7), 8),
        colormap="tab20", width=0.85,
        edgecolor="white", linewidth=0.3,
        )
ax.set_title(f"{STATE_NAME.replace('_',' ')} — Land-Use Composition per District  ({yr})", fontsize=13, fontweight="bold")
ax.set_xlabel("District", fontsize=9)
ax.set_ylabel("% of district total", fontsize=9)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6, title="Sub-Category")
ax.tick_params(axis="x", rotation=45, labelsize=6)
plt.tight_layout()
_save(f"08_stacked_bar_{yr.replace('-','_')}.png")

print("[Plot 9] Per-district time-series …")
ts_dir = os.path.join(OUTPUT_DIR, "09_timeseries_per_district")
os.makedirs(ts_dir, exist_ok=True)

all_districts = sorted(set().union(*[df.columns for df in DATA.values()]))

for district in all_districts:

        trend_data = {}

        for yr, df in DATA.items():

            if district in df.columns:

                vals = df[district]

                if isinstance(vals, pd.Series):
                    trend_data[yr] = vals.sum()
                else:
                   trend_data[yr] = float(vals)

        if len(trend_data) == 0:
            continue

        trend = pd.Series(trend_data)

        plt.figure(figsize=(8, 4))
        trend.plot(marker='o')

        plt.title(f"{district} : Total Area Trend")
        plt.xlabel("Year")
        plt.ylabel("Area")

        plt.grid(True)

        save_path = os.path.join(
        ts_dir,
        f"{district.replace('/', '_')}.png" )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

print(f"  Saved {len(all_districts)} district time-series to {ts_dir}")


print("[Plot 10] Year-vs-year scatter plots …")
scatter_dir = os.path.join(OUTPUT_DIR, "10_scatter_year_vs_year")
os.makedirs(scatter_dir, exist_ok=True)

pairs = [(YEAR_LABELS[i], YEAR_LABELS[j])
         for i in range(len(YEAR_LABELS))
         for j in range(i+1, len(YEAR_LABELS))]

for (y1, y2) in pairs:
        df1, df2 = DATA[y1], DATA[y2]
        common_dists = df1.columns.intersection(df2.columns)
        common_cats  = df1.index.intersection(df2.index)

for cat in common_cats:
        v1 = df1.loc[cat, common_dists].fillna(0)
        v2 = df2.loc[cat, common_dists].fillna(0)
        if v1.sum() == 0 and v2.sum() == 0:
            continue

        mask = np.isfinite(v1) & np.isfinite(v2)
        if mask.sum() < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(v1, v2, s=55, alpha=0.75, edgecolors="k", linewidths=0.5, color="steelblue")

        for dist in common_dists:
            ax.annotate(dist, (v1[dist], v2[dist]),
                        fontsize=5, alpha=0.65,
                        xytext=(3, 2), textcoords="offset points")

        mask = (v1 > 0) | (v2 > 0)
        if mask.sum() > 2:
            m, b = np.polyfit(v1[mask], v2[mask], 1)
            xs = np.linspace(v1[mask].min(), v1[mask].max(), 100)
            ax.plot(xs, m*xs + b, "r--", linewidth=1.3, label=f"y = {m:.2f}x + {b:.1f}")
            r, p = pearsonr(v1[mask], v2[mask])
            title_extra = f"r = {r:.2f}  (p = {p:.3f})"
        else:
            title_extra = ""

        ax.set_title(f"{cat}\n{y1}  vs  {y2}   {title_extra}", fontsize=10, fontweight="bold")
        ax.set_xlabel(f"Area sq km — {y1}", fontsize=9)
        ax.set_ylabel(f"Area sq km — {y2}", fontsize=9)
        ax.legend(fontsize=8)
        plt.tight_layout()

        safe_cat = cat.replace(" ", "_").replace("/", "-").replace(">", "").replace("›","")[:50]
        path = os.path.join(scatter_dir, f"{safe_cat}_{y1}_{y2}.png".replace("-","_"))
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()

print(f"  Saved scatter plots to {scatter_dir}/")

print("[Plot 11] Dominant category per district …")
fig, axes = plt.subplots(len(DATA), 1, figsize=(max(14, len(ALL_DISTS)*0.65), 5*len(DATA)))
if len(DATA) == 1:
       axes = [axes]

palette = plt.cm.get_cmap("tab20", len(ALL_CATS))
cat_colors = {cat: palette(i) for i, cat in enumerate(ALL_CATS)}

for ax, (yr, df) in zip(axes, DATA.items()):
        dominant = df.fillna(0).idxmax(axis=0)      
        dom_val  = df.fillna(0).max(axis=0)          
        colors = [cat_colors.get(c, "grey") for c in dominant]
        ax.bar(df.columns, dom_val, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_title(f"Dominant Land-Use Category per District — {yr}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Area of dominant category (sq km)", fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=5)

seen = {}
for cat, col in zip(dominant, colors):
        if cat not in seen:
            seen[cat] = col
from matplotlib.patches import Patch
handles = [Patch(color=col, label=cat) for cat, col in seen.items()]
ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6, title="Category")

plt.suptitle(f"{STATE_NAME.replace('_',' ')} — Dominant Land Use per District", fontsize=13, fontweight="bold")
plt.tight_layout()
_save("11_dominant_category_per_district.png")


print("[Plot 12] L1 category totals over time …")

fig, ax = plt.subplots(figsize=(13, 6))
width = 0.25
all_l1 = set()

yearly_l1 = {}

for yr, df in DATA.items():

        l1 = df.index.str.split(" › ").str[0]

        l1_totals = (
        df.assign(_l1=l1)
          .groupby("_l1")
          .sum(numeric_only=True)
          .sum(axis=1))

        yearly_l1[yr] = l1_totals
        all_l1.update(l1_totals.index)

all_l1 = sorted(all_l1)

x = np.arange(len(all_l1))

for i, (yr, totals) in enumerate(yearly_l1.items()):

        aligned = totals.reindex(all_l1, fill_value=0)

        ax.bar(
        x + i*width,
        aligned.values,
        width=width,
        label=yr,
        edgecolor="white")

ax.set_xticks(x + width)
ax.set_xticklabels(all_l1, rotation=25, ha="right", fontsize=8)

ax.set_title(
    f"{STATE_NAME.replace('_',' ')} — Total Area by L1 Category Across Years",
    fontsize=13,
    fontweight="bold")

ax.set_ylabel("Total area (sq km)")
ax.legend(fontsize=9)

plt.tight_layout()

_save("12_L1_category_totals_by_year.png")

print("[Export] Summary statistics …")
frames = []
for yr, df in DATA.items():
       s = df.describe().T.copy()
       s.insert(0, "year", yr)
       frames.append(s)
       summary = pd.concat(frames)
       out = os.path.join(OUTPUT_DIR, "13_summary_statistics.csv")
       summary.to_csv(out)
print(f"  Saved in {out}")


print(f"\n  All done!  Outputs in: {OUTPUT_DIR}/")
print("""
    Output files:
  01_raw_heatmaps.png                  — absolute area heatmap per year
  02_normalised_heatmaps.png           — % composition per district per year
  03_change_heatmap.png                — absolute change (first to last year)
  04_pct_change_heatmap.png            — % change (first to last year)
  05_category_correlation.png          — which land types co-occur (per year)
  06_district_correlation_*.png        — which districts are similar (per year)
  07_clustermap_*.png                  — ward-clustered profiles (per year)
  08_stacked_bar_*.png                 — composition bars per district (per year)
  09_timeseries_per_district/          — one trend chart per district
  10_scatter_year_vs_year/             — category-level year-vs-year scatter
  11_dominant_category_per_district.png— which category dominates each district
  12_L1_category_totals_by_year.png    — top-level group totals across years
  13_summary_statistics.csv            — mean/std/min/max per category per year
    """)

print("Successfully Completed:", STATE_NAME)

