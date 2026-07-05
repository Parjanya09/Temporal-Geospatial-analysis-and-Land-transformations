import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = "refined data"

YEAR_FOLDERS = {
    "2005-06": os.path.join(BASE_DIR, "csv of 2005-2006"),
    "2011-12": os.path.join(BASE_DIR, "csv of 2011-2012"),
    "2015-16": os.path.join(BASE_DIR, "csv of 2015-2016"),
}

OUTPUT_ROOT = "state_analysis_output"
YEAR_LABELS = list(YEAR_FOLDERS.keys())


def load_state(state: str) -> dict[str, pd.DataFrame]:
    data = {}
    for label, folder in YEAR_FOLDERS.items():
        path = os.path.join(folder, f"{state}.csv")
        if not os.path.exists(path):
            print(f"  WARN Missing: {path}")
            continue
        raw = pd.read_csv(path, header=0)
        l1_col, l2_col = raw.columns[0], raw.columns[1]
        raw[l1_col] = raw[l1_col].ffill()
        raw["_label"] = raw[l1_col].str.strip() + " › " + raw[l2_col].str.strip()
        raw = raw.set_index("_label").drop(columns=[l1_col, l2_col])
        raw = raw.loc[:, ~raw.columns.str.contains("Grand Total", case=False)]
        raw = raw.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        data[label] = raw
    return data


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    return df.div(df.sum(axis=0), axis=1).fillna(0) * 100


def _l1_totals(df: pd.DataFrame) -> pd.Series:
    l1 = df.index.str.split(" › ").str[0]
    return df.assign(_l1=l1).groupby("_l1").sum(numeric_only=True).sum(axis=1)


def _save(out_dir: str, fname: str):
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved in {path}")



def process_state(state_name: str):
    print(f"\n{'='*60}")
    print(f"  STATE: {state_name}")
    print(f"{'='*60}")

    out_dir = os.path.join(OUTPUT_ROOT, state_name.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    DATA = load_state(state_name)
    if len(DATA) < 2:
        print(f"  SKIP Not enough years loaded for {state_name}")
        return

    y_prev   = YEAR_LABELS[-2]          
    y_recent = YEAR_LABELS[-1]          
    y_first  = YEAR_LABELS[0]           

    df_recent = DATA[y_recent]
    df_prev   = DATA[y_prev]
    df_first  = DATA[y_first]

    yearly_l1: dict[str, pd.Series] = {yr: _l1_totals(df) for yr, df in DATA.items()}
    all_l1 = sorted(set().union(*[s.index for s in yearly_l1.values()]))
    l1_table = pd.DataFrame(
        {yr: s.reindex(all_l1, fill_value=0) for yr, s in yearly_l1.items()},
        index=all_l1,
    )

    l1_table["Δ_abs"]   = l1_table[y_recent] - l1_table[y_first]
    l1_table["Δ_pct"]   = (l1_table["Δ_abs"] / l1_table[y_first].replace(0, np.nan)) * 100
    l1_table.index.name = "L1_Category"

    print("1. L1 totals grouped bar …")
    n_years = len(YEAR_LABELS)
    x = np.arange(len(all_l1))
    width = 0.8 / n_years
    colors = plt.cm.Set2(np.linspace(0, 1, n_years))

    fig, ax = plt.subplots(figsize=(max(14, len(all_l1) * 1.1), 7))
    for i, yr in enumerate(YEAR_LABELS):
        vals = l1_table[yr].values
        bars = ax.bar(x + i * width, vals, width=width,
                      label=yr, color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + vals.max() * 0.005,
                        f"{v/1000:.1f}k", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x + width * (n_years - 1) / 2)
    ax.set_xticklabels(all_l1, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Total Area (sq km)", fontsize=10)
    ax.set_title(f"{state_name} — Total Area by L1 Category Across All Years",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, title="Year")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    plt.tight_layout()
    _save(out_dir, "01_l1_totals_all_years.png")
    
    print("2. L2 breakdown per L1 group (state totals) …")
 
    ser_first_state  = df_first.sum(axis=1)   
    ser_recent_state = df_recent.sum(axis=1)
 
    all_l1_groups = sorted(set(
        ser_first_state.index.str.split(" › ").str[0].tolist() +
        ser_recent_state.index.str.split(" › ").str[0].tolist()
    ))
 
    for l1_cat in all_l1_groups:
 
        mask_f = ser_first_state.index.str.startswith(l1_cat + " › ")
        mask_r = ser_recent_state.index.str.startswith(l1_cat + " › ")
 
        l2_first  = ser_first_state[mask_f].copy()
        l2_recent = ser_recent_state[mask_r].copy()
 
        l2_first.index  = l2_first.index.str.split(" › ").str[1]
        l2_recent.index = l2_recent.index.str.split(" › ").str[1]
 
        all_l2 = sorted(l2_first.index.union(l2_recent.index))
        l2_first  = l2_first.reindex(all_l2, fill_value=0)
        l2_recent = l2_recent.reindex(all_l2, fill_value=0)
 
        l2_delta_abs = l2_recent - l2_first
        l2_delta_pct = (l2_delta_abs / l2_first.replace(0, np.nan)) * 100
 
        yearly_l2 = {}
        for yr, df in DATA.items():
            s = df.sum(axis=1)
            mask = s.index.str.startswith(l1_cat + " › ")
            sub = s[mask].copy()
            sub.index = sub.index.str.split(" › ").str[1]
            yearly_l2[yr] = sub.reindex(all_l2, fill_value=0)
 
        n_l2  = len(all_l2)
        fig_h = max(16, n_l2 * 1.8)
        fig_w = max(12, n_l2 * 1.1)
 
        fig, (ax_bar, ax_abs, ax_pct) = plt.subplots(
            3, 1,
            figsize=(fig_w, fig_h),
            gridspec_kw={"height_ratios": [1.4, 1, 1]},
        )
 
        x         = np.arange(n_l2)
        n_yrs     = len(YEAR_LABELS)
        w         = 0.75 / n_yrs
        yr_colors = plt.cm.Set2(np.linspace(0, 1, n_yrs))
 
        for i, yr in enumerate(YEAR_LABELS):
            vals = yearly_l2[yr].values
            bars = ax_bar.bar(x + i * w, vals, width=w,
                              label=yr, color=yr_colors[i],
                              edgecolor="white", linewidth=0.4)
            max_v = vals.max() if vals.max() > 0 else 1
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max_v * 0.008,
                        f"{v:,.0f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=90,
                    )
 
        ax_bar.set_xticks(x + w * (n_yrs - 1) / 2)
        ax_bar.set_xticklabels(all_l2, rotation=35, ha="right", fontsize=8)
        ax_bar.set_ylabel("Total Area (sq km)", fontsize=9)
        ax_bar.set_title("All Years — State Total", fontsize=10, fontweight="bold")
        ax_bar.legend(fontsize=8, title="Year")
        ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
 
        colors_abs = ["#d73027" if v < 0 else "#4dac26" for v in l2_delta_abs]
        abs_bars   = ax_abs.bar(x, l2_delta_abs.values,
                                color=colors_abs, edgecolor="white", linewidth=0.4)
        ax_abs.axhline(0, color="black", linewidth=0.9)
        max_abs = l2_delta_abs.abs().max() if l2_delta_abs.abs().max() > 0 else 1
        for bar, v in zip(abs_bars, l2_delta_abs.values):
            if np.isfinite(v) and v != 0:
                offset = max_abs * 0.02
                va = "bottom" if v >= 0 else "top"
                ax_abs.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + (offset if v >= 0 else -offset),
                    f"{v:+,.0f} sq km", ha="center", va=va,
                    fontsize=7, rotation=90,
                )
        ax_abs.set_xticks(x)
        ax_abs.set_xticklabels(all_l2, rotation=35, ha="right", fontsize=8)
        ax_abs.set_ylabel("Δ Area (sq km)", fontsize=9)
        ax_abs.set_title(f"Absolute Change  ({y_first} to {y_recent})",
                         fontsize=10, fontweight="bold")
        ax_abs.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
 
        valid_pct  = l2_delta_pct.replace([np.inf, -np.inf], np.nan)
        colors_pct = [
            "#d73027" if (pd.notna(v) and v < 0) else "#4dac26"
            for v in valid_pct
        ]
        pct_bars = ax_pct.bar(x, valid_pct.fillna(0).values,
                              color=colors_pct, edgecolor="white", linewidth=0.4)
        ax_pct.axhline(0, color="black", linewidth=0.9)
        max_pct = valid_pct.abs().max() if valid_pct.abs().max() > 0 else 1
        for bar, v in zip(pct_bars, valid_pct.values):
            if np.isfinite(v) and v != 0:
                offset = max_pct * 0.02
                va = "bottom" if v >= 0 else "top"
                ax_pct.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + (offset if v >= 0 else -offset),
                    f"{v:+.1f}%", ha="center", va=va,
                    fontsize=7, rotation=90,
                )
        ax_pct.set_xticks(x)
        ax_pct.set_xticklabels(all_l2, rotation=35, ha="right", fontsize=8)
        ax_pct.set_ylabel("% Change", fontsize=9)
        ax_pct.set_title(f"Percentage Change  ({y_first} to {y_recent})",
                         fontsize=10, fontweight="bold")
 
        gain_p = mpatches.Patch(color="#4dac26", label="Gain")
        loss_p = mpatches.Patch(color="#d73027", label="Loss")
        for ax_ in (ax_abs, ax_pct):
            ax_.legend(handles=[gain_p, loss_p], fontsize=8, loc="upper right")
 
        fig.suptitle(
            f"{state_name}  —  {l1_cat}\nL2 Sub-Category State Totals & Change",
            fontsize=13, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
 
        safe_l1 = l1_cat.replace(" ", "_").replace("/", "-")[:40]
        _save(out_dir, f"2_l2_breakdown_{safe_l1}.png")
     

    print("3. L1 absolute change …")
    sorted_abs = l1_table["Δ_abs"].sort_values()
    colors_abs = ["#d73027" if v < 0 else "#4dac26" for v in sorted_abs]

    fig, ax = plt.subplots(figsize=(10, max(5, len(all_l1) * 0.55)))
    bars = ax.barh(sorted_abs.index, sorted_abs.values,
                   color=colors_abs, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, v in zip(bars, sorted_abs.values):
        offset = sorted_abs.abs().max() * 0.01
        ha = "left" if v >= 0 else "right"
        ax.text(v + (offset if v >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+,.0f}", va="center", ha=ha, fontsize=8)
    ax.set_xlabel("Δ Area (sq km)", fontsize=10)
    ax.set_title(f"{state_name} — L1 Absolute Change  ({y_first} to {y_recent})\n"
                 "Green = gain   Red = loss", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(out_dir, "03_l1_absolute_change.png")

    print("4. L1 percent change …")
    sorted_pct = l1_table["Δ_pct"].dropna().sort_values()
    colors_pct = ["#d73027" if v < 0 else "#4dac26" for v in sorted_pct]

    fig, ax = plt.subplots(figsize=(10, max(5, len(sorted_pct) * 0.55)))
    bars = ax.barh(sorted_pct.index, sorted_pct.values,
                   color=colors_pct, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, v in zip(bars, sorted_pct.values):
        offset = sorted_pct.abs().max() * 0.01
        ha = "left" if v >= 0 else "right"
        ax.text(v + (offset if v >= 0 else -offset),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}%", va="center", ha=ha, fontsize=8)
    ax.set_xlabel("% Change", fontsize=10)
    ax.set_title(f"{state_name} — L1 Percentage Change  ({y_first} to {y_recent})\n"
                 "Green = gain   Red = loss", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(out_dir, "04_l1_percent_change.png")

    csv_path = os.path.join(out_dir, "04_l1_change_table.csv")
    l1_table.round(2).to_csv(csv_path)
    print(f"  Saved in {csv_path}")

    print("5. % composition heatmap (recent) …")
    norm_recent = _norm(df_recent)
    fig, ax = plt.subplots(figsize=(max(14, df_recent.shape[1] * 0.65),
                                    max(8, df_recent.shape[0] * 0.45)))
    sns.heatmap(norm_recent, ax=ax, cmap="Blues", vmin=0, vmax=70,
                linewidths=0.25, linecolor="white",
                cbar_kws={"label": "% of district total", "shrink": 0.6})
    ax.set_title(f"{state_name} — % Composition per District  ({y_recent})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("District", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    _save(out_dir, "05_pct_composition_recent.png")

    print("6. Ward clustermap (recent) …")
    norm_r = norm_recent.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0)
    if norm_r.shape[0] >= 2 and norm_r.shape[1] >= 2:
        try:
            g = sns.clustermap(
                norm_r,
                cmap="YlGnBu",
                figsize=(max(16, norm_r.shape[1] * 0.65),
                         max(10, norm_r.shape[0] * 0.5)),
                linewidths=0.2, linecolor="white",
                method="ward", metric="euclidean",
                cbar_kws={"label": "% share"},
                dendrogram_ratio=(0.12, 0.12),
                xticklabels=True, yticklabels=True,
            )
            g.ax_heatmap.tick_params(axis="x", labelsize=5, rotation=45)
            g.ax_heatmap.tick_params(axis="y", labelsize=6, rotation=0)
            g.fig.suptitle(f"{state_name} — Ward-Clustered Land Use Profiles  ({y_recent})",
                           fontsize=12, fontweight="bold", y=1.01)
            path = os.path.join(out_dir, "06_clustermap_recent.png")
            g.fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved in {path}")
        except Exception as e:
            print(f"  WARN Clustermap failed: {e}")
    else:
        print("  SKIP Not enough data for clustermap")

    print("7 Stacked bar (recent) …")
    ax = _norm(df_recent).T.plot(
        kind="bar", stacked=True,
        figsize=(max(14, df_recent.shape[1] * 0.75), 8),
        colormap="tab20", width=0.85,
        edgecolor="white", linewidth=0.3,
    )
    ax.set_title(f"{state_name} — Land-Use Composition per District  ({y_recent})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("District", fontsize=9)
    ax.set_ylabel("% of district total", fontsize=9)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6, title="Sub-Category")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    plt.tight_layout()
    _save(out_dir, "07_stacked_bar_recent.png")


    print("8 Dominant category (recent) …")
    all_cats = df_recent.index.tolist()
    palette = plt.cm.get_cmap("tab20", len(all_cats))
    cat_colors = {cat: palette(i) for i, cat in enumerate(all_cats)}

    dominant = df_recent.fillna(0).idxmax(axis=0)
    dom_val  = df_recent.fillna(0).max(axis=0)
    colors   = [cat_colors.get(c, "grey") for c in dominant]

    fig, ax = plt.subplots(figsize=(max(14, df_recent.shape[1] * 0.7), 6))
    bars = ax.bar(df_recent.columns, dom_val, color=colors,
                  edgecolor="white", linewidth=0.4)
    for bar, cat in zip(bars, dominant):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                cat.split(" › ")[-1][:15], ha="center", va="bottom",
                fontsize=4.5, rotation=90, color="dimgrey")

    seen: dict = {}
    for cat, col in zip(dominant, colors):
        seen.setdefault(cat, col)
    handles = [mpatches.Patch(color=col, label=cat) for cat, col in seen.items()]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=6, title="Dominant Category")

    ax.set_title(f"{state_name} — Dominant Land-Use Category per District  ({y_recent})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Area of dominant category (sq km)", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=5)
    plt.tight_layout()
    _save(out_dir, "08_dominant_category_recent.png")
 
    print(f"\n   All outputs saved to: {out_dir}/")


if __name__ == "__main__":
    sample_folder = list(YEAR_FOLDERS.values())[0]
    ALL_STATES = [f.replace(".csv", "") for f in os.listdir(sample_folder) if f.endswith(".csv")]
    print(f"States found: {ALL_STATES}\n")

    for state in ALL_STATES:
        try:
            process_state(state)
        except Exception as e:
            print(f"ERROR {state}: {e}")