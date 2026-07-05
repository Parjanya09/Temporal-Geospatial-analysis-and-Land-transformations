import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

BASE_DIR = "refined data"

YEAR_FOLDERS = {
    "2005-06": os.path.join(BASE_DIR, "csv of 2005-06"),
    "2011-12": os.path.join(BASE_DIR, "csv of 2011-12"),
    "2015-16": os.path.join(BASE_DIR, "csv of 2015-16"),
}

OUTPUT_ROOT = "district_analysis_output"
YEAR_LABELS = list(YEAR_FOLDERS.keys())

MANUAL_CHANGES: dict[str, str] = {
    "Nellore": "Sri Potti Sriramulu Nellore",
    "Potti Sriramulu Nellore": "Sri Potti Sriramulu Nellore",
    "Muzzafarnagar": "Muzaffarnagar",
    "Pondicherry": "Puducherry",
    "Visakhapatnam": "Vishakhapatnam",
    "Bagalkot": "Bagalkote",
    "Bagalakote": "Bagalkote",
    "Bengaluru ": "Bengaluru Urban",
    "Bangalore Urban": "Bengaluru Urban",
    "Bangalore Rural": "Bengaluru Rural",
    "Belgaum": "Belagavi",
    "Bellary": "Ballari",
    "Gulbarga": "Kalaburagi",
    "Dakshin Kannad": "Dakshina Kannada",
    "Rangareddi": "Rangareddy",
    "Mysore": "Mysuru",
    "Ramanagara": "Bengaluru South",
    "Shimoga": "Shivamogga",
    "Tumkur": "Tumakuru",
    "Uttar Kannad": "Uttara Kannada",
    "Uttarakhand": "Uttara Kannada",
    "Bijapur": "Vijayapura",
    "Lahul & Spiti": "Lahaul And Spiti",
    "Ananthapuram": "Anantapur",
    "Anantapuram": "Anantapur",
    "Kupwara (Muzaffarabad)": "Kupwara",
    "Karimganj": "Sribhumi",
    "Sibsagar": "sivasagar",
    "Anantnag (Kashmir South)": "Anantnag",
    "Bagdam": "Budgam",
    "Badgam": "Budgam",
    "Baramula (Kashmir North)": "Baramulla",
    "Baramula": "Baramulla",
    " Ladakh (Leh)": "Leh",
    "Punch": "Poonch",
    "Rajauri ": "Rajouri",
    "Udhampure": "Udhampur",
    "Janjgir Champa": "Janjgir-Champa",
    "Yamuna Nagar": "Yamunanagar",
    "Sonepat": "Sonipat",
    "Gurgaon": "Gurugram",
    "Chikmagalur": "Chikkamagaluru",
    " Bengaluru Urban": "Bengaluru Urban",
    "Chamarajanagar": "Chamrajnagar",
    "AlLaphuzha": "Alappuzha",
    "Ahmadnagar": "Ahmednagar",
    "West-Imphal": "West Imphal",
    "S. Garo": "South Garo",
    "W. Garo": "West Garo",
    "W. Khasi": "West Khasi",
    "EKhasi": "East Khasi",
    "EastGaro": "East Garo",
    "East-Sikkim": "East Sikkim",
    "North-Sikkim": "North Sikkim",
    "South-Sikkim": "South Sikkim",
    "West-Sikkim": "West Sikkim",
    "West": "West Tripura",
    "South": "South Tripura",
    "North": "North Tripura",
    "Kanshiramnagar": "Kanshiram Nagar",
    "Kansiramnagar": "Kanshiram Nagar",
    "Naini Tal": "Nainital",
}
  
def clean_l1(x):
    if pd.isna(x):
        return x

    x = str(x).strip()
    x = x.replace("_", " ")
    x = x.replace("-", " ")
    x = " ".join(x.split())
    if "barren" in x.lower():
        return "Barren unculturable Wastelands"
    if "wetland" in x.lower() or "wet lands" in x.lower():
        return "Wetlands Water bodies"

    replacements = {
        "Wetlands Water bodies": "Wetlands Water bodies",
        "Wet lands Water bodies": "Wetlands Water bodies",
        "Forest": "Forest",
        "Agriculture": "Agriculture",
        "Builtup": "Builtup",
    }
    return replacements.get(x, x)

def load_state(state: str) -> dict[str, pd.DataFrame]:
    data = {}
    for label, folder in YEAR_FOLDERS.items():
        path = os.path.join(folder, f"{state}.csv")
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path, header=0)
        l1_col, l2_col = raw.columns[0], raw.columns[1]
        raw[l1_col] = raw[l1_col].ffill()
        raw["_label"] = raw[l1_col].str.strip() + " › " + raw[l2_col].str.strip()
        raw = raw.set_index("_label").drop(columns=[l1_col, l2_col])
        raw.columns = (raw.columns.astype(str).str.strip().str.replace("_", " ", regex=False).str.replace(r"\s+", " ", regex=True))
        raw = raw.loc[:, ~raw.columns.str.contains("Grand Total", case=False)]
        raw = raw.apply(pd.to_numeric, errors="coerce").astype(np.float64).dropna(how="all")
        
        l1_vals = raw.index.str.split(" › ").str[0].map(clean_l1)
        l2_vals = raw.index.str.split(" › ").str[1].str.strip()
        raw.index = l1_vals + " › " + l2_vals
        data[label] = raw
    return data

def _save(out_dir: str, fname: str):
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved in {path}")

def _add_side_table(ax, cell_text, col_labels, title_text="Data Summary"):
    """Helper layout to draw a nice, cleanly padded table on an axis slot."""
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.4)
    
    # Style headers gracefully
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2c3e50")
    ax.set_title(title_text, fontsize=10, fontweight="bold", pad=10)

def _hbar_change(values: pd.Series, title: str, xlabel: str,
                 out_dir: str, fname: str, unit: str = ""):
    sorted_v = values.sort_values()
    colors = ["#d73027" if v < 0 else "#4dac26" for v in sorted_v]

    # Side-by-side grid structure (Chart left, Table right)
    fig, (ax_chart, ax_tbl) = plt.subplots(1, 2, figsize=(16, max(4, len(sorted_v) * 0.55)), 
                                           gridspec_kw={"width_ratios": [2.2, 1]})
    
    bars = ax_chart.barh(sorted_v.index, sorted_v.values,
                         color=colors, edgecolor="white", linewidth=0.4)
    ax_chart.axvline(0, color="black", linewidth=0.9)

    max_abs = sorted_v.abs().max() if sorted_v.abs().max() > 0 else 1
    for bar, v in zip(bars, sorted_v.values):
        if np.isfinite(v):
            offset = max_abs * 0.012
            ha = "left" if v >= 0 else "right"
            label = f"{v:+,.2f}{unit}" 
            ax_chart.text(v + (offset if v >= 0 else -offset),
                          bar.get_y() + bar.get_height() / 2,
                          label, va="center", ha=ha, fontsize=8)

    ax_chart.set_xlabel(xlabel, fontsize=10)
    ax_chart.set_title(title, fontsize=11, fontweight="bold")

    gain_patch = mpatches.Patch(color="#4dac26", label="Gain")
    loss_patch = mpatches.Patch(color="#d73027", label="Loss")
    ax_chart.legend(handles=[gain_patch, loss_patch], fontsize=9, loc="lower right")

    # Draw side table
    cell_data = [[idx, f"{val:+,.2f}{unit}"] for idx, val in sorted_v.items()]
    _add_side_table(ax_tbl, cell_data, ["Category", f"Change ({unit.strip()})"])

    plt.tight_layout()
    _save(out_dir, fname)

def _normalise(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace(".", "")
        .replace("  ", " ")
    )

def build_canonical_map(DATA: dict[str, pd.DataFrame]) -> dict[str, str]:
    all_raw: list[str] = []
    for df in DATA.values():
        all_raw.extend(df.columns.tolist())

    all_raw = list(dict.fromkeys(all_raw))
    norm_to_canonical: dict[str, str] = {}

    for k, v in MANUAL_CHANGES.items():
        norm_to_canonical[_normalise(k)] = v
        norm_to_canonical[_normalise(v)] = v

    canonical_map: dict[str, str] = {}
    merges: list[str] = []
    for name in all_raw:
        key = _normalise(name)

        if key in norm_to_canonical:
            canonical = norm_to_canonical[key]
        else:
            canonical = name.title()

        canonical_map[name] = canonical

        if canonical != name:
            merges.append(f" MERGE '{name}' into '{canonical}'")

    if merges:
        print(f"  Name merges detected ({len(merges)}):")
        for msg in merges:
            print(msg)
    else:
        print("  No district name merges needed.")
    return canonical_map

def apply_canonical(DATA: dict[str, pd.DataFrame], canon_map: dict[str, str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for yr, df in DATA.items():
        renamed = df.rename(columns=canon_map)
        renamed = renamed.T.groupby(level=0).sum().T
        out[yr] = renamed
    return out

def l1_totals(ser: pd.Series) -> pd.Series:
    l1 = (
        ser.index.str.split(" › ").str[0]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("_", " ", regex=False)
        .str.replace(" - ", "-", regex=False)
    )
    return ser.groupby(l1).sum()

def process_district(district: str, DATA: dict[str, pd.DataFrame], state_name: str, out_root: str):
    years_with_dist = [yr for yr, df in DATA.items() if district in df.columns]
    if len(years_with_dist) < 2:
        print(f"  SKIP {district}: found in < 2 years")
        return

    out_dir = os.path.join(out_root, district.replace("/", "_").replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    y_first  = YEAR_LABELS[0]
    y_recent = YEAR_LABELS[-1]
    y_previous = YEAR_LABELS[-2]
    dist_data: dict[str, pd.Series] = {}
    for yr in years_with_dist:
        dist_data[yr] = DATA[yr][district].fillna(0)

    ser_first  = dist_data.get(y_first,  pd.Series(dtype=float))
    ser_recent = dist_data.get(y_recent, pd.Series(dtype=float))

    print(f"    Pie charts for L2 composition …")
    yr = y_recent
    yr_ser = dist_data[yr]
    for l1_cat in sorted(set(yr_ser.index.str.split(" › ").str[0])):
        mask = yr_ser.index.str.startswith(l1_cat + " › ")
        l2_ser = yr_ser[mask].copy()
        if l2_ser.empty:
            continue
        l2_ser.index = (
            l2_ser.index.str.split(" › ").str[1].str.strip()
        )
        l2_ser = l2_ser[l2_ser > 0]
        if l2_ser.empty:
            continue
        l2_ser = l2_ser.sort_values(ascending=False)
        fig, (ax_pie, ax_tbl) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1.5, 1]})
        
        _, texts, autotexts = ax_pie.pie(
            l2_ser.values,
            labels=l2_ser.index,
            autopct=lambda p: f"{p:.2f}%" if p >= 0.05 else "",  
            startangle=90,
            pctdistance=0.78,
            wedgeprops=dict(width=0.42, edgecolor="white")
        )
        centre_circle = plt.Circle((0, 0), 0.45, fc="white")
        ax_pie.add_artist(centre_circle)
        total_val = l2_ser.sum()
        ax_pie.text(0, 0, f"{l1_cat}\n\n{total_val:,.2f}", ha="center", va="center", fontsize=10, fontweight="bold")
        ax_pie.set_title(f"{district} ({state_name})\n"f"{l1_cat} — L2 Composition ({yr})", fontsize=11, fontweight="bold")
        
        total_l2_sum = l2_ser.sum()
        cell_data = [[idx, f"{val:,.2f}", f"{(val/total_l2_sum)*100:.2f}%"] for idx, val in l2_ser.items()]
        _add_side_table(ax_tbl, cell_data, ["L2 Sub-Category", "Area (sq km)", "Share in %"], "Composition Summary")

        safe_l1 = (l1_cat.strip().replace(" ", "_").replace("/", "-"))[:40]
        plt.tight_layout()
        _save(out_dir, f"01_donut_{yr}_{safe_l1}.png")

    print(f" 2 Grouped bar L1 change from baseline …")
    all_l1 = sorted(set().union(*[l1_totals(dist_data[yr]).index for yr in years_with_dist]))
    l1_ts = pd.DataFrame({yr: l1_totals(dist_data[yr]).reindex(all_l1, fill_value=0) for yr in years_with_dist}, index=all_l1).T
    baseline = l1_ts.iloc[0]
    l1_delta = l1_ts - baseline
    l1_delta = l1_delta.iloc[1:]  

    n_years = len(l1_delta)
    n_l1 = len(all_l1)
    x = np.arange(n_l1)
    width = 0.35
    palette = plt.cm.get_cmap("tab10", n_years)

    fig, (ax_chart, ax_tbl) = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [2.2, 1.2]})
    
    for i, yr in enumerate(l1_delta.index):
        offset = (i - n_years / 2 + 0.5) * width
        bars = ax_chart.bar(x + offset, l1_delta.loc[yr].values, width, label=yr, color=palette(i), edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, l1_delta.loc[yr].values):
            if v != 0:
                offset_val = abs(v) * 0.02
                ax_chart.text(bar.get_x() + bar.get_width() / 2, v + (offset_val if v >= 0 else -offset_val), 
                        f"{v:+,.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7, rotation=90)

    ax_chart.axhline(0, color="black", linewidth=0.9)
    ax_chart.set_title(f"{district}  ({state_name})\nL1 Category Area Change from {years_with_dist[0]}", fontsize=11, fontweight="bold")
    ax_chart.set_ylabel(f"Δ Area from {years_with_dist[0]} (sq km)", fontsize=10)
    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels(all_l1, rotation=30, ha="right", fontsize=9)
    ax_chart.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+,.2f}"))  
    ax_chart.legend(title="Year", fontsize=9)
    ax_chart.grid(axis="y", linestyle="--", alpha=0.4)
    
    tbl_rows = []
    tbl_cols = ["Category"] + [f"Change {yr}" for yr in l1_delta.index]
    for cat in all_l1:
        row_vals = [cat] + [f"{l1_delta.loc[yr, cat]:+,.2f}" for yr in l1_delta.index]
        tbl_rows.append(row_vals)
    _add_side_table(ax_tbl, tbl_rows, tbl_cols, "Baseline Delta Summary")

    plt.tight_layout()
    _save(out_dir, "02_timeseries_total_area.png")
    
    print(f"    3 Time-series per L1 …")
    all_l1 = sorted(set().union(*[l1_totals(dist_data[yr]).index for yr in years_with_dist]))
    l1_ts = pd.DataFrame({yr: l1_totals(dist_data[yr]).reindex(all_l1, fill_value=0) for yr in years_with_dist}, index=all_l1).T

    fig, (ax_chart, ax_tbl) = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [2.2, 1.2]})
    palette = plt.cm.get_cmap("tab10", len(all_l1))
    
    for i, cat in enumerate(all_l1):
        ax_chart.plot(l1_ts.index, l1_ts[cat], marker="o", linewidth=1.8, label=cat, color=palette(i), markersize=6)

    ax_chart.set_title(f"{district}  ({state_name})\nL1 Category Area Trends", fontsize=11, fontweight="bold")
    ax_chart.set_ylabel("Area (sq km)", fontsize=10)
    ax_chart.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.2f}")) 
    ax_chart.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title="L1 Category")
    ax_chart.grid(axis="y", linestyle="--", alpha=0.4)
    
    # Trend Summary Table Data Layout
    tbl_rows = []
    tbl_cols = ["Category"] + list(l1_ts.index)
    for cat in all_l1:
        row_vals = [cat] + [f"{l1_ts.loc[yr, cat]:,.2f}" for yr in l1_ts.index]
        tbl_rows.append(row_vals)
    _add_side_table(ax_tbl, tbl_rows, tbl_cols, "Trend Area History (sq km)")

    plt.tight_layout()
    _save(out_dir, "03_timeseries_per_l1.png")

    if ser_first.empty or ser_recent.empty:
        print(f"    SKIP Change plots skipped (missing {y_first} or {y_recent})")
        return

    l1_first  = l1_totals(ser_first)
    l1_recent = l1_totals(ser_recent)
    common_l1 = l1_first.index.union(l1_recent.index)
    l1_first  = l1_first.reindex(common_l1, fill_value=0)
    l1_recent = l1_recent.reindex(common_l1, fill_value=0)

    l1_delta_abs = l1_recent - l1_first
    l1_delta_pct = (l1_delta_abs / l1_first.replace(0, np.nan)) * 100

    print(f"  4 L1 absolute change …")
    _hbar_change(l1_delta_abs, title=f"{district} — L1 Change in Area ({y_previous} to {y_recent})", xlabel="Change in Area (sq km)", out_dir=out_dir, fname="04_l1_change_abs_numbers.png", unit=" sq km")

    print(f"  5 L1 percent change …")
    _hbar_change(l1_delta_pct.dropna(), title=f"{district} — L1 Percentage Change ({y_previous} to {y_recent})", xlabel="% Change", out_dir=out_dir, fname="05_l1_change_pct.png", unit="%")

    print(f" L2 change per L1 …")
    ser_previous = dist_data.get(y_previous, pd.Series(dtype=float))

    for l1_cat in sorted(common_l1):
        ser_prev_idx_str = ser_previous.index.astype(str)
        ser_rec_idx_str = ser_recent.index.astype(str)

        mask_first  = ser_prev_idx_str.str.startswith(l1_cat + " › ")
        mask_recent = ser_rec_idx_str.str.startswith(l1_cat + " › ")

        l2_first  = ser_previous[mask_first].copy()
        l2_recent = ser_recent[mask_recent].copy()

        if l2_first.empty and l2_recent.empty:
            continue

        l2_first.index  = l2_first.index.astype(str).str.split(" › ").str[1]
        l2_recent.index = l2_recent.index.astype(str).str.split(" › ").str[1]

        all_l2 = l2_first.index.union(l2_recent.index).astype(str)
        l2_first  = l2_first.reindex(all_l2, fill_value=0)
        l2_recent = l2_recent.reindex(all_l2, fill_value=0)

        l2_delta_abs = l2_recent - l2_first
        l2_delta_pct = (l2_delta_abs / l2_first.replace(0, np.nan)) * 100
        if l2_delta_abs.abs().sum() == 0:
            continue

        safe_l1 = (l1_cat.strip().replace(" ", "_").replace("/", "-").replace("__", "_"))[:40]
        fig, (ax_abs, ax_pct, ax_tbl) = plt.subplots(1, 3, figsize=(21, max(5, len(all_l2) * 0.55)), gridspec_kw={"width_ratios": [1.5, 1.5, 1.1]})

        x = np.arange(len(all_l2))
        w = 0.4
        colors_a = ["#d73027" if v < 0 else "#4dac26" for v in l2_delta_abs]
        ax_abs.bar(x, l2_delta_abs.values, color=colors_a, edgecolor="white", linewidth=0.4, width=w * 1.8)
        ax_abs.axhline(0, color="black", linewidth=0.8)
        for xi, v in zip(x, l2_delta_abs.values):
            if np.isfinite(v) and v != 0:
                va = "bottom" if v >= 0 else "top"
                offset = l2_delta_abs.abs().max() * 0.02 if l2_delta_abs.abs().max() > 0 else 1
                ax_abs.text(xi, v + (offset if v >= 0 else -offset), f"{v:+,.2f}", ha="center", va=va, fontsize=7, rotation=90)
        ax_abs.set_xticks(x)
        ax_abs.set_xticklabels(all_l2, rotation=40, ha="right", fontsize=8)
        ax_abs.set_ylabel("Δ Area (sq km)", fontsize=9)
        ax_abs.set_title(f"Absolute Change\nfrom {y_previous} to {y_recent}", fontsize=9, fontweight="bold")
        ax_abs.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.2f}"))
        valid_pct = l2_delta_pct.replace([np.inf, -np.inf], np.nan)
        colors_p = ["#d73027" if (pd.notna(v) and v < 0) else "#4dac26" for v in valid_pct]
        ax_pct.bar(x, valid_pct.fillna(0).values, color=colors_p, edgecolor="white", linewidth=0.4, width=w * 1.8)
        ax_pct.axhline(0, color="black", linewidth=0.8)
        for xi, v in zip(x, valid_pct.values):
            if np.isfinite(v) and v != 0:
                va = "bottom" if v >= 0 else "top"
                offset = valid_pct.abs().max() * 0.02 if valid_pct.abs().max() > 0 else 1
                ax_pct.text(xi, v + (offset if v >= 0 else -offset), f"{v:+.2f}%", ha="center", va=va, fontsize=7, rotation=90)  
        ax_pct.set_xticks(x)
        ax_pct.set_xticklabels(all_l2, rotation=40, ha="right", fontsize=8)
        ax_pct.set_ylabel("% Change", fontsize=9)
        ax_pct.set_title(f"Percentage Change\nfrom {y_previous} to {y_recent}", fontsize=9, fontweight="bold")
        ax_pct.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+,.2f}%"))  
        gain_patch = mpatches.Patch(color="#4dac26", label="Gain")
        loss_patch = mpatches.Patch(color="#d73027", label="Loss")
        fig.legend(handles=[gain_patch, loss_patch], loc="upper right", fontsize=9, ncol=2)

        # Build Side Data Summary Table
        tbl_rows = []
        for idx in all_l2:
            abs_str = f"{l2_delta_abs[idx]:+,.2f}"
            pct_val = valid_pct[idx]
            pct_str = f"{pct_val:+.2f}%" if pd.notna(pct_val) else "N/A"
            tbl_rows.append([idx, abs_str, pct_str])
        _add_side_table(ax_tbl, tbl_rows, ["L2 Category", " Absolute change(sqkm)", "% Change"], "L2 Change Breakdown")

        fig.suptitle(f"{district}  ({state_name})  —  {l1_cat}\nL2 Sub-Category Changes", fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        _save(out_dir, f"06_l2_change_{safe_l1}.png")

def process_state(state_name: str):
    print(f" STATE: {state_name}")

    DATA = load_state(state_name)
    if not DATA:
        print(" SKIP No data loaded.")
        return

    canon_map = build_canonical_map(DATA)
    DATA = apply_canonical(DATA, canon_map)

    out_root = os.path.join(OUTPUT_ROOT, state_name.replace(" ", "_"))
    os.makedirs(out_root, exist_ok=True)

    all_districts = sorted(set().union(*[df.columns for df in DATA.values()]))
    print(f"  Districts after normalisation: {len(all_districts)}")

    for district in all_districts:
        print(f"\n District: {district}")
        try:
            process_district(district, DATA, state_name, out_root)
        except Exception as e:
            print(f"ERROR {district}: {e}")

    print(f"\n  State complete. Outputs in: {out_root}/")

if __name__ == "__main__":
    sample_folder = list(YEAR_FOLDERS.values())[0]
    ALL_STATES = [f.replace(".csv", "") for f in os.listdir(sample_folder) if f.endswith(".csv")]
    print(f"States found: {ALL_STATES}\n")

    for state in ALL_STATES:  
            try:
                process_state(state)
            except Exception as e:
                print(f"ERROR {state}: {e}")