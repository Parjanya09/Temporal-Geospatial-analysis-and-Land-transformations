import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = "refined data"

YEAR_FOLDERS = {
    "2005-06": os.path.join(BASE_DIR, "csv of 2005-2006"),
    "2011-12": os.path.join(BASE_DIR, "csv of 2011-2012"),
    "2015-16": os.path.join(BASE_DIR, "csv of 2015-2016"),
}

YEAR_LABELS = list(YEAR_FOLDERS.keys())

sample_folder = list(YEAR_FOLDERS.values())[0]

ALL_STATES = [
    f.replace(".csv", "")
    for f in os.listdir(sample_folder)
    if f.endswith(".csv")
]

print("States found:")
print(ALL_STATES)


def clean_text(x):

    if pd.isna(x):
        return ""

    x = str(x).upper()

    x = x.replace("_", " ")
    x = x.replace("-", " ")

    x = " ".join(x.split())

    return x


def make_key(x):

    x = str(x).upper()

    x = x.replace(" ", "")
    x = x.replace("/", "")
    x = x.replace("-", "")
    x = x.replace("_", "")
    x = x.replace(",", "")
    x = x.replace(".", "")

    x = x.replace("MANGROVES", "MANGROVE")
    x = x.replace("WATERBODIES", "WATERBODY")
    x = x.replace("CANALS", "CANAL")
    x = x.replace("LANDS", "LAND")

    x = x.replace(
        "GULLIEDRAVINOUSLAND",
        "GULLIEDRAVINOUS"
    )

    x = x.replace(
        "GRASSGRAZINGLAND",
        "GRASSGRAZING"
    )

    x = x.replace(
        "SNOWANDGLACIER",
        "SNOW"
    )

    x = x.replace(
        "CURRENTSHIFTINGCULTIVATIO",
        "CURRENTSHIFTINGCULTIVATION"
    )

    x = x.replace(
        "EVERGREENSEMIEVERGREE",
        "EVERGREENSEMIEVERGREEN"
    )

    return x


def load_state(state, year_folders):

    data = {}

    for label, folder in year_folders.items():

        path = os.path.join(folder, f"{state}.csv")

        if not os.path.exists(path):

            print(f"[WARN] Missing: {path}")

            continue

        raw = pd.read_csv(path, header=0)

        l1_col = raw.columns[0]
        l2_col = raw.columns[1]

        raw[l1_col] = raw[l1_col].ffill()

        raw["_label"] = (
            raw[l1_col].astype(str).str.strip()
            + " › "
            + raw[l2_col].astype(str).str.strip()
        )

        raw = raw.set_index("_label")

        raw = raw.drop(columns=[l1_col, l2_col])

        raw = raw.loc[
            :,
            ~raw.columns.str.contains(
                "Grand Total",
                case=False
            )
        ]

        raw = raw.apply(
            pd.to_numeric,
            errors="coerce"
        )

        raw = raw.fillna(0)

        raw.columns = (
            raw.columns
            .astype(str)
            .map(clean_text)
        )

        l1_vals = (
            raw.index
            .str.split(" › ")
            .str[0]
            .map(clean_text)
        )

        l2_vals = (
            raw.index
            .str.split(" › ")
            .str[1]
            .map(clean_text)
        )

        raw.index = (
            l1_vals.astype(str)
            + " › "
            + l2_vals.astype(str)
        )

        raw["_match_key"] = (
            raw.index.map(make_key)
        )

        raw = raw.set_index("_match_key")

        raw = raw.groupby(raw.index).sum()

        data[label] = raw

    return data


def process_state(state_name):

    print(f"\nProcessing: {state_name}")

    output_dir = os.path.join(
        "scatterplot_analysis_output",
        state_name.replace(" ", "_")
    )

    os.makedirs(output_dir, exist_ok=True)

    print("Loading data …")

    data = load_state(
        state_name,
        YEAR_FOLDERS
    )

    print("Year-vs-year scatter plots …")

    pairs = [ (YEAR_LABELS[-2], YEAR_LABELS[-1]) ]

    total_plots = 0

    for (y1, y2) in pairs:

        if y1 not in data or y2 not in data:
            continue

        df1 = data[y1]
        df2 = data[y2]

        common_dists = (
            df1.columns.intersection(df2.columns)
        )

        common_cats = (
            df1.index.intersection(df2.index)
        )

        for cat in common_cats:

            try:

                v1 = (
                    df1.loc[cat, common_dists]
                    .astype(float)
                    .fillna(0)
                )

                v2 = (
                    df2.loc[cat, common_dists]
                    .astype(float)
                    .fillna(0)
                )

                fig, ax = plt.subplots(
                    figsize=(8, 7)
                )

                ax.scatter(
                    v1,
                    v2,
                    s=45,
                    alpha=0.75,
                    edgecolors="k",
                    linewidths=0.4
                )

                shown_positions = []

                x_range = max(
                    v1.max() - v1.min(),
                    1
                )

                y_range = max(
                    v2.max() - v2.min(),
                    1
                )

                x_thresh = x_range * 0.015
                y_thresh = y_range * 0.015

                for dist in common_dists:

                    x = float(v1[dist])
                    y = float(v2[dist])

                    too_close = False

                    for px, py in shown_positions:

                        if (
                            abs(x - px) < x_thresh
                            and abs(y - py) < y_thresh
                        ):
                            too_close = True
                            break

                    if not too_close:

                        ax.annotate(
                            dist,
                            (x, y),
                            fontsize=5,
                            alpha=0.7,
                            xytext=(3, 2),
                            textcoords="offset points"
                        )

                        shown_positions.append((x, y))

                title_extra = ""

                mask = (
                    np.isfinite(v1)
                    & np.isfinite(v2)
                )

                xvals = v1[mask].astype(float)
                yvals = v2[mask].astype(float)

                if len(xvals) >= 2:

                    try:

                        unique_x = len(
                            np.unique(xvals)
                        )

                        unique_y = len(
                            np.unique(yvals)
                        )

                        if unique_x >= 2:

                            m, b = np.polyfit(
                                xvals,
                                yvals,
                                1
                            )

                            xs = np.linspace(
                                xvals.min(),
                                xvals.max(),
                                100
                            )

                            ys = m * xs + b

                            ax.plot(
                                xs,
                                ys,
                                linestyle="--",
                                linewidth=1.2
                            )

                            if unique_y >= 2:

                                r, p = pearsonr(
                                    xvals,
                                    yvals
                                )

                                title_extra = (
                                    f"r = {r:.2f}   "
                                    f"p = {p:.3f}"
                                )

                    except Exception:
                        pass

                ax.set_title(
                    f"{cat}\n"
                    f"{y1} vs {y2}   "
                    f"{title_extra}",
                    fontsize=10,
                    fontweight="bold"
                )

                ax.set_xlabel(
                    f"Area sq km — {y1}",
                    fontsize=9
                )

                ax.set_ylabel(
                    f"Area sq km — {y2}",
                    fontsize=9
                )

                plt.tight_layout()

                safe_cat = (
                    cat.replace(" ", "_")
                    .replace("/", "-")
                    .replace(">", "")
                    .replace("›", "")
                )[:60]

                filename = (
                    f"{safe_cat}_{y1}_{y2}.png"
                    .replace("-", "_")
                )

                path = os.path.join(
                    output_dir,
                    filename
                )

                plt.savefig(
                    path,
                    dpi=120,
                    bbox_inches="tight"
                )

                plt.close()

                total_plots += 1

            except Exception as e:
                print(
                    f"SKIPPED {state_name} | "
                    f"{cat} | "
                    f"{y1}-{y2} : {e}"
                )

    print(f"Saved {total_plots} scatter plots")

    print(
        f"Successfully Completed: "
        f"{state_name}"
    )


for state in ALL_STATES:

    try:
        process_state(state)
    except Exception as e:

        print(f"ERROR {state}: {e}")