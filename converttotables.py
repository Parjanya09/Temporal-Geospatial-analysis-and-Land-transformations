import os
import re
import pdfplumber
import pandas as pd

RAW_ROOT = "raw data"
REFINED_ROOT = "refined data"

KEY_COLS = ["L 1", "L 2"]

MASTER_KEYS = {}


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def standardize_keys(df):
    if "L 1" not in df.columns or "L 2" not in df.columns:
        return df
    fixed_l1 = []
    fixed_l2 = []
    for l1, l2 in zip(df["L 1"], df["L 2"]):
        norm_l1 = normalize_text(l1)
        norm_l2 = normalize_text(l2)
        key = (norm_l1, norm_l2)
        if key not in MASTER_KEYS:
            MASTER_KEYS[key] = (str(l1).strip(), str(l2).strip())
        canon_l1, canon_l2 = MASTER_KEYS[key]
        fixed_l1.append(canon_l1)
        fixed_l2.append(canon_l2)
    df["L 1"] = fixed_l1
    df["L 2"] = fixed_l2
    return df


def fix_split_cells(table):
    fixed = []
    for row in table:
        new_row = list(row)
        result = []
        i = 0
        while i < len(new_row):
            cell = str(new_row[i]).strip() if new_row[i] else ""
            if i + 1 < len(new_row):
                nxt = str(new_row[i + 1]).strip() if new_row[i + 1] else ""
                if nxt and re.search(r"\d$", cell) and re.match(r"^[a-z]", nxt):
                    merged = (cell + nxt).strip()
                    m = re.match(r"^(.*\D)\s+([\d.]+)$", merged)
                    if m:
                        result.append(m.group(1).strip())
                        tail = list(new_row[i + 2:])
                        tail.insert(0, m.group(2))
                        new_row = new_row[: i + 2] + tail
                    else:
                        result.append(merged)
                    i += 2
                    continue
            result.append(cell)
            i += 1
        fixed.append(result)
    return fixed


def looks_like_number(val):
    return bool(re.match(r"^\s*[\d,.\-]+\s*$", str(val))) or str(val).strip() == ""


def classify_page(df):
    cols = [str(c).strip() for c in df.columns]

    if cols[:2] == KEY_COLS:
        return "normal"

    first_two_blank = all(c == "" for c in cols[:2])

    if first_two_blank:
        sample = df.iloc[:min(5, len(df))]
        col0_numeric = all(looks_like_number(v) for v in sample.iloc[:, 0])
        col1_numeric = all(looks_like_number(v) for v in sample.iloc[:, 1])
        if col0_numeric and col1_numeric:
            return "no_keys"
        return "blank_keys"

    return "no_keys"


def normalise_key_cols(df, page_num=None):
    kind = classify_page(df)

    if kind == "normal":
        return df

    if kind == "blank_keys":
        cols = list(df.columns)
        df = df.rename(columns={cols[0]: "L 1", cols[1]: "L 2"})
        print(f"    Page {page_num}: blank key headers — renamed by position.")
        return df

    return None


def extract_page(page, page_num=None):
    tables = page.extract_tables()
    if not tables:
        return None, "skip"
    table = max(tables, key=len)
    table = fix_split_cells(table)
    if len(table) < 2:
        return None, "skip"

    raw_headers = table[0]
    headers = [
        str(h).strip().replace("\n", " ") if h else ""
        for h in raw_headers
    ]
    df = pd.DataFrame(table[1:], columns=headers)

    kind = classify_page(df)

    if kind == "no_keys":
        print(f"    Page {page_num}: no L 1/L 2 found — treating all columns as district data, will append right.")
        return df, "no_keys"

    df = normalise_key_cols(df, page_num=page_num)
    if df is None:
        return None, "skip"

    return df, "normal"


def forward_fill_l1(df):
    if "L 1" in df.columns:
        df["L 1"] = (
            df["L 1"]
            .replace("", pd.NA)
            .ffill()
            .fillna("")
            .str.replace("\n", " ", regex=False)
            .str.strip()
        )
    return df


def clean_l2(df):
    if "L 2" in df.columns:
        df["L 2"] = (
            df["L 2"]
            .fillna("")
            .str.replace("\n", " ", regex=False)
            .str.strip()
        )
    return df


def drop_empty_rows(df):
    if "L 2" not in df.columns:
        return df
    mask = df["L 2"].replace("", pd.NA).isna()
    return df[~mask].reset_index(drop=True)


def clean_page(df):
    return df.pipe(forward_fill_l1).pipe(clean_l2).pipe(drop_empty_rows)


def merge_pages(pages):
    # pages is a list of (df, kind) tuples
    # kind == "normal"  → merge on L1+L2 keys
    # kind == "no_keys" → concat columns directly by row position onto accumulated result

    if not pages:
        return None

    result, _ = pages[0]

    for idx, (page_df, kind) in enumerate(pages[1:], start=2):

        if kind == "no_keys":
            # All columns in this page are district data, no keys present
            # Row order matches the reference page exactly — concat by position
            if len(page_df) != len(result):
                print(f"    Page {idx}: row count mismatch ({len(page_df)} vs {len(result)}) on no-key page — skipped.")
                continue
            page_df = page_df.reset_index(drop=True)
            result = result.reset_index(drop=True)
            result = pd.concat([result, page_df], axis=1)
            print(f"    Page {idx}: appended {len(page_df.columns)} district cols directly by position.")
            continue

        missing_keys = [k for k in KEY_COLS if k not in page_df.columns]
        if missing_keys:
            print(f"    Page {idx}: key columns {missing_keys} still missing — skipped.")
            continue

        new_district_cols = [c for c in page_df.columns if c not in KEY_COLS]
        if not new_district_cols:
            print(f"    Page {idx}: no district columns found — skipped.")
            continue

        merge_slice = page_df[KEY_COLS + new_district_cols]
        result = pd.merge(
            result,
            merge_slice,
            on=KEY_COLS,
            how="outer",
            suffixes=("", "_dup"),
        )
        result = result.loc[:, ~result.columns.str.endswith("_dup")]

    return result


def process_pdf(pdf_path):
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                df, kind = extract_page(page, page_num=page_num)

                if df is None:
                    print(f"    Page {page_num}: no table found — skipped.")
                    continue

                if kind != "no_keys":
                    df = clean_page(df)

                df = standardize_keys(df)

                if df.empty:
                    print(f"    Page {page_num}: table empty after cleaning — skipped.")
                    continue

                pages.append((df, kind))

                district_cols = [c for c in df.columns if c not in KEY_COLS]
                preview = ", ".join(district_cols[:3])
                if len(district_cols) > 3:
                    preview += "..."
                print(f"    Page {page_num}: {len(df)} rows, {len(district_cols)} district cols ({preview})")

            except Exception as exc:
                print(f"    Page {page_num}: ERROR — {exc}")

    if not pages:
        return None

    merged = merge_pages(pages)

    merged["_key"] = (
        merged["L 1"].str.lower().str.strip()
        + "||"
        + merged["L 2"].str.lower().str.strip()
    )
    merged = merged.drop_duplicates(subset="_key", keep="first")
    merged = merged.drop(columns="_key")
    merged = merged.reset_index(drop=True)
    return merged


for folder_name in sorted(os.listdir(RAW_ROOT)):
    pdf_folder = os.path.join(RAW_ROOT, folder_name)
    if not os.path.isdir(pdf_folder):
        continue
    out_folder_name = folder_name.replace("data from", "csv of")
    output_folder = os.path.join(REFINED_ROOT, out_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n{'═'*60}")
    print(f"FOLDER: {folder_name}  to  {output_folder}")
    for pdf_file in sorted(os.listdir(pdf_folder)):
        if not pdf_file.endswith(".pdf"):
            continue
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"Processing: {pdf_file}")
        result = process_pdf(pdf_path)
        if result is None:
            print("   No usable data found.")
            continue
        out_name = pdf_file.replace(".pdf", ".csv")
        out_path = os.path.join(output_folder, out_name)
        try:
            result.to_csv(out_path, index=False)
            print(f"   Saved: {out_path}")
            print(f"     Shape: {result.shape[0]} feature rows × {result.shape[1]} columns ({result.shape[1] - len(KEY_COLS)} district/total cols)")
        except PermissionError:
            print(f"   File is open elsewhere — skipped: {out_path}")

print("Saved successfully.")