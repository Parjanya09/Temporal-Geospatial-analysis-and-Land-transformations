import os
import io
import json
import traceback
import re
import math
import tempfile
import shutil
import subprocess
import warnings
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, send_file, session
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile

os.environ.setdefault("PROJ_NETWORK", "OFF")
warnings.filterwarnings("ignore")

upload_bp = Blueprint("ingest", __name__)

UPLOAD_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

LU_WEBCODE = {
    1:  ("Builtup",                          "Urban",              (255, 0,   0)),
    2:  ("Builtup",                          "Rural",              (205, 92,  92)),
    3:  ("Builtup",                          "Mining",             (139, 0,   0)),
    4:  ("Agriculture",                      "Cropland",           (255, 255, 0)),
    5:  ("Agriculture",                      "Plantation",         (255, 165, 0)),
    6:  ("Agriculture",                      "Fallow",             (255, 215, 0)),
    8:  ("Forest",                           "Evergreen/Semi Evergreen", (0, 128, 0)),
    9:  ("Forest",                           "Deciduous",          (34,  139, 34)),
    10: ("Forest",                           "Forest Plantation",  (0,   100, 0)),
    11: ("Forest",                           "Scrub Forest",       (144, 238, 144)),
    12: ("Forest",                           "Swamp/Mangrove",     (0,   80,  0)),
    13: ("Grass-Grazing",                    "Grass-Grazing",      (128, 128, 0)),
    14: ("Barren/Unculturable/Wastelands",   "Salt Affected",      (188, 143, 143)),
    15: ("Barren/Unculturable/Wastelands",   "Gullied/Ravinous",   (160, 82,  45)),
    16: ("Barren/Unculturable/Wastelands",   "Scrubland",          (169, 196, 108)),
    17: ("Barren/Unculturable/Wastelands",   "Sandy Area",         (210, 180, 140)),
    18: ("Barren/Unculturable/Wastelands",   "Barren Rocks",       (245, 245, 220)),
    20: ("Wetlands/Waterbodies",             "Inland Wetland",     (135, 206, 235)),
    22: ("Wetlands/Waterbodies",             "River/Stream/Canal", (30,  144, 255)),
    23: ("Wetlands/Waterbodies",             "Waterbodies",        (0,   191, 255)),
}

CATEGORY_COLORS = [
    (["cropland"],                               "#ffff00"),
    (["fallow"],                                 "#ffd700"),
    (["forest plantation","forestplantation"],   "#006400"),
    (["plantation"],                             "#ffa500"),
    (["evergreen","semievergreen"],              "#008000"),
    (["decidious","deciduous"],                  "#228b22"),
    (["scrub forest","scrubforest"],             "#90ee90"),
    (["swamp","mangrove"],                       "#2e8b57"),
    (["scrubland"],                              "#a9c46c"),
    (["urban"],                                  "#ff0000"),
    (["rural"],                                  "#cd5c5c"),
    (["mining"],                                 "#8b0000"),
    (["river","canal","stream"],                 "#1e90ff"),
    (["inland wetland","inlandwetland"],         "#87ceeb"),
    (["waterbod"],                               "#00bfff"),
    (["barren rock","barrenrock"],               "#f5f5dc"),
    (["sandyland","sandy","sand"],               "#d2b48c"),
    (["salt"],                                   "#bc8f8f"),
    (["gullied","ravine"],                       "#a0522d"),
    (["scrub"],                                  "#c8e6c9"),
    (["grass","grazing"],                        "#808000"),
    (["snow","glacier"],                         "#fffafa"),
    (["forest"],                                 "#228b22"),
    (["agriculture","agri"],                     "#ffe066"),
    (["buildup","builtup"],                      "#ff6666"),
    (["wetland","waterbod"],                     "#4fc3f7"),
    (["barren"],                                 "#e8dcc8"),
]

def color_for_label(label):
    lbl_lower = label.lower()
    for words, hex_color in CATEGORY_COLORS:
        if any(w in lbl_lower for w in words):
            return hex_color
    return "#888888"

def color_for_code(code):
    info = LU_WEBCODE.get(code)
    if info and len(info) > 2:
        return info[2]
    return (128, 128, 128)

def label_for_code(code):
    info = LU_WEBCODE.get(code)
    if info:
        return f"{info[0]} > {info[1]}"
    return f"Class {code}"

def state_key_from_name(display_name):
    cleaned = re.sub(r'[^a-zA-Z0-9\s_-]', '', display_name)
    return cleaned.strip().lower().replace(" ", "_").replace("-", "_")

def code_from_key(state_key):
    words = state_key.split("_")
    return "".join(w[0].upper() for w in words if w)[:4]

def _session_id():
    try:
        from flask import session as flask_session
        sid = flask_session.get("sid")
        if not sid:
            import uuid
            sid = uuid.uuid4().hex
            flask_session["sid"] = sid
        return sid
    except RuntimeError:
        return "default"

def upload_folder_for_key(state_key, sid=None):
    if sid is None:
        sid = _session_id()
    path = os.path.join(UPLOAD_ROOT, sid, state_key)
    os.makedirs(path, exist_ok=True)
    return path

def session_root(sid=None):
    if sid is None:
        sid = _session_id()
    return os.path.join(UPLOAD_ROOT, sid)

def meta_path(state_key, sid=None):
    return os.path.join(upload_folder_for_key(state_key, sid), "_meta.json")

def load_meta(state_key, sid=None):
    mp = meta_path(state_key, sid)
    if os.path.exists(mp):
        with open(mp) as f:
            return json.load(f)
    return {}

def save_meta(state_key, meta, sid=None):
    with open(meta_path(state_key, sid), "w") as f:
        json.dump(meta, f, indent=2)

_PERIOD_CODE_MAP = {
    "0506": "2005-06",
    "1112": "2011-12",
    "1516": "2015-16",
    "1920": "2019-20",
    "2021": "2020-21",
}
_PERIOD_CODE_ORDER = {"0506": 0, "1112": 1, "1516": 2, "1920": 3, "2021": 4}

def _period_sort_key(code):
    if code in _PERIOD_CODE_ORDER:
        return _PERIOD_CODE_ORDER[code]
    m = re.fullmatch(r'(\d{2})(\d{2})', str(code))
    if m:
        y1 = int(m.group(1))
        return (2000 + y1) if y1 < 50 else (1900 + y1)
    m2 = re.search(r'(20\d{2}|19\d{2})', str(code))
    if m2:
        return int(m2.group(1))
    return 9999

def sort_periods(codes):
    return sorted(set(codes), key=_period_sort_key)

def make_suffix(p1, p2):
    a, b = sorted([p1, p2], key=_period_sort_key)
    return f"{a}_to_{b}"

def period_label(code_or_str):
    s = str(code_or_str).strip()
    if "_to_" in s:
        parts = s.split("_to_")
        labels = [period_label(p) for p in parts]
        years = []
        for lbl in labels:
            m = re.search(r'(\d{4})', lbl)
            years.append(int(m.group(1)) if m else 9999)
        if len(labels) == 2 and years[0] > years[1]:
            labels = labels[::-1]
        return " to ".join(labels)
    if re.fullmatch(r'20\d{2}-\d{2}', s):
        return s
    if s in _PERIOD_CODE_MAP:
        return _PERIOD_CODE_MAP[s]
    m = re.fullmatch(r'(\d{2})(\d{2})', s)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        full_y1 = (2000 + y1) if y1 < 50 else (1900 + y1)
        return f"{full_y1}-{m.group(2)}"
    return s

def normalize_period_label(label):
    if not label or "to" not in label.lower():
        return label
    parts = re.split(r'\s+to\s+', label.strip(), flags=re.IGNORECASE)
    if len(parts) != 2:
        return label
    def yr(p):
        m = re.search(r'(\d{4})', p)
        return int(m.group(1)) if m else 9999
    if yr(parts[0]) > yr(parts[1]):
        parts = parts[::-1]
    return f"{parts[0]} to {parts[1]}"

def infer_period(filename):
    n = filename.lower().replace("-", "").replace("_", "").replace(" ", "")
    for code in ["0506", "1112", "1516", "1920", "2021"]:
        if code in n:
            return code
    four_digit_matches = re.findall(r'(20\d{2}|19\d{2})', filename)
    if four_digit_matches:
        years = sorted(set(int(y) for y in four_digit_matches))
        if len(years) == 1:
            y = years[0]
            return f"{str(y)[2:]}{str(y + 1)[2:]}"
        elif len(years) == 2:
            s1 = str(years[0])[2:] + str(years[0]+1)[2:]
            s2 = str(years[1])[2:] + str(years[1]+1)[2:]
            return make_suffix(s1, s2)
    return "unknown"

def is_change_filename(filename):
    n = filename.lower()
    return "change" in n or "trans" in n or ("to" in n and re.search(r'\d{2}', n))

def calculate_pixel_area_sqkm(src):
    bounds = src.bounds
    center_lat = (bounds.bottom + bounds.top) / 2.0
    res_x, res_y = src.res
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))
    return (res_x * km_per_deg_lon) * (res_y * km_per_deg_lat)

def build_change_tif(tif1_path, tif2_path, out_path):
    with rasterio.open(tif1_path) as src1:
        arr1 = src1.read(1).astype(np.int32)
        profile = src1.profile.copy()
        tf1 = src1.transform
        h, w = src1.height, src1.width

    with rasterio.open(tif2_path) as src2:
        arr2 = np.zeros((h, w), dtype=np.int32)
        reproject(
            source=rasterio.band(src2, 1), destination=arr2,
            src_transform=src2.transform, src_crs=src2.crs,
            dst_transform=tf1, dst_crs=src1.crs, resampling=Resampling.nearest,
        )

    change = arr1 * 100 + arr2
    change[arr1 == 0] = 0
    change[arr2 == 0] = 0

    profile.update({"dtype": "int32", "compress": "deflate", "nodata": 0, "tiled": True, "blockxsize": 512, "blockysize": 512})
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(change, 1)

def generate_decoded_csv(tif_path, period_suffix, state_key, out_folder, sid=None):
    meta = load_meta(state_key, sid)
    state_code = meta.get("code", code_from_key(state_key))
    out_name = f"changes_{state_code}_{period_suffix}_decoded.csv"
    out_path = os.path.join(out_folder, out_name)
    is_change = "_to_" in period_suffix

    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.int32)
        pixel_sqkm = calculate_pixel_area_sqkm(src)

    mask = (data != 0)
    unique_vals, counts = np.unique(data[mask], return_counts=True)
    rows = []

    for val, cnt in zip(unique_vals, counts):
        iv = int(val)
        sqkm_v = float(cnt) * pixel_sqkm

        if is_change:
            fc = iv // 100
            tc = iv % 100
            if fc == 0 or tc == 0:
                continue
            from_lbl = label_for_code(fc)
            to_lbl = label_for_code(tc)
        else:
            from_lbl = label_for_code(iv)
            to_lbl = from_lbl

        rows.append({
            "value": iv, "From category": from_lbl, "To category": to_lbl,
            "count": int(cnt), "sqkm_total": round(sqkm_v, 4), "pixel_sqkm": round(pixel_sqkm, 8)
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["value", "From category", "To category", "count", "sqkm_total", "pixel_sqkm"])
    df.to_csv(out_path, index=False)
    return out_path, len(df)

def _validate_geo_bounds(min_x, min_y, max_x, max_y, label="layer"):
    if math.isclose(min_x, max_x, abs_tol=1e-9) or math.isclose(min_y, max_y, abs_tol=1e-9):
        raise ValueError(
            f"{label} spatial bounds collapsed into a degenerate point: "
            f"({min_x:.6f}, {min_y:.6f}) to ({max_x:.6f}, {max_y:.6f})."
        )
    if (max_x - min_x) < 1e-6 or (max_y - min_y) < 1e-6:
        raise ValueError(
            f"{label} extent is smaller than one pixel at the requested resolution: "
            f"width={max_x-min_x:.8f}, height={max_y-min_y:.8f}."
        )

def _refactor_lu_webcode_to_int32(shp_path, out_dir, lulc_field="LU_Webcode"):
    import geopandas as gpd
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Uploaded shapefile vector layer contains zero valid features.")

    if gdf.crs is None:
        bounds_raw = gdf.total_bounds
        if abs(bounds_raw[0]) > 180 or abs(bounds_raw[2]) > 180:
            gdf.set_crs(epsg=32643, inplace=True)
        else:
            gdf.set_crs(epsg=4326, inplace=True)

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    min_x, min_y, max_x, max_y = gdf.total_bounds
    _validate_geo_bounds(min_x, min_y, max_x, max_y, label="Shapefile")

    columns_lower = {col.lower(): col for col in gdf.columns}
    burn_field = columns_lower.get(lulc_field.lower())
    FALLBACK_FIELDS = ("lu_webcode", "lulc_code", "lulc", "lucode", "value", "gridcode")
    if not burn_field:
        for fb in FALLBACK_FIELDS:
            if fb in columns_lower:
                burn_field = columns_lower[fb]
                break
    if not burn_field:
        raise ValueError(
            f"Could not find LU_Webcode (or fallback) attribute field in shapefile. "
            f"Available fields: {list(gdf.columns)}"
        )

    def refactor_val(raw):
        try:
            s = str(raw).strip()
            return int(float(s))
        except Exception:
            return 0

    gdf["LU_Webcode_i32"] = gdf[burn_field].apply(refactor_val).astype(np.int32)
    gdf = gdf[gdf["LU_Webcode_i32"] > 0]
    if gdf.empty:
        raise ValueError("After refactoring LU_Webcode to int32, no valid (>0) features remained.")

    os.makedirs(out_dir, exist_ok=True)
    refactored_shp = os.path.join(out_dir, "refactored.shp")
    gdf.to_file(refactored_shp)
    return refactored_shp, gdf.total_bounds

def gdal_rasterize_layer(shp_path, out_tif_path, burn_field="LU_Webcode_i32", resolution=0.0005):
    layer_name = os.path.splitext(os.path.basename(shp_path))[0]
    cmd = [
        "gdal_rasterize",
        "-l", layer_name,
        "-a", burn_field,
        "-tr", str(resolution), str(resolution),
        "-a_nodata", "0",
        "-ot", "Int32",
        "-of", "GTiff",
        shp_path,
        out_tif_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise RuntimeError(
            "gdal_rasterize executable not found on PATH. Install GDAL command line tools "
            "(e.g. 'conda install -c conda-forge gdal' or OSGeo4W on Windows) and ensure "
            "gdal_rasterize.exe is reachable from this process's PATH."
        )
    if result.returncode != 0:
        raise RuntimeError(f"gdal_rasterize failed: {result.stderr.strip()}")
    if not os.path.exists(out_tif_path):
        raise RuntimeError("gdal_rasterize reported success but no output file was produced.")
    with rasterio.open(out_tif_path) as check:
        if check.width <= 1 or check.height <= 1:
            raise ValueError(
                f"gdal_rasterize produced a degenerate {check.width}x{check.height} raster. "
                f"Check the input shapefile extent and -tr resolution."
            )
    return out_tif_path

def shapefile_to_raster(shp_path, out_tif_path, lulc_field="LU_Webcode", resolution=0.0005):
    tmp_dir = tempfile.mkdtemp()
    try:
        refactored_shp, bounds = _refactor_lu_webcode_to_int32(shp_path, tmp_dir, lulc_field=lulc_field)
        gdal_rasterize_layer(refactored_shp, out_tif_path, burn_field="LU_Webcode_i32", resolution=resolution)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_tif_path

def raster_to_cog(src_path, dst_path):
    with rasterio.open(src_path) as src:
        if src.crs is None or not src.crs.is_valid:
            if src.bounds.left > 180 or src.bounds.right > 180:
                src_crs = CRS.from_epsg(32643)
            else:
                src_crs = CRS.from_epsg(4326)
        else:
            src_crs = src.crs

        dst_crs = CRS.from_epsg(4326)

        _validate_geo_bounds(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top, label="Source raster")

        if src_crs == dst_crs:
            transform = src.transform
            width, height = src.width, src.height
            if width <= 1 or height <= 1:
                raise ValueError(
                    f"Source raster {src_path} already has degenerate dimensions "
                    f"{width}x{height} before any reprojection — the upstream rasterization step is broken."
                )
            profile = src.profile.copy()
            profile.update({
                "crs": dst_crs, "transform": transform,
                "width": width, "height": height,
                "driver": "GTiff", "dtype": "int32", "compress": "deflate",
                "tiled": True, "blockxsize": 512, "blockysize": 512,
                "BIGTIFF": "IF_SAFER", "nodata": 0
            })
            with MemoryFile() as mf:
                with mf.open(**profile) as mem:
                    for i in range(1, src.count + 1):
                        mem.write(src.read(i).astype(np.int32), i)
                    mem_bytes = mf.read()
        else:
            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, src.width, src.height, *src.bounds
            )
            if width <= 1 or height <= 1:
                raise ValueError(
                    f"Reprojecting {src_path} from {src_crs} to {dst_crs} produced a "
                    f"degenerate {width}x{height} raster — check the source CRS/bounds."
                )
            profile = src.profile.copy()
            profile.update({
                "crs": dst_crs, "transform": transform,
                "width": width, "height": height,
                "driver": "GTiff", "dtype": "int32", "compress": "deflate",
                "tiled": True, "blockxsize": 512, "blockysize": 512,
                "BIGTIFF": "IF_SAFER", "nodata": 0
            })
            with MemoryFile() as mf:
                with mf.open(**profile) as mem:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i), destination=rasterio.band(mem, i),
                            src_transform=src.transform, src_crs=src_crs,
                            dst_transform=transform, dst_crs=dst_crs,
                            resampling=Resampling.nearest,
                        )
                    mem_bytes = mf.read()
    with open(dst_path, "wb") as f:
        f.write(mem_bytes)

def convert_to_cog_inplace(src_path, dst_path):
    cmd = [
        "gdal_translate",
        "-of", "COG",
        "-co", "COMPRESS=DEFLATE",
        "-co", "BIGTIFF=IF_SAFER",
        src_path, dst_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and os.path.exists(dst_path):
            with rasterio.open(dst_path) as check:
                if check.width > 1 and check.height > 1:
                    return dst_path
    except FileNotFoundError:
        pass
    raster_to_cog(src_path, dst_path)
    return dst_path

def generate_cog_profile(transform, width, height):
    return {
        "driver": "GTiff",
        "dtype": "int32",
        "nodata": 0,
        "width": width,
        "height": height,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "interleave": "band"
    }

SESSION_TTL_SECONDS = 90


def _heartbeat_path(sid):
    return os.path.join(session_root(sid), "_last_seen.txt")


def _touch_heartbeat(sid):
    root = session_root(sid)
    os.makedirs(root, exist_ok=True)
    with open(_heartbeat_path(sid), "w") as f:
        f.write(str(__import__("time").time()))


def _session_is_expired(sid):
    hp = _heartbeat_path(sid)
    if not os.path.exists(hp):
        return False
    try:
        with open(hp) as f:
            last_seen = float(f.read().strip())
    except Exception:
        return False
    return (__import__("time").time() - last_seen) > SESSION_TTL_SECONDS


def _reap_expired_sessions():
    if not os.path.isdir(UPLOAD_ROOT):
        return
    for entry in os.listdir(UPLOAD_ROOT):
        entry_path = os.path.join(UPLOAD_ROOT, entry)
        if not os.path.isdir(entry_path):
            continue
        if _session_is_expired(entry):
            shutil.rmtree(entry_path, ignore_errors=True)


@upload_bp.route("/api/session/new", methods=["POST"])
def new_session():
    from flask import session as flask_session
    _reap_expired_sessions()

    sid = flask_session.get("sid")
    if sid and not _session_is_expired(sid):
        _touch_heartbeat(sid)
        flask_session.permanent = True
        return jsonify({"status": "ok", "session_id": sid, "reused": True})

    if sid:
        old_root = session_root(sid)
        if os.path.exists(old_root):
            shutil.rmtree(old_root, ignore_errors=True)

    import uuid
    new_sid = uuid.uuid4().hex
    flask_session["sid"] = new_sid
    flask_session.permanent = True
    _touch_heartbeat(new_sid)
    return jsonify({"status": "ok", "session_id": new_sid, "reused": False})


@upload_bp.route("/api/session/touch", methods=["POST"])
def touch_session():
    sid = _session_id()
    _touch_heartbeat(sid)
    _reap_expired_sessions()
    return jsonify({"status": "ok", "session_id": sid, "ttl_seconds": SESSION_TTL_SECONDS})

@upload_bp.route("/api/upload/files", methods=["POST"])
def upload_files():
    display_name = (request.form.get("display_name") or "").strip()
    if not display_name:
        return jsonify({"error": "display_name required"}), 400

    sid = _session_id()
    state_key = state_key_from_name(display_name)
    state_code = (request.form.get("state_code") or code_from_key(state_key)).strip().upper()
    lulc_field = (request.form.get("lulc_field") or "LU_Webcode").strip()
    resolution = float(request.form.get("resolution") or 0.0005)
    uploaded = request.files.getlist("files")

    out_folder = upload_folder_for_key(state_key, sid)
    tmp_dir = os.path.join(out_folder, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    meta = load_meta(state_key, sid)
    meta.update({
        "display_name": display_name,
        "code": state_code,
        "periods": meta.get("periods", []),
        "change_periods": meta.get("change_periods", [])
    })

    for f in uploaded:
        if f.filename:
            f.save(os.path.join(tmp_dir, f.filename))

    results = []
    for f in uploaded:
        name = f.filename
        if not name:
            continue
        ext = os.path.splitext(name)[1].lower()
        tmp_path = os.path.join(tmp_dir, name)

        if ext in (".dbf", ".shx", ".prj", ".cpg", ".fix"):
            continue

        period = infer_period(name)
        is_chg = is_change_filename(name)
        if period == "unknown":
            results.append({
                "file": name, "status": "error",
                "error": (f"Could not infer a survey period/year from filename '{name}'. "
                          f"Rename it to include the years, e.g. '..._2005-06.tif' or '..._2005_06.shp'.")
            })
            continue
        if is_chg and "_to_" in period:
            dst_name = f"changes_{state_code}_{period}.tif"
        else:
            dst_name = f"{state_code}_{period}_raster.tif"
        dst_tif = os.path.join(out_folder, dst_name)

        try:
            if ext in (".shp", ".fgb"):
                raw_tif = os.path.join(tmp_dir, f"_raw_{period}.tif")
                if ext == ".fgb":
                    import geopandas as gpd
                    gdf = gpd.read_file(tmp_path)
                    converted_shp = os.path.join(tmp_dir, f"_conv_{period}.shp")
                    gdf.to_file(converted_shp)
                    tmp_path = converted_shp
                shapefile_to_raster(tmp_path, raw_tif, lulc_field=lulc_field, resolution=resolution)
                convert_to_cog_inplace(raw_tif, dst_tif)
                type_lbl = "shp_to_cog"
            elif ext in (".tif", ".tiff"):
                convert_to_cog_inplace(tmp_path, dst_tif)
                type_lbl = "raster_to_cog"
            else:
                continue

            entry = {"file": name, "status": "ok", "type": type_lbl, "output": dst_name}
            if period and not is_chg and period not in meta["periods"]:
                meta["periods"].append(period)
            if is_chg and period and "_to_" in period:
                csv_path, n = generate_decoded_csv(dst_tif, period, state_key, out_folder, sid)
                entry.update({"csv": os.path.basename(csv_path), "csv_rows": n})
                if period not in meta["change_periods"]:
                    meta["change_periods"].append(period)
            results.append(entry)
        except Exception as e:
            results.append({"file": name, "status": "error", "error": str(e), "trace": traceback.format_exc()})

    meta["periods"] = sort_periods(meta["periods"])
    all_periods = meta["periods"]
    for i in range(len(all_periods)):
        for j in range(i + 1, len(all_periods)):
            p1, p2 = all_periods[i], all_periods[j]
            suffix = make_suffix(p1, p2)
            change_tif = os.path.join(out_folder, f"changes_{state_code}_{suffix}.tif")
            t1 = os.path.join(out_folder, f"{state_code}_{p1}_raster.tif")
            t2 = os.path.join(out_folder, f"{state_code}_{p2}_raster.tif")
            if os.path.exists(t1) and os.path.exists(t2) and not os.path.exists(change_tif):
                try:
                    raw_change = os.path.join(tmp_dir, f"_rawchg_{suffix}.tif")
                    build_change_tif(t1, t2, raw_change)
                    convert_to_cog_inplace(raw_change, change_tif)
                    csv_path, n = generate_decoded_csv(change_tif, suffix, state_key, out_folder, sid)
                    if suffix not in meta["change_periods"]:
                        meta["change_periods"].append(suffix)
                    results.append({
                        "file": f"(auto) changes_{state_code}_{suffix}.tif",
                        "status": "ok", "type": "change_built",
                        "output": f"changes_{state_code}_{suffix}.tif",
                        "csv": os.path.basename(csv_path), "csv_rows": n
                    })
                except Exception as e:
                    results.append({"file": f"(auto) changes_{state_code}_{suffix}", "status": "error", "error": str(e)})

    meta["change_periods"] = sort_periods_suffixes(meta["change_periods"]) if False else sorted(set(meta["change_periods"]), key=lambda s: tuple(_period_sort_key(p) for p in s.split("_to_")))
    save_meta(state_key, meta, sid)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return jsonify({
        "state_key": state_key, "display_name": display_name, "code": state_code,
        "periods": meta["periods"], "change_periods": meta["change_periods"],
        "results": results,
        "ok_count": sum(1 for r in results if r["status"] == "ok"),
        "err_count": sum(1 for r in results if r["status"] == "error")
    })

@upload_bp.route("/api/upload/status")
def upload_status():
    sid = _session_id()
    sroot = session_root(sid)
    if not os.path.exists(sroot):
        return jsonify({})
    result = {}
    for entry in os.listdir(sroot):
        entry_path = os.path.join(sroot, entry)
        if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "_meta.json")):
            meta = load_meta(entry, sid)
            result[entry] = {
                "display_name": meta.get("display_name", entry.replace("_", " ").title()),
                "code": meta.get("code", code_from_key(entry)),
                "periods": meta.get("periods", []),
                "change_periods": meta.get("change_periods", []),
            }
    return jsonify(result)

@upload_bp.route("/api/gsdp/lookup/<state_key>")
def lookup_gsdp(state_key):
    sid = _session_id()
    meta = load_meta(state_key, sid)
    display_name = meta.get("display_name", state_key.replace("_", " ").title())
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GSDP-current-all.csv")

    if not os.path.exists(csv_path):
        return jsonify({"error": "GSDP reference metrics sheet missing."}), 404
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        if "State" not in df.columns:
            return jsonify({"error": "GSDP CSV missing 'State' column."}), 500

        target = re.sub(r'\s+', '', display_name.strip().lower())

        df_clean = df.copy()
        df_clean['_state_norm'] = (df_clean['State'].astype(str)
                                    .str.replace(r'\s+', '', regex=True)
                                    .str.lower())

        df_state = df_clean[df_clean['_state_norm'] == target]
        if df_state.empty:
            words = [w for w in target.split() if len(w) >= 4] if ' ' in target else [target[:5]]
            for word in words:
                df_state = df_clean[df_clean['_state_norm'].str.contains(word[:5], na=False)]
                if not df_state.empty:
                    break

        if df_state.empty:
            return jsonify({"state": display_name, "data": [], "message": "No matching GSDP data found."})

        df_state = df_state.drop(columns=['_state_norm'])
        return jsonify({"state": display_name, "data": df_state.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)})

@upload_bp.route("/api/download/transition", methods=["GET"])
def download_transition_table():
    state_key = request.args.get("state_key")
    period = request.args.get("period")
    fmt = request.args.get("format", "csv").lower()

    if not state_key or not period:
        return jsonify({"error": "Parameters state_key and period are required."}), 400

    sid = _session_id()
    meta = load_meta(state_key, sid)
    state_code = meta.get("code", code_from_key(state_key))
    out_folder = upload_folder_for_key(state_key, sid)
    csv_name = f"changes_{state_code}_{period}_decoded.csv"
    csv_path = os.path.join(out_folder, csv_name)

    if not os.path.exists(csv_path):
        return jsonify({"error": "Transition matrix data not generated yet."}), 404

    if fmt == "xlsx":
        try:
            df = pd.read_csv(csv_path)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Transitions')
            output.seek(0)
            return send_file(output, download_name=f"Transitions_{state_code}_{period}.xlsx", as_attachment=True)
        except Exception as e:
            return jsonify({"error": f"Failed to generate Excel format: {str(e)}"}), 500

    if fmt == "pdf":
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, A3, landscape
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
            page_size = A3 if request.args.get("page", "a4").lower() == "a3" else A4
            df = pd.read_csv(csv_path)
            output = io.BytesIO()
            doc = SimpleDocTemplate(output, pagesize=landscape(page_size))
            data = [list(df.columns)] + df.astype(str).values.tolist()
            table = Table(data, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#161b22")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]))
            doc.build([table])
            output.seek(0)
            return send_file(output, download_name=f"Transitions_{state_code}_{period}.pdf", as_attachment=True)
        except Exception as e:
            return jsonify({"error": f"Failed to generate PDF format: {str(e)}"}), 500

    return send_file(csv_path, download_name=csv_name, as_attachment=True)

@upload_bp.route("/api/session/clear", methods=["POST"])
def clear_session():
    sid = _session_id()
    sroot = session_root(sid)
    if os.path.exists(sroot):
        shutil.rmtree(sroot, ignore_errors=True)
    return jsonify({"status": "session_cleared", "session_id": sid})

def generate_embedded_stats_matrix(change_arr, transform, target_dir, state_code, period_suffix):
    res_x, res_y = transform[0], -transform[4]
    km_per_deg = 111.32
    pixel_area_sqkm = (res_x * km_per_deg) * (res_y * km_per_deg)

    unique_vals, cell_counts = np.unique(change_arr, return_counts=True)
    matrix_records = []

    for val, count in zip(unique_vals, cell_counts):
        ival = int(val)
        if ival == 0:
            continue
        from_class = ival // 100
        to_class = ival % 100
        if from_class not in LU_WEBCODE or to_class not in LU_WEBCODE:
            continue

        f_lbl = f"{LU_WEBCODE[from_class][0]} > {LU_WEBCODE[from_class][1]}"
        t_lbl = f"{LU_WEBCODE[to_class][0]} > {LU_WEBCODE[to_class][1]}"
        matrix_records.append({
            "value": ival,
            "From category": f_lbl,
            "To category": t_lbl,
            "count": int(count),
            "sqkm_total": round(float(count) * pixel_area_sqkm, 4),
            "pixel_sqkm": round(pixel_area_sqkm, 8)
        })
    csv_out_path = os.path.join(target_dir, f"changes_{state_code}_{period_suffix}_decoded.csv")
    pd.DataFrame(matrix_records).to_csv(csv_out_path, index=False)