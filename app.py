import os, io, re, base64, uuid, shutil, math
import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_file, render_template, request, session
from flask_cors import CORS
import rasterio
from rasterio.warp import calculate_default_transform, reproject, transform_bounds, transform as warp_transform
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from PIL import Image
import tempfile
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = "super_secret_spatial_key_changein_production"

app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"]   = False
app.config["PERMANENT_SESSION_LIFETIME"] = 86400

from upload import (
    upload_bp, LU_WEBCODE, label_for_code, color_for_label,
    upload_folder_for_key, load_meta, code_from_key,
    session_root, _session_id, period_label, normalize_period_label,
    sort_periods, make_suffix, UPLOAD_ROOT, _touch_heartbeat
)
app.register_blueprint(upload_bp)

from shapefile_upload import shapefile_bp
app.register_blueprint(shapefile_bp)

@app.before_request
def ensure_session():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
        session.permanent = True
    try:
        _touch_heartbeat(session["sid"])
    except Exception:
        pass

BASE_DIR         = r"C:\Users\bhuvan.NRSCADMIN\Desktop\GeoSpatial Analysis _Analysis part"
SCATTER_DIR      = os.path.join(BASE_DIR, "scatterplot_analysis_output")
STATE_OUT_DIR    = os.path.join(BASE_DIR, "state_analysis_output")
DISTRICT_OUT_DIR = os.path.join(BASE_DIR, "district_analysis_output")
GSDP_CSV         = os.path.join(BASE_DIR, "GSDP-current-all.csv")
GEOJSON_PATH     = os.path.join(BASE_DIR, "india_states.geojson")
TIF_BASE         = r"C:\LULC"

DEG2_TO_SQKM  = 12321.0
RENDER_MAX_DIM = 2048

STATE_FOLDER = {
    "haryana":        "Haryana",
    "madhya_pradesh": "Madhya Pradesh",
    "uttar_pradesh":  "Uttar Pradesh",
}
STATE_CODE = {
    "haryana":        "HR",
    "madhya_pradesh": "MP",
    "uttar_pradesh":  "UP",
}

PERIOD_LABEL_TO_SUFFIX = {
    "2005-06 to 2011-12": "0506_to_1112",
    "2011-12 to 2015-16": "1112_to_1516",
    "2005-06 to 2015-16": "0506_to_1516",
}
PERIOD_SUFFIX_TO_LABEL = {v: k for k, v in PERIOD_LABEL_TO_SUFFIX.items()}

_SESSION_STATES = {}
_pixel_cache    = {}

STATIC_CLASSES = {code: (rgb, l1 + " > " + l2) for code, (l1, l2, rgb) in LU_WEBCODE.items()}

RETENTION_COLORS = {
    "agriculture":              "#e6b800",
    "forest":                   "#005c00",
    "builtup":                  "#cc0000",
    "wetlands/waterbodies":     "#0077b6",
    "barren/unculturable":      "#c2a87a",
    "grass-grazing":            "#556b2f",
    "snow":                     "#cce5ff",
}


def retention_color(l1_str):
    low = l1_str.lower().strip()
    for k, c in RETENTION_COLORS.items():
        if k in low:
            return c
    return "#888888"


def nodata_zero_mask(data, nodata):
    if nodata is not None and abs(float(nodata)) > 1e10:
        nd = np.abs(data.astype(float) - float(nodata)) < 1e30
    elif nodata is not None:
        nd = np.abs(data.astype(float) - float(nodata)) < 0.1
    else:
        nd = np.zeros(data.shape, dtype=bool)
    return (data == 0) | nd


def png_from_rgba(rgba):
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _scale_arr(data, max_dim=RENDER_MAX_DIM):
    h, w = data.shape
    if max(h, w) <= max_dim:
        return data
    scale = max_dim / max(h, w)
    ri = np.round(np.linspace(0, h-1, max(1, int(h*scale)))).astype(int)
    ci = np.round(np.linspace(0, w-1, max(1, int(w*scale)))).astype(int)
    return data[np.ix_(ri, ci)]


def read_raster_scaled(tif_path, max_dim=RENDER_MAX_DIM):
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            oh = max(1, int(h*scale))
            ow = max(1, int(w*scale))
            data = src.read(1, out_shape=(1, oh, ow), resampling=Resampling.nearest)
        else:
            data = src.read(1)
        nodata = src.nodata
        bnds   = transform_bounds(src.crs, "EPSG:4326",
                                  src.bounds.left, src.bounds.bottom,
                                  src.bounds.right, src.bounds.top)
        bounds = {"west": bnds[0], "south": bnds[1], "east": bnds[2], "north": bnds[3]}
    return data, nodata, bounds


def _colorise(data_s, zm_s, is_change, pixel_lookup):
    rgba        = np.zeros((*data_s.shape, 4), dtype=np.uint8)
    legend_cats = {}
    if is_change and pixel_lookup:
        for v in np.unique(data_s):
            fv = float(v)
            if abs(fv) > 1e10:
                continue
            iv   = int(round(fv))
            mask = (~zm_s) & (np.abs(data_s - fv) < 0.5)
            if not mask.any():
                continue
            info = pixel_lookup.get(iv)
            if info is None or info["is_same"]:
                continue
            hc = color_for_label(info["to_label"])
            rgba[mask] = [int(hc[1:3], 16), int(hc[3:5], 16), int(hc[5:7], 16), 220]
            legend_cats[info["to_label"]] = hc
    else:
        for code, (rgb, lbl) in STATIC_CLASSES.items():
            mask = (~zm_s) & (np.abs(data_s - float(code)) < 0.5)
            if mask.any():
                rgba[mask] = [rgb[0], rgb[1], rgb[2], 230]
                legend_cats[lbl] = "#{:02x}{:02x}{:02x}".format(*rgb)
    rgba[zm_s] = [0, 0, 0, 0]
    return rgba, legend_cats


def render_raster(tif_path, is_change=False, pixel_lookup=None):
    data, nodata, bounds = read_raster_scaled(tif_path)
    zm   = nodata_zero_mask(data, nodata)
    rgba, lc = _colorise(data, zm, is_change, pixel_lookup or {})
    return png_from_rgba(rgba), bounds, [{"label": k, "color": v} for k, v in sorted(lc.items())]


def render_raster_from_array(arr, bounds_4326, is_change=False, pixel_lookup=None):
    data_s = _scale_arr(arr)
    zm_s   = _scale_arr((arr == 0).astype(np.uint8)).astype(bool)
    w, s, e, n = bounds_4326
    bounds = {"west": w, "south": s, "east": e, "north": n}
    rgba, lc = _colorise(data_s, zm_s, is_change, pixel_lookup or {})
    return png_from_rgba(rgba), bounds, [{"label": k, "color": v} for k, v in sorted(lc.items())]


def render_transition_subset(tif_path, pixel_lookup, matcher, mode="change"):
    data, nodata, bounds = read_raster_scaled(tif_path)
    zm   = nodata_zero_mask(data, nodata)
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    legend_cats = {}
    stats       = {}
    for val, info in pixel_lookup.items():
        if not matcher(info, mode):
            continue
        fv   = float(val)
        mask = (~zm) & (np.abs(data - fv) < 0.5)
        if not mask.any():
            continue
        if mode == "retention":
            color = retention_color(info["from_l1"])
            lbl   = info["from_l1"] + " (retained)"
            stats[info["from_l1"]] = stats.get(info["from_l1"], 0.0) + info["sqkm_total"]
        else:
            color = color_for_label(info["to_label"])
            lbl   = info["to_label"]
        rgba[mask] = [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16), 220]
        legend_cats[lbl] = color
    legend = [{"label": k, "color": v} for k, v in sorted(legend_cats.items())]
    return png_from_rgba(rgba), bounds, legend, stats


def render_transition_subset_from_array(change_arr, bounds_4326, pixel_lookup, matcher, mode="change"):
    data_s = _scale_arr(change_arr.astype(np.float32))
    zm_s   = _scale_arr((change_arr == 0).astype(np.uint8)).astype(bool)
    w, s, e, n = bounds_4326
    bounds = {"west": w, "south": s, "east": e, "north": n}
    rgba        = np.zeros((*data_s.shape, 4), dtype=np.uint8)
    legend_cats = {}
    stats       = {}
    for val, info in pixel_lookup.items():
        if not matcher(info, mode):
            continue
        fv   = float(val)
        mask = (~zm_s) & (np.abs(data_s - fv) < 0.5)
        if not mask.any():
            continue
        if mode == "retention":
            color = retention_color(info["from_l1"])
            lbl   = info["from_l1"] + " (retained)"
            stats[info["from_l1"]] = stats.get(info["from_l1"], 0.0) + info["sqkm_total"]
        else:
            color = color_for_label(info["to_label"])
            lbl   = info["to_label"]
        rgba[mask] = [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16), 220]
        legend_cats[lbl] = color
    legend = [{"label": k, "color": v} for k, v in sorted(legend_cats.items())]
    return png_from_rgba(rgba), bounds, legend, stats

def _upload_folder(key):
    sid = _session_id()
    p = os.path.join(UPLOAD_ROOT, sid, key.lower())
    return p if os.path.isdir(p) else None


def _load_upload_meta(key):
    sid = _session_id()
    mp = os.path.join(UPLOAD_ROOT, sid, key.lower(), "_meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            return __import__("json").load(f)
    return {}


def _tif_folder(key):
    k = key.lower()
    if k in STATE_FOLDER:
        return os.path.join(TIF_BASE, STATE_FOLDER[k])
    up = _upload_folder(k)
    if up:
        return up
    return os.path.join(TIF_BASE, k)


def _state_code(key):
    k = key.lower()
    if k in STATE_CODE:
        return STATE_CODE[k]
    meta = _load_upload_meta(k)
    return meta.get("code", "".join(w[0].upper() for w in k.split("_") if w)[:4])

def build_pixel_lookup(state_key, period_suffix):
    k  = state_key.lower()
    ck = (k, period_suffix)
    if ck in _pixel_cache:
        return _pixel_cache[ck]

    sess = _SESSION_STATES.get(k)
    if sess:
        lu = sess.get("lookup", {}).get(period_suffix, {})
        if lu:
            return lu

    code   = _state_code(k)
    folder = _tif_folder(k)

    candidates = [
        os.path.join(folder, f"changes_{code}_{period_suffix}_decoded.csv"),
        os.path.join(folder, f"RasterChanges_{code}_{period_suffix}_decoded.csv"),
    ]
    fpath = next((c for c in candidates if os.path.exists(c)), None)
    if not fpath:
        return {}

    df = pd.read_csv(fpath)
    if "deg2" in df.columns:
        df["sqkm_total"] = (df["deg2"] * DEG2_TO_SQKM).round(4)
    elif "sqkm_total" not in df.columns and "sqkm" in df.columns:
        df["sqkm_total"] = df["sqkm"]

    from_col = next((c for c in df.columns if "from" in c.lower() and "cat" in c.lower()), None)
    to_col   = next((c for c in df.columns if "to"   in c.lower() and "cat" in c.lower()), None)
    if not from_col or not to_col or "value" not in df.columns:
        return {}

    lookup = {}
    for _, row in df.iterrows():
        try:
            pv = int(round(float(row["value"])))
        except Exception:
            continue
        from_lbl = str(row[from_col]).strip()
        to_lbl   = str(row[to_col]).strip()
        sqkm     = float(row.get("sqkm_total", 0))
        count    = int(row.get("count", 0)) if "count" in df.columns else 0
        from_l1  = from_lbl.split(">")[0].strip()
        to_l1    = to_lbl.split(">")[0].strip()
        lookup[pv] = {
            "from_label":  from_lbl,
            "to_label":    to_lbl,
            "from_l1":     from_l1,
            "to_l1":       to_l1,
            "sqkm_total":  sqkm,
            "pixel_sqkm":  round(sqkm / count, 6) if count else 0.0,
            "count":       count,
            "is_same":     (from_lbl.lower() == to_lbl.lower()),
            "is_retained": (from_l1.lower() == to_l1.lower()),
        }
    _pixel_cache[ck] = lookup
    return lookup


def _build_lookup_from_arrays(arr1, arr2, pixel_size_deg=0.0005):
    px_sqkm = (pixel_size_deg * 111.32) ** 2
    change  = arr1.astype(np.int32) * 100 + arr2.astype(np.int32)
    uvals, counts = np.unique(change, return_counts=True)
    lookup = {}
    for uv, cnt in zip(uvals, counts):
        iuv = int(uv)
        if iuv == 0:
            continue
        fc = iuv // 100
        tc = iuv % 100
        if fc == 0 or tc == 0:
            continue
        fl  = label_for_code(fc)
        tl  = label_for_code(tc)
        fl1 = fl.split(">")[0].strip()
        tl1 = tl.split(">")[0].strip()
        lookup[iuv] = {
            "from_label":  fl,
            "to_label":    tl,
            "from_l1":     fl1,
            "to_l1":       tl1,
            "sqkm_total":  round(float(cnt) * px_sqkm, 4),
            "pixel_sqkm":  round(px_sqkm, 6),
            "count":       int(cnt),
            "is_same":     (fc == tc),
            "is_retained": (fl1.lower() == tl1.lower()),
        }
    return lookup, change


def _recompute_session_lookups(k):
    sess = _SESSION_STATES.get(k)
    if not sess:
        return
    rasters = sess["rasters"]
    periods = sort_periods(sess["periods"])
    sess["periods"] = periods
    for i in range(len(periods)):
        for j in range(i + 1, len(periods)):
            p1     = periods[i]
            p2     = periods[j]
            suffix = make_suffix(p1, p2)
            if p1 not in rasters or p2 not in rasters:
                continue
            a1 = rasters[p1].astype(np.int32)
            a2 = rasters[p2].astype(np.int32)
            h  = min(a1.shape[0], a2.shape[0])
            w  = min(a1.shape[1], a2.shape[1])
            a1, a2 = a1[:h, :w], a2[:h, :w]

            bnds  = sess["bounds_4326"].get(p1, (0, 0, 1, 1))
            pxsz  = (bnds[2] - bnds[0]) / max(a1.shape[1], 1)
            if pxsz <= 0 or pxsz > 5:
                pxsz = 0.0005

            lkup, change_arr = _build_lookup_from_arrays(a1, a2, pxsz)
            sess["lookup"][suffix] = lkup
            sess.setdefault("change_arrays", {})[suffix] = {"arr": change_arr, "bounds_4326": bnds}
            _pixel_cache.pop((k, suffix), None)


def _tif_bytes_to_array(raw_bytes):
    with rasterio.MemoryFile(raw_bytes) as mf:
        with mf.open() as src:
            try:
                if src.crs is None or not src.crs.is_valid:
                    if src.bounds.left > 180 or src.bounds.right > 180:
                        from rasterio.crs import CRS as RCrs
                        src_crs = RCrs.from_epsg(32643)
                    else:
                        from rasterio.crs import CRS as RCrs
                        src_crs = RCrs.from_epsg(4326)
                else:
                    src_crs = src.crs
                epsg = src_crs.to_epsg()
            except Exception:
                from rasterio.crs import CRS as RCrs
                src_crs = RCrs.from_epsg(4326)
                epsg = 4326

            if epsg == 4326:
                data = src.read(1).astype(np.int32)
                b    = src.bounds
                return data, src.transform, (b.left, b.bottom, b.right, b.top)

            return _reproject_to_4326_with_crs(src, src_crs)


def _reproject_to_4326_with_crs(src_ds, source_crs):
    dst_crs = "EPSG:4326"
    dst_tf, dst_w, dst_h = calculate_default_transform(
        source_crs, dst_crs, src_ds.width, src_ds.height, *src_ds.bounds)
    meta = src_ds.meta.copy()
    meta.update({"crs": dst_crs, "transform": dst_tf, "width": dst_w, "height": dst_h,
                 "count": 1, "dtype": "int32", "nodata": 0})
    buf = io.BytesIO()
    with rasterio.MemoryFile(buf) as mf:
        with mf.open(**meta) as dst:
            reproject(source=rasterio.band(src_ds, 1), destination=rasterio.band(dst, 1),
                      src_transform=src_ds.transform, src_crs=source_crs,
                      dst_transform=dst_tf, dst_crs=dst_crs, resampling=Resampling.nearest)
            data = dst.read(1)
            tfm  = dst.transform
            b    = dst.bounds
    return data, tfm, (b.left, b.bottom, b.right, b.top)

#updated on 18-06
def _shapefile_bytes_to_raster(shp_b, dbf_b, shx_b, prj_b=None,
                                field="LU_Webcode", resolution=0.0005):
    import geopandas as gpd
    from rasterio.features import rasterize as rio_rasterize

    tmpdir = tempfile.mkdtemp()
    try:
        for name, data in [("up.shp", shp_b), ("up.dbf", dbf_b), ("up.shx", shx_b)]:
            with open(os.path.join(tmpdir, name), "wb") as fh:
                fh.write(data)
        if prj_b:
            with open(os.path.join(tmpdir, "up.prj"), "wb") as fh:
                fh.write(prj_b)

        gdf = gpd.read_file(os.path.join(tmpdir, "up.shp"))
        if gdf.empty:
            raise ValueError("Shapefile contains no readable features.")

        if gdf.crs is None:
            b = gdf.total_bounds
            gdf = gdf.set_crs(epsg=32643 if (abs(b[0]) > 180 or abs(b[2]) > 180) else 4326)
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        col_map    = {c.lower(): c for c in gdf.columns}
        burn_field = col_map.get(field.lower())
        if not burn_field:
            for fb in ("lu_webcode", "lulc_code", "lulc", "lucode", "value", "gridcode"):
                if fb in col_map:
                    burn_field = col_map[fb]
                    break
        if not burn_field:
            raise ValueError(f"LU_Webcode field not found. Available: {list(gdf.columns)}")

        def _toint(v):
            try:
                return int(float(str(v).strip()))
            except Exception:
                return 0

        gdf["_burn"] = gdf[burn_field].apply(_toint).astype(np.int32)
        gdf = gdf[gdf["_burn"] > 0]
        if gdf.empty:
            raise ValueError("No valid LU_Webcode (>0) features after int32 conversion.")

        west, south, east, north = gdf.total_bounds
        if abs(east - west) < 1e-9 or abs(north - south) < 1e-9:
            raise ValueError(
                f"Shapefile extent is degenerate: "
                f"({west:.6f},{south:.6f}) -> ({east:.6f},{north:.6f}). Check CRS."
            )

        width  = max(1, int(round((east  - west)  / resolution)))
        height = max(1, int(round((north - south) / resolution)))
        if max(width, height) > 8000:
            scale = 4000.0 / max(width, height)
            resolution = resolution / scale
            width  = max(1, int(round((east  - west)  / resolution)))
            height = max(1, int(round((north - south) / resolution)))

        tfm = from_bounds(west, south, east, north, width, height)

        shapes = (
            (geom, int(val))
            for geom, val in zip(gdf.geometry, gdf["_burn"])
            if geom is not None and not geom.is_empty
        )
        data = rio_rasterize(
            shapes, out_shape=(height, width),
            transform=tfm, fill=0, dtype=np.int32,
        )

        if data.max() == 0:
            raise ValueError(
                "Rasterization produced an all-zero array. "
                "Verify the shapefile geometry and LU_Webcode values."
            )

        return data, tfm, (west, south, east, north)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _suffix_to_label(suffix):
    if suffix in PERIOD_SUFFIX_TO_LABEL:
        return PERIOD_SUFFIX_TO_LABEL[suffix]
    return normalize_period_label(period_label(suffix))


def load_all_csvs(state_key, period_filter=None):
    k    = state_key.lower()
    sess = _SESSION_STATES.get(k)
    if sess:
        frames = []
        for psfx, lkup in sess.get("lookup", {}).items():
            plabel = _suffix_to_label(psfx)
            if period_filter and plabel not in period_filter:
                continue
            rows = [
                {
                    "From category": info["from_label"],
                    "To category":   info["to_label"],
                    "sqkm":          info["sqkm_total"],
                    "Period":        plabel,
                }
                for info in lkup.values()
            ]
            if rows:
                frames.append(pd.DataFrame(rows))
        if frames:
            return frames

    code   = _state_code(k)
    folder = _tif_folder(k)
    meta   = _load_upload_meta(k)

    change_periods = meta.get("change_periods", [])
    if not change_periods:
        change_periods = list(PERIOD_LABEL_TO_SUFFIX.values())

    frames = []
    for psfx in change_periods:
        plabel = _suffix_to_label(psfx)
        if period_filter and plabel not in period_filter:
            continue
        for fname in [
            f"changes_{code}_{psfx}_decoded.csv",
            f"RasterChanges_{code}_{psfx}_decoded.csv",
        ]:
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                df["Period"] = plabel
                if "deg2" in df.columns:
                    df["sqkm"] = (df["deg2"] * DEG2_TO_SQKM).round(4)
                elif "sqkm_total" in df.columns and "sqkm" not in df.columns:
                    df["sqkm"] = df["sqkm_total"]
                frames.append(df)
                break
    return frames


def _norm_frames(frames):
    ALL      = pd.concat(frames, ignore_index=True)
    ALL["Period"] = ALL["Period"].apply(normalize_period_label)
    from_col = next((c for c in ALL.columns if "from" in c.lower() and "cat" in c.lower()), None)
    to_col   = next((c for c in ALL.columns if "to"   in c.lower() and "cat" in c.lower()), None)
    return ALL, from_col, to_col


def _available_period_labels(state_key):
    k    = state_key.lower()
    sess = _SESSION_STATES.get(k)
    if sess:
        suffixes = list(sess.get("lookup", {}).keys())
    else:
        meta = _load_upload_meta(k)
        suffixes = meta.get("change_periods") or list(PERIOD_LABEL_TO_SUFFIX.values())

    def _start_year(sfx):
        m = re.search(r"(20\d{2}|19\d{2})", sfx)
        return int(m.group(1)) if m else 9999

    suffixes = sorted(set(suffixes), key=_start_year)
    return [(sfx, _suffix_to_label(sfx)) for sfx in suffixes]


def _class_name_mask(series_l1, series_full, names):
    mask = pd.Series(False, index=series_l1.index)
    for n in names:
        if not n:
            continue
        mask |= (series_l1 == n) | series_full.str.contains(re.escape(n), na=False)
    return mask


def _net_affected_area(grouped_by_period, all_labels):
    labels_present = [lbl for _, lbl in all_labels if lbl in grouped_by_period]
    if len(labels_present) < 2:
        return sum(grouped_by_period.values())

    def _years(lbl):
        ys = re.findall(r"(\d{4})", lbl)
        return (int(ys[0]), int(ys[1])) if len(ys) == 2 else (None, None)

    span = {lbl: _years(lbl) for lbl in labels_present}
    year_set = sorted(set(y for pair in span.values() for y in pair if y is not None))
    consecutive_pairs = set(zip(year_set[:-1], year_set[1:]))

    consecutive, spanning = [], []
    for lbl, (y1, y2) in span.items():
        if (y1, y2) in consecutive_pairs:
            consecutive.append(lbl)
        else:
            spanning.append(lbl)

    if not consecutive or not spanning:
        return sum(grouped_by_period.values())

    total  = sum(grouped_by_period[lbl] for lbl in consecutive)
    total -= sum(grouped_by_period[lbl] for lbl in spanning)
    return total

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/geojson")
def get_geojson():
    if not os.path.exists(GEOJSON_PATH):
        return jsonify({"error": "india_states.geojson not found"}), 404
    return send_file(GEOJSON_PATH, mimetype="application/json")


@app.route("/api/states")
def get_states():
    if not os.path.exists(SCATTER_DIR):
        return jsonify([])
    return jsonify(sorted(d for d in os.listdir(SCATTER_DIR)
                          if os.path.isdir(os.path.join(SCATTER_DIR, d))))


@app.route("/api/districts/<state>")
def get_districts(state):
    d = os.path.join(DISTRICT_OUT_DIR, state)
    if not os.path.exists(d):
        return jsonify([])
    return jsonify(sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))))

@app.route("/api/raster_states")
def get_raster_states():
    result = {}

    for key, folder_name in STATE_FOLDER.items():
        if os.path.isdir(os.path.join(TIF_BASE, folder_name)):
            result[key] = {
                "display_name": folder_name,
                "code":          STATE_CODE[key],
                "source":        "local",
                "periods":       ["0506", "1112", "1516"],
                "change_periods": ["0506_to_1112", "1112_to_1516", "0506_to_1516"],
            }

    for key, info in _SESSION_STATES.items():
        ps = sort_periods(info.get("periods", []))
        cp = [make_suffix(p1, p2) for i, p1 in enumerate(ps) for p2 in ps[i+1:]]

        result[key] = {
            "display_name": info["display_name"],
            "code":          info["code"],
            "source":        "session",
            "periods":       ps,
            "change_periods": cp,
        }

    sid = _session_id()
    sroot = os.path.join(UPLOAD_ROOT, sid)
    if os.path.isdir(sroot):
        for entry in os.listdir(sroot):
            ep = os.path.join(sroot, entry)
            if not os.path.isdir(ep) or entry in result:
                continue

            meta = _load_upload_meta(entry)
            tifs = [f for f in os.listdir(ep) if f.endswith(".tif") and not f.startswith("_")]
            if not tifs:
                continue

            uploaded_ps = sort_periods(meta.get("periods", []))
            uploaded_cp = [make_suffix(p1, p2) for i, p1 in enumerate(uploaded_ps) for p2 in uploaded_ps[i+1:]]

            result[entry] = {
                "display_name":   meta.get("display_name", entry.replace("_", " ").title()),
                "code":           meta.get("code", "".join(w[0].upper() for w in entry.split("_") if w)[:4]),
                "source":         "uploaded",
                "periods":        uploaded_ps,
                "change_periods": uploaded_cp,
            }

    return jsonify(result)


@app.route("/api/upload/raster", methods=["POST"])
def upload_raster():
    state_name = request.form.get("state_name", "").strip()
    state_code = request.form.get("state_code", "").strip().upper()
    period_str = request.form.get("period", "").strip()
    files      = request.files.getlist("files[]")
    if not state_name or not files:
        return jsonify({"error": "state_name and files[] required"}), 400
    k            = state_name.lower().replace(" ", "_")
    periods_list = [p.strip() for p in period_str.split(",") if p.strip()]
    if k not in _SESSION_STATES:
        _SESSION_STATES[k] = {
            "display_name": state_name,
            "code":         state_code or state_name[:2].upper(),
            "periods":      [], "rasters":   {},
            "transforms":   {}, "bounds_4326": {},
            "lookup":       {}, "change_arrays": {},
        }
    sess   = _SESSION_STATES[k]
    loaded = []
    for i, f in enumerate(files):
        pcode = periods_list[i] if i < len(periods_list) else f"p{i}"
        try:
            data, tfm, bnds = _tif_bytes_to_array(f.read())
        except Exception as e:
            return jsonify({"error": f"Period {pcode}: {e}"}), 400
        sess["rasters"][pcode]     = data
        sess["transforms"][pcode]  = tfm
        sess["bounds_4326"][pcode] = bnds
        if pcode not in sess["periods"]:
            sess["periods"].append(pcode)
        loaded.append(pcode)
    _recompute_session_lookups(k)
    return jsonify({"status": "ok", "state_key": k, "loaded_periods": loaded, "all_periods": sess["periods"]})

#updated on 18-06
@app.route("/api/upload/shapefile", methods=["POST"])
def upload_shapefile():
    state_name  = request.form.get("state_name",  "").strip()
    state_code  = request.form.get("state_code",  "").strip().upper()
    period_code = request.form.get("period",       "").strip()
    lulc_field  = request.form.get("lulc_field",  "LU_Webcode").strip()
    shp = request.files.get("shp")
    dbf = request.files.get("dbf")
    shx = request.files.get("shx")
    prj = request.files.get("prj")

    if not state_name:
        return jsonify({"error": "state_name required"}), 400
    if not period_code:
        return jsonify({"error": "period required"}), 400
    if not (shp and dbf and shx):
        return jsonify({"error": ".shp, .dbf, and .shx all required"}), 400

    k   = state_name.lower().replace(" ", "_")
    sid = _session_id()
    if not state_code:
        state_code = "".join(w[0].upper() for w in k.split("_") if w)[:4]

    try:
        data, tfm, bnds = _shapefile_bytes_to_raster(
            shp.read(), dbf.read(), shx.read(),
            prj_b=(prj.read() if prj else None),
            field=lulc_field,
        )
    except Exception as e:
        return jsonify({"error": str(e), "trace": __import__("traceback").format_exc()}), 400

    if k not in _SESSION_STATES:
        _SESSION_STATES[k] = {
            "display_name": state_name,
            "code":         state_code,
            "periods":      [], "rasters":   {},
            "transforms":   {}, "bounds_4326": {},
            "lookup":       {}, "change_arrays": {},
        }
    sess = _SESSION_STATES[k]
    sess["rasters"][period_code]     = data
    sess["transforms"][period_code]  = tfm
    sess["bounds_4326"][period_code] = bnds
    if period_code not in sess["periods"]:
        sess["periods"].append(period_code)
    _recompute_session_lookups(k)

    try:
        out_folder = upload_folder_for_key(k, sid)
        meta = load_meta(k, sid)
        meta.update({
            "display_name": state_name,
            "code": state_code,
            "periods": sort_periods(meta.get("periods", []) + [period_code]),
            "change_periods": meta.get("change_periods", []),
        })
        dst_tif = os.path.join(out_folder, f"{state_code}_{period_code}_raster.tif")
        profile = {
            "driver": "GTiff", "dtype": "int32", "nodata": 0,
            "width": data.shape[1], "height": data.shape[0], "count": 1,
            "crs": "EPSG:4326", "transform": tfm,
            "tiled": True, "blockxsize": 512, "blockysize": 512, "compress": "deflate",
        }
        with rasterio.open(dst_tif, "w", **profile) as dst:
            dst.write(data, 1)

        for i in range(len(meta["periods"])):
            for j in range(i + 1, len(meta["periods"])):
                p1, p2 = meta["periods"][i], meta["periods"][j]
                sfx    = make_suffix(p1, p2)
                t1     = os.path.join(out_folder, f"{state_code}_{p1}_raster.tif")
                t2     = os.path.join(out_folder, f"{state_code}_{p2}_raster.tif")
                chg    = os.path.join(out_folder, f"changes_{state_code}_{sfx}.tif")
                if os.path.exists(t1) and os.path.exists(t2) and not os.path.exists(chg):
                    try:
                        from upload import build_change_tif, generate_decoded_csv
                        build_change_tif(t1, t2, chg)
                        generate_decoded_csv(chg, sfx, k, out_folder, sid)
                        if sfx not in meta["change_periods"]:
                            meta["change_periods"].append(sfx)
                    except Exception:
                        pass

        meta["change_periods"] = sorted(set(meta["change_periods"]))
        from upload import save_meta as _save_meta
        _save_meta(k, meta, sid)
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "state_key": k,
        "loaded_period": period_code,
        "all_periods": sess["periods"],
        "shape": list(data.shape),
        "bounds": list(bnds),
    })


@app.route("/api/upload/clear/<state_key>", methods=["POST"])
def clear_session_state(state_key):
    k = state_key.lower()
    _SESSION_STATES.pop(k, None)
    for ck in [c for c in list(_pixel_cache) if c[0] == k]:
        _pixel_cache.pop(ck, None)
    return jsonify({"status": "ok", "removed": k})


@app.route("/api/tif/<state_key>/<tif_name>")
def get_tif(state_key, tif_name):
    k = state_key.lower().strip()
    sess = _SESSION_STATES.get(k)
    tif_name_clean = tif_name.replace(".tif", "")

    if sess:
        is_change = (tif_name_clean.startswith("Change_") or
                     tif_name_clean.startswith("changes_") or
                     "to" in tif_name_clean.lower())
        if is_change:
            period_suffix = re.sub(r'^(?:changes?_[A-Z]{1,4}_|Change_)', '', tif_name_clean)
            period_suffix = period_suffix.replace("_raster", "")

            ca = sess.get("change_arrays", {}).get(period_suffix)
            if not ca:
                parts = re.split(r'_to_', period_suffix)
                if len(parts) == 2:
                    canon_suffix = make_suffix(parts[0], parts[1])
                    ca = sess.get("change_arrays", {}).get(canon_suffix)
                    if ca:
                        period_suffix = canon_suffix
            if not ca:
                alt_suffix = period_suffix.replace("-", "").replace("20", "")
                ca = sess.get("change_arrays", {}).get(alt_suffix)
                if ca:
                    period_suffix = alt_suffix

            if ca:
                lookup = sess.get("lookup", {}).get(period_suffix, {})
                b64, bounds, legend = render_raster_from_array(
                    ca["arr"], ca["bounds_4326"], is_change=True, pixel_lookup=lookup)
                return jsonify({"image": b64, "bounds": bounds, "legend": legend})

            return jsonify({"error": f"Transition array for '{period_suffix}' not found. "
                                     f"Upload rasters for both periods first. "
                                     f"Available: {list(sess.get('change_arrays', {}).keys())}"}), 404
        else:
            pcode = None
            match = re.search(r'(\d{2,4})', tif_name_clean)
            if match:
                pcode = match.group(1)
            long_pcode_map = {
                "0506": "2005-06", "1112": "2011-12", "1516": "2015-16",
                "2005": "2005-06", "2011": "2011-12", "2015": "2015-16"
            }
            target_key = pcode
            if pcode not in sess["rasters"]:
                target_key = long_pcode_map.get(pcode)
                if not target_key or target_key not in sess["rasters"]:
                    for existing_key in sess["rasters"]:
                        if existing_key in tif_name_clean or existing_key.replace("-", "") in tif_name_clean:
                            target_key = existing_key
                            break
            if not target_key or target_key not in sess["rasters"]:
                return jsonify({"error": f"Period '{pcode}' not found. Available: {list(sess['rasters'].keys())}"}), 404
            raster_matrix = sess["rasters"][target_key]
            geo_bounds = sess["bounds_4326"].get(target_key, next(iter(sess["bounds_4326"].values())))
            b64, bounds, legend = render_raster_from_array(raster_matrix, geo_bounds, is_change=False)
            return jsonify({"image": b64, "bounds": bounds, "legend": legend})
    folder = _tif_folder(state_key)
    code   = _state_code(state_key)
    candidates = [
        os.path.join(folder, tif_name_clean + ".tif"),
        os.path.join(folder, tif_name_clean.replace("Change_", f"changes_{code}_") + ".tif"),
    ]
    fpath = next((c for c in candidates if os.path.exists(c)), None)
    if not fpath:
        return jsonify({"error": f"TIF not found (tried {candidates})"}), 404

    is_change = tif_name_clean.startswith("Change_") or tif_name_clean.startswith("changes_")
    period_suffix = None
    if is_change:
        period_suffix = re.sub(r'^(?:changes?_[A-Z]{1,4}_|Change_)', '', tif_name_clean)
        _pixel_cache.pop((state_key, period_suffix), None)

    pixel_lookup = build_pixel_lookup(state_key, period_suffix) if is_change and period_suffix else {}
    try:
        b64, bounds, legend = render_raster(fpath, is_change=is_change, pixel_lookup=pixel_lookup)
        return jsonify({"image": b64, "bounds": bounds, "legend": legend})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/sample_pixel/<state_key>/<tif_name>")
def sample_pixel(state_key, tif_name):
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon required"}), 400

    k    = state_key.lower()
    sess = _SESSION_STATES.get(k)

    if sess:
        is_change     = tif_name.startswith("Change_") or tif_name.startswith("changes_")
        period_suffix = re.sub(r'^(?:changes?_[A-Z]{1,4}_|Change_)', '', tif_name) if is_change else None
        if is_change and period_suffix:
            ca = sess.get("change_arrays", {}).get(period_suffix)
            if not ca:
                return jsonify({"value": None, "message": "No change array"})
            arr  = ca["arr"]
            west, south, east, north = ca["bounds_4326"]
        else:
            parts = tif_name.split("_")
            pcode = next((p for p in parts if re.match(r'^\d{4}$', p)), None)
            if not pcode or pcode not in sess["rasters"]:
                return jsonify({"value": None, "message": "Period not loaded"})
            arr  = sess["rasters"][pcode]
            west, south, east, north = sess["bounds_4326"][pcode]
        h, w = arr.shape
        if not (west <= lon <= east and south <= lat <= north):
            return jsonify({"value": None, "message": "Outside extent"})
        col = max(0, min(w-1, int((lon-west) / (east-west) * w)))
        row = max(0, min(h-1, int((north-lat) / (north-south) * h)))
        val = int(arr[row, col])
        if val == 0:
            return jsonify({"value": None, "message": "Zero"})
        return jsonify({"value": val})

    folder = _tif_folder(state_key)
    code   = _state_code(state_key)
    candidates = [
        os.path.join(folder, tif_name + ".tif"),
        os.path.join(folder, re.sub(r'^Change_', f"changes_{code}_", tif_name) + ".tif"),
    ]
    fpath = next((c for c in candidates if os.path.exists(c)), None)
    if not fpath:
        return jsonify({"value": None, "message": "TIF not found"})
    try:
        with rasterio.open(fpath) as src:
            try:
                src_epsg = src.crs.to_epsg()
            except Exception:
                src_epsg = None
            if src_epsg and src_epsg != 4326:
                xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
                x_s, y_s = xs[0], ys[0]
            else:
                x_s, y_s = lon, lat
            b = src.bounds
            if not (b.left <= x_s <= b.right and b.bottom <= y_s <= b.top):
                return jsonify({"value": None, "message": "Outside extent"})
            row, col = src.index(x_s, y_s)
            if not (0 <= row < src.height and 0 <= col < src.width):
                return jsonify({"value": None, "message": "Outside extent"})
            val = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0][0]
            nd  = src.nodata
            if nd is not None:
                thresh = 1e30 if abs(float(nd)) > 1e10 else 0.1
                if abs(float(val) - float(nd)) < thresh:
                    return jsonify({"value": None, "message": "NoData"})
            iv = int(round(float(val)))
            if iv == 0:
                return jsonify({"value": None, "message": "Zero"})
            return jsonify({"value": iv})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pixel_info/<state_key>/<period_suffix>/<int:pixel_value>")
def pixel_info(state_key, period_suffix, pixel_value):
    lookup = build_pixel_lookup(state_key, period_suffix)
    info   = lookup.get(pixel_value)
    if info is None:
        return jsonify({"value": None, "is_same": True})
    return jsonify({
        "from_label":  info["from_label"],
        "to_label":    info["to_label"],
        "from_l1":     info["from_l1"],
        "to_l1":       info["to_l1"],
        "pixel_sqkm":  info["pixel_sqkm"],
        "sqkm_total":  info["sqkm_total"],
        "count":       info["count"],
        "is_same":     info["is_same"],
        "is_retained": info["is_retained"],
    })


def _read_period_array(state_key, period_suffix):
    """Returns the raw LULC class array (int codes) for a single survey
    period, whether it lives in an active in-memory upload session or on
    disk (curated or uploaded-and-saved state)."""
    k    = state_key.lower()
    sess = _SESSION_STATES.get(k)
    if sess and period_suffix in sess.get("rasters", {}):
        return sess["rasters"][period_suffix]

    folder = _tif_folder(k)
    code   = _state_code(k)
    candidates = [
        os.path.join(folder, f"{code}_{period_suffix}_raster.tif"),
        os.path.join(folder, f"{period_suffix}.tif"),
    ]
    fpath = next((c for c in candidates if os.path.exists(c)), None)
    if not fpath:
        return None
    with rasterio.open(fpath) as src:
        return src.read(1).astype(np.int32)


def _pixel_area_sqkm_for_array(state_key, period_suffix, arr_shape):
    k    = state_key.lower()
    sess = _SESSION_STATES.get(k)
    if sess and period_suffix in sess.get("bounds_4326", {}):
        w, s, e, n = sess["bounds_4326"][period_suffix]
    else:
        folder = _tif_folder(k)
        code   = _state_code(k)
        candidates = [
            os.path.join(folder, f"{code}_{period_suffix}_raster.tif"),
            os.path.join(folder, f"{period_suffix}.tif"),
        ]
        fpath = next((c for c in candidates if os.path.exists(c)), None)
        if not fpath:
            return 0.0005 ** 2 * (111.32 ** 2)
        with rasterio.open(fpath) as src:
            b = src.bounds
            w, s, e, n = b.left, b.bottom, b.right, b.top
    px_w = (e - w) / max(arr_shape[1], 1)
    px_h = (n - s) / max(arr_shape[0], 1)
    center_lat = (s + n) / 2.0
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))
    return (px_w * km_per_deg_lon) * (px_h * km_per_deg_lat)


@app.route("/api/l2_timeline/<state_key>")
def l2_timeline(state_key):
    """L2-class area (sqkm) for every available survey period, for the
    timeline chart that replaces the GSDP chart on the Transition page."""
    k    = state_key.lower()
    sess = _SESSION_STATES.get(k)
    if sess:
        periods = sort_periods(sess.get("periods", []))
    else:
        meta = _load_upload_meta(k)
        periods = sort_periods(meta.get("periods", [])) if meta else ["0506", "1112", "1516"]

    rows = []
    for p in periods:
        arr = _read_period_array(k, p)
        if arr is None:
            continue
        px_sqkm = _pixel_area_sqkm_for_array(k, p, arr.shape)
        vals, counts = np.unique(arr, return_counts=True)
        plabel = period_label(p)
        for v, c in zip(vals, counts):
            iv = int(v)
            if iv == 0 or iv not in LU_WEBCODE:
                continue
            l1, l2, _ = LU_WEBCODE[iv]
            rows.append({
                "period": plabel,
                "l1": l1,
                "l2": l2,
                "sqkm": round(float(c) * px_sqkm, 4),
            })

    if not rows:
        return jsonify({"periods": [], "l2_classes": [], "series": {}})

    df = pd.DataFrame(rows)
    period_order = sorted(df["period"].unique().tolist(),
                          key=lambda lbl: int(re.search(r'(\d{4})', lbl).group(1)) if re.search(r'(\d{4})', lbl) else 9999)
    l2_classes = sorted(df["l2"].unique().tolist())

    series = {}
    for l2 in l2_classes:
        sub = df[df["l2"] == l2].set_index("period")["sqkm"]
        series[l2] = [round(float(sub.get(p, 0.0)), 4) for p in period_order]

    return jsonify({"periods": period_order, "l2_classes": l2_classes, "series": series})


@app.route("/api/transitions/<state_key>")
def get_transitions(state_key):
    frames = load_all_csvs(state_key)
    if not frames:
        return jsonify({"error": "No transition data available"}), 404
    ALL, from_col, to_col = _norm_frames(frames)
    if not from_col or not to_col:
        return jsonify({"error": f"Columns not found: {list(ALL.columns)}"}), 500
    ALL["From_L1"]   = ALL[from_col].astype(str).str.split(">").str[0].str.strip()
    ALL["To_L1"]     = ALL[to_col].astype(str).str.split(">").str[0].str.strip()
    l1c              = ALL[ALL["From_L1"] != ALL["To_L1"]].copy()
    l1c["Transition"] = l1c["From_L1"] + " to " + l1c["To_L1"]
    l1s              = l1c.groupby(["Period", "Transition"])["sqkm"].sum().reset_index()
    periods_found    = sorted(l1s["Period"].unique().tolist())
    pivot = l1s.pivot_table(index="Transition", columns="Period",
                            values="sqkm", aggfunc="sum", fill_value=0).reset_index()
    pivot.columns.name = None
    pivot = pivot.fillna(0)

    detail = (ALL.groupby(["Period", from_col, to_col], as_index=False)["sqkm"]
                 .sum().rename(columns={from_col: "from_cat", to_col: "to_cat"}))
    detail["from_cat"] = detail["from_cat"].astype(str).str.strip()
    detail["to_cat"]   = detail["to_cat"].astype(str).str.strip()
    detail = detail[detail["from_cat"].str.lower() != detail["to_cat"].str.lower()]
    detail["From_L1"]   = detail["from_cat"].str.split(">").str[0].str.strip()
    detail["To_L1"]      = detail["to_cat"].str.split(">").str[0].str.strip()
    detail["Transition"] = detail["from_cat"] + " to " + detail["to_cat"]
    detail["L1_Transition"] = detail["From_L1"] + " to " + detail["To_L1"]
    detail = detail[detail["sqkm"] > 0.001].sort_values(["Period", "sqkm"], ascending=[True, False])

    return jsonify({
        "l1":      pivot.round(2).to_dict(orient="records"),
        "periods": periods_found,
        "detail":  detail.round(4).to_dict(orient="records"),
    })


@app.route("/api/search_transitions/<state_key>")
def search_transitions(state_key):
    from_q = (request.args.get("from_class") or "").strip().lower()
    to_q   = (request.args.get("to_class")   or "").strip().lower()
    period = (request.args.get("period")     or "").strip()
    if period:
        period = normalize_period_label(period)
    if not from_q and not to_q:
        return jsonify({"error": "Provide from_class and/or to_class"}), 400
    frames = load_all_csvs(state_key, period_filter=[period] if period else None)
    if not frames:
        return jsonify({"rows": [], "legend": [], "total_sqkm": 0, "periods": []})
    ALL, from_col, to_col = _norm_frames(frames)
    if not from_col or not to_col:
        return jsonify({"rows": [], "legend": [], "total_sqkm": 0, "periods": []})
    ALL["from_cat"] = ALL[from_col].astype(str).str.strip()
    ALL["to_cat"]   = ALL[to_col].astype(str).str.strip()
    ALL = ALL[ALL["from_cat"].str.lower() != ALL["to_cat"].str.lower()]
    mask = pd.Series(True, index=ALL.index)
    if from_q: mask &= ALL["from_cat"].str.lower().str.contains(from_q, na=False)
    if to_q:   mask &= ALL["to_cat"].str.lower().str.contains(to_q,   na=False)
    filt = ALL[mask]
    if filt.empty:
        return jsonify({"rows": [], "legend": [], "total_sqkm": 0, "periods": []})
    grp = (filt.groupby(["Period", "from_cat", "to_cat"])["sqkm"]
               .sum().reset_index().sort_values("sqkm", ascending=False))
    grp = grp[grp["sqkm"] > 0.001]
    legend, seen = [], set()
    for _, row in grp.iterrows():
        cat = row["to_cat"]
        if cat not in seen:
            seen.add(cat)
            legend.append({"category": cat, "color": color_for_label(cat)})
        periods_present = set(grp["Period"].unique())
        if "2005-06 to 2011-12" in periods_present and "2011-12 to 2015-16" in periods_present:
            total_sqkm = grp[grp["Period"].isin(["2005-06 to 2011-12","2011-12 to 2015-16"])]["sqkm"].sum()
        else:
            total_sqkm = grp["sqkm"].sum()
    return jsonify({
        "rows":       grp.round(4).to_dict(orient="records"),
        "legend":     legend,
        "total_sqkm": round(float(total_sqkm), 4),
        "periods":    sorted(grp["Period"].unique().tolist()),
    })


@app.route("/api/class_transitions/<state_key>")
def class_transitions(state_key):
    class_name = request.args.get("class_name", "").strip().lower()
    direction  = request.args.get("direction", "both").lower()
    period     = request.args.get("period", "").strip()
    if period:
        period = normalize_period_label(period)
    if not class_name:
        return jsonify({"rows": [], "periods": [], "total_sqkm": 0})
    frames = load_all_csvs(state_key, period_filter=[period] if period else None)
    if not frames:
        return jsonify({"rows": [], "periods": [], "total_sqkm": 0})
    ALL, from_col, to_col = _norm_frames(frames)
    if not from_col or not to_col:
        return jsonify({"rows": [], "periods": [], "total_sqkm": 0})
    ALL["from_cat"] = ALL[from_col].astype(str).str.strip()
    ALL["to_cat"]   = ALL[to_col].astype(str).str.strip()
    ALL["from_l1"]  = ALL["from_cat"].str.split(">").str[0].str.strip().str.lower()
    ALL["to_l1"]    = ALL["to_cat"].str.split(">").str[0].str.strip().str.lower()
    ALL = ALL[ALL["from_cat"].str.lower() != ALL["to_cat"].str.lower()]
    names = [n.strip().lower() for n in class_name.split(",") if n.strip()]
    fm    = _class_name_mask(ALL["from_l1"], ALL["from_cat"].str.lower(), names)
    tm    = _class_name_mask(ALL["to_l1"],   ALL["to_cat"].str.lower(),   names)
    filt  = ALL[fm] if direction == "from" else (ALL[tm] if direction == "to" else ALL[fm | tm])
    if filt.empty:
        return jsonify({"rows": [], "periods": [], "total_sqkm": 0})
    grp = (filt.groupby(["Period", "from_cat", "to_cat"], as_index=False)["sqkm"]
               .sum().sort_values("sqkm", ascending=False))
    grp = grp[grp["sqkm"] > 0.001]
    period_sums = grp.groupby("Period")["sqkm"].sum().to_dict()
    all_labels  = _available_period_labels(state_key)
    periods_present = set(grp["Period"].unique())
    if "2005-06 to 2011-12" in periods_present and "2011-12 to 2015-16" in periods_present:
        total_sqkm = grp[grp["Period"].isin(["2005-06 to 2011-12","2011-12 to 2015-16"])]["sqkm"].sum()
    else:
        total_sqkm = grp["sqkm"].sum()
    return jsonify({
        "rows":       grp.round(4).to_dict(orient="records"),
        "periods":    sorted(grp["Period"].unique().tolist()),
        "total_sqkm": round(float(total_sqkm), 4),
    })


@app.route("/api/retention/<state_key>")
def get_retention(state_key):
    class_name = request.args.get("class_name", "").strip().lower()
    period     = request.args.get("period", "").strip()
    if period:
        period = normalize_period_label(period)
    if not class_name:
        return jsonify({"rows": [], "periods": [], "total_sqkm": 0})
    frames = load_all_csvs(state_key, period_filter=[period] if period else None)
    if not frames:
        return jsonify({"rows": [], "periods": [], "total_sqkm": 0})
    ALL, from_col, to_col = _norm_frames(frames)
    if not from_col or not to_col:
        return jsonify({"rows": [], "periods": [], "total_sqkm": 0})
    ALL["from_cat"] = ALL[from_col].astype(str).str.strip()
    ALL["to_cat"]   = ALL[to_col].astype(str).str.strip()
    ALL["from_l1"]  = ALL["from_cat"].str.split(">").str[0].str.strip().str.lower()
    ALL["to_l1"]    = ALL["to_cat"].str.split(">").str[0].str.strip().str.lower()
    names    = [n.strip().lower() for n in class_name.split(",") if n.strip()]
    same_l1  = ALL["from_l1"] == ALL["to_l1"]
    name_hit = _class_name_mask(ALL["from_l1"], ALL["from_cat"].str.lower(), names)
    retained = ALL[same_l1 & name_hit].copy()
    if retained.empty:
        return jsonify({"rows": [], "periods": [], "total_sqkm": 0})
    grp = (retained.groupby(["Period", "from_cat", "to_cat"], as_index=False)["sqkm"]
                   .sum().sort_values("sqkm", ascending=False))
    grp = grp[grp["sqkm"] > 0.001]
    period_sums = grp.groupby("Period")["sqkm"].sum().to_dict()
    all_labels  = _available_period_labels(state_key)
    net_total   = _net_affected_area(period_sums, all_labels)
    return jsonify({
        "rows":       grp.round(4).to_dict(orient="records"),
        "periods":    sorted(grp["Period"].unique().tolist()),
        "total_sqkm": round(float(net_total), 4),
    })

@app.route("/api/render_transition_subset_raster/<state_key>")
def render_transition_subset_raster_api(state_key):
    from_q        = (request.args.get("from_class", "")  or "").strip().lower()
    to_q          = (request.args.get("to_class", "")    or "").strip().lower()
    class_names_p = (request.args.get("class_names", "") or "").strip().lower()
    period_suffix = (request.args.get("period", "")      or "0506_to_1112").strip()
    mode          = (request.args.get("mode", "")        or "change").strip()

    parts = re.split(r'_to_', period_suffix)
    if len(parts) == 2:
        period_suffix = make_suffix(parts[0], parts[1])

    def _name_matches(label_l1, label_full, _names):
        ll1, lfull = label_l1.lower(), label_full.lower()
        return any(n == ll1 or n in lfull for n in _names)

    if mode == "retention":
        _names = [n.strip() for n in class_names_p.split(",") if n.strip()] if class_names_p else []
        def matcher(info, m):
            return info["is_retained"] and (not _names or _name_matches(info["from_l1"], info["from_label"], _names))
    elif class_names_p:
        _names = [n.strip() for n in class_names_p.split(",") if n.strip()]
        def matcher(info, m):
            return (not info["is_same"]) and \
                   (_name_matches(info["from_l1"], info["from_label"], _names) or
                    _name_matches(info["to_l1"],   info["to_label"],   _names))
    else:
        def matcher(info, m):
            if info["is_same"]:
                return False
            return ((not from_q) or from_q in info["from_label"].lower()) and \
                   ((not to_q)   or to_q   in info["to_label"].lower())

    sess = _SESSION_STATES.get(state_key.lower())
    if sess:
        lookup = sess.get("lookup", {}).get(period_suffix, {})
        ca     = sess.get("change_arrays", {}).get(period_suffix)
        if not ca:
            return jsonify({"error": "Upload both period rasters first"}), 404
        try:
            img, bnds, leg, stats = render_transition_subset_from_array(
                ca["arr"], ca["bounds_4326"], lookup, matcher, mode=mode)
            return jsonify({"image": img, "bounds": bnds, "legend": leg, "stats": stats})
        except Exception as e:
            import traceback
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    folder = _tif_folder(state_key)
    code   = _state_code(state_key)
    for fname in [f"Change_{period_suffix}.tif", f"changes_{code}_{period_suffix}.tif"]:
        tif_path = os.path.join(folder, fname)
        if os.path.exists(tif_path):
            break
    else:
        return jsonify({"error": f"Change raster not found for period '{period_suffix}'"}), 404

    lookup = build_pixel_lookup(state_key, period_suffix)
    if not lookup:
        return jsonify({"error": "No pixel lookup data for this period"}), 404
    try:
        img, bnds, leg, stats = render_transition_subset(tif_path, lookup, matcher, mode=mode)
        return jsonify({"image": img, "bounds": bnds, "legend": leg, "stats": stats})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/class_names/<state_key>")
def get_class_names(state_key):
    state_key = state_key.lower().replace(" ", "_").strip()
    sid = _session_id()
    uploaded_dir = os.path.join(UPLOAD_ROOT, sid, state_key)
    meta = _load_upload_meta(state_key)
    if meta and "periods" in meta:
        cats = set()
        state_code = meta.get("code", "UNKNOWN")
        for chg_p in meta.get("change_periods", []):
            for fname in [
                f"changes_{state_code}_{chg_p}_decoded.csv",
                f"RasterChanges_{state_code}_{chg_p}_decoded.csv",
            ]:
                csv_path = os.path.join(uploaded_dir, fname)
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        df.columns = df.columns.str.strip()
                        for col in ["From category", "To category", "from_category", "to_category"]:
                            if col in df.columns:
                                cats.update(df[col].dropna().astype(str).str.strip().tolist())
                    except Exception as e:
                        print(f"Error reading class names CSV: {e}")
                    break
        if cats:
            return jsonify(sorted(list(set(c.upper() for c in cats if c and c.lower() != "nan"))))

    frames = load_all_csvs(state_key)
    if not frames:
        return jsonify([])
    ALL, from_col, to_col = _norm_frames(frames)
    cats = set()
    if from_col:
        cats.update(ALL[from_col].dropna().astype(str).str.strip().tolist())
    if to_col:
        cats.update(ALL[to_col].dropna().astype(str).str.strip().tolist())
    return jsonify(sorted(c.upper() for c in cats if c and c.lower() != "nan"))


@app.route("/api/state_analysis/<state>/images")
def state_images(state):
    d = os.path.join(STATE_OUT_DIR, state)
    if not os.path.exists(d):
        return jsonify({"files": [], "groups": {}})
    files  = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    groups = {
        "L1 Overview":  [f for f in files if any(f.startswith(p) for p in ["01_", "03_", "04_"])],
        "Composition":  [f for f in files if any(f.startswith(p) for p in ["05_", "06_", "07_", "08_"])],
        "L2 Breakdown": [f for f in files if f.startswith("2_")],
    }
    return jsonify({"files": files, "groups": groups})


@app.route("/api/state_analysis/<state>/image/<path:filename>")
def state_image(state, filename):
    fpath = os.path.join(STATE_OUT_DIR, state, filename)
    if not os.path.exists(fpath):
        return jsonify({"error": "not found"}), 404
    return send_file(fpath, mimetype="image/png")


@app.route("/api/district_analysis/<state>/<district>/images")
def district_images(state, district):
    d = os.path.join(DISTRICT_OUT_DIR, state, district)
    if not os.path.exists(d):
        return jsonify({"files": [], "groups": {}})
    files  = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    groups = {
        "Donut Charts": [f for f in files if f.startswith("01_")],
        "Time Series":  [f for f in files if any(f.startswith(p) for p in ["02_", "03_"])],
        "L1 Changes":   [f for f in files if any(f.startswith(p) for p in ["04_", "05_"])],
        "L2 Changes":   [f for f in files if f.startswith("06_")],
    }
    return jsonify({"files": files, "groups": groups})


@app.route("/api/district_analysis/<state>/<district>/image/<path:filename>")
def district_image(state, district, filename):
    fpath = os.path.join(DISTRICT_OUT_DIR, state, district, filename)
    if not os.path.exists(fpath):
        return jsonify({"error": "not found"}), 404
    return send_file(fpath, mimetype="image/png")


@app.route("/api/scatter/<state>")
def scatter_list(state):
    d = os.path.join(SCATTER_DIR, state)
    if not os.path.exists(d):
        return jsonify([])
    result = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".png"):
            continue
        name  = f[:-4]
        parts = name.split("_")
        year_part  = "_".join(parts[-4:])
        class_part = "_".join(parts[:-4])
        l1 = next((p for p in ["AGRICULTURE", "BARRENUNCULTURABLE", "BUILTUP",
                                "FOREST", "GRASS", "SNOW", "WETLAND"]
                   if class_part.startswith(p)), "")
        result.append({"file": f, "class": class_part, "l1": l1, "year_range": year_part})
    return jsonify(result)


@app.route("/api/scatter/<state>/image/<path:filename>")
def scatter_image(state, filename):
    fpath = os.path.join(SCATTER_DIR, state, filename)
    if not os.path.exists(fpath):
        return jsonify({"error": "not found"}), 404
    return send_file(fpath, mimetype="image/png")

@app.route("/api/gsdp")
def get_gsdp():
    try:
        target_state_query = request.args.get("state", "").strip().lower().replace(" ", "_")
        df = pd.read_csv(GSDP_CSV)
        df.columns = df.columns.str.strip()
        result = {}

        for _, row in df.iterrows():
            s = str(row["State"]).strip().lower().replace(" ", "_")
            result[s] = {
                "2005-06": float(row.get("2005-06", 0) or 0),
                "2011-12": float(row.get("2011-12", 0) or 0),
                "2015-16": float(row.get("2015-16", 0) or 0),
            }
            if target_state_query and s == target_state_query:
                result[target_state_query] = row.to_dict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query/cog-layer", methods=["GET"])
def query_cog_layer():
    state_key  = request.args.get("state", "").strip().lower()
    period     = request.args.get("period", "").strip()
    class_code = request.args.get("class_code", type=int)

    parts = re.split(r'_to_', period)
    if len(parts) == 2:
        period = make_suffix(parts[0], parts[1])

    folder = _tif_folder(state_key)
    code   = _state_code(state_key)

    for fname in [f"changes_{code}_{period}.tif", f"Change_{period}.tif"]:
        change_tif = os.path.join(folder, fname)
        if os.path.exists(change_tif):
            break
    else:
        return jsonify({"error": "Requested change COG is not available"}), 404

    with rasterio.open(change_tif) as src:
        data = src.read(1)
        if class_code:
            mask = (data % 100 == class_code) & (data != 0)
            render_arr = np.where(mask, data, 0)
        else:
            render_arr = data
        bnds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        bounds_4326 = [bnds[0], bnds[1], bnds[2], bnds[3]]
        img_data, bounds, legend = render_raster_from_array(render_arr, bounds_4326, is_change=True)

    return jsonify({"raster": img_data, "bounds": bounds, "legend": legend})

# Animation part
@app.route("/api/l1_animation/<state_key>")
def l1_animation(state_key):
    l1_class = (request.args.get("l1_class") or "").strip().lower()
    if not l1_class:
        return jsonify({"error": "l1_class required"}), 400

    k = state_key.lower()
    sess = _SESSION_STATES.get(k)

    periods = []
    if sess:
        periods = sort_periods(sess.get("periods", []))
    else:
        meta = _load_upload_meta(k)
        if meta and meta.get("periods"):
            periods = sort_periods(meta["periods"])
        elif k in STATE_FOLDER:
            periods = ["0506", "1112", "1516"]

    if len(periods) < 2:
        return jsonify({"error": "Need at least 2 periods for animation"}), 400

    frames = []

    base_period = periods[0]

    if sess:
        base_arr = sess["rasters"].get(base_period)
        base_bounds = sess["bounds_4326"].get(base_period)
    else:
        code = _state_code(k)
        folder = _tif_folder(k)
        tif_path = os.path.join(folder, f"{code}_{base_period}_raster.tif")
        if not os.path.exists(tif_path):
            return jsonify({"error": f"Base raster not found for period {base_period}"}), 404
        base_arr, _, raw_bounds = read_raster_scaled(tif_path)
        base_bounds = (raw_bounds["west"], raw_bounds["south"], raw_bounds["east"], raw_bounds["north"])

    if base_arr is None:
        return jsonify({"error": "Base raster not loaded"}), 404

    base_l1_mask = np.zeros(base_arr.shape, dtype=bool)
    for code, (l1, l2, rgb) in LU_WEBCODE.items():
        if l1_class in l1.lower():
            base_l1_mask |= (base_arr == code)

    if not base_l1_mask.any():
        return jsonify({"error": f"No pixels found for l1_class '{l1_class}' in base period"}), 404

    rgba_base = np.zeros((*base_arr.shape, 4), dtype=np.uint8)
    for code, (l1, l2, rgb) in LU_WEBCODE.items():
        if l1_class in l1.lower():
            mask = base_arr == code
            if mask.any():
                rgba_base[mask] = [rgb[0], rgb[1], rgb[2], 220]
    rgba_base[~base_l1_mask] = [0, 0, 0, 0]

    w, s, e, n = base_bounds
    frames.append({
        "period": period_label(base_period) if hasattr(period_label, "__call__") else base_period,
        "period_code": base_period,
        "type": "base",
        "label": f"Base: {l1_class.title()} distribution in {period_label(base_period) if hasattr(period_label, '__call__') else base_period}",
        "image": png_from_rgba(rgba_base),
        "bounds": {"west": w, "south": s, "east": e, "north": n},
        "pixel_count": int(base_l1_mask.sum()),
    })

    for i in range(1, len(periods)):
        p2 = periods[i]
        suffix = make_suffix(base_period, p2)
        plabel = _suffix_to_label(suffix)

        lookup = build_pixel_lookup(k, suffix)

        if sess:
            ca = sess.get("change_arrays", {}).get(suffix)
            if ca:
                change_arr = ca["arr"]
                frame_bounds = ca["bounds_4326"]
            else:
                change_arr = None
                frame_bounds = base_bounds
        else:
            code_k = _state_code(k)
            folder = _tif_folder(k)
            chg_tif = None
            for fname in [f"changes_{code_k}_{suffix}.tif", f"Change_{suffix}.tif"]:
                fp = os.path.join(folder, fname)
                if os.path.exists(fp):
                    chg_tif = fp
                    break
            if chg_tif:
                chg_data, chg_nodata, raw_b = read_raster_scaled(chg_tif)
                change_arr = chg_data
                frame_bounds = (raw_b["west"], raw_b["south"], raw_b["east"], raw_b["north"])
            else:
                change_arr = None
                frame_bounds = base_bounds

        if change_arr is None or not lookup:
            frames.append({
                "period": plabel, "period_code": p2, "type": "change",
                "label": f"Change {plabel} (no data)", "image": None,
                "bounds": {"west": frame_bounds[0], "south": frame_bounds[1],
                           "east": frame_bounds[2], "north": frame_bounds[3]},
                "pixel_count": 0,
            })
            continue

        rgba = np.zeros((*change_arr.shape, 4), dtype=np.uint8)
        lost_count = 0
        gained_count = 0
        retained_count = 0

        for val, info in lookup.items():
            fv = float(val)
            mask = np.abs(change_arr.astype(float) - fv) < 0.5
            if not mask.any():
                continue

            from_is_l1 = l1_class in info["from_l1"].lower()
            to_is_l1 = l1_class in info["to_l1"].lower()

            if from_is_l1 and to_is_l1:
                rgba[mask] = [63, 185, 80, 200]
                retained_count += int(mask.sum())
            elif from_is_l1 and not to_is_l1:
                rgba[mask] = [247, 129, 102, 230]
                lost_count += int(mask.sum())
            elif not from_is_l1 and to_is_l1:
                rgba[mask] = [88, 166, 255, 230]
                gained_count += int(mask.sum())

        fw, fs, fe, fn = frame_bounds
        frames.append({
            "period": plabel, "period_code": p2, "type": "change",
            "label": f"Transitions {plabel}",
            "image": png_from_rgba(rgba),
            "bounds": {"west": fw, "south": fs, "east": fe, "north": fn},
            "pixel_count": lost_count + gained_count + retained_count,
            "stats": {
                "lost": lost_count,
                "gained": gained_count,
                "retained": retained_count,
            },
            "legend": [
                {"label": f"{l1_class.title()} retained", "color": "#3fb950"},
                {"label": f"{l1_class.title()} lost (converted away)", "color": "#f78166"},
                {"label": f"Gained {l1_class.title()} (converted from other)", "color": "#58a6ff"},
            ],
        })

    return jsonify({
        "l1_class": l1_class,
        "base_period": base_period,
        "periods": periods,
        "frames": frames,
    })

# Downloads part

@app.route("/api/download/transition_table")
def download_transition_table_v2():
    state_key     = (request.args.get("state_key")     or "").strip()
    period        = (request.args.get("period")        or "").strip()
    period_filter = (request.args.get("period_filter") or "").strip()
    fmt           = (request.args.get("format")        or "csv").lower()
    from_filter   = (request.args.get("from_filter")   or "").strip().lower()
    to_filter     = (request.args.get("to_filter")     or "").strip().lower()

    if not state_key:
        return jsonify({"error": "state_key required"}), 400

    frames = load_all_csvs(state_key)
    if not frames:
        return jsonify({"error": "No transition data found for this state."}), 404

    ALL, from_col, to_col = _norm_frames(frames)
    if not from_col or not to_col:
        return jsonify({"error": "Cannot identify from/to columns in data."}), 500

    ALL["from_cat"] = ALL[from_col].astype(str).str.strip()
    ALL["to_cat"]   = ALL[to_col].astype(str).str.strip()
    df = ALL.copy()

    eff_period = period_filter or period
    if eff_period and eff_period not in ("", "all"):
        if "_to_" in eff_period:
            p_label = _suffix_to_label(eff_period)
        else:
            try:
                p_label = normalize_period_label(eff_period)
            except Exception:
                p_label = eff_period
        norm_periods = df["Period"].apply(normalize_period_label)
        if p_label in norm_periods.values:
            df = df[norm_periods == p_label].copy()

    df = df[df["from_cat"].str.lower() != df["to_cat"].str.lower()]
    if from_filter:
        df = df[df["from_cat"].str.lower().str.contains(from_filter, na=False)]
    if to_filter:
        df = df[df["to_cat"].str.lower().str.contains(to_filter, na=False)]

    export_df = df[["Period", "from_cat", "to_cat", "sqkm"]].copy()
    export_df.columns = ["Period", "From Category", "To Category", "Area (km2)"]
    export_df = export_df.sort_values(
        ["Period", "Area (km2)"], ascending=[True, False]
    ).round(4)

    code = _state_code(state_key)
    name_parts = [code]
    if from_filter:
        name_parts.append(re.sub(r"[^a-zA-Z0-9]", "_", from_filter)[:20])
    if to_filter:
        name_parts.append(re.sub(r"[^a-zA-Z0-9]", "_", to_filter)[:20])
    if eff_period and eff_period not in ("", "all"):
        name_parts.append(eff_period.replace(" ", "_")[:30])
    fname_base = "_".join(name_parts)

    if fmt == "xlsx":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Transitions")
            wb  = writer.book
            ws  = writer.sheets["Transitions"]
            hdr = wb.add_format({"bold": True, "bg_color": "#161b22",
                                  "font_color": "#ffffff", "border": 1, "font_size": 9})
            cel = wb.add_format({"border": 1, "font_size": 9})
            for cn, val in enumerate(export_df.columns):
                ws.write(0, cn, val, hdr)
            for rn in range(len(export_df)):
                for cn in range(len(export_df.columns)):
                    ws.write(rn + 1, cn, export_df.iloc[rn, cn], cel)
            ws.set_column(0, 0, 18)
            ws.set_column(1, 2, 40)
            ws.set_column(3, 3, 14)
        output.seek(0)
        return send_file(
            output, download_name=fname_base + ".xlsx", as_attachment=True,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    output = io.BytesIO()
    export_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, download_name=fname_base + ".csv",as_attachment=True, mimetype="text/csv")


# updated on 18-06
@app.route("/api/download/transition_pdf", methods=["POST"])
def download_transition_pdf():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, A3, landscape, portrait
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle,
            Paragraph, Spacer, Image as RLImage, PageBreak,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
    except ImportError:
        return jsonify({"error": "reportlab not installed"}), 500

    body          = request.get_json(force=True, silent=True) or {}
    state_key     = body.get("state_key", "")
    period        = (body.get("period")        or "").strip()
    page_size_s   = (body.get("page_size")     or "a4l").lower()
    from_filter   = (body.get("from_filter")   or "").strip().lower()
    to_filter     = (body.get("to_filter")     or "").strip().lower()
    period_filter = (body.get("period_filter") or "").strip()
    headers_in    = body.get("headers") or []
    rows_in       = body.get("rows")    or []
    map_image_b64 = body.get("map_image")
    legend_items  = body.get("legend_items") or []

    PAGE_MAP = {
        "a4":  portrait(A4),  "a4l": landscape(A4),
        "a3":  portrait(A3),  "a3l": landscape(A3),
    }
    page_size = PAGE_MAP.get(page_size_s, landscape(A4))
    pw = page_size[0] - 3.0 * cm
    ph = page_size[1] - 4.0 * cm

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=page_size,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )

    styles    = getSampleStyleSheet()
    title_sty = ParagraphStyle("t", parent=styles["Normal"], fontSize=12,
                                fontName="Helvetica-Bold",
                                textColor=colors.HexColor("#58a6ff"), spaceAfter=4)
    sub_sty   = ParagraphStyle("s", parent=styles["Normal"], fontSize=8,
                                fontName="Helvetica",
                                textColor=colors.HexColor("#8b949e"), spaceAfter=4)
    cell_sty  = ParagraphStyle("c", parent=styles["Normal"], fontSize=7,
                                fontName="Helvetica",
                                textColor=colors.HexColor("#e6edf3"), leading=8)
    leg_sty   = ParagraphStyle("l", parent=styles["Normal"], fontSize=6,
                                fontName="Helvetica",
                                textColor=colors.HexColor("#010911"), leading=8)

    story = []

    display    = state_key.replace("_", " ").title()
    eff_period = period_filter or period
    if eff_period and "_to_" in eff_period:
        period_lbl = _suffix_to_label(eff_period)
    elif eff_period:
        try:
            period_lbl = normalize_period_label(eff_period)
        except Exception:
            period_lbl = eff_period
    else:
        period_lbl = "All periods"

    story.append(Paragraph(f"LULC Transition Analysis — {display}", title_sty))
    sub_parts = [f"Period: {period_lbl}"]
    if from_filter:
        sub_parts.append(f"From: {from_filter}")
    if to_filter:
        sub_parts.append(f"To: {to_filter}")
    story.append(Paragraph("  |  ".join(sub_parts), sub_sty))
    story.append(Spacer(1, 0.3 * cm))

    if map_image_b64:
        try:
            img_bytes = base64.b64decode(map_image_b64)
            img_buf   = io.BytesIO(img_bytes)
            if legend_items:
                map_w = pw * 0.72
                map_h = min(map_w * 0.60, ph * 0.55)
                leg_w = pw * 0.26
                rl_img    = RLImage(img_buf, width=map_w, height=map_h)
                leg_rows  = []
                for item in legend_items:
                    hex_c = item.get("color", "#888888")
                    lbl   = item.get("label", "")
                    leg_rows.append([
                        Paragraph(
                            f'<font color="{hex_c}">\u25a0 </font>{lbl}',
                            leg_sty,
                        )
                    ])
                leg_tbl = Table(leg_rows, colWidths=[leg_w])
                leg_tbl.setStyle(TableStyle([
                    ("VALIGN",        (0,0), (-1,-1), "TOP"),
                    ("TOPPADDING",    (0,0), (-1,-1), 2),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 2),
                ]))
                combined = Table([[rl_img, leg_tbl]], colWidths=[map_w, leg_w])
                combined.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
                story.append(combined)
            else:
                map_h = min(pw * 0.55, ph * 0.55)
                story.append(RLImage(img_buf, width=pw, height=map_h))
            story.append(Spacer(1, 0.3 * cm))
        except Exception:
            pass

    story.append(PageBreak())
    story.append(Paragraph("Transition Table", title_sty))
    story.append(Spacer(1, 0.15 * cm))

    final_headers = headers_in
    final_rows    = rows_in

    if not final_rows:
        frames = load_all_csvs(state_key)
        if frames:
            ALL, from_col, to_col = _norm_frames(frames)
            if from_col and to_col:
                ALL["from_cat"] = ALL[from_col].astype(str).str.strip()
                ALL["to_cat"]   = ALL[to_col].astype(str).str.strip()
                df = ALL.copy()

                if eff_period and eff_period not in ("", "all"):
                    if "_to_" in eff_period:
                        p_lbl = _suffix_to_label(eff_period)
                    else:
                        try:
                            p_lbl = normalize_period_label(eff_period)
                        except Exception:
                            p_lbl = eff_period
                    norm_p = df["Period"].apply(normalize_period_label)
                    if p_lbl in norm_p.values:
                        df = df[norm_p == p_lbl].copy()

                df = df[df["from_cat"].str.lower() != df["to_cat"].str.lower()]
                if from_filter:
                    df = df[df["from_cat"].str.lower().str.contains(from_filter, na=False)]
                if to_filter:
                    df = df[df["to_cat"].str.lower().str.contains(to_filter, na=False)]
                df = (df[["Period", "from_cat", "to_cat", "sqkm"]]
                      .sort_values(["Period", "sqkm"], ascending=[True, False])
                      .round(4))
                final_headers = ["Period", "From Category", "To Category", "Area (km2)"]
                final_rows    = df.astype(str).values.tolist()

    if final_rows:
        col_count = len(final_headers) if final_headers else (len(final_rows[0]) if final_rows else 4)
        col_w     = pw / col_count

        def _p(txt):
            return Paragraph(str(txt), cell_sty)

        tbl_data = [[_p(h) for h in final_headers]] + [[_p(c) for c in row] for row in final_rows]
        tbl = Table(tbl_data, colWidths=[col_w] * col_count, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),  (-1,0),  colors.HexColor("#161b22")),
            ("TEXTCOLOR",      (0,0),  (-1,0),  colors.white),
            ("FONTNAME",       (0,0),  (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),  (-1,-1), 7),
            ("GRID",           (0,0),  (-1,-1), 0.25, colors.HexColor("#30363d")),
            ("ROWBACKGROUNDS", (0,1),  (-1,-1), [colors.HexColor("#1c2128"),
                                                  colors.HexColor("#161b22")]),
            ("TEXTCOLOR",      (0,1),  (-1,-1), colors.HexColor("#e6edf3")),
            ("TOPPADDING",     (0,0),  (-1,-1), 3),
            ("BOTTOMPADDING",  (0,0),  (-1,-1), 3),
            ("LEFTPADDING",    (0,0),  (-1,-1), 4),
            ("RIGHTPADDING",   (0,0),  (-1,-1), 4),
            ("VALIGN",         (0,0),  (-1,-1), "TOP"),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("No transition data matched the current filters.", sub_sty))

    doc.build(story)
    output.seek(0)

    code = _state_code(state_key)
    parts = [code]
    if from_filter:
        parts.append(re.sub(r"[^a-zA-Z0-9]", "_", from_filter)[:20])
    if to_filter:
        parts.append(re.sub(r"[^a-zA-Z0-9]", "_", to_filter)[:20])
    if eff_period and eff_period not in ("", "all"):
        parts.append(eff_period.replace(" ", "_")[:30])
    parts.append(page_size_s)
    fname = "_".join(parts) + ".pdf"

    return send_file(output, download_name=fname, as_attachment=True, mimetype="application/pdf")

if __name__ == "__main__":
    app.run(host='172.31.4.129', debug=True, port=7000)