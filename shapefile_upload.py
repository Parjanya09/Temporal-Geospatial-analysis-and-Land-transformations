import os
import io
import json
import math
import shutil
import tempfile
import traceback
import warnings
import re
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.features import rasterize as rio_rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling

os.environ.setdefault("PROJ_NETWORK", "OFF")
warnings.filterwarnings("ignore")

shapefile_bp = Blueprint("shapefile_ingest", __name__)

from upload import (
    LU_WEBCODE,
    label_for_code,
    UPLOAD_ROOT,
    _session_id,
    upload_folder_for_key,
    load_meta,
    save_meta,
    sort_periods,
    make_suffix,
    code_from_key,
    state_key_from_name,
    build_change_tif,
    generate_decoded_csv,
    _validate_geo_bounds,
    infer_period,
    period_label,
)


def _period_sort_key(code):
    ORDER = {"0506": 0, "1112": 1, "1516": 2, "1920": 3, "2021": 4}
    if code in ORDER:
        return ORDER[code]
    m = re.search(r"(20\d{2}|19\d{2})", str(code))
    return int(m.group(1)) if m else 9999


def _read_vector_file(path):
    import geopandas as gpd
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fgb":
        return gpd.read_file(path, driver="FlatGeobuf")
    return gpd.read_file(path)


def _normalise_to_4326(gdf):
    if gdf.crs is None:
        b = gdf.total_bounds
        gdf = gdf.set_crs(epsg=32643 if (abs(b[0]) > 180 or abs(b[2]) > 180) else 4326)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def _find_burn_field(gdf, preferred="LU_Webcode"):
    col_map = {c.lower(): c for c in gdf.columns}
    if preferred.lower() in col_map:
        return col_map[preferred.lower()]
    for fb in ("lu_webcode", "lulc_code", "lulc", "lucode", "value", "gridcode"):
        if fb in col_map:
            return col_map[fb]
    raise ValueError(
        "Cannot find LU_Webcode field (or fallback). "
        "Available columns: " + str(list(gdf.columns))
    )


def _to_int32(raw):
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return 0


def _vector_to_tif(src_path, lulc_field, resolution, dst_tif):
    gdf = _read_vector_file(src_path)
    if gdf.empty:
        raise ValueError("Vector layer contains no features.")

    gdf = _normalise_to_4326(gdf)
    burn_col = _find_burn_field(gdf, lulc_field)

    gdf = gdf.copy()
    gdf["_burn"] = gdf[burn_col].apply(_to_int32).astype(np.int32)
    gdf = gdf[gdf["_burn"] > 0]
    if gdf.empty:
        raise ValueError("No valid LU_Webcode (>0) features after int32 conversion.")

    west, south, east, north = gdf.total_bounds
    _validate_geo_bounds(west, south, east, north, label="Vector layer")

    width  = max(1, int(round((east  - west)  / resolution)))
    height = max(1, int(round((north - south) / resolution)))
    if max(width, height) > 8000:
        scale = 4000.0 / max(width, height)
        resolution = resolution / scale
        width  = max(1, int(round((east  - west)  / resolution)))
        height = max(1, int(round((north - south) / resolution)))

    transform = from_bounds(west, south, east, north, width, height)

    shapes = (
        (geom, int(val))
        for geom, val in zip(gdf.geometry, gdf["_burn"])
        if geom is not None and not geom.is_empty
    )
    arr = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
    )

    profile = {
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
    }
    os.makedirs(os.path.dirname(dst_tif), exist_ok=True)
    with rasterio.open(dst_tif, "w", **profile) as dst:
        dst.write(arr, 1)

    return dst_tif, (west, south, east, north)


def _save_as_flatgeobuf(gdf, out_path):
    gdf.to_file(out_path, driver="FlatGeobuf")
    return out_path


def _build_change_csvs(out_folder, state_code, state_key, all_periods, sid, tmp_dir):
    change_periods = []
    for i in range(len(all_periods)):
        for j in range(i + 1, len(all_periods)):
            p1, p2 = all_periods[i], all_periods[j]
            suffix  = make_suffix(p1, p2)
            t1      = os.path.join(out_folder, f"{state_code}_{p1}_raster.tif")
            t2      = os.path.join(out_folder, f"{state_code}_{p2}_raster.tif")
            chg_tif = os.path.join(out_folder, f"changes_{state_code}_{suffix}.tif")
            if os.path.exists(t1) and os.path.exists(t2) and not os.path.exists(chg_tif):
                try:
                    raw = os.path.join(tmp_dir, f"_raw_{suffix}.tif")
                    build_change_tif(t1, t2, raw)
                    shutil.copy2(raw, chg_tif)
                    generate_decoded_csv(chg_tif, suffix, state_key, out_folder, sid)
                    change_periods.append(suffix)
                except Exception:
                    pass
    return change_periods


@shapefile_bp.route("/api/upload/shapefile", methods=["POST"])
def upload_shapefile_pure():
    state_name  = (request.form.get("state_name")  or "").strip()
    state_code  = (request.form.get("state_code")  or "").strip().upper()
    period_code = (request.form.get("period")       or "").strip()
    lulc_field  = (request.form.get("lulc_field")  or "LU_Webcode").strip()
    resolution  = float(request.form.get("resolution") or 0.0005)

    shp_f = request.files.get("shp")
    dbf_f = request.files.get("dbf")
    shx_f = request.files.get("shx")

    if not state_name:
        return jsonify({"error": "state_name required"}), 400
    if not period_code:
        return jsonify({"error": "period required"}), 400
    if not (shp_f and dbf_f and shx_f):
        return jsonify({"error": ".shp, .dbf, and .shx all required"}), 400

    sid        = _session_id()
    state_key  = state_key_from_name(state_name)
    if not state_code:
        state_code = code_from_key(state_key)

    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "upload.shp")
        shp_f.save(src)
        dbf_f.save(os.path.join(tmp, "upload.dbf"))
        shx_f.save(os.path.join(tmp, "upload.shx"))
        if request.files.get("prj"):
            request.files["prj"].save(os.path.join(tmp, "upload.prj"))
        if request.files.get("cpg"):
            request.files["cpg"].save(os.path.join(tmp, "upload.cpg"))

        import geopandas as gpd
        gdf = gpd.read_file(src)
        if gdf.empty:
            return jsonify({"error": "Shapefile contains no features."}), 400
        gdf = _normalise_to_4326(gdf)

        fgb_path = os.path.join(tmp, "layer.fgb")
        _save_as_flatgeobuf(gdf, fgb_path)

        out_folder = upload_folder_for_key(state_key, sid)
        dst_tif    = os.path.join(out_folder, f"{state_code}_{period_code}_raster.tif")
        _, bounds  = _vector_to_tif(fgb_path, lulc_field, resolution, dst_tif)

        meta = load_meta(state_key, sid)
        meta.update({
            "display_name": state_name,
            "code": state_code,
            "periods": meta.get("periods", []),
            "change_periods": meta.get("change_periods", []),
        })
        if period_code not in meta["periods"]:
            meta["periods"].append(period_code)
        meta["periods"] = sort_periods(meta["periods"])

        new_cp = _build_change_csvs(out_folder, state_code, state_key, meta["periods"], sid, tmp)
        for cp in new_cp:
            if cp not in meta["change_periods"]:
                meta["change_periods"].append(cp)
        meta["change_periods"] = sorted(
            set(meta["change_periods"]),
            key=lambda s: tuple(_period_sort_key(p) for p in s.split("_to_")),
        )
        save_meta(state_key, meta, sid)

        with rasterio.open(dst_tif) as chk:
            shape = [chk.height, chk.width]

        return jsonify({
            "status": "ok",
            "state_key": state_key,
            "display_name": state_name,
            "code": state_code,
            "loaded_period": period_code,
            "all_periods": meta["periods"],
            "change_periods": meta["change_periods"],
            "shape": shape,
            "bounds": list(bounds),
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@shapefile_bp.route("/api/upload/vector_batch", methods=["POST"])
def upload_vector_batch():
    state_name  = (request.form.get("state_name")  or "").strip()
    state_code  = (request.form.get("state_code")  or "").strip().upper()
    lulc_field  = (request.form.get("lulc_field")  or "LU_Webcode").strip()
    resolution  = float(request.form.get("resolution") or 0.0005)
    uploaded    = request.files.getlist("files")

    if not state_name:
        return jsonify({"error": "state_name required"}), 400
    if not uploaded:
        return jsonify({"error": "No files received"}), 400

    sid        = _session_id()
    state_key  = state_key_from_name(state_name)
    if not state_code:
        state_code = code_from_key(state_key)

    out_folder = upload_folder_for_key(state_key, sid)
    tmp        = tempfile.mkdtemp()
    try:
        saved = {}
        for f in uploaded:
            if f.filename:
                dest = os.path.join(tmp, f.filename)
                f.save(dest)
                saved[f.filename] = dest

        meta = load_meta(state_key, sid)
        meta.update({
            "display_name": state_name,
            "code": state_code,
            "periods": meta.get("periods", []),
            "change_periods": meta.get("change_periods", []),
        })

        VECTOR_EXTS  = (".shp", ".fgb", ".gpkg", ".geojson", ".json")
        SIDECAR_EXTS = (".dbf", ".shx", ".prj", ".cpg", ".fix")
        results = []

        for fname, fpath in saved.items():
            ext = os.path.splitext(fname)[1].lower()
            if ext in SIDECAR_EXTS:
                continue
            if ext not in VECTOR_EXTS:
                continue

            period = infer_period(fname)
            if period == "unknown":
                results.append({"file": fname, "status": "error",
                                 "error": f"Cannot infer period from '{fname}'."})
                continue

            is_chg = "change" in fname.lower() or "trans" in fname.lower()
            dst_name = (f"changes_{state_code}_{period}.tif" if (is_chg and "_to_" in period)
                        else f"{state_code}_{period}_raster.tif")
            dst_tif = os.path.join(out_folder, dst_name)

            try:
                import geopandas as gpd
                gdf = gpd.read_file(fpath)
                gdf = _normalise_to_4326(gdf)
                fgb = os.path.join(tmp, os.path.splitext(fname)[0] + ".fgb")
                _save_as_flatgeobuf(gdf, fgb)
                _vector_to_tif(fgb, lulc_field, resolution, dst_tif)

                entry = {"file": fname, "status": "ok", "output": dst_name}
                if not is_chg and period not in meta["periods"]:
                    meta["periods"].append(period)
                if is_chg and "_to_" in period:
                    csv_path, n = generate_decoded_csv(dst_tif, period, state_key, out_folder, sid)
                    entry.update({"csv": os.path.basename(csv_path), "csv_rows": n})
                    if period not in meta["change_periods"]:
                        meta["change_periods"].append(period)
                results.append(entry)
            except Exception as e:
                results.append({"file": fname, "status": "error", "error": str(e)})

        meta["periods"] = sort_periods(meta["periods"])
        new_cp = _build_change_csvs(out_folder, state_code, state_key, meta["periods"], sid, tmp)
        for cp in new_cp:
            if cp not in meta["change_periods"]:
                meta["change_periods"].append(cp)
        meta["change_periods"] = sorted(
            set(meta["change_periods"]),
            key=lambda s: tuple(_period_sort_key(p) for p in s.split("_to_")),
        )
        save_meta(state_key, meta, sid)

        return jsonify({
            "state_key": state_key,
            "display_name": state_name,
            "code": state_code,
            "periods": meta["periods"],
            "change_periods": meta["change_periods"],
            "results": results,
            "ok_count":  sum(1 for r in results if r["status"] == "ok"),
            "err_count": sum(1 for r in results if r["status"] == "error"),
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400
    finally:
        shutil.rmtree(tmp, ignore_errors=True)