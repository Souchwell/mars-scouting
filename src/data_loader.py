import re
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds


def _decode(data: bytes | bytearray | memoryview) -> str:
    return bytes(data).decode("latin-1", errors="replace")


def _header_value(text: str, key: str, default: str | None = None) -> str | None:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*(.*?)\s*$", text)
    return match.group(1).strip() if match else default


def _numeric_label_value(text: str, key: str, default: float) -> float:
    match = re.search(
        rf"(?im)^\s*{re.escape(key)}\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
        text,
    )
    return float(match.group(1)) if match else default


def _label_text_value(text: str, key: str) -> str | None:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*(.*?)\s*$", text)
    if not match:
        return None
    return match.group(1).strip().strip('"').strip("'")


def _parse_map_info(text: str) -> dict[str, Any]:
    match = re.search(
        r"(?im)^\s*map\s+info\s*=\s*\{([^}]*)\}",
        text,
    )

    if not match:
        raise ValueError("ENVI HDR içinde map info bulunamadı.")

    raw = match.group(1)
    parts = [part.strip() for part in raw.split(",")]

    if len(parts) < 7:
        raise ValueError("ENVI map info alanı eksik.")

    try:
        return {
            "raw": raw,
            "projection_name": parts[0],
            "reference_col": float(parts[1]),
            "reference_row": float(parts[2]),
            "map_x": float(parts[3]),
            "map_y": float(parts[4]),
            "pixel_x": abs(float(parts[5])),
            "pixel_y": abs(float(parts[6])),
            "units": parts[-1].split("=", 1)[-1].strip()
            if "units=" in parts[-1].lower()
            else None,
        }
    except ValueError as exc:
        raise ValueError(
            "ENVI map info koordinatları okunamadı."
        ) from exc

def _parse_projection_info(text: str) -> dict[str, Any]:
    # Önce klasik ENVI projection info alanını kontrol et.
    match = re.search(
        r"(?im)^\s*projection\s+info\s*=\s*\{([^}]*)\}",
        text,
    )

    if match:
        parts = [part.strip() for part in match.group(1).split(",")]

        if len(parts) < 4:
            raise ValueError("ENVI projection info eksik.")

        try:
            return {
                "projection_code": int(parts[0]),
                "semi_major_axis": float(parts[1]),
                "central_meridian": float(parts[3]),
            }
        except ValueError as exc:
            raise ValueError(
                "Projeksiyon parametreleri okunamadı."
            ) from exc

    # ENVI projection info yoksa coordinate system string alanını oku.
    coord_match = re.search(
        r'(?im)^\s*coordinate\s+system\s+string\s*=\s*\{(.*?)\}\s*$',
        text,
    )

    if not coord_match:
        raise ValueError(
            "ENVI projection info veya coordinate system string bulunamadı."
        )

    coordinate_text = coord_match.group(1)

    radius_match = re.search(
        r'SPHEROID\s*\[\s*"[^"]+"\s*,\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)',
        coordinate_text,
        re.IGNORECASE,
    )

    if not radius_match:
        raise ValueError("Mars referans yarıçapı bulunamadı.")

    radius = float(radius_match.group(1))

    if radius <= 0:
        raise ValueError("Mars referans yarıçapı geçersiz.")

    meridian_match = re.search(
        r'PRIMEM\s*\[\s*"[^"]+"\s*,\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)',
        coordinate_text,
        re.IGNORECASE,
    )

    central_meridian = (
        float(meridian_match.group(1))
        if meridian_match
        else 0.0
    )

    return {
        "projection_code": 0,
        "semi_major_axis": radius,
        "central_meridian": central_meridian,
    }


def parse_envi_header(hdr_bytes: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Parse the small ENVI header and return raster layout information."""
    text = _decode(hdr_bytes)
    required = ("samples", "lines", "data type")
    for key in required:
        if _header_value(text, key) is None:
            raise ValueError(f"HDR dosyasında zorunlu '{key}' alanı bulunamadı.")

    samples = int(_header_value(text, "samples"))
    lines = int(_header_value(text, "lines"))
    bands = int(_header_value(text, "bands", "1"))
    data_type = int(_header_value(text, "data type"))
    byte_order = int(_header_value(text, "byte order", "0"))
    header_offset = int(_header_value(text, "header offset", "0"))
    nodata_raw = _header_value(text, "data ignore value")
    nodata = float(nodata_raw) if nodata_raw is not None else None

    envi_types = {
        1: ("u1", "uint8"),
        2: ("i2", "int16"),
        3: ("i4", "int32"),
        4: ("f4", "float32"),
        5: ("f8", "float64"),
        12: ("u2", "uint16"),
        13: ("u4", "uint32"),
        14: ("i8", "int64"),
        15: ("u8", "uint64"),
    }
    if data_type not in envi_types:
        raise ValueError(f"Desteklenmeyen ENVI data type: {data_type}.")
    if byte_order not in (0, 1):
        raise ValueError("ENVI byte order yalnızca 0 veya 1 olabilir.")
    if samples <= 0 or lines <= 0 or bands <= 0 or header_offset < 0:
        raise ValueError("HDR içindeki raster boyutu veya header offset geçersiz.")

    code, dtype_name = envi_types[data_type]
    dtype = np.dtype("u1") if data_type == 1 else np.dtype((">" if byte_order == 1 else "<") + code)
    map_info = _parse_map_info(text)
    projection_info = _parse_projection_info(text)
    return {
        "samples": samples,
        "lines": lines,
        "bands": bands,
        "data_type": data_type,
        "byte_order": byte_order,
        "header_offset": header_offset,
        "nodata": nodata,
        "dtype": dtype,
        "dtype_name": dtype_name,
        "pixel_x": map_info["pixel_x"],
        "pixel_y": map_info["pixel_y"],
        "map_info_raw": map_info["raw"],
        "map_reference_col": map_info["reference_col"],
        "map_reference_row": map_info["reference_row"],
        "map_x": map_info["map_x"],
        "map_y": map_info["map_y"],
        "map_units": map_info["units"],
        "projection_name": map_info["projection_name"],
        "projection_code": projection_info["projection_code"],
        "semi_major_axis": projection_info["semi_major_axis"],
        "central_meridian": projection_info["central_meridian"],
    }


def _parse_pds_label(lbl_bytes: bytes | bytearray | memoryview | None) -> dict[str, Any]:
    if not lbl_bytes:
        return {
            "present": False,
            "scaling_factor": 1.0,
            "offset": 0.0,
            "missing_constant": None,
            "unit": "METER",
            "unit_source": "MOLA MEGDR ürün metadata bilgisi",
            "sample_type": None,
            "sample_bits": None,
            "line_samples": None,
            "lines": None,
        }

    text = _decode(lbl_bytes)

    unit = (
        _label_text_value(text, "UNIT")
        or _label_text_value(text, "UNITS")
    )

    if unit:
        unit = unit.upper()
        unit_source = "PDS LBL dosyası"
    else:
        unit = "METER"
        unit_source = "MOLA MEGDR ürün metadata bilgisi"

    missing_raw = _label_text_value(text, "MISSING_CONSTANT")

    try:
        missing_constant = (
            float(missing_raw)
            if missing_raw is not None
            else None
        )
    except ValueError:
        missing_constant = None

    bits_raw = _label_text_value(text, "SAMPLE_BITS")
    samples_raw = _label_text_value(text, "LINE_SAMPLES")
    lines_raw = _label_text_value(text, "LINES")

    return {
        "present": True,
        "scaling_factor": _numeric_label_value(
            text, "SCALING_FACTOR", 1.0
        ),
        "offset": _numeric_label_value(
            text, "OFFSET", 0.0
        ),
        "missing_constant": missing_constant,
        "unit": unit,
        "unit_source": unit_source,
        "sample_type": _label_text_value(text, "SAMPLE_TYPE"),
        "sample_bits": (
            int(bits_raw)
            if bits_raw and bits_raw.isdigit()
            else None
        ),
        "line_samples": (
            int(samples_raw)
            if samples_raw and samples_raw.isdigit()
            else None
        ),
        "lines": (
            int(lines_raw)
            if lines_raw and lines_raw.isdigit()
            else None
        ),
    }

def _make_metadata(header: dict[str, Any], label: dict[str, Any], file_size: int) -> dict[str, Any]:
    if label["line_samples"] and label["line_samples"] != header["samples"]:
        raise ValueError("LBL içindeki LINE_SAMPLES değeri HDR içindeki samples ile uyuşmuyor.")
    if label["lines"] and label["lines"] != header["lines"]:
        raise ValueError("LBL içindeki LINES değeri HDR içindeki lines ile uyuşmuyor.")

    nodata = header["nodata"]
    if nodata is None and label["missing_constant"] is not None:
        nodata = label["missing_constant"]

    return {
        **{key: value for key, value in header.items() if key != "dtype"},
        "scale_factor": label["scaling_factor"],
        "offset": label["offset"],
        "height_unit": label["unit"],
        "vertical_scale_to_m": 1.0 if label["unit"].upper() in (
    "M", "METER", "METERS", "METRE", "METRES"
) else None,

"height_unit_source": label.get(
    "unit_source",
    "PDS LBL dosyası",
),
        "nodata": nodata,
        "label_metadata": label,
        "file_size_bytes": int(file_size),
    }


def load_geotiff_dem_from_path(path: str | Path):
    """Load a single-band geographic Mars DEM from a standard GeoTIFF."""
    from src.study_area import DEFAULT_STUDY_AREA_VERTICES

    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError("GeoTIFF DEM tek bantlı olmalı.")
        if src.crs is None or not src.crs.is_geographic:
            raise ValueError("GeoTIFF CRS'i coğrafi enlem-boylam (Mars) olmalı.")
        transform = src.transform
        if not np.isclose(transform.b, 0) or not np.isclose(transform.d, 0):
            raise ValueError("Döndürülmüş GeoTIFF gridleri henüz desteklenmiyor.")
        if transform.a <= 0 or transform.e >= 0:
            raise ValueError("GeoTIFF kuzey-yukarı grid yönünde olmalı (boylam sağa, enlem yukarı artar).")
        unit = str(src.units[0] or src.tags().get("units", "")).strip()
        if not unit:
            raise ValueError("DEM yükseklik birimi GeoTIFF metadata içinde belirtilmeli.")
        vertices = []
        for longitude, latitude in DEFAULT_STUDY_AREA_VERTICES:
            longitude = (float(longitude) + 180.0) % 360.0 - 180.0
            if src.bounds.left >= 0 and src.bounds.right > 180:
                longitude %= 360.0
            vertices.append((longitude, float(latitude)))
        base_window = from_bounds(
            min(point[0] for point in vertices),
            min(point[1] for point in vertices),
            max(point[0] for point in vertices),
            max(point[1] for point in vertices),
            src.transform,
        )
        triangle_covered = not (
            base_window.col_off < 0 or base_window.row_off < 0
            or base_window.col_off + base_window.width > src.width
            or base_window.row_off + base_window.height > src.height
        )
        if triangle_covered:
            pad_x = math.ceil(base_window.width * 0.2)
            pad_y = math.ceil(base_window.height * 0.2)
            col0 = max(0, math.floor(base_window.col_off) - pad_x)
            row0 = max(0, math.floor(base_window.row_off) - pad_y)
            col1 = min(src.width, math.ceil(base_window.col_off + base_window.width) + pad_x)
            row1 = min(src.height, math.ceil(base_window.row_off + base_window.height) + pad_y)
            window = Window(col0, row0, col1 - col0, row1 - row0)
        else:
            # A regional alternate DEM may cover a different candidate area.
            window = Window(0, 0, src.width, src.height)
        data = src.read(1, window=window, masked=True).astype(np.float32).filled(np.nan)
        data = data * float(src.scales[0]) + float(src.offsets[0])
        data[~np.isfinite(data)] = np.nan
        transform = src.window_transform(window)
        radius = 3_396_190.0
        metadata = {
            "samples": data.shape[1], "lines": data.shape[0], "bands": 1,
            "dtype_name": src.dtypes[0], "data_type": src.dtypes[0],
            "byte_order": "GeoTIFF", "header_offset": 0,
            "projection_name": "Geographic Lat/Lon",
            "map_units": "degrees", "semi_major_axis": radius,
            "central_meridian": float(transform.c + transform.a * 0.5),
            "map_reference_col": 1.0, "map_reference_row": 1.0,
            "map_x": float(transform.c + transform.a * 0.5),
            "map_y": float(transform.f + transform.e * 0.5),
            "pixel_x": abs(float(transform.a)),
            "pixel_y": abs(float(transform.e)),
            "map_info_raw": str(transform), "nodata": np.nan,
            "scale_factor": 1.0, "offset": 0.0,
            "height_unit": unit,
            "vertical_scale_to_m": {
                "m": 1.0, "meter": 1.0, "meters": 1.0,
                "metre": 1.0, "metres": 1.0, "km": 1000.0,
                "ft": 0.3048, "feet": 0.3048,
            }.get(unit.lower()),
            "label_metadata": {"crs": str(src.crs), "source": "GeoTIFF"},
            "file_size_bytes": Path(path).stat().st_size,
            "source_mode": "GeoTIFF yüklemesi",
        }
    if metadata["vertical_scale_to_m"] is None:
        raise ValueError(f"GeoTIFF yükseklik birimi desteklenmiyor: {unit}")
    if data.shape[0] < 2 or data.shape[1] < 2 or not np.any(np.isfinite(data)):
        raise ValueError("GeoTIFF rasterında yeterli geçerli yükseklik verisi yok.")
    return data, metadata


def load_mola_envi(
    img_buffer: memoryview | bytes | bytearray,
    hdr_bytes: bytes | bytearray | memoryview,
    lbl_bytes: bytes | bytearray | memoryview | None = None,
):
    """Open an uploaded ENVI IMG as a zero-copy NumPy view."""
    header = parse_envi_header(hdr_bytes)
    if header["bands"] != 1:
        raise ValueError("Bu sürüm tek bantlı DEM dosyalarını destekliyor.")

    label = _parse_pds_label(lbl_bytes)
    expected_pixels = header["samples"] * header["lines"]
    expected_bytes = expected_pixels * header["dtype"].itemsize
    available_bytes = len(img_buffer) - header["header_offset"]
    if available_bytes < expected_bytes:
        raise ValueError(
            f"IMG eksik veya yanlış dosya: {expected_bytes:,} veri baytı bekleniyor, "
            f"{max(available_bytes, 0):,} bayt mevcut."
        )

    raw = np.frombuffer(
        img_buffer,
        dtype=header["dtype"],
        count=expected_pixels,
        offset=header["header_offset"],
    ).reshape((header["lines"], header["samples"]))
    metadata = _make_metadata(header, label, len(img_buffer))
    return raw, metadata


def load_mola_envi_from_path(
    img_path: str | Path,
    hdr_bytes: bytes | bytearray | memoryview,
    lbl_bytes: bytes | bytearray | memoryview | None = None,
):
    """Memory-map an IMG already downloaded to disk; raster data is not copied into RAM."""
    path = Path(img_path)
    header = parse_envi_header(hdr_bytes)
    if header["bands"] != 1:
        raise ValueError("Bu sürüm tek bantlı DEM dosyalarını destekliyor.")

    label = _parse_pds_label(lbl_bytes)
    expected_pixels = header["samples"] * header["lines"]
    expected_bytes = header["header_offset"] + expected_pixels * header["dtype"].itemsize
    file_size = path.stat().st_size
    if file_size < expected_bytes:
        raise ValueError(
            f"İndirilen IMG eksik: en az {expected_bytes:,} bayt bekleniyor, "
            f"{file_size:,} bayt bulundu."
        )

    dem = np.memmap(
        path,
        dtype=header["dtype"],
        mode="r",
        offset=header["header_offset"],
        shape=(header["lines"], header["samples"]),
        order="C",
    )
    metadata = _make_metadata(header, label, file_size)
    metadata["storage_mode"] = "memory-mapped file"
    return dem, metadata
