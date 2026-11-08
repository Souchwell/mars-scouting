import re
from typing import Any

import numpy as np


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


def _parse_map_info(text: str) -> tuple[float, float, str | None]:
    match = re.search(r"(?im)^\s*map info\s*=\s*\{([^}]*)\}", text)
    if not match:
        return 463.0576672, 463.0576672, None

    raw = match.group(1)
    parts = [part.strip() for part in raw.split(",")]
    pixel_x = pixel_y = 463.0576672
    if len(parts) >= 7:
        try:
            pixel_x = abs(float(parts[5]))
            pixel_y = abs(float(parts[6]))
        except ValueError as exc:
            raise ValueError("ENVI map info içindeki piksel boyutu okunamadı.") from exc
    if pixel_x <= 0 or pixel_y <= 0:
        raise ValueError("ENVI başlığındaki piksel boyutu sıfır veya negatif olamaz.")
    return pixel_x, pixel_y, raw


def _parse_envi(hdr_bytes: bytes | bytearray | memoryview) -> dict[str, Any]:
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
        raise ValueError("ENVI byte order yalnızca 0 (Little Endian) veya 1 (Big Endian) olabilir.")
    if samples <= 0 or lines <= 0 or bands <= 0 or header_offset < 0:
        raise ValueError("HDR içindeki raster boyutu veya header offset geçersiz.")

    code, dtype_name = envi_types[data_type]
    dtype = np.dtype("u1") if data_type == 1 else np.dtype((">" if byte_order == 1 else "<") + code)
    pixel_x, pixel_y, map_info_raw = _parse_map_info(text)

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
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "map_info_raw": map_info_raw,
    }


def _parse_pds_label(lbl_bytes: bytes | bytearray | memoryview | None) -> dict[str, Any]:
    if not lbl_bytes:
        return {
            "present": False,
            "scaling_factor": 1.0,
            "offset": 0.0,
            "missing_constant": None,
            "unit": "etikette belirtilmemiş",
            "sample_type": None,
            "sample_bits": None,
            "line_samples": None,
            "lines": None,
        }

    text = _decode(lbl_bytes)
    unit = _label_text_value(text, "UNIT") or _label_text_value(text, "UNITS")
    if unit:
        unit = unit.upper()
    else:
        unit = "etikette belirtilmemiş"

    missing_raw = _label_text_value(text, "MISSING_CONSTANT")
    try:
        missing_constant = float(missing_raw) if missing_raw is not None else None
    except ValueError:
        missing_constant = None

    bits_raw = _label_text_value(text, "SAMPLE_BITS")
    samples_raw = _label_text_value(text, "LINE_SAMPLES")
    lines_raw = _label_text_value(text, "LINES")

    return {
        "present": True,
        "scaling_factor": _numeric_label_value(text, "SCALING_FACTOR", 1.0),
        "offset": _numeric_label_value(text, "OFFSET", 0.0),
        "missing_constant": missing_constant,
        "unit": unit,
        "sample_type": _label_text_value(text, "SAMPLE_TYPE"),
        "sample_bits": int(bits_raw) if bits_raw and bits_raw.isdigit() else None,
        "line_samples": int(samples_raw) if samples_raw and samples_raw.isdigit() else None,
        "lines": int(lines_raw) if lines_raw and lines_raw.isdigit() else None,
    }


def load_mola_envi(
    img_buffer: memoryview | bytes | bytearray,
    hdr_bytes: bytes | bytearray | memoryview,
    lbl_bytes: bytes | bytearray | memoryview | None = None,
):
    """Create a zero-copy NumPy view over an uploaded ENVI IMG raster."""
    header = _parse_envi(hdr_bytes)
    if header["bands"] != 1:
        raise ValueError("Bu sürüm tek bantlı DEM dosyalarını destekliyor.")

    label = _parse_pds_label(lbl_bytes)
    if label["line_samples"] and label["line_samples"] != header["samples"]:
        raise ValueError("LBL içindeki LINE_SAMPLES değeri HDR içindeki samples ile uyuşmuyor.")
    if label["lines"] and label["lines"] != header["lines"]:
        raise ValueError("LBL içindeki LINES değeri HDR içindeki lines ile uyuşmuyor.")

    expected_pixels = header["samples"] * header["lines"]
    expected_bytes = expected_pixels * header["dtype"].itemsize
    available_bytes = len(img_buffer) - header["header_offset"]
    if available_bytes < expected_bytes:
        raise ValueError(
            f"IMG eksik veya yanlış dosya: {expected_bytes:,} veri baytı bekleniyor, "
            f"{max(available_bytes, 0):,} bayt mevcut."
        )

    # UploadedFile.getbuffer() bir memoryview verir. frombuffer büyük IMG'yi
    # ikinci kez RAM'e kopyalamadan raster görünümü oluşturur.
    raw = np.frombuffer(
        img_buffer,
        dtype=header["dtype"],
        count=expected_pixels,
        offset=header["header_offset"],
    ).reshape((header["lines"], header["samples"]))

    nodata = header["nodata"]
    if nodata is None and label["missing_constant"] is not None:
        nodata = label["missing_constant"]

    metadata = {
        **{key: value for key, value in header.items() if key != "dtype"},
        "scale_factor": label["scaling_factor"],
        "offset": label["offset"],
        "height_unit": label["unit"],
        "nodata": nodata,
        "label_metadata": label,
        "file_size_bytes": len(img_buffer),
    }
    return raw, metadata
