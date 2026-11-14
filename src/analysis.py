import math
import numpy as np


def _valid_mask(values, nodata):
    valid = np.isfinite(values)
    if nodata is None:
        return valid
    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
        return valid
    return valid & (values != nodata)


def get_dem_statistics(dem, metadata, chunk_pixels=1_000_000, median_step=16):
    """Exact min/max/mean/std in chunks; approximate median from a regular sample."""
    flat = dem.reshape(-1)
    nodata = metadata.get("nodata")
    scale = float(metadata.get("scale_factor", 1.0))
    offset = float(metadata.get("offset", 0.0))
    count = 0
    minimum = math.inf
    maximum = -math.inf
    total = 0.0
    total_sq = 0.0
    for start in range(0, flat.size, chunk_pixels):
        part = flat[start:start + chunk_pixels]
        valid = _valid_mask(part, nodata)
        if not np.any(valid):
            continue
        values = part[valid].astype(np.float64, copy=False)
        if scale != 1.0:
            values *= scale
        if offset != 0.0:
            values += offset
        count += int(values.size)
        minimum = min(minimum, float(np.min(values)))
        maximum = max(maximum, float(np.max(values)))
        total += float(np.sum(values, dtype=np.float64))
        total_sq += float(np.sum(values * values, dtype=np.float64))
    if count == 0:
        raise ValueError("Rasterda geçerli piksel bulunamadı. NoData değerini kontrol et.")

    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    median_source = dem[::median_step, ::median_step]
    median_valid = _valid_mask(median_source, nodata)
    median_values = median_source[median_valid].astype(np.float32)
    if scale != 1.0:
        median_values *= scale
    if offset != 0.0:
        median_values += offset
    median_estimate = float(np.median(median_values)) if median_values.size else float("nan")
    return {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "median_sampled": median_estimate,
        "std": math.sqrt(variance),
        "valid_pixels": count,
        "valid_percent": count / flat.size * 100.0,
        "median_sample_step": median_step,
    }


def make_display_sample(dem, metadata, step=8):
    if step < 1:
        raise ValueError("Örnekleme aralığı en az 1 olmalıdır.")
    sample = dem[::step, ::step].astype(np.float32)
    nodata = metadata.get("nodata")
    if nodata is not None and not (
        isinstance(nodata, (float, np.floating)) and np.isnan(nodata)
    ):
        sample[sample == nodata] = np.nan
    sample[~np.isfinite(sample)] = np.nan
    sample *= float(metadata.get("scale_factor", 1.0))
    sample += float(metadata.get("offset", 0.0))
    return sample


def calculate_sampled_slope(dem, metadata, step=1):
    """
    Mars DEM verisinden eğim hesaplar.
    Çıktı boyutu make_display_sample() ile aynıdır.
    """

    if step < 1:
        raise ValueError("Örnekleme aralığı en az 1 olmalıdır.")

    rows_count, cols_count = dem.shape

    if rows_count < 2 or cols_count < 2:
        raise ValueError("Eğim için raster en az 2x2 piksel olmalıdır.")

    # Yükseklik birimini metreye dönüştür.
    unit = str(metadata.get("height_unit", "")).strip().upper()

    meter_units = {"M", "METER", "METERS", "METRE", "METRES"}
    feet_units = {"FT", "FOOT", "FEET"}

    if unit in meter_units:
        vertical_to_m = 1.0
    elif unit in feet_units:
        vertical_to_m = 0.3048
    elif metadata.get("vertical_scale_to_m") is not None:
        vertical_to_m = float(metadata["vertical_scale_to_m"])
    else:
        raise ValueError(
            "DEM yüksekliğinin birimi doğrulanamadı. "
            f"Metadata birimi: {unit or 'belirtilmemiş'}."
        )

    scale = float(metadata.get("scale_factor", 1.0))
    offset = float(metadata.get("offset", 0.0))
    nodata = metadata.get("nodata")

    # Örnekleme koordinatları.
    rows = np.arange(0, rows_count, step, dtype=np.int64)
    cols = np.arange(0, cols_count, step, dtype=np.int64)

    # Komşu piksel indeksleri.
    left_cols = np.maximum(cols - 1, 0)
    right_cols = np.minimum(cols + 1, cols_count - 1)

    up_rows = np.maximum(rows - 1, 0)
    down_rows = np.minimum(rows + 1, rows_count - 1)

    # İki boyutlu indeksleme.
    center_raw = dem[np.ix_(rows, cols)]
    left_raw = dem[np.ix_(rows, left_cols)]
    right_raw = dem[np.ix_(rows, right_cols)]
    up_raw = dem[np.ix_(up_rows, cols)]
    down_raw = dem[np.ix_(down_rows, cols)]

    # NoData kontrolü.
    valid = np.ones(center_raw.shape, dtype=bool)

    if nodata is not None:
        if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
            valid &= ~np.isnan(center_raw)
            valid &= ~np.isnan(left_raw)
            valid &= ~np.isnan(right_raw)
            valid &= ~np.isnan(up_raw)
            valid &= ~np.isnan(down_raw)
        else:
            valid &= center_raw != nodata
            valid &= left_raw != nodata
            valid &= right_raw != nodata
            valid &= up_raw != nodata
            valid &= down_raw != nodata
    valid &= (
        np.isfinite(center_raw)
        & np.isfinite(left_raw)
        & np.isfinite(right_raw)
        & np.isfinite(up_raw)
        & np.isfinite(down_raw)
    )

    # Yükseklikleri metreye dönüştür.
    center = (center_raw.astype(np.float64) * scale + offset) * vertical_to_m
    left = (left_raw.astype(np.float64) * scale + offset) * vertical_to_m
    right = (right_raw.astype(np.float64) * scale + offset) * vertical_to_m
    up = (up_raw.astype(np.float64) * scale + offset) * vertical_to_m
    down = (down_raw.astype(np.float64) * scale + offset) * vertical_to_m

    # Piksel aralıkları.
    x_pixel_span = right_cols - left_cols
    y_pixel_span = down_rows - up_rows

    projection_name = str(
        metadata.get("projection_name", "")
    ).lower()

    map_units = str(
        metadata.get("map_units") or ""
    ).lower()

    is_geographic = (
        "geographic" in projection_name
        or "lat/lon" in projection_name
        or "degree" in map_units
        or "degrees" in map_units
    )

    pixel_x = abs(float(metadata["pixel_x"]))
    pixel_y = abs(float(metadata["pixel_y"]))

    if is_geographic:
        radius = float(metadata["semi_major_axis"])

        reference_row = float(metadata["map_reference_row"])
        map_y = float(metadata["map_y"])

        latitude = map_y - (
            rows - (reference_row - 1.0)
        ) * float(metadata["pixel_y"])

        latitude_rad = np.radians(latitude)

        dx_per_pixel = (
            radius
            * np.cos(latitude_rad)
            * np.radians(pixel_x)
        )

        dy_per_pixel = radius * np.radians(pixel_y)

        dx = (
            dx_per_pixel[:, np.newaxis]
            * x_pixel_span[np.newaxis, :]
        )

        dy = (
            dy_per_pixel
            * y_pixel_span[:, np.newaxis]
        )

    else:
        if map_units and not any(
            name in map_units
            for name in ("meter", "metre")
        ):
            raise ValueError(
                f"Desteklenmeyen yatay koordinat birimi: {map_units}"
            )

        dx = pixel_x * x_pixel_span[np.newaxis, :]
        dy = pixel_y * y_pixel_span[:, np.newaxis]

    if np.any(dx <= 0) or np.any(dy <= 0):
        raise ValueError("Piksel mesafesi sıfır veya negatif olamaz.")

    # Eğim hesaplama.
    dz_dx = (right - left) / dx
    dz_dy = (down - up) / dy

    slope = np.degrees(
        np.arctan(
            np.sqrt(dz_dx ** 2 + dz_dy ** 2)
        )
    )

    slope = slope.astype(np.float32)
    slope[~valid] = np.nan

    return slope
