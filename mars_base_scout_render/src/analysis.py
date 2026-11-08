import math
import numpy as np


def _valid_mask(values, nodata):
    if nodata is None:
        return np.ones(values.shape, dtype=bool)
    return values != nodata


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
    sample = dem[::step, ::step].astype(np.float32)
    nodata = metadata.get("nodata")
    if nodata is not None:
        sample[sample == nodata] = np.nan
    sample *= float(metadata.get("scale_factor", 1.0))
    sample += float(metadata.get("offset", 0.0))
    return sample


def calculate_sampled_slope(dem, metadata, step=8):
    """Calculate local central-difference slope at regularly spaced sample cells.

    The derivative uses adjacent native raster pixels; `step` only controls output
    sample spacing, keeping peak memory much lower than a full-resolution gradient.
    """
    rows_count, cols_count = dem.shape
    if rows_count < 3 or cols_count < 3:
        raise ValueError("Eğim için raster en az 3x3 piksel olmalı.")

    rows = np.arange(1, rows_count - 1, step, dtype=np.int32)
    cols = np.arange(1, cols_count - 1, step, dtype=np.int32)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Seçilen örnekleme aralığı için yeterli piksel yok.")

    row_grid, col_grid = np.ix_(rows, cols)
    nodata = metadata.get("nodata")

    center = dem[row_grid, col_grid]
    left = dem[row_grid, col_grid - 1].astype(np.float32)
    right = dem[row_grid, col_grid + 1].astype(np.float32)
    up = dem[row_grid - 1, col_grid].astype(np.float32)
    down = dem[row_grid + 1, col_grid].astype(np.float32)

    valid = np.ones(center.shape, dtype=bool)
    if nodata is not None:
        valid &= center != nodata
        valid &= dem[row_grid, col_grid - 1] != nodata
        valid &= dem[row_grid, col_grid + 1] != nodata
        valid &= dem[row_grid - 1, col_grid] != nodata
        valid &= dem[row_grid + 1, col_grid] != nodata

    scale = float(metadata.get("scale_factor", 1.0))
    dz_dx = ((right - left) * scale) / (2.0 * float(metadata["pixel_x"]))
    dz_dy = ((down - up) * scale) / (2.0 * float(metadata["pixel_y"]))
    slope = np.degrees(np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)))
    slope = slope.astype(np.float32, copy=False)
    slope[~valid] = np.nan
    return slope
