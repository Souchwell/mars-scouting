from __future__ import annotations

import re

import numpy as np


# Lycus Sulci'nin güneyindeki seçtiğimiz üçgen çalışma alanı.
# Koordinat sırası: (Doğu boylamı, kuzey enlemi)
# Boylam sistemi: 0-360 derece.
DEFAULT_STUDY_AREA_VERTICES = [
    (223.563, 26.797),  # A
    (221.563, 25.788),  # B
    (217.781, 22.317),  # C
]


def parse_geographic_map_info(metadata: dict) -> dict:
    projection = metadata.get("projection_name", "")

    if "equirectangular" not in projection.lower():
        raise ValueError(
            f"Desteklenmeyen projeksiyon: {projection}"
        )

    if metadata.get("map_units", "").lower() != "meters":
        raise ValueError(
            "Bu dönüşüm yalnızca metre birimli rasterları destekliyor."
        )

    radius = float(metadata["semi_major_axis"])

    if radius <= 0:
        raise ValueError("Mars referans yarıçapı geçersiz.")

    return {
        "radius": radius,
        "central_meridian": float(metadata["central_meridian"]),
        "reference_col": float(metadata["map_reference_col"]),
        "reference_row": float(metadata["map_reference_row"]),
        "map_x": float(metadata["map_x"]),
        "map_y": float(metadata["map_y"]),
        "pixel_x": float(metadata["pixel_x"]),
        "pixel_y": float(metadata["pixel_y"]),
    }


def geographic_to_pixel(
    longitude: float,
    latitude: float,
    map_info: dict,
) -> tuple[float, float]:

    radius = map_info["radius"]
    central_meridian = map_info["central_meridian"]

    delta_lon = longitude - central_meridian

    # Boylam farkını [-180, 180) aralığına getir.
    delta_lon = (delta_lon + 180.0) % 360.0 - 180.0

    x = radius * np.radians(delta_lon)
    y = radius * np.radians(latitude)

    col = (
        map_info["reference_col"] - 1
        + (x - map_info["map_x"]) / map_info["pixel_x"]
    )

    row = (
        map_info["reference_row"] - 1
        - (y - map_info["map_y"]) / map_info["pixel_y"]
    )

    return row, col


def point_in_polygon(x, y, polygon):
    """
    Ray casting yöntemiyle noktaların poligon içerisinde
    olup olmadığını belirler.

    x ve y NumPy dizileri olabilir.
    """

    inside = np.zeros(x.shape, dtype=bool)

    polygon = list(polygon)
    vertex_count = len(polygon)

    if vertex_count < 3:
        raise ValueError("Çalışma alanı en az üç köşeli olmalı.")

    for i in range(vertex_count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % vertex_count]

        crosses = (y1 > y) != (y2 > y)

        denominator = y2 - y1

        if denominator == 0:
            continue

        intersection_x = (
            (x2 - x1) * (y - y1) / denominator + x1
        )

        inside ^= crosses & (x < intersection_x)

    return inside


def extract_study_area(
    dem,
    metadata: dict,
    vertices=None,
) -> dict:
    """
    MOLA DEM içerisinden üçgen çalışma alanını keser.

    Rasterın tamamını RAM'e kopyalamaz.
    Önce yalnızca gerekli satır ve sütun aralığını okur.
    """

    if vertices is None:
        vertices = DEFAULT_STUDY_AREA_VERTICES

    vertices = [(float(lon), float(lat)) for lon, lat in vertices]

    if len(vertices) < 3:
        raise ValueError("Çalışma alanı en az üç köşe içermeli.")

    map_info = parse_geographic_map_info(metadata)

    rows_count, cols_count = dem.shape

    pixel_coordinates = [
        geographic_to_pixel(lon, lat, map_info)
        for lon, lat in vertices
    ]

    polygon_rows = [point[0] for point in pixel_coordinates]
    polygon_cols = [point[1] for point in pixel_coordinates]

    row_start = max(0, int(np.floor(min(polygon_rows))))
    row_stop = min(
        rows_count,
        int(np.ceil(max(polygon_rows))) + 1,
    )

    col_start = max(0, int(np.floor(min(polygon_cols))))
    col_stop = min(
        cols_count,
        int(np.ceil(max(polygon_cols))) + 1,
    )

    if row_start >= row_stop or col_start >= col_stop:
        raise ValueError(
            "Çalışma alanı MOLA rasterının dışında kalıyor."
        )

    if (
        min(polygon_rows) < 0
        or max(polygon_rows) >= rows_count
        or min(polygon_cols) < 0
        or max(polygon_cols) >= cols_count
    ):
        raise ValueError(
            "Üçgenin bir veya daha fazla köşesi MOLA rasterının "
            "dışında. Koordinatları ve raster kapsamını kontrol et."
        )

    # Yalnızca üçgenin çevresindeki dikdörtgen pencere okunur.
    raw_window = dem[
        row_start:row_stop,
        col_start:col_stop,
    ]

    # Pencerenin her pikselinin coğrafi koordinatını hesapla.
    rows = np.arange(row_start, row_stop)
    cols = np.arange(col_start, col_stop)

    x_values = (
    map_info["map_x"]
    + (
        cols - (map_info["reference_col"] - 1)
    ) * map_info["pixel_x"]
)

    y_values = (
    map_info["map_y"]
    + (
        map_info["reference_row"] - 1 - rows
    ) * map_info["pixel_y"]
)

    lon_values = (
    map_info["central_meridian"]
    + np.degrees(x_values / map_info["radius"])
)

    lat_values = np.degrees(y_values / map_info["radius"])

    lon_grid, lat_grid = np.meshgrid(
        lon_values,
        lat_values,
    )

    # Üçgen içerisinde kalan pikseller.
    polygon_mask = point_in_polygon(
        lon_grid,
        lat_grid,
        vertices,
    )

    # NoData ve geçersiz sayısal değerleri ayıkla.
    nodata = metadata.get("nodata")

    valid_mask = np.isfinite(raw_window)

    if nodata is not None:
        if np.isnan(nodata):
            valid_mask &= ~np.isnan(raw_window)
        else:
            valid_mask &= raw_window != nodata

    final_mask = polygon_mask & valid_mask

    # Yalnızca kesilen küçük alan float32 dizisine dönüştürülür.
    scale = float(metadata.get("scale_factor", 1.0))
    offset = float(metadata.get("offset", 0.0))

    elevation = raw_window.astype(np.float32)

    elevation *= scale
    elevation += offset

    elevation[~final_mask] = np.nan

    valid_elevations = elevation[final_mask]

    if valid_elevations.size == 0:
        raise ValueError(
            "Üçgen çalışma alanında geçerli MOLA verisi bulunamadı."
        )

    # Çalışma alanının yaklaşık coğrafi sınırları.
    bounds = {
        "min_lon": float(np.min(lon_values)),
        "max_lon": float(np.max(lon_values)),
        "min_lat": float(np.min(lat_values)),
        "max_lat": float(np.max(lat_values)),
    }

    return {
        "elevation": elevation,
        "mask": final_mask,
        "polygon_mask": polygon_mask,
        "longitude": lon_grid,
        "latitude": lat_grid,
        "vertices": vertices,
        "bounds": bounds,
        "row_start": row_start,
        "row_stop": row_stop,
        "col_start": col_start,
        "col_stop": col_stop,
        "valid_pixels": int(valid_elevations.size),
        "total_polygon_pixels": int(np.count_nonzero(polygon_mask)),
        "min_elevation": float(np.min(valid_elevations)),
        "max_elevation": float(np.max(valid_elevations)),
        "mean_elevation": float(np.mean(valid_elevations)),
        "median_elevation": float(np.median(valid_elevations)),
        "map_info": map_info,
    }