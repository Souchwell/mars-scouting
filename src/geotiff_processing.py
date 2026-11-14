
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds


# Eski merkezli piksel koordinatlarından türetilen
# genişletilmiş çalışma alanı.
# Koordinatlar: (boylam, enlem), derece.

OUTER_AREA_LONLAT = [
    (-142.219, 26.797),
    (-135.341, 26.797),
    (-135.341, 20.890),
    (-142.219, 20.890),
]

LANDING_TRIANGLE_LONLAT = [
    (-136.437, 23.8435),
    (-138.437, 25.788),
    (-142.219, 22.317)
]

def inspect_geotiff(path: str | Path) -> dict:
    """GeoTIFF metadata bilgilerini RAM'e raster yüklemeden okur."""

    with rasterio.open(path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": str(src.crs),
            "transform": src.transform,
            "bounds": src.bounds,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "block_shapes": src.block_shapes,
            "is_tiled": src.is_tiled,
            "compression": str(src.compression),
            "units": src.units,
            "scales": src.scales,
            "offsets": src.offsets,
        }


def _make_polygon(vertices):
    """Koordinat listesinden GeoJSON polygon geometrisi oluşturur."""

    coordinates = list(vertices)

    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])

    return {
        "type": "Polygon",
        "coordinates": [coordinates],
    }


def _calculate_window(src, vertices):
    """Coğrafi koordinatlardan raster okuma penceresi hesaplar."""

    if not src.crs or not src.crs.is_geographic:
        raise ValueError(
            "GeoTIFF coğrafi koordinat sisteminde değil. "
            "Koordinat dönüşümü yapılmadan kesim gerçekleştirilemez."
        )

    longitudes = [point[0] for point in vertices]
    latitudes = [point[1] for point in vertices]

    bounds = from_bounds(
        min(longitudes),
        min(latitudes),
        max(longitudes),
        max(latitudes),
        src.transform,
    )

    # Pencere sınırlarını dışarı doğru yuvarla.
    col_start = math.floor(bounds.col_off)
    row_start = math.floor(bounds.row_off)

    col_end = math.ceil(bounds.col_off + bounds.width)
    row_end = math.ceil(bounds.row_off + bounds.height)

    window = Window(
        col_start,
        row_start,
        col_end - col_start,
        row_end - row_start,
    )

    # Raster sınırlarıyla kesiştir.
    raster_bounds = Window(0, 0, src.width, src.height)
    window = window.intersection(raster_bounds)

    if window.width <= 0 or window.height <= 0:
        raise ValueError("Çalışma alanı GeoTIFF sınırlarının dışında.")

    return window


def crop_geotiff_study_area(
    source_path: str | Path,
    output_path: str | Path,
    vertices=OUTER_AREA_LONLAT,
    progress_callback=None,
) -> dict:
    """
    Küresel GeoTIFF içerisinden çalışma alanını çıkarır.

    Kaynak rasterın tamamını RAM'e yüklemez.
    Yalnızca gerekli raster penceresini okur.
    """

    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Kaynak GeoTIFF bulunamadı: {source_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(vertices) < 3:
        raise ValueError("Çalışma alanı en az üç köşe içermeli.")

    with rasterio.Env(GDAL_CACHEMAX=128):

        with rasterio.open(source_path) as src:

            if src.count != 1:
                raise ValueError(
                    "Şu anda yalnızca tek bantlı DEM destekleniyor."
                )

            if src.dtypes[0] != "int16":
                raise ValueError(
                    f"Beklenmeyen veri tipi: {src.dtypes[0]}"
                )

            if src.nodata is None:
                raise ValueError(
                    "Kaynak DEM'de NoData değeri tanımlanmamış."
                )

            window = _calculate_window(src, vertices)

            height = int(window.height)
            width = int(window.width)

            # Yalnızca gerekli bölgeyi oku.
            dem = src.read(
                1,
                window=window,
                out_dtype="int16",
            )

            # Kesilmiş alanın gerçek affine transformu.
            output_transform = src.window_transform(window)

            polygon = _make_polygon(vertices)

            # Dörtgen dışındaki pikselleri maskele.
            inside_mask = geometry_mask(
                [polygon],
                out_shape=(height, width),
                transform=output_transform,
                invert=True,
                all_touched=False,
            )

            nodata = np.int16(src.nodata)

            dem[~inside_mask] = nodata

            valid_pixels = int(np.count_nonzero(
                inside_mask & (dem != nodata)
            ))

            if valid_pixels == 0:
                raise ValueError(
                    "Çalışma alanında geçerli DEM verisi bulunamadı."
                )

            # Kaynak metadata bilgilerini koru.
            profile = src.profile.copy()

            profile.update({
                "driver": "GTiff",
                "width": width,
                "height": height,
                "count": 1,
                "dtype": "int16",
                "transform": output_transform,
                "nodata": nodata,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "compress": "DEFLATE",
                "predictor": 2,
                "BIGTIFF": "IF_SAFER",
            })

            # Önce geçici dosyaya yaz.
            fd, temp_name = tempfile.mkstemp(
                prefix="mars_dem_",
                suffix=".tif",
                dir=output_path.parent,
            )
            os.close(fd)

            try:
                with rasterio.open(
                    temp_name,
                    "w",
                    **profile,
                ) as dst:

                    dst.write(dem, 1)

                    dst.update_tags(
                        AREA_NAME="Mars expanded study area",
                        SOURCE_FILE=source_path.name,
                    )

                    # Küçük ölçekli görüntüleme için overview.
                    overview_levels = [
                        factor
                        for factor in [2, 4, 8, 16]
                        if min(width, height) // factor >= 1
                    ]

                    if overview_levels:
                        dst.build_overviews(
                            overview_levels,
                            Resampling.average,
                        )

                        dst.update_tags(
                            ns="rio_overview",
                            resampling="average",
                        )

                os.replace(temp_name, output_path)

            except Exception:
                if os.path.exists(temp_name):
                    os.remove(temp_name)
                raise

            if progress_callback:
                progress_callback(1.0)

            return {
                "output_path": str(output_path),
                "width": width,
                "height": height,
                "valid_pixels": valid_pixels,
                "total_pixels": width * height,
                "valid_percent": (
                    valid_pixels / (width * height) * 100
                ),
                "bounds": src.window_bounds(window),
                "crs": str(src.crs),
                "nodata": int(nodata),
            }
