
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


# Mars 2000 Sphere yarıçapı (metre)
MARS_RADIUS_M = 3_396_190.0


@dataclass(frozen=True)
class TerrainCostConfig:
    """
    İlk prototip için geçici rover maliyet parametreleri.

    Bu değerler NASA tarafından onaylanmış operasyonel
    güvenlik limitleri değildir. Rover özellikleri ve
    daha yüksek çözünürlüklü arazi verileriyle kalibre edilmelidir.
    """

    # Eğim
    slope_limit_deg: float = 10.0
    slope_scale_deg: float = 5.0

    # 3x3 komşulukta merkez pikselin yerel düzlemden
    # sapmasını temsil eden pürüzlülük göstergesi
    roughness_limit_m: float = 5.0
    roughness_scale_m: float = 5.0

    # Dört komşu hücre arasındaki en büyük yükseklik farkı
    step_limit_m: float = 10.0
    step_scale_m: float = 8.0

    # Kaynak ürün için bildirilen minimum yükseklik
    # belirsizliği. Bu değer doğrudan 1-sigma değildir.
    dem_vertical_uncertainty_m: float = 3.0

    # Belirsizlik payı ayrıca doğrulanana kadar sıfır.
    # Hata korelasyonu bilinmeden ±3 m'yi her piksele
    # bağımsız hata olarak uygulamak doğru değildir.
    step_uncertainty_margin_m: float = 0.0

    # Maliyet ağırlıkları
    slope_weight: float = 0.55
    roughness_weight: float = 0.25
    step_weight: float = 0.20


def load_geotiff_dem(path: str | Path):
    """Yerel GeoTIFF DEM'i ve coğrafi metadata bilgilerini yükler."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"DEM bulunamadı: {path}")

    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError("Tek bantlı DEM bekleniyor.")

        raw = src.read(1, masked=True).astype(np.float64)
        raw = raw.filled(np.nan)
        raw = raw * float(src.scales[0]) + float(src.offsets[0])
        unit = str(src.units[0] or "").strip().lower()
        unit_scales = {
            "m": 1.0, "meter": 1.0, "meters": 1.0,
            "metre": 1.0, "metres": 1.0,
            "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
            "kilometre": 1000.0, "kilometres": 1000.0,
            "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
        }
        if unit not in unit_scales:
            raise ValueError(
                f"DEM yükseklik birimi metreye çevrilemiyor: {unit or 'belirtilmemiş'}."
            )
        dem = (raw * unit_scales[unit]).astype(np.float32)

        metadata = {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "bounds": src.bounds,
            "unit": "m",
            "source_unit": unit,
        }

    if metadata["crs"] is None or not metadata["crs"].is_geographic:
        raise ValueError("Mars coğrafi koordinat sistemi bekleniyor.")

    return dem, metadata


def calculate_ground_pixel_size(metadata):
    """
    Her raster satırında doğu-batı ve kuzey-güney
    piksel mesafesini Mars yarıçapıyla hesaplar.
    """

    transform = metadata["transform"]
    height = metadata["height"]

    if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
        raise ValueError("Döndürülmüş raster gridleri henüz desteklenmiyor.")

    latitudes = (
        transform.f
        + (np.arange(height) + 0.5) * transform.e
    )

    delta_lon = np.radians(abs(transform.a))
    delta_lat = np.radians(abs(transform.e))

    dx = (
        MARS_RADIUS_M
        * np.cos(np.radians(latitudes))
        * delta_lon
    )

    dy = MARS_RADIUS_M * delta_lat

    return dx.astype(np.float32), float(dy)


def calculate_slope(dem, metadata):
    """
    Merkezi fark yöntemiyle eğim açısını hesaplar.

    Yatay mesafeler metre cinsindendir.
    Dikey DEM değerlerinin metre olduğu varsayımı
    ürün metadata doğrulamasına bağlıdır.
    """

    height, width = dem.shape
    dx, dy = calculate_ground_pixel_size(metadata)

    slope = np.full(dem.shape, np.nan, dtype=np.float32)

    if height < 3 or width < 3:
        raise ValueError("Eğim hesabı için DEM en az 3x3 olmalı.")

    center = dem[1:-1, 1:-1]
    left = dem[1:-1, :-2]
    right = dem[1:-1, 2:]
    up = dem[:-2, 1:-1]
    down = dem[2:, 1:-1]

    valid = (
        np.isfinite(center)
        & np.isfinite(left)
        & np.isfinite(right)
        & np.isfinite(up)
        & np.isfinite(down)
    )

    dz_dx = (
        (right - left)
        / (2.0 * dx[1:-1, None])
    )

    dz_dy = (down - up) / (2.0 * dy)

    local_slope = np.degrees(
        np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    )

    target = slope[1:-1, 1:-1]
    target[valid] = local_slope[valid]

    return slope



def calculate_local_roughness(dem: np.ndarray) -> np.ndarray:
    """
    3x3 komşulukta yerel düzlemden sapmanın RMS değerini hesaplar.

    Sonuç metre cinsinden yerel pürüzlülük göstergesidir.
    Düzlemsel eğim etkisini azaltmak için her pencereye
    en küçük kareler yöntemiyle bir düzlem uydurulur.

    Pencere içindeki 9 pikselin tamamı geçerli olmalıdır.
    Kenar pikselleri NaN bırakılır.
    """

    from numpy.lib.stride_tricks import sliding_window_view

    dem = np.asarray(dem, dtype=np.float64)

    height, width = dem.shape
    roughness = np.full(dem.shape, np.nan, dtype=np.float64)

    if height < 3 or width < 3:
        return roughness

    windows = sliding_window_view(dem, (3, 3))

    # Yalnızca dokuz pikselin tamamı geçerli olan pencereler
    valid = np.all(np.isfinite(windows), axis=(-2, -1))

    # Pencere ortalaması
    mean_z = np.mean(windows, axis=(-2, -1))

    # Yüksekliklerin karelerinin ortalaması
    mean_z2 = np.mean(windows ** 2, axis=(-2, -1))

    # x yönündeki doğrusal değişim
    sum_xz = (
        np.sum(windows[:, :, :, 2], axis=2)
        - np.sum(windows[:, :, :, 0], axis=2)
    )

    # y yönündeki doğrusal değişim
    sum_yz = (
        np.sum(windows[:, :, 2, :], axis=2)
        - np.sum(windows[:, :, 0, :], axis=2)
    )

    # 3x3 pencerede sum(x²) = sum(y²) = 6
    residual_variance = (
        mean_z2
        - mean_z ** 2
        - (sum_xz ** 2) / 54.0
        - (sum_yz ** 2) / 54.0
    )

    residual_variance = np.maximum(residual_variance, 0.0)

    local_roughness = np.sqrt(residual_variance)

    local_roughness[~valid] = np.nan

    # Sonuçları orijinal DEM boyutuna yerleştir
    roughness[1:-1, 1:-1] = local_roughness

    return roughness



def calculate_neighbor_height_change(dem):
    """Dört ana yöndeki en büyük komşu yükseklik farkı."""

    height, width = dem.shape
    result = np.full(dem.shape, np.nan, dtype=np.float32)

    center = dem[1:-1, 1:-1]
    left = dem[1:-1, :-2]
    right = dem[1:-1, 2:]
    up = dem[:-2, 1:-1]
    down = dem[2:, 1:-1]

    valid = (
        np.isfinite(center)
        & np.isfinite(left)
        & np.isfinite(right)
        & np.isfinite(up)
        & np.isfinite(down)
    )

    difference = np.maximum.reduce([
        np.abs(center - left),
        np.abs(center - right),
        np.abs(center - up),
        np.abs(center - down),
    ])

    target = result[1:-1, 1:-1]
    target[valid] = difference[valid]

    return result


def soft_threshold_cost(
    values,
    threshold,
    scale,
    uncertainty_margin=0.0,
):
    """
    Eşik çevresinde yumuşak geçişli, 0-1 arası maliyet.

    Eşiğin altında maliyet sıfırdır.
    Eşik üzerinde maliyet sürekli ve yumuşak artar.
    """

    if scale <= 0:
        raise ValueError("scale pozitif olmalı.")

    excess = np.maximum(
        values + uncertainty_margin - threshold,
        0.0,
    )

    normalized = excess / scale

    return 1.0 - np.exp(-0.5 * normalized**2)


def calculate_terrain_costs(
    dem,
    metadata,
    config=TerrainCostConfig(),
):
    """Arazi analiz katmanlarını ve toplam maliyet haritasını üretir."""

    weights = np.array([
        config.slope_weight,
        config.roughness_weight,
        config.step_weight,
    ], dtype=np.float64)

    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Maliyet ağırlıkları geçersiz.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Maliyet ağırlıkları sonlu sayılar olmalıdır.")
    thresholds_and_scales = np.asarray([
        config.slope_limit_deg, config.slope_scale_deg,
        config.roughness_limit_m, config.roughness_scale_m,
        config.step_limit_m, config.step_scale_m,
        config.dem_vertical_uncertainty_m,
        config.step_uncertainty_margin_m,
    ], dtype=float)
    if not np.all(np.isfinite(thresholds_and_scales)):
        raise ValueError("Arazi maliyet ayarları sonlu sayılar olmalıdır.")
    if min(config.slope_scale_deg, config.roughness_scale_m, config.step_scale_m) <= 0:
        raise ValueError("Maliyet ölçekleri pozitif olmalıdır.")

    weights /= weights.sum()

    slope = calculate_slope(dem, metadata)
    roughness = calculate_local_roughness(dem)
    height_change = calculate_neighbor_height_change(dem)

    slope_cost = soft_threshold_cost(
        slope,
        config.slope_limit_deg,
        config.slope_scale_deg,
    )

    roughness_cost = soft_threshold_cost(
        roughness,
        config.roughness_limit_m,
        config.roughness_scale_m,
    )

    step_cost = soft_threshold_cost(
        height_change,
        config.step_limit_m,
        config.step_scale_m,
        config.step_uncertainty_margin_m,
    )

    valid = (
        np.isfinite(slope)
        & np.isfinite(roughness)
        & np.isfinite(height_change)
    )

    total_cost = np.full(dem.shape, np.nan, dtype=np.float32)

    combined = (
        weights[0] * slope_cost
        + weights[1] * roughness_cost
        + weights[2] * step_cost
    )

    total_cost[valid] = combined[valid]

    return {
        "elevation": dem,
        "slope": slope,
        "roughness": roughness,
        "height_change": height_change,
        "slope_cost": slope_cost,
        "roughness_cost": roughness_cost,
        "step_cost": step_cost,
        "total_cost": total_cost,
        "valid_mask": valid,
        "config": config,
    }


def analyze_geotiff(path, config=TerrainCostConfig()):
    """DEM yükleme ve bütün arazi analizini tek çağrıda çalıştırır."""

    dem, metadata = load_geotiff_dem(path)
    layers = calculate_terrain_costs(dem, metadata, config)

    return layers, metadata
