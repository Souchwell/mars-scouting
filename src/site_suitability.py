"""Soft, uncertainty-aware site screening from terrain and atmosphere layers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SiteSuitabilityConfig:
    """Illustrative screening ranges; they are not mission safety limits."""

    slope_ideal_deg: tuple[float, float] = (0.0, 10.0)
    slope_transition_deg: float = 3.0
    roughness_ideal_m: tuple[float, float] = (0.0, 2.0)
    roughness_transition_m: float = 1.5
    temperature_ideal_k: tuple[float, float] = (180.0, 260.0)
    temperature_transition_k: float = 20.0
    wind_ideal_m_s: tuple[float, float] = (0.0, 10.0)
    wind_transition_m_s: float = 5.0
    dust_ideal: tuple[float, float] = (0.0, 0.5)
    dust_transition: float = 0.5
    minimum_samples: int = 16
    weights: tuple[float, float, float, float, float] = (0.25, 0.20, 0.20, 0.20, 0.15)


def soft_range_cost(values, ideal_range, transition_scale):
    """Return continuous [0,1] cost, rising smoothly outside an ideal interval."""
    low, high = map(float, ideal_range)
    if not np.isfinite([low, high, transition_scale]).all() or low > high or transition_scale <= 0:
        raise ValueError("İdeal aralık veya geçiş ölçeği geçersiz.")
    array = np.asarray(values, dtype=float)
    below = np.maximum(low - array, 0.0) / transition_scale
    above = np.maximum(array - high, 0.0) / transition_scale
    cost = 1.0 - np.exp(-0.5 * (below**2 + above**2))
    return np.where(np.isfinite(array), cost, np.nan)


def _summarize(values, ideal_range, transition_scale, minimum_samples):
    if values is None:
        return {"cost": 0.5, "coverage": 0.0, "valid_samples": 0, "total_samples": 0, "available": False}
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"cost": 0.5, "coverage": 0.0, "valid_samples": 0, "total_samples": 0, "available": False}
    valid = array[np.isfinite(array)]
    count = int(valid.size)
    coverage = count / array.size
    if not count:
        return {"cost": 0.5, "coverage": 0.0, "valid_samples": 0, "total_samples": int(array.size), "available": False}
    costs = soft_range_cost(valid, ideal_range, transition_scale)
    # The 90th percentile preserves sensitivity to hazardous tails without a hard cutoff.
    cost = float(np.percentile(costs, 90))
    adequacy = min(1.0, np.sqrt(count / max(1, minimum_samples)))
    return {"cost": cost, "coverage": float(coverage * adequacy), "valid_samples": count, "total_samples": int(array.size), "available": True}


def calculate_site_suitability(
    slope_deg=None,
    roughness_m=None,
    surface_temperature_k=None,
    wind_speed_m_s=None,
    dust_optical_depth=None,
    config: SiteSuitabilityConfig = SiteSuitabilityConfig(),
):
    """Combine five site factors into one soft screening score (0–100).

    Atmospheric cost combines wind speed and dust opacity. Missing factors use a
    neutral cost while reducing reliability, so missing values cannot imply safety.
    """
    if len(config.weights) != 5 or not np.isfinite(config.weights).all() or min(config.weights) < 0 or sum(config.weights) <= 0:
        raise ValueError("Beş değişken için ağırlıklar geçersiz.")
    minimum = int(config.minimum_samples)
    if minimum < 1:
        raise ValueError("minimum_samples en az 1 olmalı.")

    parts = {
        "slope": _summarize(slope_deg, config.slope_ideal_deg, config.slope_transition_deg, minimum),
        "roughness": _summarize(roughness_m, config.roughness_ideal_m, config.roughness_transition_m, minimum),
        "thermal": _summarize(surface_temperature_k, config.temperature_ideal_k, config.temperature_transition_k, minimum),
        "wind": _summarize(wind_speed_m_s, config.wind_ideal_m_s, config.wind_transition_m_s, minimum),
        "dust": _summarize(dust_optical_depth, config.dust_ideal, config.dust_transition, minimum),
    }
    atmospheric_available = [parts[k] for k in ("wind", "dust") if parts[k]["available"]]
    atmospheric = {
        "cost": float(np.mean([part["cost"] for part in atmospheric_available])) if atmospheric_available else 0.5,
        # Missing wind or dust is itself a coverage gap, even if the other exists.
        "coverage": float(np.mean([parts["wind"]["coverage"], parts["dust"]["coverage"]])),
        "available": bool(atmospheric_available),
    }
    groups = {
        "slope": parts["slope"], "roughness": parts["roughness"],
        "thermal": parts["thermal"], "atmospheric": atmospheric,
    }
    reliability = float(np.mean([group["coverage"] for group in groups.values()]))
    scores = {name: (1.0 - group["cost"]) * 100.0 for name, group in groups.items()}
    scores["data_reliability"] = reliability * 100.0
    weights = np.asarray(config.weights, dtype=float)
    weights /= weights.sum()
    ordered_scores = [scores["slope"], scores["roughness"], scores["thermal"], scores["atmospheric"], scores["data_reliability"]]
    return {
        "overall_score": float(np.dot(weights, ordered_scores)),
        "component_scores": scores,
        "data_reliability": reliability * 100.0,
        "coverage": {
            **{name: group["coverage"] * 100.0 for name, group in groups.items()},
            "wind": parts["wind"]["coverage"] * 100.0,
            "dust": parts["dust"]["coverage"] * 100.0,
        },
        "sample_counts": {
            name: {
                "valid": group.get("valid_samples", 0),
                "total": group.get("total_samples", 0),
            }
            for name, group in parts.items()
        },
        "available": {name: group["available"] for name, group in groups.items()},
        "config": config,
    }


def regrid_geographic_field(values, longitudes, latitudes, target_longitudes, target_latitudes, values_order="latlon"):
    """Linearly sample a regular geographic field onto a target lat/lon grid."""
    from scipy.interpolate import RegularGridInterpolator

    data = np.squeeze(np.asarray(values, dtype=float))
    lon = ((np.asarray(longitudes, dtype=float) + 180.0) % 360.0) - 180.0
    lat = np.asarray(latitudes, dtype=float)
    latlon_shape = (len(lat), len(lon))
    lonlat_shape = (len(lon), len(lat))
    if data.shape == latlon_shape == lonlat_shape:
        if values_order == "lonlat":
            data = data.T
    elif data.shape == latlon_shape:
        pass
    elif data.shape == lonlat_shape:
        data = data.T
    else:
        raise ValueError(
            f"Kaynak grid {data.shape}; koordinatlar {latlon_shape} (enlem-boylam) "
            f"veya {lonlat_shape} (boylam-enlem) bekliyor."
        )
    lat_order, lon_order = np.argsort(lat), np.argsort(lon)
    lat, lon = lat[lat_order], lon[lon_order]
    data = data[np.ix_(lat_order, lon_order)]
    lat, lat_indices = np.unique(lat, return_index=True)
    lon, lon_indices = np.unique(lon, return_index=True)
    data = data[np.ix_(lat_indices, lon_indices)]
    if len(lat) < 2 or len(lon) < 2:
        return np.full(np.shape(target_latitudes), np.nan, dtype=float)
    finite = np.isfinite(data)
    points = np.column_stack((np.asarray(target_latitudes).ravel(), ((np.asarray(target_longitudes).ravel() + 180) % 360) - 180))
    numerator = RegularGridInterpolator(
        (lat, lon), np.where(finite, data, 0.0), bounds_error=False, fill_value=np.nan
    )(points)
    weights = RegularGridInterpolator(
        (lat, lon), finite.astype(float), bounds_error=False, fill_value=np.nan
    )(points)
    result = np.divide(numerator, weights, out=np.full_like(numerator, np.nan), where=weights > 0.999)
    return result.reshape(np.shape(target_latitudes))


def rank_landing_sites(
    slope_deg,
    roughness_m,
    surface_temperature_k,
    wind_speed_m_s,
    dust_optical_depth,
    polygon_mask,
    longitude_grid,
    latitude_grid,
    config: SiteSuitabilityConfig = SiteSuitabilityConfig(),
    maximum_sites: int = 10,
    minimum_separation_km: float = 2.0,
    resolution_factors=(1.0, 1.0, 1.0, 1.0, 1.0),
):
    """Create a soft-cost suitability raster and spatially separated landing candidates."""
    mask = np.asarray(polygon_mask, dtype=bool)
    shape = mask.shape

    def matching(values):
        if values is None:
            return np.full(shape, np.nan, dtype=float)
        array = np.asarray(values, dtype=float)
        if array.shape != shape:
            raise ValueError(f"Katman şekli {array.shape}; hedef grid {shape} olmalı.")
        return array

    slope = matching(slope_deg)
    roughness = matching(roughness_m)
    thermal = matching(surface_temperature_k)
    wind = matching(wind_speed_m_s)
    dust = matching(dust_optical_depth)
    available = [np.isfinite(v) for v in (slope, roughness, thermal, wind, dust)]
    factors = np.asarray(resolution_factors, dtype=float)
    if factors.shape != (5,) or not np.all(np.isfinite(factors)) or np.any((factors < 0) | (factors > 1)):
        raise ValueError("Her katman için 0–1 aralığında grid güvenilirlik katsayısı gerekli.")
    # Match the summary metric: terrain, thermal, and atmosphere are four
    # groups. Wind and dust share one atmospheric group so that atmosphere is
    # not counted twice in candidate-level reliability.
    layer_reliability = np.stack([
        np.asarray(valid, dtype=float) * factors[index]
        for index, valid in enumerate(available)
    ], axis=0)
    atmospheric_reliability = np.mean(layer_reliability[3:5], axis=0)
    reliability = np.mean(np.stack([
        layer_reliability[0],
        layer_reliability[1],
        layer_reliability[2],
        atmospheric_reliability,
    ], axis=0), axis=0)
    group_costs = [
        soft_range_cost(slope, config.slope_ideal_deg, config.slope_transition_deg),
        soft_range_cost(roughness, config.roughness_ideal_m, config.roughness_transition_m),
        soft_range_cost(thermal, config.temperature_ideal_k, config.temperature_transition_k),
    ]
    atmospheric_costs = []
    if np.any(np.isfinite(wind)):
        atmospheric_costs.append(soft_range_cost(wind, config.wind_ideal_m_s, config.wind_transition_m_s))
    if np.any(np.isfinite(dust)):
        atmospheric_costs.append(soft_range_cost(dust, config.dust_ideal, config.dust_transition))
    atmospheric = np.nanmean(np.stack(atmospheric_costs), axis=0) if atmospheric_costs else np.full(shape, 0.5)
    costs = [*group_costs, atmospheric, 1.0 - reliability]
    costs = [np.where(np.isfinite(cost), cost, 0.5) for cost in costs]
    weights = np.asarray(config.weights, dtype=float)
    weights = weights / weights.sum()
    suitability = (1.0 - np.tensordot(weights, np.stack(costs), axes=(0, 0))) * 100.0
    # A landing site represents a small local footprint, not a single isolated cell.
    from scipy.ndimage import uniform_filter
    neighborhood_weight = uniform_filter(mask.astype(float), size=3, mode="constant", cval=0.0)
    suitability = np.divide(
        uniform_filter(np.where(mask, suitability, 0.0), size=3, mode="constant", cval=0.0),
        neighborhood_weight,
        out=np.full(shape, np.nan, dtype=float),
        where=neighborhood_weight > 0,
    )
    reliability = np.divide(
        uniform_filter(np.where(mask, reliability, 0.0), size=3, mode="constant", cval=0.0),
        neighborhood_weight,
        out=np.zeros(shape, dtype=float),
        where=neighborhood_weight > 0,
    )
    candidate_mask = mask & np.isfinite(slope) & np.isfinite(roughness)
    suitability[~mask] = np.nan
    rows, cols = np.where(candidate_mask)
    if not len(rows):
        raise ValueError("Çalışma alanında geçerli eğim ve pürüzlülük hücresi bulunamadı.")
    order = np.argsort(suitability[rows, cols])[::-1]
    selected = []
    mars_radius_km = 3396.19
    min_sep_rad = float(minimum_separation_km) / mars_radius_km
    lon_grid = np.asarray(longitude_grid, dtype=float)
    lat_grid = np.asarray(latitude_grid, dtype=float)
    for index in order:
        row, col = int(rows[index]), int(cols[index])
        lon, lat = float(lon_grid[row, col]), float(lat_grid[row, col])
        separated = True
        for site in selected:
            dlat = np.radians(lat - site["latitude"])
            dlon = np.radians(((lon - site["longitude"] + 180) % 360) - 180)
            mean_lat = np.radians((lat + site["latitude"]) / 2.0)
            if np.hypot(dlat, dlon * np.cos(mean_lat)) < min_sep_rad:
                separated = False
                break
        if separated:
            selected.append({
                "rank": len(selected) + 1,
                "row": row,
                "col": col,
                "longitude": lon,
                "latitude": lat,
                "score": float(suitability[row, col]),
                "data_reliability": float(reliability[row, col] * 100.0),
            })
            if len(selected) >= int(maximum_sites):
                break
    return {"suitability_map": suitability, "reliability_map": reliability * 100.0, "candidates": selected}
