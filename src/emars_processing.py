
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from src.i18n import translate


def open_emars(path: str | Path) -> xr.Dataset:
    """EMARS NetCDF dosyasını tembel (lazy) biçimde açar."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"EMARS dosyası bulunamadı: {path}")

    ds = xr.open_dataset(
        path,
        engine="netcdf4",
        chunks=None
    )

    required_coords = {"lat", "lon", "time", "pfull"}

    missing = required_coords - set(ds.coords)

    if missing:
        ds.close()
        raise ValueError(
            f"Eksik EMARS koordinatları: {sorted(missing)}"
        )

    if ds.sizes.get("time", 0) < 1 or ds.sizes.get("pfull", 0) < 1:
        ds.close()
        raise ValueError("EMARS dosyasında zaman ve basınç seviyeleri boş olamaz.")

    return ds


def get_emars_metadata(ds: xr.Dataset) -> dict:
    """Dosyanın koordinat ve değişken bilgilerini döndürür."""

    return {
        "dimensions": dict(ds.sizes),
        "latitude": ds["lat"].values.tolist(),
        "longitude": ds["lon"].values.tolist(),
        "variables": list(ds.data_vars),
        "time_count": ds.sizes.get("time", 0),
        "pressure_levels": ds["pfull"].values.tolist(),
    }


def get_surface_temperature(
    ds: xr.Dataset,
    time_index: int
) -> dict:
    """Yüzey sıcaklığını Kelvin ve Celsius olarak döndürür."""

    if "ts" not in ds:
        raise ValueError("EMARS dosyasında ts değişkeni bulunamadı.")

    if not 0 <= time_index < ds.sizes["time"]:
        raise IndexError("Geçersiz zaman indeksi.")

    variable = ds["ts"]
    temperature = np.asarray(variable.isel(time=time_index).values, dtype=float)
    temperature_c = _temperature_to_celsius(temperature, variable.attrs.get("units", "K"))

    return {
        "kelvin": temperature_c + 273.15,
        "celsius": temperature_c,
        "unit": variable.attrs.get("units", "K"),
    }


def get_atmospheric_temperature(
    ds: xr.Dataset,
    time_index: int,
    level_index: int
) -> dict:
    """Seçilen atmosfer basınç seviyesindeki sıcaklığı döndürür."""

    if "t" not in ds:
        raise ValueError("EMARS dosyasında t değişkeni bulunamadı.")

    if not 0 <= time_index < ds.sizes["time"]:
        raise IndexError("Geçersiz zaman indeksi.")

    if not 0 <= level_index < ds.sizes["pfull"]:
        raise IndexError("Geçersiz atmosfer seviyesi.")

    variable = ds["t"]
    temperature = np.asarray(variable.isel(
        time=time_index,
        pfull=level_index
    ).values, dtype=float)
    temperature_c = _temperature_to_celsius(temperature, variable.attrs.get("units", "K"))

    return {
        "kelvin": temperature_c + 273.15,
        "celsius": temperature_c,
        "pressure_mb": get_pressure_level_info(ds, level_index)[0],
    }


def get_surface_pressure_level_index(ds: xr.Dataset) -> int:
    """Return the pressure level closest to the surface (highest pressure)."""
    pressure = np.asarray(ds["pfull"].values, dtype=float)
    valid = np.isfinite(pressure)
    if pressure.ndim != 1 or not np.any(valid):
        raise ValueError("pfull basınç seviyeleri geçerli tek boyutlu değerler içermiyor.")
    valid_indices = np.flatnonzero(valid)
    return int(valid_indices[np.argmax(pressure[valid])])


def get_pressure_level_info(ds: xr.Dataset, level_index: int) -> tuple[float, str]:
    """Return a pressure level in mbar when units are known, otherwise its native value and unit."""
    coordinate = ds["pfull"]
    if not 0 <= level_index < coordinate.size:
        raise IndexError("Geçersiz atmosfer seviyesi.")
    value = float(coordinate.isel(pfull=level_index).values)
    unit = str(coordinate.attrs.get("units", "")).strip().lower()
    normalized = unit.replace(" ", "").replace("_", "")
    if normalized in {"pa", "pascal", "pascals"}:
        return value / 100.0, "mbar"
    if normalized in {"hpa", "mbar", "mb", "millibar", "millibars"}:
        return value, "mbar"
    if normalized in {"kpa", "kilopascal", "kilopascals"}:
        return value * 10.0, "mbar"
    if normalized in {"bar", "bars"}:
        return value * 1000.0, "mbar"
    if unit:
        return value, unit
    return value, "pfull birimi belirtilmemiş"


ATMOSPHERIC_MAP_VARIABLES = {
    "W-E bileşeni (u)": ("u", "uwind", "ucomp", "zonwind"),
    "N-S bileşeni (v)": ("v", "vwind", "vcomp", "merwind"),
    "Toz optik derinliği": (
        "tauref", "tau_ref", "tod", "dod", "dust_optical_depth", "dustopacity",
    ),
}


def resolve_atmospheric_map_variables(ds: xr.Dataset) -> dict[str, str | None]:
    """Resolve the supported fields against case-insensitive NetCDF names."""
    available = {str(name).lower(): str(name) for name in ds.data_vars}
    resolved = {}
    for label, candidates in ATMOSPHERIC_MAP_VARIABLES.items():
        resolved[label] = next(
            (available[name.lower()] for name in candidates if name.lower() in available),
            None,
        )
    return resolved


def get_atmospheric_map_field(
    ds: xr.Dataset,
    variable_name: str,
    time_index: int,
    level_index: int,
) -> np.ndarray:
    """Select a time/pressure slice and align staggered winds to lat/lon centers."""
    if not 0 <= time_index < ds.sizes.get("time", 0):
        raise IndexError("Geçersiz zaman indeksi.")
    if not 0 <= level_index < ds.sizes.get("pfull", 0):
        raise IndexError("Geçersiz atmosfer seviyesi.")

    field = ds[variable_name]
    selectors = {}
    if "time" in field.dims:
        selectors["time"] = time_index
    if "pfull" in field.dims:
        selectors["pfull"] = level_index
    field = field.isel(selectors)

    lat_dim = next((dim for dim in field.dims if dim.lower() in {"lat", "latitude", "latu", "latv"}), None)
    lon_dim = next((dim for dim in field.dims if dim.lower() in {"lon", "longitude", "lonu", "lonv"}), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"{variable_name} değişkeni 2B enlem-boylam haritasına indirgenemedi.")

    if lat_dim != "lat":
        field = field.interp({lat_dim: np.asarray(ds["lat"].values)}).rename({lat_dim: "lat"})
    if lon_dim != "lon":
        field = field.interp({lon_dim: np.asarray(ds["lon"].values)}).rename({lon_dim: "lon"})
    field = field.transpose("lat", "lon")
    values = np.asarray(field.values, dtype=float)
    if values.shape != (ds.sizes["lat"], ds.sizes["lon"]):
        raise ValueError(
            f"{variable_name} harita boyutu {values.shape}; beklenen "
            f"({ds.sizes['lat']}, {ds.sizes['lon']})."
        )
    return values


def _temperature_to_celsius(values, units):
    unit = str(units or "K").strip().lower().replace("°", "deg")
    unit = unit.replace(" ", "_")
    if unit in {"k", "kelvin", "kelvins"}:
        return values - 273.15
    if unit in {"c", "degc", "celsius", "degree_c", "degrees_c", "degree_celsius", "degrees_celsius"}:
        return values
    if unit in {"f", "degf", "fahrenheit", "degree_f", "degrees_f", "degree_fahrenheit", "degrees_fahrenheit"}:
        return (values - 32.0) * (5.0 / 9.0)
    raise ValueError(f"Desteklenmeyen sıcaklık birimi: {units}")


def get_wind_data(
    ds: xr.Dataset,
    time_index: int,
    level_index: int
) -> dict:
    """Yatay rüzgâr bileşenlerinden hız ve yön hesaplar."""

    if "u" not in ds or "v" not in ds:
        raise ValueError("EMARS u veya v değişkeni bulunamadı.")

    if not 0 <= time_index < ds.sizes.get("time", 0):
        raise IndexError("Geçersiz zaman indeksi.")
    if not 0 <= level_index < ds.sizes.get("pfull", 0):
        raise IndexError("Geçersiz atmosfer seviyesi.")

    u = ds["u"].isel(
        time=time_index,
        pfull=level_index
    ).values

    v = ds["v"].isel(
        time=time_index,
        pfull=level_index
    ).values

    speed = np.sqrt(u**2 + v**2)

    # Meteorolojik yön: rüzgârın geldiği yön.
    direction = (
        np.degrees(np.arctan2(-u, -v)) + 360
    ) % 360

    return {
        "u": u,
        "v": v,
        "speed": speed,
        "direction": direction,
        "unit": "m/s",
    }


def get_dust_opacity(
    ds: xr.Dataset,
    time_index: int
) -> dict:
    """Toplam sütun toz optik derinliğini döndürür."""

    if "dod" not in ds:
        raise ValueError("EMARS dod değişkeni bulunamadı.")

    if not 0 <= time_index < ds.sizes.get("time", 0):
        raise IndexError("Geçersiz zaman indeksi.")

    opacity = ds["dod"].isel(time=time_index).values

    return {
        "opacity": opacity,
        "unit": "dimensionless",
    }


def get_time_information(
    ds: xr.Dataset,
    time_index: int
) -> dict:
    """Seçilen zaman örneğinin Mars zamanı bilgilerini döndürür."""

    if not 0 <= time_index < ds.sizes["time"]:
        raise IndexError("Geçersiz zaman indeksi.")

    return {
        "mars_year": float(ds["MY"].isel(time=time_index).values),
        "solar_longitude": float(ds["Ls"].isel(time=time_index).values),
        "mars_hour": float(ds["mars_hour"].isel(time=time_index).values),
        "mars_sol": float(ds["emars_sol"].isel(time=time_index).values),
    }

import matplotlib.pyplot as plt


def plot_emars_map(
    ds,
    values,
    title,
    colorbar_label,
    cmap="viridis",
    u=None,
    v=None,
    vertices=None,
    vmin=None,
    vmax=None,
    language="tr",
):
    """
    EMARS verisini gerçek Mars enlem-boylam koordinatlarında gösterir.

    values: (latitude, longitude) boyutlarında veri dizisi.
    u, v: İsteğe bağlı rüzgâr bileşenleri (m/s).
    vertices: Çalışma alanının lon-lat köşe koordinatları.
    """

    import matplotlib.pyplot as plt
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)

    # EMARS 0-360 derece doğu boylamlarını -180/+180 sistemine çevir.
    lon = ((lon + 180) % 360) - 180

    values = np.asarray(values, dtype=float)

    if values.shape != (len(lat), len(lon)):
        raise ValueError(
            f"Veri boyutu {values.shape}; beklenen "
            f"({len(lat)}, {len(lon)})."
        )

    # Make both axes increasing and remove duplicated coordinates.
    lat_order = np.argsort(lat)
    lat = lat[lat_order]
    values = values[lat_order, :]
    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    values = values[:, lon_order]

    lat, lat_indices = np.unique(lat, return_index=True)
    values = values[lat_indices, :]

    # A global grid may include both 0° and 360° (or both -180° and 180°).
    # They become the same longitude after wrapping; keep one seam column.
    lon, unique_indices = np.unique(lon, return_index=True)
    values = values[:, unique_indices]

    if u is not None and v is not None:
        u = np.asarray(u, dtype=float)[lat_order, :][:, lon_order]
        v = np.asarray(v, dtype=float)[lat_order, :][:, lon_order]
        u = u[lat_indices, :]
        v = v[lat_indices, :]
        u = u[:, unique_indices]
        v = v[:, unique_indices]

    if len(lon) < 2 or len(lat) < 2 or not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError("Harita için en az iki geçerli enlem ve boylam koordinatı gerekir.")

    # Densify the display grid only. The source values remain untouched;
    # bilinear interpolation removes the blocky appearance of sparse grids.
    from scipy.interpolate import RegularGridInterpolator

    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("Enterpolasyon için en az iki farklı enlem ve boylam gerekir.")

    longitude_step = float(np.median(np.diff(lon)))
    is_global_grid = (lon[-1] - lon[0] + longitude_step) >= 359.0
    interp_lon = lon
    interp_values = values
    if is_global_grid:
        interp_lon = np.concatenate(([lon[-1] - 360.0], lon, [lon[0] + 360.0]))
        interp_values = np.concatenate((values[:, -1:], values, values[:, :1]), axis=1)
        display_lon_min, display_lon_max = -180.0, 180.0
        lon_count = min(1600, max(360, (len(lon) - 1) * 4 + 1))
    else:
        display_lon_min, display_lon_max = float(lon.min()), float(lon.max())
        lon_count = min(1000, max(240, (len(lon) - 1) * 8 + 1))

    fine_lats = np.linspace(float(lat.min()), float(lat.max()), min(800, max(200, (len(lat) - 1) * 8 + 1)))
    fine_lons = np.linspace(display_lon_min, display_lon_max, lon_count)
    fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lons, fine_lats)
    finite_values = np.isfinite(interp_values)
    value_interpolator = RegularGridInterpolator(
        (lat, interp_lon), np.where(finite_values, interp_values, 0.0),
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    weight_interpolator = RegularGridInterpolator(
        (lat, interp_lon), finite_values.astype(float),
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    points = np.column_stack((fine_lat_grid.ravel(), fine_lon_grid.ravel()))
    fine_numerator = value_interpolator(points).reshape(fine_lat_grid.shape)
    fine_weights = weight_interpolator(points).reshape(fine_lat_grid.shape)
    fine_values = np.divide(
        fine_numerator, fine_weights,
        out=np.full_like(fine_numerator, np.nan),
        where=fine_weights > 0.999,
    )

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    mesh = ax.imshow(
        np.ma.masked_invalid(fine_values),
        extent=[fine_lons.min(), fine_lons.max(), fine_lats.min(), fine_lats.max()],
        origin="lower", aspect="auto", cmap=cmap, interpolation="bilinear",
        vmin=vmin, vmax=vmax,
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label=colorbar_label,
        shrink=0.85,
    )

    # Çalışma alanı üçgeni.
    if vertices is not None:
        polygon = np.asarray(vertices, dtype=float)

        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError(
                "vertices, (boylam, enlem) çiftlerinden oluşmalıdır."
            )

        polygon[:, 0] = ((polygon[:, 0] + 180) % 360) - 180

        polygon = np.vstack((polygon, polygon[0]))

        ax.plot(
            polygon[:, 0],
            polygon[:, 1],
            color="red",
            linewidth=2,
            linestyle="--",
            label=translate("Lycus Sulci çalışma alanı", language),
        )

        ax.legend(loc="upper right")

    # Rüzgâr vektörleri.
    if u is not None and v is not None:

        if u.shape != values.shape or v.shape != values.shape:
            raise ValueError("Rüzgâr bileşenlerinin boyutları uyuşmuyor.")

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Boylam yönündeki açısal bileşeni enlemle düzelt.
        u_geo = u / np.cos(np.radians(lat_grid))
        v_geo = v

        magnitude = np.hypot(u_geo, v_geo)

        u_direction = np.divide(
            u_geo,
            magnitude,
            out=np.zeros_like(u_geo),
            where=magnitude > 0,
        )

        v_direction = np.divide(
            v_geo,
            magnitude,
            out=np.zeros_like(v_geo),
            where=magnitude > 0,
        )

        stride = max(1, int(np.ceil(max(len(lat), len(lon)) / 6)))
        lat_indices = np.arange(len(lat))
        lon_indices = np.arange(len(lon))
        if len(lat_indices) > 2:
            lat_indices = lat_indices[1:-1]
        if len(lon_indices) > 2:
            lon_indices = lon_indices[1:-1]
        lat_indices = lat_indices[::stride]
        lon_indices = lon_indices[::stride]
        ax.quiver(
            lon_grid[np.ix_(lat_indices, lon_indices)],
            lat_grid[np.ix_(lat_indices, lon_indices)],
            u_direction[np.ix_(lat_indices, lon_indices)],
            v_direction[np.ix_(lat_indices, lon_indices)],
            color="white",
            pivot="middle",
            scale=14,
            width=0.003,
        )

    ax.set_title(title)
    if language == "en":
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
    else:
        ax.set_xlabel("Boylam (°)")
        ax.set_ylabel("Enlem (°)")

    ax.set_xticks(np.linspace(lon.min(), lon.max(), min(7, len(lon))))
    ax.set_yticks(np.linspace(lat.min(), lat.max(), min(7, len(lat))))

    ax.grid(alpha=0.25)

    if is_global_grid:
        ax.set_xlim(-180, 180)
    else:
        ax.set_xlim(fine_lons.min(), fine_lons.max())
    ax.set_ylim(lat.min(), lat.max())

    # Mars yüzeyinde yaklaşık coğrafi en-boy oranı.
    mean_lat = np.mean(lat)
    ax.set_aspect(1 / np.cos(np.radians(mean_lat)))

    fig.tight_layout()

    return fig


def plot_surface_temperature_map(ds, time_index, vertices=None, language="tr"):
    """Yüzey sıcaklığını Celsius cinsinden haritalar."""

    data = get_surface_temperature(ds, time_index)

    return plot_emars_map(
        ds=ds,
        values=data["celsius"],
        title=translate("Mars Yüzey Sıcaklığı", language),
        colorbar_label=translate("Sıcaklık (°C)", language),
        cmap="inferno",
        vertices=vertices,
        language=language,
    )


def plot_atmospheric_temperature_map(
    ds,
    time_index,
    level_index,
    vertices=None,
    language="tr",
):
    """Seçilen basınç seviyesindeki atmosfer sıcaklığını haritalar."""

    data = get_atmospheric_temperature(
        ds,
        time_index,
        level_index,
    )

    return plot_emars_map(
        ds=ds,
        values=data["celsius"],
        title=translate("Mars Atmosfer Sıcaklığı", language),
        colorbar_label=translate("Sıcaklık (°C)", language),
        cmap="coolwarm",
        vertices=vertices,
        language=language,
    )


def plot_wind_map(
    ds,
    time_index,
    level_index,
    vertices=None,
    language="tr",
):
    """Rüzgâr hızını renklerle, yönünü vektörlerle gösterir."""

    data = get_wind_data(
        ds,
        time_index,
        level_index,
    )

    return plot_emars_map(
        ds=ds,
        values=data["speed"],
        title=translate("Mars Atmosferik Rüzgâr", language),
        colorbar_label=translate("Rüzgâr hızı (m/s)", language),
        cmap="plasma",
        u=data["u"],
        v=data["v"],
        vertices=vertices,
        language=language,
    )


def plot_dust_opacity_map(ds, time_index, vertices=None, language="tr"):
    """Atmosferik toz optik derinliğini haritalar."""

    data = get_dust_opacity(ds, time_index)

    return plot_emars_map(
        ds=ds,
        values=data["opacity"],
        title=translate("Mars Atmosferik Toz Optik Derinliği", language),
        colorbar_label=translate("Toz optik derinliği", language),
        cmap="cividis",
        vertices=vertices,
        language=language,
    )
