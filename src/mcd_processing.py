
from __future__ import annotations

import re
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def _infer_unit(description: str) -> str | None:
    match = re.search(r"[([]\s*(K|kelvin|degC|°C|m/s)\s*[)\]]", description, re.I)
    if match:
        value = match.group(1).lower()
        return "m/s" if value == "m/s" else ("K" if value in {"k", "kelvin"} else "°C")
    text = description.lower()
    if "temperature" in text or "temp" in text:
        return "K"
    if "wind" in text:
        return "m/s"
    return None


def _extract_numbers(text: str) -> np.ndarray:
    """Metindeki ondalıklı ve bilimsel gösterimli sayıları çıkarır."""
    return np.array(
        [float(value) for value in NUMBER_PATTERN.findall(text)],
        dtype=float,
    )



def parse_mcd_text(text: str, source: str = "") -> list[dict]:
    """
    MCD metin çıktısındaki veri bloklarını ayrıştırır.
    Her blokta ilk sütun boylam, ilk satır enlem koordinatlarıdır.
    """

    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()

    layers = []
    index = 0
    description = "MCD değişkeni"

    while index < len(lines):

        line = lines[index].strip()

        # Değişken açıklamasını takip et.
        if "Columns 2+" in line:
            description = line.split("Columns 2+", 1)[1].strip()
            description = re.sub(r"^(?:are|is)\s+", "", description, flags=re.IGNORECASE)
            index += 1
            continue

        # Veri bloğunun başlangıcını bul.
        if not line.startswith("----") or "||" not in line:
            index += 1
            continue

        latitude_part = line.split("||", 1)[1]
        latitudes = _extract_numbers(latitude_part)

        if latitudes.size == 0:
            raise ValueError(
                f"Enlem koordinatları okunamadı: {description}"
            )

        longitudes = []
        rows = []

        index += 1

        # Veri satırlarını oku.
        while index < len(lines):

            row_line = lines[index].strip()

            if not row_line:
                index += 1
                continue

            # Yeni MCD bölümüne ulaşıldı.
            if row_line.startswith("###"):
                break

            # Yeni veri bloğu başlangıcı.
            if row_line.startswith("----"):

                if "||" in row_line:
                    break

                index += 1
                continue

            if "||" not in row_line:
                index += 1
                continue

            longitude_part, values_part = row_line.split("||", 1)

            longitude_numbers = _extract_numbers(longitude_part)
            # MCD tables commonly mark missing values with ``*****``. Keep
            # the grid shape and represent those cells as NaN.
            tokens = values_part.split()
            values = np.asarray([
                float(token) if NUMBER_PATTERN.fullmatch(token) else np.nan
                for token in tokens
            ], dtype=float)

            if longitude_numbers.size == 0:
                index += 1
                continue

            longitude = float(longitude_numbers[0])

            if values.size != latitudes.size:
                raise ValueError(
                    f"{description}: {longitude}° boylamındaki "
                    f"veri sayısı ({values.size}) enlem sayısıyla "
                    f"({latitudes.size}) uyuşmuyor."
                )

            longitudes.append(longitude)
            rows.append(values)

            # ÖNEMLİ: Her başarılı satırdan sonra ilerle.
            index += 1

        if not rows:
            raise ValueError(
                f"Veri satırları bulunamadı: {description}"
            )

        values_array = np.asarray(rows, dtype=float)

        if values_array.shape != (
            len(longitudes),
            len(latitudes),
        ):
            raise ValueError(
                f"{description}: Veri matrisi boyutları tutarsız."
            )

        layers.append({
            "description": description,
            "unit": _infer_unit(description),
            "latitudes": latitudes,
            "longitudes": np.asarray(longitudes, dtype=float),
            "values": values_array,
            "source": source,
        })

    if not layers:
        raise ValueError(
            "Dosyada ayrıştırılabilir MCD veri bloğu bulunamadı."
        )

    return layers


def load_mcd_from_urls(urls: list[str]) -> list[dict]:
    """Bir veya daha fazla MCD TXT bağlantısını indirip ayrıştırır."""
    all_layers = []

    for url in urls:
        url = url.strip()

        if not url:
            continue

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "MarsBaseScout/1.0"},
        )

        with urllib.request.urlopen(request, timeout=45) as response:
            raw_data = response.read()

        text = raw_data.decode("utf-8-sig", errors="replace")

        layers = parse_mcd_text(text, source=url)
        all_layers.extend(layers)

    if not all_layers:
        raise ValueError("İşlenecek MCD bağlantısı verilmedi.")

    return all_layers




def plot_mcd_layer(
    layer: dict,
    vertices: list[tuple[float, float]] | None = None,
    language: str = "tr",
):
    """
    MCD katmanını dikdörtgen coğrafi alanda gösterir.
    Bilineer enterpolasyon yalnızca görselleştirme için kullanılır.
    Tekrarlanan koordinatlar ortalama alınarak birleştirilir.
    """

    from scipy.interpolate import RegularGridInterpolator

    latitudes = np.asarray(layer["latitudes"], dtype=float)
    longitudes = np.asarray(layer["longitudes"], dtype=float)
    values = np.asarray(layer["values"], dtype=float)

    # Koordinatların sıralama indeksleri
    lat_order = np.argsort(latitudes)
    lon_order = np.argsort(longitudes)

    latitudes = latitudes[lat_order]
    longitudes = longitudes[lon_order]

    # Veri matrisini (enlem, boylam) düzenine getir.
    if values.shape == (len(longitudes), len(latitudes)):
        map_values = values[np.ix_(lon_order, lat_order)].T

    elif values.shape == (len(latitudes), len(longitudes)):
        map_values = values[np.ix_(lat_order, lon_order)]

    else:
        raise ValueError(
            f"Veri boyutları koordinatlarla uyuşmuyor: "
            f"values={values.shape}, "
            f"latitudes={len(latitudes)}, "
            f"longitudes={len(longitudes)}"
        )

    # Tekrarlanan koordinatları tespit et.
    unique_latitudes, lat_inverse = np.unique(
        latitudes,
        return_inverse=True,
    )

    unique_longitudes, lon_inverse = np.unique(
        longitudes,
        return_inverse=True,
    )

    # Aynı koordinata karşılık gelen değerleri ortala.
    valid_mask = np.isfinite(map_values)

    sums = np.zeros(
        (len(unique_latitudes), len(unique_longitudes)),
        dtype=float,
    )

    counts = np.zeros_like(sums)

    np.add.at(
        sums,
        (lat_inverse[:, None], lon_inverse[None, :]),
        np.where(valid_mask, map_values, 0.0),
    )

    np.add.at(
        counts,
        (lat_inverse[:, None], lon_inverse[None, :]),
        valid_mask.astype(float),
    )

    map_values = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )

    latitudes = unique_latitudes
    longitudes = unique_longitudes

    if len(latitudes) < 2 or len(longitudes) < 2:
        raise ValueError(
            "Enterpolasyon için en az 2 farklı enlem ve boylam gerekir."
        )

    valid_values = map_values[np.isfinite(map_values)]

    if valid_values.size == 0:
        raise ValueError("Geçerli MCD verisi bulunamadı.")

    # Enterpolasyon için ince görselleştirme ızgarası.
    fine_latitudes = np.linspace(
        latitudes.min(),
        latitudes.max(),
        max(200, len(latitudes) * 8),
    )

    fine_longitudes = np.linspace(
        longitudes.min(),
        longitudes.max(),
        max(200, len(longitudes) * 8),
    )

    fine_lon_grid, fine_lat_grid = np.meshgrid(
        fine_longitudes,
        fine_latitudes,
    )

    valid_mask = np.isfinite(map_values)

    interpolator = RegularGridInterpolator(
        (latitudes, longitudes),
        np.where(valid_mask, map_values, 0.0),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    weight_interpolator = RegularGridInterpolator(
        (latitudes, longitudes),
        valid_mask.astype(float),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    points = np.column_stack([
        fine_lat_grid.ravel(),
        fine_lon_grid.ravel(),
    ])

    interpolated_values = interpolator(points).reshape(
        fine_lat_grid.shape
    )

    interpolated_weights = weight_interpolator(points).reshape(
        fine_lat_grid.shape
    )

    fine_values = np.divide(
        interpolated_values,
        interpolated_weights,
        out=np.full_like(interpolated_values, np.nan),
        where=interpolated_weights > 0.999,
    )

    # Harita
    fig, ax = plt.subplots(
        figsize=(11, 8),
        dpi=180,
    )

    image = ax.imshow(
        fine_values,
        extent=[
            longitudes.min(),
            longitudes.max(),
            latitudes.min(),
            latitudes.max(),
        ],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        interpolation="bilinear",
    )

    ax.set_xlim(longitudes.min(), longitudes.max())
    ax.set_ylim(latitudes.min(), latitudes.max())

    if language == "en":
        ax.set_xlabel("East longitude (°E)")
        ax.set_ylabel("North latitude (°N)")
    else:
        ax.set_xlabel("Doğu boylamı (°E)")
        ax.set_ylabel("Kuzey enlemi (°N)")
    ax.set_title(layer["description"])

    ax.grid(
        True,
        linestyle="--",
        alpha=0.2,
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(
        f"Value ({layer.get('unit') or 'unit not specified'})" if language == "en"
        else f"Değer ({layer.get('unit') or 'birim belirtilmemiş'})"
    )

    fig.tight_layout()

    statistics = {
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "mean": float(np.mean(valid_values)),
        "median": float(np.median(valid_values)),
        "valid_cells": int(valid_values.size),
    }

    return fig, statistics

def calculate_wind_speed_layer(layers):
    """
    MCD u ve v bileşenlerinden yatay rüzgâr hızını hesaplar.
    Birim: m/s
    """

    u_layer = None
    v_layer = None

    for layer in layers:
        description = layer.get("description", "").lower()

        if "wind component" in description and ("w-e" in description or "west-east" in description or "zonal" in description):
            u_layer = layer

        elif "wind component" in description and ("s-n" in description or "south-north" in description or "meridional" in description):
            v_layer = layer

    if u_layer is None or v_layer is None:
        raise ValueError("MCD dosyasında u veya v rüzgâr bileşeni bulunamadı.")

    u = np.asarray(u_layer["values"], dtype=float)
    v = np.asarray(v_layer["values"], dtype=float)

    if u.shape != v.shape:
        raise ValueError("u ve v veri boyutları eşleşmiyor.")

    if not np.allclose(
        u_layer["latitudes"],
        v_layer["latitudes"]
    , equal_nan=True):
        raise ValueError("u ve v enlem koordinatları eşleşmiyor.")

    if not np.allclose(
        u_layer["longitudes"],
        v_layer["longitudes"]
    , equal_nan=True):
        raise ValueError("u ve v boylam etiketleri eşleşmiyor.")

    wind_speed = np.sqrt(u**2 + v**2)

    return {
        "description": "Horizontal wind speed",
        "unit": "m/s",
        "values": wind_speed,
        "latitudes": np.asarray(u_layer["latitudes"]),
        "longitudes": np.asarray(u_layer["longitudes"]),
        "source": "MCD 6.2 - calculated from u and v",
    }


def plot_wind_speed_layer(layer):
    """Rüzgâr hızını kaynak MCD boylam ve enlem koordinatlarında çizer."""
    fig, stats = plot_mcd_layer(layer)
    stats = {
        "minimum": stats["min"],
        "maximum": stats["max"],
        "mean": stats["mean"],
    }
    return fig, stats
def reconstruct_longitudes(
    longitude_min: float,
    longitude_max: float,
    row_count: int,
) -> np.ndarray:
    """
    MCD veri satırlarının gerçek boylam koordinatlarını oluşturur.
    """

    if row_count < 2:
        raise ValueError("En az iki boylam satırı gerekli.")

    return np.linspace(
        longitude_min,
        longitude_max,
        row_count,
        dtype=float,
    )
