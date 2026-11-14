
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _base_map(ax, data, metadata, title, colorbar_label, cmap):
    transform = metadata["transform"]

    left = transform.c
    right = left + data.shape[1] * transform.a
    top = transform.f
    bottom = top + data.shape[0] * transform.e

    image = ax.imshow(
        data,
        extent=[left, right, bottom, top],
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("Boylam (°)")
    ax.set_ylabel("Enlem (°)")
    ax.set_aspect("auto")

    colorbar = plt.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label(colorbar_label)

    return image


def plot_elevation(layers, metadata):
    fig, ax = plt.subplots(figsize=(11, 8))

    _base_map(
        ax,
        layers["elevation"],
        metadata,
        "Mars - Yüzey Yüksekliği",
        "Yükseklik (m, birim doğrulamasına bağlı)",
        "terrain",
    )

    return fig


def plot_slope(layers, metadata):
    fig, ax = plt.subplots(figsize=(11, 8))

    _base_map(
        ax,
        layers["slope"],
        metadata,
        "Mars - Eğim Haritası",
        "Eğim (derece)",
        "magma",
    )

    return fig


def plot_roughness(layers, metadata):
    fig, ax = plt.subplots(figsize=(11, 8))

    _base_map(
        ax,
        layers["roughness"],
        metadata,
        "Mars - Yerel Yüzey Pürüzlülüğü Göstergesi",
        "Yerel yükseklik sapması (m)",
        "inferno",
    )

    return fig


def plot_height_change(layers, metadata):
    fig, ax = plt.subplots(figsize=(11, 8))

    _base_map(
        ax,
        layers["height_change"],
        metadata,
        "Mars - Komşu Hücre Yükseklik Farkı",
        "En büyük komşu farkı (m)",
        "viridis",
    )

    return fig


def plot_total_cost(layers, metadata):
    fig, ax = plt.subplots(figsize=(11, 8))

    _base_map(
        ax,
        layers["total_cost"],
        metadata,
        "Mars - Birleşik Rover Geçiş Maliyeti",
        "Normalize edilmiş maliyet (0-1)",
        "inferno",
    )

    return fig


def plot_landing_zone(
    layers,
    metadata,
    triangle_vertices,
):
    fig, ax = plt.subplots(figsize=(11, 8))

    _base_map(
        ax,
        layers["elevation"],
        metadata,
        "Mars - İniş Bölgesi",
        "Yükseklik (m, birim doğrulamasına bağlı)",
        "terrain",
    )

    vertices = np.asarray(triangle_vertices, dtype=float)
    closed = np.vstack([vertices, vertices[0]])

    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color="cyan",
        linewidth=2,
        label="İniş üçgeni",
    )

    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        color="cyan",
        edgecolors="black",
        s=45,
        zorder=5,
    )

    ax.legend()
    return fig
