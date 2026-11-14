"""Grid-based, risk-weighted Dijkstra routing to a polygonal research zone."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import binary_dilation

from src.study_area import point_in_polygon


# User supplied Mars map coordinates are milli-degrees: X=longitude*1000,
# Y=latitude*1000. Vertex order is clockwise around the target rectangle.
RESEARCH_ZONE_XY = [
    (-136343, 23683),  # A
    (-136328, 23515),  # B
    (-136735, 23449),  # C
    (-136735, 23660),  # D
]
RESEARCH_ZONE_LONLAT = [(x / 1000.0, y / 1000.0) for x, y in RESEARCH_ZONE_XY]
RESEARCH_ZONE_LABELS = ["A", "B", "C", "D"]


def _distance_km(lon1, lat1, lon2, lat2):
    mean_lat = np.radians((lat1 + lat2) / 2.0)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    return 3396.19 * float(np.hypot(dlat, dlon * np.cos(mean_lat)))


def _nearest_zone_boundary(lon, lat, vertices=RESEARCH_ZONE_LONLAT):
    """Project an approach cell onto the closest polygon edge or corner."""
    cosine = np.cos(np.radians(lat))
    point = np.asarray([lon * cosine, lat], dtype=float)
    best = None
    for i, (start, end) in enumerate(zip(vertices, vertices[1:] + vertices[:1])):
        a = np.asarray([start[0] * cosine, start[1]], dtype=float)
        b = np.asarray([end[0] * cosine, end[1]], dtype=float)
        delta = b - a
        fraction = float(np.clip(np.dot(point - a, delta) / max(np.dot(delta, delta), 1e-15), 0.0, 1.0))
        nearest = a + fraction * delta
        distance = float(np.linalg.norm(point - nearest))
        if best is None or distance < best[0]:
            if fraction <= 0.08:
                label, out_lon, out_lat = f"Köşe {RESEARCH_ZONE_LABELS[i]}", start[0], start[1]
            elif fraction >= 0.92:
                next_label = RESEARCH_ZONE_LABELS[(i + 1) % len(RESEARCH_ZONE_LABELS)]
                label, out_lon, out_lat = f"Köşe {next_label}", end[0], end[1]
            else:
                label = f"Kenar {RESEARCH_ZONE_LABELS[i]}–{RESEARCH_ZONE_LABELS[(i + 1) % 4]}"
                out_lon, out_lat = float(nearest[0] / cosine), float(nearest[1])
            best = (distance, label, out_lon, out_lat)
    return best[1:]


def build_target_masks(longitude_grid, latitude_grid, traversable_mask, vertices=RESEARCH_ZONE_LONLAT):
    """Rasterize the destination region and choose traversable cells along its approach ring."""
    lon_grid = ((np.asarray(longitude_grid, dtype=float) + 180.0) % 360.0) - 180.0
    lat_grid = np.asarray(latitude_grid, dtype=float)
    polygon = [(((lon + 180.0) % 360.0) - 180.0, lat) for lon, lat in vertices]
    inside = point_in_polygon(lon_grid, lat_grid, polygon)
    traversable = np.asarray(traversable_mask, dtype=bool)
    if inside.shape != traversable.shape:
        raise ValueError("Hedef alan grid boyutları rota gridine uymuyor.")
    ring = binary_dilation(inside, structure=np.ones((3, 3), dtype=bool)) & ~inside
    goals = ring & traversable
    if not np.any(goals):
        goals = inside & traversable
    if not np.any(inside):
        raise ValueError("Araştırma bölgesi mevcut DEM/grid kapsamının dışında kalıyor.")
    if not np.any(goals):
        raise ValueError("Araştırma bölgesinin kenarında geçilebilir grid hücresi yok.")
    return inside, goals


def dijkstra_routes(cost_grid, traversable_mask, goal_mask, candidates, longitude_grid, latitude_grid):
    """Run one multi-source Dijkstra search and reconstruct routes for all candidates."""
    risk = np.asarray(cost_grid, dtype=float)
    valid = np.asarray(traversable_mask, dtype=bool) & np.isfinite(risk)
    goals = np.asarray(goal_mask, dtype=bool) & valid
    lon = np.asarray(longitude_grid, dtype=float)
    lat = np.asarray(latitude_grid, dtype=float)
    if not (risk.shape == valid.shape == goals.shape == lon.shape == lat.shape):
        raise ValueError("Rota katmanları aynı boyutta olmalı.")
    goal_rows, goal_cols = np.where(goals)
    if not len(goal_rows):
        raise ValueError("Dijkstra için hedef bölgeye yakın geçerli başlangıç hücresi yok.")

    distances = np.full(risk.shape, np.inf, dtype=np.float64)
    next_rows = np.full(risk.shape, -1, dtype=np.int32)
    next_cols = np.full(risk.shape, -1, dtype=np.int32)
    queue = []
    for row, col in zip(goal_rows, goal_cols):
        distances[row, col] = 0.0
        heapq.heappush(queue, (0.0, int(row), int(col)))

    source_keys = {(int(site["row"]), int(site["col"])) for site in candidates}
    pending = set(source_keys)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    height, width = risk.shape
    while queue and pending:
        current_distance, row, col = heapq.heappop(queue)
        if current_distance != distances[row, col]:
            continue
        pending.discard((row, col))
        for drow, dcol in neighbors:
            nr, nc = row + drow, col + dcol
            if nr < 0 or nr >= height or nc < 0 or nc >= width or not valid[nr, nc]:
                continue
            step = _distance_km(lon[row, col], lat[row, col], lon[nr, nc], lat[nr, nc])
            edge_risk = 0.05 + (float(risk[row, col]) + float(risk[nr, nc])) * 0.5
            proposed = current_distance + step * edge_risk
            if proposed < distances[nr, nc]:
                distances[nr, nc] = proposed
                next_rows[nr, nc] = row
                next_cols[nr, nc] = col
                heapq.heappush(queue, (proposed, nr, nc))

    routes = []
    for site in candidates:
        row, col = int(site["row"]), int(site["col"])
        if (row, col) in pending or not np.isfinite(distances[row, col]):
            continue
        path = [(row, col)]
        seen = {(row, col)}
        while not goals[path[-1]]:
            current = path[-1]
            nr, nc = int(next_rows[current]), int(next_cols[current])
            if nr < 0 or (nr, nc) in seen:
                path = []
                break
            path.append((nr, nc))
            seen.add((nr, nc))
        if not path:
            continue
        path_lons = [float(lon[r, c]) for r, c in path]
        path_lats = [float(lat[r, c]) for r, c in path]
        length = sum(_distance_km(*a, *b) for a, b in zip(
            zip(path_lons[:-1], path_lats[:-1]), zip(path_lons[1:], path_lats[1:])
        ))
        endpoint_label, target_lon, target_lat = _nearest_zone_boundary(path_lons[-1], path_lats[-1])
        connector = _distance_km(path_lons[-1], path_lats[-1], target_lon, target_lat)
        path_lons.append(target_lon)
        path_lats.append(target_lat)
        routes.append({
            "site": site,
            "path": path,
            "longitudes": path_lons,
            "latitudes": path_lats,
            "path_cost": float(distances[row, col]),
            "distance_km": float(length + connector),
            "target_entry": endpoint_label,
            "target_longitude": target_lon,
            "target_latitude": target_lat,
        })
    routes.sort(key=lambda route: (route["path_cost"], -route["site"]["score"]))
    for i, route in enumerate(routes):
        route["label"] = chr(ord("A") + i)
    return routes
