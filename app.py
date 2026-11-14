
import os
import shutil
import tempfile
import importlib
import time

from pathlib import Path
from src.emars_processing import (
    open_emars,
    get_time_information,
    get_surface_pressure_level_index,
    get_pressure_level_info,
    resolve_atmospheric_map_variables,
    get_atmospheric_map_field,
    plot_emars_map,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, to_rgba

from src.mcd_processing import (
    parse_mcd_text,
    plot_mcd_layer,
)
from src.terrain_analysis import calculate_local_roughness
from src.site_suitability import (
    calculate_site_suitability,
    regrid_geographic_field,
    rank_landing_sites,
)
from src.route_planning import (
    RESEARCH_ZONE_LONLAT,
    build_target_masks,
    dijkstra_routes,
)


from src.study_area import (
    DEFAULT_STUDY_AREA_VERTICES,
    extract_study_area,
    geographic_to_pixel,
    parse_geographic_map_info,
    point_in_polygon,
)


try:
    st = importlib.import_module("streamlit")
except ModuleNotFoundError as exc:
    if exc.name != "streamlit":
        raise
    raise RuntimeError(
        "Streamlit is required to run this app. Install it with `python -m pip install streamlit`."
    ) from exc

from src.i18n import LocalizedStreamlit, translate

from src.data_loader import load_mola_envi_from_path, load_geotiff_dem_from_path
from src.pds_download import download_pds_product, download_pds_asset
from src.analysis import (
    calculate_sampled_slope,
    get_dem_statistics,
    make_display_sample,
)


st.set_page_config(
    page_title="Mars Base Scout",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    language = st.selectbox(
        "Dil / Language",
        options=["tr", "en"],
        format_func=lambda code: "Türkçe" if code == "tr" else "English",
        key="ui_language",
    )
st = LocalizedStreamlit(st, language)
tx = lambda text: translate(text, language)

st.title("Mars Base Scout")
st.caption("Mars topografya verisi · DEM görüntüleme · Bölgesel eğim incelemesi")
st.info(
    "Render kullanımı için önerilen yöntem PDS arşivinden doğrudan indirmedir. "
    "IMG dosyası geçici diske parça parça indirilir ve RAM'e bütünüyle kopyalanmadan açılır. "
    "Sonuçlar ön inceleme amaçlıdır; tek başına iniş güvenliği kararı üretmez."
)

with st.sidebar:
    st.header("Veri kaynağı")
    source_mode = st.radio(
        "MOLA verisini nereden alalım?",
        options=["pds", "local"],
        format_func=lambda mode: translate(
            "PDS arşivinden indir (Render için önerilen)" if mode == "pds" else "Bilgisayardan yükle (yerel kullanım)",
            language,
        ),
        index=1,
    )
    st.divider()

    st.header("Görüntüleme")

    display_step = st.select_slider(
        "Harita örnekleme aralığı",
        options=[1, 2, 4, 8, 16],
        value=1,
        help="1: Bütün raster pikselleri görüntülenir.",
    )

    slope_step = st.select_slider(
        "Eğim haritası örnekleme aralığı",
        options=[1, 2, 4, 8, 16],
        value=2,
        help="1: Her pikselde eğim hesaplanır.",
    )



MAX_LOCAL_IMG_SIZE = 50 * 1024 * 1024
MAX_METADATA_SIZE = 5 * 1024 * 1024
MAX_EMARS_UPLOAD_SIZE = 140 * 1024 * 1024
MAX_MCD_TEXT_SIZE = 20 * 1024 * 1024

# Current-session score inputs; absent sources remain explicitly unavailable.
score_slope = score_roughness = score_temperature = None
score_u = score_v = score_dust = None
score_temperature_source = None
landing_roughness = None


def identify_uploads(files):
    found = {"img": None, "hdr": None, "lbl": None}

    for uploaded in files or []:
        suffix = Path(uploaded.name).suffix.lower().lstrip(".")

        if suffix not in found:
            continue

        if found[suffix] is not None:
            raise ValueError(f"Birden fazla .{suffix} dosyası seçildi.")
        found[suffix] = uploaded

    if found["img"] is None or found["hdr"] is None:
        raise ValueError(
            "MOLA verileri için .img ve .hdr dosyaları zorunludur."
        )

    img_stem = Path(found["img"].name).stem.lower()
    hdr_stem = Path(found["hdr"].name).stem.lower()

    if img_stem != hdr_stem:
        raise ValueError(
            ".img ve .hdr dosyalarının dosya adları eşleşmiyor."
        )

    if found["lbl"] is not None:
        lbl_stem = Path(found["lbl"].name).stem.lower()

        if lbl_stem != img_stem:
            raise ValueError(
                ".lbl dosyasının adı .img dosyasıyla eşleşmiyor."
            )
    return found


def save_uploaded_img_to_disk(uploaded_file):
    if uploaded_file.size > MAX_LOCAL_IMG_SIZE:
        raise ValueError(
            "IMG dosyası 50 MiB sınırını aşıyor."
        )

    return save_uploaded_file_to_disk(uploaded_file, prefix="mars_dem_")


def save_uploaded_file_to_disk(uploaded_file, prefix):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    img_path = Path(temp_dir) / Path(uploaded_file.name).name

    try:
        uploaded_file.seek(0)

        with open(img_path, "wb") as output:
            shutil.copyfileobj(
                uploaded_file,
                output,
                length=1024 * 1024
            )

        return temp_dir, img_path

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def mask_geographic_grid(values, longitudes, latitudes, vertices=DEFAULT_STUDY_AREA_VERTICES, values_order="latlon"):
    """Keep only cells inside the analysis polygon when source coordinates exist."""
    data = np.squeeze(np.asarray(values, dtype=float))
    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)
    if data.shape == (len(lat), len(lon)) == (len(lon), len(lat)):
        if values_order == "lonlat":
            data = data.T
    elif data.shape == (len(lon), len(lat)):
        data = data.T
    if data.shape != (len(lat), len(lon)):
        return data
    lon_grid, lat_grid = np.meshgrid(((lon + 180) % 360) - 180, lat)
    polygon = [(((float(x) + 180) % 360) - 180, float(y)) for x, y in vertices]
    inside = point_in_polygon(lon_grid, lat_grid, polygon)
    return data[inside]


def raster_coverage_vertices(dem, metadata):
    """Return the raster footprint polygon for alternate regional DEMs."""
    info = parse_geographic_map_info(metadata)
    height, width = dem.shape
    corners = [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]
    vertices = []
    for row, col in corners:
        if info["coordinate_units"] == "degrees":
            lon = info["map_x"] + (col - (info["reference_col"] - 1)) * info["pixel_x"]
            lat = info["map_y"] - (row - (info["reference_row"] - 1)) * info["pixel_y"]
        else:
            x = info["map_x"] + (col - (info["reference_col"] - 1)) * info["pixel_x"]
            y = info["map_y"] + (info["reference_row"] - 1 - row) * info["pixel_y"]
            lon = info["central_meridian"] + np.degrees(x / info["radius"])
            lat = np.degrees(y / info["radius"])
        vertices.append((float(lon), float(lat)))
    return vertices


def native_grid_support(longitudes, latitudes, values, target_longitude, target_latitude, target_mask, values_order="latlon"):
    """Conservative support factor from native finite grid cells in the ROI."""
    data = np.squeeze(np.asarray(values, dtype=float))
    lon = ((np.asarray(longitudes, dtype=float) + 180) % 360) - 180
    lat = np.asarray(latitudes, dtype=float)
    if data.shape == (len(lat), len(lon)) == (len(lon), len(lat)):
        if values_order == "lonlat":
            data = data.T
    elif data.shape == (len(lon), len(lat)):
        data = data.T
    if data.shape != (len(lat), len(lon)):
        return 0.0
    t_lon = np.asarray(target_longitude, dtype=float)[target_mask]
    t_lat = np.asarray(target_latitude, dtype=float)[target_mask]
    if not t_lon.size:
        return 0.0
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    in_bounds = (
        (lon_grid >= np.nanmin(t_lon)) & (lon_grid <= np.nanmax(t_lon))
        & (lat_grid >= np.nanmin(t_lat)) & (lat_grid <= np.nanmax(t_lat))
    )
    native_cells = int(np.count_nonzero(in_bounds & np.isfinite(data)))
    return float(min(1.0, np.sqrt(native_cells / 16.0)))

# Local uploads are shown only when local data source mode is selected.
st.divider()
upload_submitted = False
project_files = []
if source_mode == "local":
    st.header("Proje verileri")
    st.caption("MOLA/DEM ve atmosfer dosyalar\u0131n\u0131 tek seferde y\u00fckleyip i\u015fleyebilirsiniz. GeoTIFF ve NetCDF ayr\u0131nt\u0131l\u0131 gridler desteklenir.")
    with st.form("project_data_upload_form"):
        project_files = st.file_uploader(
            "DEM (.img + .hdr [+ .lbl], .tif/.tiff) ve atmosfer (.nc, .txt) dosyalar\u0131n\u0131 se\u00e7in",
            type=["img", "hdr", "lbl", "tif", "tiff", "nc", "txt"],
            accept_multiple_files=True,
            key="project_data_files",
        )
        upload_submitted = st.form_submit_button("Se\u00e7ilen verileri birlikte y\u00fckle ve i\u015fle", type="primary")

if upload_submitted:
    temp_dir = None
    nc_dir = None
    new_ds = None
    try:
        files = project_files or []
        by_suffix = {}
        for uploaded in files:
            suffix = Path(uploaded.name).suffix.lower().lstrip(".")
            by_suffix.setdefault(suffix, []).append(uploaded)
        for suffix in ("img", "hdr", "lbl", "tif", "tiff", "nc"):
            if len(by_suffix.get(suffix, [])) > (1 if suffix in ("img", "hdr", "lbl") else 1):
                raise ValueError(f"Birden fazla .{suffix} dosyası seçildi.")
        if by_suffix.get("tif") and by_suffix.get("tiff"):
            raise ValueError("Tek seferde yalnızca bir GeoTIFF seçin.")
        new_dem = new_metadata = None
        img_files = by_suffix.get("img", [])
        if img_files:
            selected = identify_uploads(files)
            for suffix in ("hdr", "lbl"):
                item = selected[suffix]
                if item is not None and item.size > MAX_METADATA_SIZE:
                    raise ValueError(f"{suffix.upper()} dosyası 5 MiB sınırını aşıyor.")
            if selected["img"].size > MAX_LOCAL_IMG_SIZE:
                raise ValueError("IMG dosyası 50 MiB sınırını aşıyor.")
            temp_dir, img_path = save_uploaded_img_to_disk(selected["img"])
            new_dem, new_metadata = load_mola_envi_from_path(
                img_path, selected["hdr"].getvalue(),
                selected["lbl"].getvalue() if selected["lbl"] else None,
            )
            new_metadata["source_mode"] = "Bilgisayardan toplu yükleme"
            new_metadata["temporary_directory"] = temp_dir
        elif by_suffix.get("hdr") or by_suffix.get("lbl"):
            raise ValueError(".hdr/.lbl dosyaları tek başına kullanılamaz; .img dosyası da seçin.")
        geotiffs = by_suffix.get("tif", []) + by_suffix.get("tiff", [])
        if geotiffs:
            if new_dem is not None:
                raise ValueError("Aynı yüklemede yalnızca bir DEM kaynağı seçin (.img veya GeoTIFF).")
            if geotiffs[0].size > 500 * 1024 * 1024:
                raise ValueError("GeoTIFF dosyası 500 MiB sınırını aşıyor.")
            temp_dir, tif_path = save_uploaded_file_to_disk(
                geotiffs[0], prefix="mars_geotiff_"
            )
            new_dem, new_metadata = load_geotiff_dem_from_path(tif_path)
            new_metadata["temporary_directory"] = temp_dir
        txt_files = by_suffix.get("txt", [])
        new_layers = []
        for uploaded in txt_files:
            if uploaded.size > MAX_MCD_TEXT_SIZE:
                raise ValueError(f"{uploaded.name}: TXT dosyası 20 MiB sınırını aşıyor.")
            try:
                text_data = uploaded.getvalue().decode("utf-8-sig")
            except UnicodeDecodeError:
                text_data = uploaded.getvalue().decode("latin-1")
            new_layers.extend(parse_mcd_text(text_data, source=uploaded.name))
        nc_files = by_suffix.get("nc", [])
        if nc_files:
            if nc_files[0].size > MAX_EMARS_UPLOAD_SIZE:
                raise ValueError("NetCDF dosyası 140 MiB sınırını aşıyor.")
            nc_dir = tempfile.mkdtemp(prefix="mars_emars_")
            nc_path = Path(nc_dir) / Path(nc_files[0].name).name
            nc_path.write_bytes(nc_files[0].getvalue())
            new_ds = open_emars(nc_path)
        if not files:
            raise ValueError("En az bir desteklenen dosya seçin.")
        st.session_state.pop("route_plan", None)
        st.session_state.pop("route_plan_signature", None)
        if new_dem is not None:
            old_temp_dir = st.session_state.get("local_dem_directory")
            st.session_state["local_dem"] = new_dem
            st.session_state["local_metadata"] = new_metadata
            st.session_state["local_dem_directory"] = temp_dir
            if old_temp_dir and old_temp_dir != temp_dir:
                shutil.rmtree(old_temp_dir, ignore_errors=True)
        if new_layers:
            st.session_state["mcd_layers"] = new_layers
        if new_ds is not None:
            old_ds = st.session_state.get("emars_ds")
            old_dir = st.session_state.get("emars_temp_dir")
            st.session_state["emars_ds"] = new_ds
            st.session_state["emars_temp_dir"] = nc_dir
            if old_ds is not None:
                old_ds.close()
            if old_dir:
                shutil.rmtree(old_dir, ignore_errors=True)
        st.success("Seçilen proje verileri yüklendi.")
    except Exception as exc:
        if new_ds is not None:
            new_ds.close()
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        if nc_dir:
            shutil.rmtree(nc_dir, ignore_errors=True)
        st.error(f"Veriler yüklenemedi veya işlenemedi: {exc}")
if st.session_state.get("emars_ds") is not None:
    ds = st.session_state["emars_ds"]
    time_count = int(ds.sizes.get("time", 0))
    if time_count and ds.sizes.get("pfull", 0):
        st.subheader("Atmosfer de\u011fi\u015fken haritalar\u0131")
        time_index = st.slider(
            "Mars zaman indeksi",
            min_value=0,
            max_value=time_count - 1,
            value=0,
            key="emars_map_time_index",
        )
        level_index = get_surface_pressure_level_index(ds)
        pressure, pressure_unit = get_pressure_level_info(ds, level_index)
        try:
            mars_time = get_time_information(ds, int(time_index))
            time_label = (
                f"Mars y\u0131l\u0131 {mars_time['mars_year']:g}; "
                f"Ls {mars_time['solar_longitude']:g}\u00b0; "
                f"sol {mars_time['mars_sol']:g}, saat {mars_time['mars_hour']:g}"
            )
        except (KeyError, ValueError, IndexError):
            time_label = f"Zaman indeksi {time_index} / {time_count - 1}"
        st.caption(
            f"{time_label}. Bas\u0131n\u00e7a ba\u011fl\u0131 katmanlarda "
            f"y\u00fczeye en yak\u0131n seviye kullan\u0131l\u0131yor: "
            f"{pressure:g} {pressure_unit}."
        )

        resolved_fields = resolve_atmospheric_map_variables(ds)
        field_labels = list(resolved_fields)
        field_tabs = st.tabs([
            f"{label} ({resolved_fields[label] or 'veri yok'})"
            for label in field_labels
        ])
        for label, tab in zip(field_labels, field_tabs):
            with tab:
                variable_name = resolved_fields[label]
                if variable_name is None:
                    st.warning(f"Dosyada {label} i\u00e7in uygun bir de\u011fi\u015fken bulunamad\u0131.")
                    continue
                try:
                    variable = ds[variable_name]
                    values = get_atmospheric_map_field(
                        ds, variable_name, int(time_index), level_index
                    )
                    if label.startswith("W-E"):
                        score_u = values
                    elif label.startswith("N-S"):
                        score_v = values
                    elif label.startswith("Toz optik"):
                        score_dust = values
                    units = str(variable.attrs.get("units", "")).strip()
                    if not units:
                        units = (
                            "m/s" if label.startswith(("W-E", "N-S"))
                            else "boyutsuz"
                        )
                    valid = values[np.isfinite(values)]
                    if valid.size == 0:
                        raise ValueError("Bu zaman ve bas\u0131n\u00e7 seviyesinde ge\u00e7erli h\u00fccre yok.")
                    component = label.startswith(("W-E", "N-S"))
                    limit = float(np.max(np.abs(valid))) if component else None
                    display_label = tx(label)
                    fig = plot_emars_map(
                        ds=ds,
                        values=values,
                        title=f"{display_label}: {variable_name}",
                        colorbar_label=f"{display_label} ({units})",
                        cmap="RdBu_r" if component else "cividis",
                        vertices=DEFAULT_STUDY_AREA_VERTICES,
                        vmin=-limit if component else None,
                        vmax=limit if component else None,
                        language=language,
                    )
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                    st.caption(
                        f"Kaynak de\u011fi\u015fken: {variable_name}; "
                        f"birim: {units}; zaman indeksi: {time_index}."
                    )
                except Exception as exc:
                    st.error(f"{label} haritas\u0131 olu\u015fturulamad\u0131: {exc}")


# MCD YATAY RÜZGÂR ANALİZİ
# --------------------------------------------------

if st.session_state.get("mcd_layers"):

    temperature_layers = [
        layer for layer in st.session_state["mcd_layers"]
        if any(word in layer.get("description", "").lower() for word in ("temperature", "temp", "t_surf", "tsurf", "sıcaklık"))
    ]
    if temperature_layers:
        st.header("MCD TXT   sıcaklık haritaları")
        st.caption(f"{len(temperature_layers)} sıcaklık katmanı bulundu; en fazla dört harita gösterilir.")
        selected_temperature_layers = temperature_layers[:4]
        surface_candidates = [
            layer for layer in temperature_layers
            if "surface temperature" in layer.get("description", "").lower()
            and "rms" not in layer.get("description", "").lower()
        ]
        if surface_candidates:
            surface_layer = surface_candidates[0]
            score_temperature_source = surface_layer
            score_temperature = mask_geographic_grid(
                surface_layer["values"],
                surface_layer["longitudes"],
                surface_layer["latitudes"],
                values_order="lonlat",
            )
        temperature_tabs = st.tabs([
            f"Sıcaklık {index + 1}"
            for index in range(len(selected_temperature_layers))
        ])
        for index, (tab, layer) in enumerate(
            zip(temperature_tabs, selected_temperature_layers), start=1
        ):
            with tab:
                st.subheader(layer["description"])
                try:
                    fig, stats = plot_mcd_layer(layer, language=language)
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                    cols = st.columns(3)
                    unit = layer.get("unit") or "birim belirtilmemiş"
                    cols[0].metric("Minimum", f"{stats['min']:.3f} {unit}")
                    cols[1].metric("Ortalama", f"{stats['mean']:.3f} {unit}")
                    cols[2].metric("Maksimum", f"{stats['max']:.3f} {unit}")
                except Exception as exc:
                    st.error(f"{layer['description']} haritası üretilemedi: {exc}")

# MCD ATMOSFERİK VERİ KATMANLARI
# --------------------------------------------------

if st.session_state.get("mcd_layers"):

    st.header("Mars Atmosferi - Veri Katmanları")

    for index, layer in enumerate(st.session_state["mcd_layers"]):
        description = layer.get("description", "").lower()
        if (
            any(word in description for word in ("temperature", "temp", "t_surf", "tsurf"))
            or "wind component" in description
            or any(word in description for word in ("rho", "density", "tauref", "tau_ref", "dust optical"))
        ):
            continue

        st.subheader(layer["description"])

        try:
            fig, stats = plot_mcd_layer(layer, language=language)

            st.pyplot(fig, width="stretch")
            plt.close(fig)

            col1, col2, col3 = st.columns(3)

            col1.metric("Minimum", f"{stats['min']:.3f}")
            col2.metric("Ortalama", f"{stats['mean']:.3f}")
            col3.metric("Maksimum", f"{stats['max']:.3f}")

        except Exception as exc:
            st.error(
                f"{layer['description']} görüntülenemedi: {exc}"
            )

dem = None
metadata = None

if source_mode == "pds":
    pds_type_labels = {
        "img": "MOLA ENVI DEM (.img + .hdr)",
        "tiff": "GeoTIFF DEM (.tif / .tiff)",
        "nc": "EMARS atmosfer verisi (.nc)",
        "txt": "MCD atmosfer verisi (.txt)",
    }
    st.subheader("PDS ar\u015fiv verileri")
    st.write("Desteklenen verilerin do\u011frudan HTTPS URL adreslerini girin. ENVI IMG adresi e\u015fle\u015fen HDR dosyas\u0131n\u0131 ve iste\u011fe ba\u011fl\u0131 LBL dosyas\u0131n\u0131 da al\u0131r.")
    st.caption("PDS ve herkese a\u00e7\u0131k GitHub Release indirme adresleri desteklenir. Ba\u015fka bir HTTPS sunucusu i\u00e7in MARS_DATA_ALLOWED_HOSTS ayar\u0131n\u0131 yap\u0131n.")
    url_count = int(st.session_state.get("pds_url_count", 1))
    if st.button("Daha fazla URL ekle", key="add_pds_url"):
        st.session_state["pds_url_count"] = url_count + 1
        st.rerun()
    url_rows = []
    for index in range(url_count):
        pds_type = st.selectbox(
            f"PDS \u00fcr\u00fcn t\u00fcr\u00fc {index + 1}",
            options=list(pds_type_labels),
            format_func=lambda code: translate(pds_type_labels[code], language),
            key=f"pds_type_{index}",
        )
        pds_url = st.text_input(
            f"PDS dosya URL\u2019si {index + 1}",
            value="",
            placeholder="https://your-host.example/path/product.ext",
            key=f"pds_img_url_{index}",
        )
        url_rows.append((pds_type, pds_url.strip()))
    start_download = st.button("PDS verilerini indir ve analize haz\u0131rla", type="primary")

    if start_download:
        requested_rows = [(kind, url) for kind, url in url_rows if url]
        downloaded_assets = []
        new_ds = None
        status = None
        try:
            if not requested_rows:
                raise ValueError("En az bir PDS URL adresi girin.")
            if sum(kind == "nc" for kind, _ in requested_rows) > 1:
                raise ValueError("Her indirmede yaln\u0131zca bir EMARS NetCDF dosyas\u0131 se\u00e7in.")
            status = st.status("PDS verileri indiriliyor...", expanded=True)
            type_extensions = {
                "tiff": (".tif", ".tiff"),
                "nc": (".nc",),
                "txt": (".txt",),
            }
            for index, (kind, url) in enumerate(requested_rows, start=1):
                status.update(label=f"PDS \u00fcr\u00fcn\u00fc indiriliyor ({index}/{len(requested_rows)})...", state="running")
                if kind == "img":
                    path, hdr, lbl, normalized_url = download_pds_product(url)
                    downloaded_assets.append({"kind": kind, "path": path, "hdr_bytes": hdr, "lbl_bytes": lbl, "url": normalized_url})
                else:
                    path, normalized_url = download_pds_asset(url, type_extensions[kind])
                    downloaded_assets.append({"kind": kind, "path": path, "url": normalized_url})
            dem_products = [item for item in downloaded_assets if item["kind"] in ("img", "tiff")]
            if not dem_products:
                raise ValueError("Analiz i\u00e7in y\u00fckseklik gridini olu\u015fturacak en az bir DEM (.img veya GeoTIFF) ekleyin.")
            if sum(item["kind"] == "tiff" for item in dem_products) > 1:
                raise ValueError("Ayn\u0131 anda yaln\u0131zca bir GeoTIFF DEM indirin.")

            new_layers = []
            for item in downloaded_assets:
                if item["kind"] == "nc":
                    new_ds = open_emars(item["path"])
                elif item["kind"] == "txt":
                    raw_text = Path(item["path"]).read_bytes()
                    try:
                        text_data = raw_text.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        text_data = raw_text.decode("latin-1")
                    new_layers.extend(parse_mcd_text(text_data, source=item["url"].rsplit("/", 1)[-1]))

            old_assets = st.session_state.get("pds_downloads", [])
            old_ds = st.session_state.get("emars_ds")
            if old_ds is not None:
                old_ds.close()
            st.session_state["pds_downloads"] = downloaded_assets
            st.session_state["pds_products"] = dem_products
            st.session_state.pop("selected_pds_product_index", None)
            st.session_state.pop("route_plan", None)
            st.session_state.pop("route_plan_signature", None)
            if new_ds is not None:
                st.session_state["emars_ds"] = new_ds
            else:
                st.session_state.pop("emars_ds", None)
            if new_layers:
                st.session_state["mcd_layers"] = new_layers
            else:
                st.session_state.pop("mcd_layers", None)
            new_paths = {item["path"] for item in downloaded_assets}
            for old in old_assets:
                if old.get("path") not in new_paths:
                    try:
                        os.remove(old["path"])
                    except OSError:
                        pass
            status.update(label=f"\u0130ndirilen PDS dosyas\u0131 say\u0131s\u0131: {len(downloaded_assets)}", state="complete")
        except Exception as exc:
            if new_ds is not None:
                new_ds.close()
            for item in downloaded_assets:
                try:
                    os.remove(item["path"])
                except OSError:
                    pass
            if status is not None:
                status.update(label="PDS indirmesi ba\u015far\u0131s\u0131z oldu.", state="error")
            st.error(f"PDS verileri indirilemedi veya i\u015flenmedi: {exc}")

    products = st.session_state.get("pds_products", [])
    if not products:
        st.info("Analizden \u00f6nce en az bir PDS DEM URL\u2019si indirmeniz gerekiyor.")
        st.stop()
    selected_product_index = st.selectbox(
        "Analiz edilecek DEM \u00fcr\u00fcn\u00fc",
        options=list(range(len(products))),
        format_func=lambda index: f"{translate(pds_type_labels[products[index]['kind']], language)} | {products[index]['url']}",
        key="selected_pds_product_index",
    )
    product = products[selected_product_index]
    if not os.path.exists(product["path"]):
        st.warning("PDS ge\u00e7ici dosyas\u0131 bulunamad\u0131. Dosyay\u0131 yeniden indirin.")
        st.stop()

    if product["kind"] == "img":
        try:
            dem, metadata = load_mola_envi_from_path(product["path"], product["hdr_bytes"], product["lbl_bytes"])
        except Exception as exc:
            st.error(f"\u0130ndirilen MOLA raster\u0131 a\u00e7\u0131lamad\u0131: {exc}")
            st.stop()
        lbl_availability = "mevcut" if product["lbl_bytes"] else "bulunamad\u0131 (iste\u011fe ba\u011fl\u0131)"
        st.caption(f"Kaynak: {product['url']} | HDR: mevcut | LBL: {lbl_availability}")
    else:
        try:
            dem, metadata = load_geotiff_dem_from_path(product["path"])
        except Exception as exc:
            st.error(f"\u0130ndirilen GeoTIFF DEM a\u00e7\u0131lamad\u0131: {exc}")
            st.stop()
        st.caption(f"Kaynak: {product['url']}")
    metadata["source_mode"] = "PDS do\u011frudan indirme"

    if st.sidebar.button("\u0130ndirilen ge\u00e7ici verileri temizle"):
        for downloaded in st.session_state.get("pds_downloads", []):
            try:
                os.remove(downloaded["path"])
            except OSError:
                pass
        old_ds = st.session_state.get("emars_ds")
        if old_ds is not None:
            old_ds.close()
        for key in ("pds_downloads", "pds_products", "pds_product", "emars_ds", "mcd_layers"):
            st.session_state.pop(key, None)
        st.rerun()

else:
    dem = st.session_state.get("local_dem")
    metadata = st.session_state.get("local_metadata")



if dem is None or metadata is None:
    st.stop()
try:
    stats = get_dem_statistics(dem, metadata)
except Exception as exc:
    st.error(f"Raster istatistikleri hesaplanamadı: {exc}")
    st.stop()

st.subheader("Veri özeti")
metric_cols = st.columns(5)
metric_cols[0].metric("Genişlik", f"{metadata['samples']:,} px")
metric_cols[1].metric("Yükseklik", f"{metadata['lines']:,} px")
pixel_units = "°" if (metadata.get("map_units") or "").lower() in ("degree", "degrees") else "m"
metric_cols[2].metric("Piksel boyutu", f"{metadata['pixel_x']:.6f} {pixel_units}")
metric_cols[3].metric("Geçerli veri", f"{stats['valid_percent']:.2f}%")
metric_cols[4].metric("Veri tipi", metadata["dtype_name"])

with st.expander("Raster metadata bilgileri"):
    st.json(metadata["label_metadata"])
    st.write("**Map info**")
    st.code(metadata.get("map_info_raw") or "Başlıkta bulunamadı")
    st.write("**Raster okuma ayarları**")
    st.json({
        "samples": metadata["samples"],
        "lines": metadata["lines"],
        "bands": metadata["bands"],
        "Raster data type": metadata["data_type"],
        "byte order": metadata["byte_order"],
        "dtype": metadata["dtype_name"],
        "NoData": metadata["nodata"],
        "scale factor": metadata["scale_factor"],
        "offset": metadata["offset"],
        "height unit": metadata["height_unit"],
        "file size (MiB)": round(metadata["file_size_bytes"] / (1024 ** 2), 2),
        "storage mode": metadata.get("storage_mode", "uploaded file buffer"),
        "source mode": metadata.get("source_mode"),
    })

st.subheader("Yükseklik istatistikleri")
st.caption("Minimum, maksimum ve ortalama geçerli raster piksellerinden; medyan ise örneklenmiş piksellerden hesaplanır.")
stat_cols = st.columns(5)
stat_cols[0].metric("Minimum", f"{stats['min']:.2f} {metadata['height_unit']}")
stat_cols[1].metric("Maksimum", f"{stats['max']:.2f} {metadata['height_unit']}")
stat_cols[2].metric("Ortalama", f"{stats['mean']:.2f} {metadata['height_unit']}")
stat_cols[3].metric("Medyan (yaklaşık)", f"{stats['median_sampled']:.2f} {metadata['height_unit']}")
stat_cols[4].metric("Standart sapma", f"{stats['std']:.2f} {metadata['height_unit']}")

preview = make_display_sample(dem, metadata, step=display_step)
fig, ax = plt.subplots(figsize=(10, 8.6), dpi=180)
image = ax.imshow(
    np.ma.masked_invalid(preview),
    cmap="terrain",
    origin="upper",
    interpolation="nearest",
)
ax.set_title(tx("Topografya rasterı   örneklenmiş görünüm"))
ax.set_xlabel(tx("Raster sütunu (piksel)"))
ax.set_ylabel(tx("Raster satırı (piksel)"))
fig.colorbar(image, ax=ax, label=f"{tx('Yükseklik')} ({metadata['height_unit']})")
fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)
# ==========================================================
# LYCUS SULCI - UCGEN CALISMA ALANI
# ==========================================================

st.divider()
st.subheader("Lycus Sulci   Üçgen çalışma alanı")

st.write(
    "Önceden belirlenen üçgen bölge DEM rasterından "
    "otomatik olarak kesilir. Yalnızca poligon içerisinde "
    "kalan geçerli pikseller analiz edilir."
)

try:
    try:
        study_area = extract_study_area(
            dem, metadata, vertices=DEFAULT_STUDY_AREA_VERTICES,
        )
        study_area["area_mode"] = "Lycus Sulci üçgeni"
    except Exception:
        # Regional alternate datasets may not overlap the preset triangle.
        study_area = extract_study_area(
            dem, metadata, vertices=raster_coverage_vertices(dem, metadata),
        )
        study_area["area_mode"] = "yüklenen raster kapsamı"
        st.info("Yüklenen DEM varsayılan üçgeni kapsamıyor; adaylar bu veri setinin geçerli raster alanında aranıyor.")

    st.success(f"Çalışma alanı DEM rasterından çıkarıldı: {study_area['area_mode']}.")

    area_cols = st.columns(5)

    area_cols[0].metric(
        "Minimum yükseklik",
        f"{study_area['min_elevation']:.2f} {metadata['height_unit']}",
    )

    area_cols[1].metric(
        "Maksimum yükseklik",
        f"{study_area['max_elevation']:.2f} {metadata['height_unit']}",
    )

    area_cols[2].metric(
        "Ortalama yükseklik",
        f"{study_area['mean_elevation']:.2f} {metadata['height_unit']}",
    )

    area_cols[3].metric(
        "Medyan yükseklik",
        f"{study_area['median_elevation']:.2f} {metadata['height_unit']}",
    )

    area_cols[4].metric(
        "Geçerli piksel",
        f"{study_area['valid_pixels']:,}",
    )

    st.caption(
        f"Raster penceresi: "
        f"{study_area['row_start']}–{study_area['row_stop']} satır, "
        f"{study_area['col_start']}–{study_area['col_stop']} sütun."
    )

    # Üçgen alanın yükseklik haritası.
    fig_area, ax_area = plt.subplots(figsize=(10, 8.6), dpi=180)

    area_image = ax_area.imshow(
        np.ma.masked_invalid(study_area["background_elevation"]),
        cmap="terrain",
        origin="upper",
        interpolation="nearest",
    )

    # Üçgen sınırını haritanın üzerine çiz.
    vertices = study_area["vertices"]

    polygon_lon = [point[0] for point in vertices]
    polygon_lat = [point[1] for point in vertices]

    polygon_lon.append(polygon_lon[0])
    polygon_lat.append(polygon_lat[0])

    # Raster koordinatlarıyla aynı eksenlerde çiz.
    map_info = study_area["map_info"]

    polygon_cols = [
        geographic_to_pixel(lon, lat, map_info)[1]
        - study_area["col_start"]
        for lon, lat in zip(polygon_lon, polygon_lat)
    ]

    polygon_rows = [
        geographic_to_pixel(lon, lat, map_info)[0]
        - study_area["row_start"]
        for lon, lat in zip(polygon_lon, polygon_lat)
    ]

    ax_area.plot(
        polygon_cols,
        polygon_rows,
        linewidth=2,
        color="red",
    )

    for index, (x, y) in enumerate(
        zip(polygon_cols[:-1], polygon_rows[:-1])
    ):
        ax_area.scatter(x, y, color="red", s=35)
        ax_area.annotate(
            chr(65 + index),
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            color="red",
            fontsize=10,
        )

    ax_area.set_title(tx("Lycus Sulci   Üçgen çalışma alanı yükseklik haritası"))

    ax_area.set_xlabel(tx("Raster sütunu"))
    ax_area.set_ylabel(tx("Raster satırı"))

    fig_area.colorbar(
        area_image,
        ax=ax_area,
        label=f"{tx('Yükseklik')} ({metadata['height_unit']})",
    )

    fig_area.tight_layout()

    st.pyplot(fig_area, width="stretch")
    plt.close(fig_area)

    st.caption(
        "Kırmızı çizgi çalışma alanının sınırlarını gösterir. "
        "Haritadaki yükseklikler seçilen DEM kaynağından alınmıştır. "
        "Bu analiz henüz rover geçiş güvenliği değerlendirmesi değildir."
    )

    # ==========================================================
    # YEREL ARAZI PÜRÜZLÜLÜĞÜ
    # ==========================================================

    st.divider()
    st.subheader("Bölgesel arazi pürüzlülüğü")

    try:
        with st.spinner("Pürüzlülük değerleri hesaplanıyor..."):

            # Üçgenin çevresindeki DEM verisini kullan.
            # Böylece üçgen sınırındaki piksellerin komşuları kaybolmaz.
            roughness_source = study_area["background_elevation"]

            roughness_native = calculate_local_roughness(
                roughness_source
            )

            # Kaynak yüksekliğinin birimini metreye dönüştür.
            height_unit = str(
                metadata.get("height_unit", "")
            ).strip().lower()

            unit_scales = {
                "m": 1.0,
                "meter": 1.0,
                "meters": 1.0,
                "metre": 1.0,
                "metres": 1.0,
                "km": 1000.0,
                "kilometer": 1000.0,
                "kilometers": 1000.0,
                "kilometre": 1000.0,
                "kilometres": 1000.0,
                "ft": 0.3048,
                "feet": 0.3048,
                "foot": 0.3048,
            }

            vertical_scale = metadata.get(
                "vertical_scale_to_m"
            )

            if vertical_scale is None:
                vertical_scale = unit_scales.get(height_unit)

            if vertical_scale is None:
                raise ValueError(
                    "DEM yükseklik birimi metreye dönüştürülemiyor. "
                    "PDS LBL/HDR ürün bilgisinden birimi doğrulamamız gerekiyor."
                )

            vertical_scale = float(vertical_scale)

            if not np.isfinite(vertical_scale) or vertical_scale <= 0:
                raise ValueError(
                    "Düşey birim dönüşüm katsayısı geçersiz."
                )

            # Metreye dönüştür.
            roughness = roughness_native * vertical_scale
            landing_roughness = roughness

        score_roughness = roughness[study_area["polygon_mask"]]

        # Haritada çevre bağlamı kalsın; sayısal özet üçgen içinde hesaplansın.
        roughness_valid = roughness[
            study_area["polygon_mask"] & np.isfinite(roughness)
        ]

        if roughness_valid.size == 0:
            st.warning(
                "Pürüzlülük hesaplamak için yeterli geçerli komşu piksel bulunamadı."
            )

        else:
            # İstatistikler
            roughness_cols = st.columns(4)

            roughness_cols[0].metric(
                "Ortalama pürüzlülük",
                f"{np.mean(roughness_valid):.3f} m",
            )

            roughness_cols[1].metric(
                "Medyan pürüzlülük",
                f"{np.median(roughness_valid):.3f} m",
            )

            roughness_cols[2].metric(
                "95. yüzdelik",
                f"{np.percentile(roughness_valid, 95):.3f} m",
            )

            roughness_cols[3].metric(
                "Maksimum",
                f"{np.max(roughness_valid):.3f} m",
            )

            # Harita
            fig_rough, ax_rough = plt.subplots(
                figsize=(10, 8.6),
                dpi=180,
            )

            rough_image = ax_rough.imshow(
                np.ma.masked_invalid(roughness),
                cmap="magma",
                origin="upper",
                interpolation="nearest",
                aspect="equal",
            )

            # Üçgen sınırı
            ax_rough.plot(
                polygon_cols,
                polygon_rows,
                color="cyan",
                linewidth=1.5,
        label=tx("Çalışma alanı sınırı"),
            )

            ax_rough.scatter(
                polygon_cols[:-1],
                polygon_rows[:-1],
                color="cyan",
                s=20,
            )

            ax_rough.legend()

            ax_rough.set_title(tx("Lycus Sulci   Yerel arazi pürüzlülüğü"))

            ax_rough.set_xlabel(tx("Raster sütunu"))
            ax_rough.set_ylabel(tx("Raster satırı"))

            fig_rough.colorbar(
                rough_image,
                ax=ax_rough,
                label=tx("Yerel düzlemden RMS sapma (m)"),
            )

            fig_rough.tight_layout()

            st.pyplot(fig_rough, width="stretch")
            plt.close(fig_rough)

            st.caption(
                "Pürüzlülük, 3×3 piksel pencerelerinde en küçük kareler "
                "yöntemiyle uydurulan yerel düzlemden RMS yükseklik sapması "
                "olarak hesaplanmıştır. Harita üçgen çevresindeki verileri de "
                "gösterir; istatistikler çalışma üçgeninin içinden hesaplanır."
            )

            st.info(
                "Bu değer bölgesel arazi pürüzlülüğünü temsil eder. "
                "DEM çözünürlüğü nedeniyle küçük kaya ve rover tekerleği "
                "ölçeğindeki engelleri belirlemek için kullanılamaz."
            )

    except Exception as exc:
        st.error(
            f"Pürüzlülük analizi başarısız oldu: {exc}"
        )

    with st.expander("Çalışma alanı koordinatları"):
        for index, (lon, lat) in enumerate(vertices):
            st.write(
                f"{chr(65 + index)}: "
                f"{lon:.3f}°E, {lat:.3f}°N"
            )

except Exception as exc:
    st.error(
        f"Üçgen çalışma alanı analiz edilemedi: {exc}"
    )



# ==========================================================
# BÖLGESEL EĞİM ANALİZİ
# ==========================================================

st.divider()
st.subheader("Bölgesel eğim analizi")

try:
    with st.spinner("Eğim örnekleri hesaplanıyor..."):
        slope = calculate_sampled_slope(
            dem,
            metadata,
            step=slope_step,
        )

    slope_rows = np.arange(
        0, dem.shape[0], slope_step, dtype=np.int64
    )
    slope_cols = np.arange(
        0, dem.shape[1], slope_step, dtype=np.int64
    )

    # Align sampled slope pixels to the triangle crop used by MOLA roughness.
    row_sel = np.flatnonzero(
        (slope_rows >= study_area["row_start"]) & (slope_rows < study_area["row_stop"])
    )
    col_sel = np.flatnonzero(
        (slope_cols >= study_area["col_start"]) & (slope_cols < study_area["col_stop"])
    )
    local_rows = slope_rows[row_sel] - study_area["row_start"]
    local_cols = slope_cols[col_sel] - study_area["col_start"]
    sampled_slope_roi = slope[np.ix_(row_sel, col_sel)]
    sampled_polygon_mask = study_area["polygon_mask"][np.ix_(local_rows, local_cols)]
    score_slope = sampled_slope_roi[sampled_polygon_mask]

    slope_valid = slope[np.isfinite(slope)]

    if slope_valid.size == 0:
        st.warning(
            "Eğim hesaplamak için yeterli geçerli komşu piksel bulunamadı."
        )

    else:
        slope_metric_cols = st.columns(3)

        slope_metric_cols[0].metric(
            "Ortalama eğim",
            f"{float(np.mean(slope_valid)):.2f}°",
        )

        slope_metric_cols[1].metric(
            "Medyan eğim",
            f"{float(np.median(slope_valid)):.2f}°",
        )

        slope_metric_cols[2].metric(
            "95. yüzdelik",
            f"{float(np.percentile(slope_valid, 95)):.2f}°",
        )

        triangle_rows = []
        triangle_cols = []

        map_info = parse_geographic_map_info(metadata)

        for lon, lat in study_area["vertices"]:
            row, col = geographic_to_pixel(
                float(lon),
                float(lat),
                map_info,
            )

            row_index = int(
                np.argmin(np.abs(slope_rows - row))
            )

            col_index = int(
                np.argmin(np.abs(slope_cols - col))
            )

            triangle_rows.append(row_index)
            triangle_cols.append(col_index)

        triangle_rows.append(triangle_rows[0])
        triangle_cols.append(triangle_cols[0])

        fig2, ax2 = plt.subplots(
            figsize=(10, 8.6),
            dpi=180,
        )

        slope_image = ax2.imshow(
            np.ma.masked_invalid(slope),
            cmap="terrain",
            origin="upper",
            interpolation="nearest",
            vmin=0,
            vmax=15,
            aspect="equal",
        )

        ax2.plot(
            triangle_cols,
            triangle_rows,
            color="cyan",
            linewidth=1.5,
                label=tx("Çalışma alanı sınırı"),
        )

        ax2.scatter(
            triangle_cols[:-1],
            triangle_rows[:-1],
            color="cyan",
            s=20,
        )

        ax2.legend()
        ax2.set_title(tx("Mars yüzey eğimi"))
        ax2.set_xlabel(tx("Örnek sütun indeksi"))
        ax2.set_ylabel(tx("Örnek satır indeksi"))

        fig2.colorbar(
            slope_image,
            ax=ax2,
            label=tx("Eğim (derece)"),
        )

        fig2.tight_layout()
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)

        st.caption(
            f"Eğim, orijinal DEM komşu piksellerinden hesaplandı. "
            f"Harita her {slope_step} pikselde bir örnek içerir. "
            "Renk aralığı 0–15° olarak sınırlandırılmıştır; "
            "15° üzerindeki değerler aynı üst renk aralığında gösterilir."
        )

except Exception as exc:
    st.error(f"Eğim analizi başarısız oldu: {exc}")


# ==========================================================
# BİRLEŞİK YER SEÇİMİ ÖN DEĞERLENDİRMESİ
# ==========================================================
st.divider()
st.subheader("Birleşik bölgesel uygunluk ön değerlendirmesi")
wind_speed_for_score = None
if score_u is not None and score_v is not None and score_u.shape == score_v.shape:
    wind_speed_for_score = mask_geographic_grid(
        np.hypot(score_u, score_v), ds["lon"].values, ds["lat"].values
    )
score_dust_for_score = score_dust
if score_dust is not None and st.session_state.get("emars_ds") is not None:
    score_dust_for_score = mask_geographic_grid(
        score_dust,
        st.session_state["emars_ds"]["lon"].values,
        st.session_state["emars_ds"]["lat"].values,
    )
try:
    suitability = calculate_site_suitability(
        slope_deg=score_slope,
        roughness_m=score_roughness,
        surface_temperature_k=score_temperature,
        wind_speed_m_s=wind_speed_for_score,
        dust_optical_depth=score_dust_for_score,
    )
    metrics = st.columns(3)
    metrics[0].metric("Birleşik ön değerlendirme", f"{suitability['overall_score']:.1f} / 100")
    metrics[1].metric("Veri güvenilirliği", f"{suitability['data_reliability']:.1f}%")
    metrics[2].metric("Atmosfer verisi", "Uygun" if suitability["available"]["atmospheric"] else "Eksik")
    details = st.columns(5)
    labels = [
        ("slope", "Eğim"), ("roughness", "Pürüzlülük"),
        ("thermal", "Termal koşullar"), ("atmospheric", "Rüzgâr + toz"),
        ("data_reliability", "Veri güvenilirliği"),
    ]
    for column, (key, title) in zip(details, labels):
        column.metric(title, f"{suitability['component_scores'][key]:.1f} / 100")
    missing_names = {
        "slope": "Eğim", "roughness": "Pürüzlülük", "thermal": "Termal koşullar",
        "atmospheric": "Rüzgâr + toz", "data_reliability": "Veri güvenilirliği",
    }
    missing = [missing_names.get(name, name) for name, available in suitability["available"].items() if not available]
    if missing:
        st.warning("Eksik bileşenler: " + ", ".join(missing) + ". Eksiklikler güvenilirlik puanını düşürür.")
    st.caption(
        "Tek fonksiyon eğim, pürüzlülük, yüzey sıcaklığı, rüzgâr/toz ve veri güvenilirliğini birleştirir. "
        "İdeal aralık dışındaki değerler kademeli maliyet alır; eşikler başlangıç varsayımıdır, görev güvenlik sınırı değildir."
    )
    with st.expander("Veri güvenilirliği ayrıntıları"):
        coverage_rows = []
        for key, title in [("slope", "Slope"), ("roughness", "Roughness"), ("thermal", "Thermal"), ("wind", "Wind"), ("dust", "Dust")]:
            count = suitability["sample_counts"][key]
            coverage_rows.append({
                "Katman": title,
                "Geçerli / toplam hücre": f"{count['valid']:,} / {count['total']:,}",
                "Kapsama": f"{suitability['coverage'].get(key, suitability['coverage'].get('atmospheric', 0)):.1f}%",
            })
        st.table(coverage_rows)
        st.caption("Hücre sayısı grid ayrıntısını; kapsama ise çalışma alanındaki geçerli veriyi ve örnek yeterliliğini yansıtır. Yüksek hücre sayısı tek başına ölçüm doğruluğunu kanıtlamaz.")
except Exception as exc:
    st.error(f"Birleşik uygunluk hesabı yapılamadı: {exc}")


# ==========================================================
# ÖNERİLEN İNİŞ ALANLARI — HEDEF VE ROTA SONRA EKLENECEK
# ==========================================================
st.divider()
st.subheader("Önerilen iniş noktaları")
st.caption("Hedef nokta bu aşamada kullanılmıyor. Aşağıdaki 10 aday, mevcut çalışma alanı verilerinde yumuşak maliyet ve veri güvenilirliği birlikte değerlendirilerek sıralanır.")
try:
    stride = int(slope_step)
    local_dem = np.asarray(study_area["background_elevation"], dtype=float)
    local_metadata = metadata.copy()
    local_metadata["map_reference_col"] = float(metadata["map_reference_col"]) - study_area["col_start"]
    local_metadata["map_reference_row"] = float(metadata["map_reference_row"]) - study_area["row_start"]
    local_metadata["scale_factor"] = 1.0
    local_metadata["offset"] = 0.0
    local_metadata["nodata"] = np.nan
    local_slope = calculate_sampled_slope(local_dem, local_metadata, step=stride)
    target_lon = study_area["longitude"][::stride, ::stride]
    target_lat = study_area["latitude"][::stride, ::stride]
    target_mask = study_area["polygon_mask"][::stride, ::stride]
    local_roughness = np.asarray(landing_roughness, dtype=float)[::stride, ::stride]

    thermal_grid = None
    thermal_factor = 0.0
    if score_temperature_source is not None:
        source = score_temperature_source
        thermal_grid = regrid_geographic_field(
            source["values"], source["longitudes"], source["latitudes"], target_lon, target_lat,
            values_order="lonlat",
        )
        thermal_factor = native_grid_support(
            source["longitudes"], source["latitudes"], source["values"], target_lon, target_lat, target_mask,
            values_order="lonlat",
        )

    wind_grid = dust_grid = None
    wind_factor = dust_factor = 0.0
    source_ds = st.session_state.get("emars_ds")
    if score_u is not None and score_v is not None and source_ds is not None:
        native_wind = np.hypot(score_u, score_v)
        wind_grid = regrid_geographic_field(
            native_wind, source_ds["lon"].values, source_ds["lat"].values, target_lon, target_lat
        )
        wind_factor = native_grid_support(
            source_ds["lon"].values, source_ds["lat"].values, native_wind,
            target_lon, target_lat, target_mask,
        )
    if score_dust is not None and source_ds is not None:
        dust_grid = regrid_geographic_field(
            score_dust, source_ds["lon"].values, source_ds["lat"].values, target_lon, target_lat
        )
        dust_factor = native_grid_support(
            source_ds["lon"].values, source_ds["lat"].values, score_dust,
            target_lon, target_lat, target_mask,
        )

    terrain_support = float(min(1.0, np.sqrt(np.count_nonzero(target_mask) / 16.0)))
    landing = rank_landing_sites(
        slope_deg=local_slope,
        roughness_m=local_roughness,
        surface_temperature_k=thermal_grid,
        wind_speed_m_s=wind_grid,
        dust_optical_depth=dust_grid,
        polygon_mask=target_mask,
        longitude_grid=target_lon,
        latitude_grid=target_lat,
        maximum_sites=10,
        minimum_separation_km=4.0,
        resolution_factors=(terrain_support, terrain_support, thermal_factor, wind_factor, dust_factor),
    )
    landing_map = landing["suitability_map"]
    map_extent = [float(np.nanmin(target_lon)), float(np.nanmax(target_lon)),
                  float(np.nanmin(target_lat)), float(np.nanmax(target_lat))]
    background_elevation = np.asarray(study_area["background_elevation"], dtype=float)[::stride, ::stride]
    valid_elevation = np.isfinite(background_elevation)
    if np.any(valid_elevation):
        low, high = np.nanpercentile(background_elevation[valid_elevation], [3, 97])
        relief_range = max(float(high - low), 1e-9)
        normalized_elevation = np.clip((background_elevation - low) / relief_range, 0.0, 1.0)
        normalized_elevation = np.where(valid_elevation, normalized_elevation, 0.0)
        relief = LightSource(azdeg=315, altdeg=38).shade(
            normalized_elevation, cmap=plt.get_cmap("terrain"), vert_exag=2.5,
            blend_mode="soft", vmin=0.0, vmax=1.0,
        )
        relief[~valid_elevation] = to_rgba("#c9c2b5")
    else:
        relief = np.broadcast_to(to_rgba("#c9c2b5"), (*background_elevation.shape, 4)).copy()
    fig_land, ax_land = plt.subplots(figsize=(11, 8.5), dpi=180)
    ax_land.imshow(relief, extent=map_extent, origin="upper", aspect="auto", interpolation="bilinear", zorder=0)
    landing_image = ax_land.imshow(
        np.ma.masked_invalid(landing_map), extent=map_extent,
        origin="upper", aspect="auto", cmap="viridis", vmin=0, vmax=100,
        interpolation="bilinear", alpha=0.42, zorder=1,
    )
    active_vertices = study_area["vertices"]
    boundary = np.asarray([
        [((lon + 180) % 360) - 180, lat] for lon, lat in active_vertices
    ] + [[((active_vertices[0][0] + 180) % 360) - 180, active_vertices[0][1]]])
    ax_land.plot(boundary[:, 0], boundary[:, 1], color="white", linewidth=1.5, linestyle="--", label=tx("Aday arama alanı"))
    zone_boundary = np.asarray(RESEARCH_ZONE_LONLAT + [RESEARCH_ZONE_LONLAT[0]])
    ax_land.fill(zone_boundary[:-1, 0], zone_boundary[:-1, 1], facecolor="#f47721", alpha=0.22, zorder=3)
    ax_land.plot(zone_boundary[:, 0], zone_boundary[:, 1], color="#ff8c00", linewidth=2, label=tx("Araştırma hedef bölgesi"))
    target_center = np.mean(zone_boundary[:-1], axis=0)
    ax_land.annotate(tx("Olympus Mons etki ara\u015ft\u0131rma b\u00f6lgesi"), target_center, xytext=(8, -18), textcoords="offset points", color="#5a2a00", fontsize=9, weight="bold", bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fff4df", "edgecolor": "#f47721", "alpha": 0.95}, zorder=6)
    label_offsets = [(-10, 9), (10, 9), (-10, -11), (10, -11)]
    for site in landing["candidates"]:
        ax_land.scatter(site["longitude"], site["latitude"], marker="*", s=145, edgecolor="black", linewidth=0.7, zorder=3)
        offset = label_offsets[(site["rank"] - 1) % len(label_offsets)]
        ax_land.annotate(str(site["rank"]), (site["longitude"], site["latitude"]), xytext=offset, textcoords="offset points", color="white", fontsize=8, weight="bold", bbox={"boxstyle": "round,pad=0.15", "facecolor": "#18212b", "edgecolor": "none", "alpha": 0.75})
    ax_land.set_title(tx("Yumuşak maliyet ve veri güvenilirliğiyle iniş adayı haritası"))
    ax_land.set_xlabel(tx("Boylam (°)"))
    ax_land.set_ylabel(tx("Enlem (°)"))
    site_lons = np.asarray([site["longitude"] for site in landing["candidates"]], dtype=float)
    site_lats = np.asarray([site["latitude"] for site in landing["candidates"]], dtype=float)
    focus_lons = np.concatenate([site_lons, zone_boundary[:, 0]]) if site_lons.size else zone_boundary[:, 0]
    focus_lats = np.concatenate([site_lats, zone_boundary[:, 1]]) if site_lats.size else zone_boundary[:, 1]
    lon_span = max(float(np.ptp(focus_lons)), 0.02)
    lat_span = max(float(np.ptp(focus_lats)), 0.02)
    ax_land.set_xlim(float(np.min(focus_lons) - lon_span * 0.06), float(np.max(focus_lons) + lon_span * 0.06))
    ax_land.set_ylim(float(np.min(focus_lats) - lat_span * 0.06), float(np.max(focus_lats) + lat_span * 0.06))
    ax_land.legend(loc="best")
    fig_land.colorbar(landing_image, ax=ax_land, label=tx("Birleşik uygunluk (0–100)"))
    fig_land.tight_layout()
    st.pyplot(fig_land, width="stretch")
    plt.close(fig_land)
    if landing["candidates"]:
        st.table([{
            "Sıra": site["rank"],
            "Boylam (°)": round(site["longitude"], 5),
            "Enlem (°)": round(site["latitude"], 5),
            "Uygunluk": round(site["score"], 1),
            "Veri güvenilirliği": f"{site['data_reliability']:.1f}%",
        } for site in landing["candidates"]])
    st.caption("Yıldızlar sıralanmış adaylardır; 4 km yakınlık eşiği kümelenmeyi azaltır. Yüksek çözünürlüklü DEM içinde arama aralığı üçgenin içi, üçgeni kapsamayan bölgesel DEM içinde arama aralığı raster kapsamıdır. Bu sonuçlar görev doğrulamalı iniş emniyeti değildir.")

    route_signature = (
        metadata.get("samples"), metadata.get("lines"), metadata.get("map_x"),
        metadata.get("map_y"), int(slope_step),
        int(st.session_state.get("emars_map_time_index", 0)),
    )
    if st.session_state.get("route_plan_signature") != route_signature:
        st.session_state.pop("route_plan", None)
        st.session_state.pop("route_plan_signature", None)
    build_routes = st.button("Rotaları oluştur", type="primary", key="create_dijkstra_routes")
    st.caption("Hedef bölge koordinatları X/1000 = boylam, Y/1000 = enlem (derece) olarak yorumlanır; turuncu dikdörtgen varış bölgesini gösterir.")
    if build_routes:
        st.session_state.pop("route_plan", None)
        route_background = np.asarray(study_area["background_elevation"], dtype=float)[::stride, ::stride]
        route_domain = (
            np.isfinite(route_background)
            & np.isfinite(local_slope)
            & np.isfinite(local_roughness)
        )
        route_model = rank_landing_sites(
            slope_deg=local_slope,
            roughness_m=local_roughness,
            surface_temperature_k=thermal_grid,
            wind_speed_m_s=wind_grid,
            dust_optical_depth=dust_grid,
            polygon_mask=route_domain,
            longitude_grid=target_lon,
            latitude_grid=target_lat,
            maximum_sites=1,
            minimum_separation_km=0,
            resolution_factors=(terrain_support, terrain_support, thermal_factor, wind_factor, dust_factor),
        )
        route_score = route_model["suitability_map"]
        route_traversable = route_domain & np.isfinite(route_score)
        target_inside, goal_mask = build_target_masks(
            target_lon, target_lat, route_traversable,
        )
        routes = dijkstra_routes(
            cost_grid=1.0 - route_score / 100.0,
            traversable_mask=route_traversable,
            goal_mask=goal_mask,
            candidates=landing["candidates"],
            longitude_grid=target_lon,
            latitude_grid=target_lat,
        )
        if not routes:
            st.error("Aday noktalarından araştırma bölgesine geçerli bir Dijkstra rotası bulunamadı.")
        else:
            route_rows = []
            for route in routes:
                indices = tuple(zip(*route["path"]))
                def route_mean(values):
                    if values is None:
                        return "Veri yok"
                    sampled = np.asarray(values, dtype=float)[indices]
                    sampled = sampled[np.isfinite(sampled)]
                    return round(float(np.mean(sampled)), 2) if sampled.size else "Veri yok"
                route_rows.append({
                    "Rota": route["label"],
                    "İniş adayı": route["site"]["rank"],
                    "Hedefe giriş": route["target_entry"],
                    "Uygunluk": round(route["site"]["score"], 1),
                    "Rota uzunluğu (km)": round(route["distance_km"], 2),
                    "Dijkstra maliyeti": round(route["path_cost"], 3),
                    "Eğim (°)": route_mean(local_slope),
                    "Pürüzlülük (m)": route_mean(local_roughness),
                    "Sıcaklık (K)": route_mean(thermal_grid),
                    "Rüzgâr (m/s)": route_mean(wind_grid),
                    "Toz optik derinliği": route_mean(dust_grid),
                    "Veri güvenilirliği (%)": round(float(np.mean(route_model["reliability_map"][indices])), 1),
                })
            st.session_state["route_plan"] = {
                "routes": routes,
                "score_map": route_score,
                "extent": [float(np.nanmin(target_lon)), float(np.nanmax(target_lon)), float(np.nanmin(target_lat)), float(np.nanmax(target_lat))],
                "route_rows": route_rows,
            }
            st.session_state["route_plan_signature"] = route_signature

    route_plan = st.session_state.get("route_plan")
    if route_plan is not None and build_routes:
        st.subheader("Dijkstra rotaları oluşturuluyor")
    if route_plan is not None:
        animation_area = st.empty()
        route_status = st.empty()
        routes_to_show = route_plan["routes"]
        if build_routes:
            animation_frames = [(route_index, part / 6.0) for route_index in range(len(routes_to_show)) for part in range(1, 7)]
        else:
            animation_frames = [(len(routes_to_show) - 1, 1.0)]
        route_colors = plt.get_cmap("tab10")
        route_styles = ["-", "--", "-.", ":"]
        for route_index, progress in animation_frames:
            visible_count = route_index + 1 if build_routes else len(routes_to_show)
            fig_route, ax_route = plt.subplots(figsize=(11, 8.5), dpi=150)
            route_extent = route_plan["extent"]
            ax_route.set_facecolor("#e9e5da")
            for route_number, route in enumerate(routes_to_show[:visible_count]):
                x = np.asarray(route["longitudes"], dtype=float)
                y = np.asarray(route["latitudes"], dtype=float)
                if build_routes and route_number == route_index:
                    stop = max(2, int(np.ceil(len(x) * progress)))
                    x, y = x[:stop], y[:stop]
                color = route_colors(route_number % 10)
                style = route_styles[route_number % len(route_styles)]
                ax_route.plot(x, y, color=color, linewidth=5, alpha=0.18, solid_capstyle="round")
                ax_route.plot(x, y, color=color, linewidth=2.4, linestyle=style, alpha=1.0, label=f"{tx('Rota')} {route['label']}")
                ax_route.scatter(x[0], y[0], color=[color], edgecolor="white", s=48, zorder=4)
                label_offsets = [(-8, 9), (8, 9), (-8, -11), (8, -11)]
                ax_route.annotate(route["label"], (x[0], y[0]), xytext=label_offsets[route_number % len(label_offsets)], textcoords="offset points", color="white", weight="bold", bbox={"boxstyle": "round,pad=0.15", "facecolor": "#18212b", "edgecolor": "none", "alpha": 0.75})
            ax_route.fill(zone_boundary[:-1, 0], zone_boundary[:-1, 1], facecolor="#ff9f1c", alpha=0.25, zorder=2)
            ax_route.plot(zone_boundary[:, 0], zone_boundary[:, 1], color="#ef6c00", linewidth=2.8, label=tx("Ara\u015ft\u0131rma b\u00f6lgesi"), zorder=3)
            route_lons = np.concatenate([np.asarray(route["longitudes"], dtype=float) for route in routes_to_show[:visible_count]])
            route_lats = np.concatenate([np.asarray(route["latitudes"], dtype=float) for route in routes_to_show[:visible_count]])
            focus_lons = np.concatenate([route_lons, zone_boundary[:, 0]])
            focus_lats = np.concatenate([route_lats, zone_boundary[:, 1]])
            lon_span = max(float(np.ptp(focus_lons)), 0.02)
            lat_span = max(float(np.ptp(focus_lats)), 0.02)
            ax_route.set_xlim(float(np.min(focus_lons) - lon_span * 0.12), float(np.max(focus_lons) + lon_span * 0.12))
            ax_route.set_ylim(float(np.min(focus_lats) - lat_span * 0.12), float(np.max(focus_lats) + lat_span * 0.12))
            ax_route.set_title(tx("Olympus Mons \u00e7evresel etkilerinin incelenece\u011fi b\u00f6lgeye Dijkstra rotalar\u0131"))
            ax_route.set_xlabel(tx("Boylam (\u00b0)"))
            ax_route.set_ylabel(tx("Enlem (\u00b0)"))
            ax_route.legend(loc="best", fontsize=8, ncol=2)
            fig_route.tight_layout()
            animation_area.pyplot(fig_route, width="stretch")
            plt.close(fig_route)
            route_progress_label = tx("Rotalar s\u0131rayla g\u00f6steriliyor: ")
            route_status.caption(f"{route_progress_label}{route_index + 1} / {len(routes_to_show)}")
            if build_routes and progress < 1.0:
                time.sleep(0.04)
        route_status.empty()
        st.subheader("En iyi rotaların sıralaması")
        st.caption("A–J sıralaması Dijkstra'nın toplam risk ağırlıklı yol maliyetine göredir; tabloda rota boyunca değişken ortalamaları, uzunluk ve iniş noktası uygunluğu birlikte sunulur.")
        st.table(route_plan["route_rows"])
except Exception as exc:
    st.error(f"İniş noktası aday haritası oluşturulamadı: {exc}")


# --------------------------------------------------

st.divider()
st.subheader("Bilimsel ve dağıtım sınırları")

st.markdown(
    """
    - MOLA/DEM \u00e7\u00f6z\u00fcn\u00fcrl\u00fc\u011fu se\u00e7ilen veri kayna\u011f\u0131n\u0131n metadata bilgilerine ba\u011fl\u0131d\u0131r.
    - Girdi çözünürlüğü küçük kaya, blok ve dar çatlakları göstermeye yetmeyebilir; haritada görülen ayrıntı kaynak gridin ölçeğine bağlıdır.
    - Birleşik uygunluk göstergesi keşif amaçlıdır; görev doğrulamalı iniş güvenliği puanı değildir.
    - NoData pikselleri güvenli arazi olarak yorumlanmaz; analiz dışında tutulur.
    - PDS indirme yöntemi IMG rasterını geçici diske indirir; dosya Render yeniden başlatıldığında kaybolabilir ve yeniden indirilmesi gerekir.
    """
)
