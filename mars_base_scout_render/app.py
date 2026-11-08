from pathlib import Path
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_mola_envi, load_mola_envi_from_path
from src.pds_download import download_pds_product
from src.analysis import (
    calculate_sampled_slope,
    get_dem_statistics,
    make_display_sample,
)

DEFAULT_IMG_URL = (
    "https://pds-geosciences.wustl.edu/mgs/urn-nasa-pds-mgs_mola_topography_derived/"
    "meg128/megt44n180hb.img"
)

st.set_page_config(
    page_title="Mars Base Scout",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Mars Base Scout")
st.caption("MOLA topografya verisi · DEM görüntüleme · Bölgesel eğim incelemesi")
st.info(
    "Render kullanımı için önerilen yöntem PDS arşivinden doğrudan indirmedir. "
    "IMG dosyası geçici diske parça parça indirilir ve RAM'e bütünüyle kopyalanmadan açılır. "
    "Sonuçlar ön inceleme amaçlıdır; tek başına iniş güvenliği kararı üretmez."
)

with st.sidebar:
    st.header("Veri kaynağı")
    source_mode = st.radio(
        "MOLA verisini nereden alalım?",
        ["PDS arşivinden indir (Render için önerilen)", "Bilgisayardan yükle (yerel kullanım)"],
        index=0,
    )
    st.divider()
    st.header("Görüntüleme")
    display_step = st.select_slider(
        "Harita örnekleme aralığı",
        options=[2, 4, 8, 16, 32],
        value=8,
        help="Büyük rasterların ek bellek kullanımını sınırlamak için harita seyreltilir.",
    )
    run_slope = st.checkbox("Eğim haritasını hesapla", value=False)
    slope_step = st.select_slider(
        "Eğim haritası örnekleme aralığı",
        options=[2, 4, 8, 16, 32],
        value=8,
        help="Türevler komşu orijinal piksellerden hesaplanır; bu ayar yalnızca sonuç haritasının örnekleme aralığını belirler.",
        disabled=not run_slope,
    )


def identify_uploads(files):
    found = {"img": None, "hdr": None, "lbl": None}
    for uploaded in files or []:
        suffix = Path(uploaded.name).suffix.lower().lstrip(".")
        if suffix not in found:
            continue
        if found[suffix] is not None:
            raise ValueError(f"Birden fazla .{suffix} dosyası seçildi. Tek ürünün dosyalarını seç.")
        found[suffix] = uploaded
    if found["img"] is None or found["hdr"] is None:
        raise ValueError("Devam etmek için .img ve eşleşen .hdr dosyaları zorunludur.")
    stems = [Path(found[key].name).stem.lower() for key in ("img", "hdr")]
    if stems[0] != stems[1]:
        raise ValueError("IMG ve HDR dosyalarının ad gövdeleri aynı değil; eşleşen dosyaları seç.")
    if found["lbl"] is not None and Path(found["lbl"].name).stem.lower() != stems[0]:
        raise ValueError("LBL dosyasının adı IMG ve HDR dosyalarıyla eşleşmiyor.")
    return found


dem = None
metadata = None

if source_mode == "PDS arşivinden indir (Render için önerilen)":
    st.subheader("PDS arşivinden MOLA verisi")
    st.write(
        "Doğrudan `.img` dosyasının HTTPS adresini gir. Uygulama aynı klasördeki "
        "`.hdr` dosyasını, varsa `.lbl` dosyasını otomatik bulur."
    )
    with st.form("pds_download_form"):
        img_url = st.text_input(
            "MOLA IMG adresi",
            value=DEFAULT_IMG_URL,
            help="Şimdilik pds-geosciences.wustl.edu üzerindeki doğrudan .img bağlantıları destekleniyor.",
        )
        start_download = st.form_submit_button("PDS verisini indir ve analize hazırla", type="primary")

    if start_download:
        progress_bar = st.progress(0.0, text="PDS dosyaları hazırlanıyor...")
        status = st.status("PDS arşivine bağlanılıyor...", expanded=True)

        def update_progress(downloaded, total):
            downloaded_mib = downloaded / (1024 ** 2)
            if total:
                fraction = min(downloaded / total, 1.0)
                total_mib = total / (1024 ** 2)
                progress_bar.progress(
                    fraction,
                    text=f"IMG indiriliyor: {downloaded_mib:.1f} / {total_mib:.1f} MiB",
                )
            else:
                progress_bar.progress(
                    0.0,
                    text=f"IMG indiriliyor: {downloaded_mib:.1f} MiB",
                )

        try:
            status.update(label="HDR/LBL metadata dosyaları alınıyor...", state="running")
            new_path, new_hdr, new_lbl, normalized_url = download_pds_product(
                img_url,
                progress_callback=update_progress,
            )
            old_product = st.session_state.get("pds_product")
            st.session_state["pds_product"] = {
                "url": normalized_url,
                "img_path": new_path,
                "hdr_bytes": new_hdr,
                "lbl_bytes": new_lbl,
            }
            if old_product and old_product.get("img_path") != new_path:
                try:
                    os.remove(old_product["img_path"])
                except OSError:
                    pass
            status.update(label="MOLA verisi indirildi ve hazır.", state="complete")
            progress_bar.progress(1.0, text="İndirme tamamlandı")
        except Exception as exc:
            status.update(label="PDS indirmesi başarısız oldu.", state="error")
            progress_bar.empty()
            st.error(f"PDS verisi alınamadı: {exc}")

    product = st.session_state.get("pds_product")
    normalized_input = (img_url if "img_url" in locals() else DEFAULT_IMG_URL).strip()
    if not product or product.get("url") != normalized_input:
        st.info("Analiz için önce PDS verisini indir düğmesine bas.")
        st.stop()
    if not os.path.exists(product["img_path"]):
        st.warning("Geçici raster dosyası artık sunucuda bulunmuyor. Render yeniden başlamış olabilir; dosyayı tekrar indir.")
        st.stop()

    st.success("PDS verisi hazır. IMG geçici diskte tutuluyor; raster RAM'e bütünüyle yüklenmiyor.")
    st.caption(f"Kaynak: {product['url']}")
    st.caption(f"HDR: otomatik alındı · LBL: {'otomatik alındı' if product['lbl_bytes'] else 'arşivde bulunamadı (isteğe bağlı)'}")

    try:
        dem, metadata = load_mola_envi_from_path(
            product["img_path"],
            product["hdr_bytes"],
            product["lbl_bytes"],
        )
    except Exception as exc:
        st.error(f"İndirilen MOLA rasterı açılamadı: {exc}")
        st.stop()
    metadata["source_mode"] = "PDS üzerinden doğrudan indirme"

    if st.sidebar.button("İndirilen geçici veriyi temizle"):
        try:
            os.remove(product["img_path"])
        except OSError:
            pass
        st.session_state.pop("pds_product", None)
        st.rerun()

else:
    st.subheader("Bilgisayardan MOLA dosyası yükle")
    st.warning(
        "Bu yöntem yerel kullanım için uygundur. Render Free üzerinde büyük IMG dosyaları "
        "tarayıcı yüklemesi sırasında 502 hatasına yol açabilir; Render'da PDS indirme yöntemini kullan."
    )
    uploaded_files = st.file_uploader(
        "IMG, HDR ve isteğe bağlı LBL dosyalarını birlikte seç",
        type=["img", "hdr", "lbl"],
        accept_multiple_files=True,
        help="Aynı ürünün aynı dosya adına sahip dosyalarını seç.",
    )
    if not uploaded_files:
        st.info("Devam etmek için .img ve .hdr dosyalarını seç.")
        st.stop()
    try:
        selected = identify_uploads(uploaded_files)
        dem, metadata = load_mola_envi(
            img_buffer=selected["img"].getbuffer(),
            hdr_bytes=selected["hdr"].getvalue(),
            lbl_bytes=selected["lbl"].getvalue() if selected["lbl"] else None,
        )
    except Exception as exc:
        st.error(f"MOLA verisi okunamadı: {exc}")
        st.stop()
    metadata["source_mode"] = "Bilgisayardan yükleme"

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
metric_cols[2].metric("Piksel boyutu", f"{metadata['pixel_x']:.2f} m")
metric_cols[3].metric("Geçerli veri", f"{stats['valid_percent']:.2f}%")
metric_cols[4].metric("Veri tipi", metadata["dtype_name"])

with st.expander("ENVI ve PDS metadata bilgileri"):
    st.json(metadata["label_metadata"])
    st.write("**ENVI map info**")
    st.code(metadata.get("map_info_raw") or "Başlıkta bulunamadı")
    st.write("**Raster okuma ayarları**")
    st.json({
        "samples": metadata["samples"],
        "lines": metadata["lines"],
        "bands": metadata["bands"],
        "ENVI data type": metadata["data_type"],
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
fig, ax = plt.subplots(figsize=(12, 6))
image = ax.imshow(
    np.ma.masked_invalid(preview),
    cmap="terrain",
    origin="upper",
    interpolation="nearest",
)
ax.set_title("MOLA topografya rasterı — örneklenmiş görünüm")
ax.set_xlabel("Raster sütunu (piksel)")
ax.set_ylabel("Raster satırı (piksel)")
fig.colorbar(image, ax=ax, label=f"Raster yüksekliği ({metadata['height_unit']})")
fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

if run_slope:
    st.divider()
    st.subheader("Bölgesel eğim analizi")
    with st.spinner("Eğim örnekleri hesaplanıyor..."):
        slope = calculate_sampled_slope(dem, metadata, step=slope_step)
    slope_valid = slope[np.isfinite(slope)]
    if slope_valid.size == 0:
        st.warning("Eğim hesaplamak için yeterli geçerli komşu piksel bulunamadı.")
    else:
        slope_cols = st.columns(3)
        slope_cols[0].metric("Ortalama eğim", f"{float(np.mean(slope_valid)):.2f}°")
        slope_cols[1].metric("Medyan eğim", f"{float(np.median(slope_valid)):.2f}°")
        slope_cols[2].metric("95. yüzdelik", f"{float(np.percentile(slope_valid, 95)):.2f}°")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        slope_image = ax2.imshow(
            np.ma.masked_invalid(slope),
            cmap="magma",
            origin="upper",
            interpolation="nearest",
            vmin=0,
            vmax=30,
        )
        ax2.set_title("Eğim haritası — örneklenmiş değerler")
        ax2.set_xlabel("Örnek sütun indeksi")
        ax2.set_ylabel("Örnek satır indeksi")
        fig2.colorbar(slope_image, ax=ax2, label="Eğim (derece)")
        fig2.tight_layout()
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)
        st.caption(
            f"Türevler orijinal komşu piksellerden hesaplandı. Haritada her {slope_step} pikselde bir eğim örneği gösteriliyor. "
            "MOLA'nın yaklaşık 463 m/piksel çözünürlüğü küçük kaya ve metre ölçekli iniş tehlikelerini göstermez."
        )

st.divider()
st.subheader("Bilimsel ve dağıtım sınırları")
st.markdown(
    """
    - MOLA 128 px/degree ürünü ekvator çevresinde yaklaşık 463 m/piksel ölçektedir.
    - Bu ölçek bölgesel topografya ve genel eğim incelemesi içindir; küçük kaya, blok ve dar çatlakları belirleyemez.
    - Bu sürüm otomatik iniş alanı seçmez ve iniş güvenliği puanı üretmez.
    - NoData pikselleri güvenli arazi olarak yorumlanmaz; analiz dışında tutulur.
    - PDS indirme yöntemi IMG rasterını geçici diske indirir; dosya Render yeniden başlatıldığında kaybolabilir ve yeniden indirilmesi gerekir.
    """
)
