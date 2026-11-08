from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_mola_envi
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

st.title("Mars Base Scout")
st.caption("MOLA topografya verisi · DEM görüntüleme · Bölgesel eğim incelemesi")

st.info(
    "Bu sürüm MOLA ENVI rasterını (.img) eşleşen başlık (.hdr) ve PDS etiketi "
    "(.lbl) ile okur. Sonuçlar ön inceleme amaçlıdır; tek başına iniş güvenliği "
    "kararı veya iniş alanı puanı üretmez."
)

with st.sidebar:
    st.header("MOLA veri dosyaları")
    uploaded_files = st.file_uploader(
        "IMG, HDR ve LBL dosyalarını birlikte seç",
        type=["img", "hdr", "lbl"],
        accept_multiple_files=True,
        help=(
            "Aynı ürünün aynı dosya adına sahip .img, .hdr ve .lbl dosyalarını "
            "seç. Örneğin megt44n180hb.img / .hdr / .lbl."
        ),
    )
    st.divider()
    st.header("Görüntüleme")
    display_step = st.select_slider(
        "Harita örnekleme aralığı",
        options=[2, 4, 8, 16, 32],
        value=8,
        help="Büyük rasterların bellekte gereksiz kopyalarını oluşturmamak için görüntü seyreltilir.",
    )
    run_slope = st.checkbox("Eğim haritasını hesapla", value=False)
    slope_step = st.select_slider(
        "Eğim haritası örnekleme aralığı",
        options=[2, 4, 8, 16, 32],
        value=8,
        help=(
            "Eğim, orijinal rasterın bitişik piksellerinden türetilir; bu ayar yalnızca "
            "hesaplanan eğim değerlerinin haritada kaç pikselde bir gösterileceğini belirler."
        ),
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

    if found["lbl"] is not None:
        if Path(found["lbl"].name).stem.lower() != stems[0]:
            raise ValueError("LBL dosyasının adı IMG ve HDR dosyalarıyla eşleşmiyor.")
    return found


files = uploaded_files or []
if not files:
    st.subheader("Başlamak için veri yükle")
    st.markdown(
        """
        Örnek dosya grubu:
        - `megt44n180hb.img` — sayısal raster
        - `megt44n180hb.hdr` — ENVI raster başlığı
        - `megt44n180hb.lbl` — PDS bilimsel etiketi (önerilir)

        IMG ve HDR zorunludur. LBL dosyası metadata doğrulaması için önerilir.
        """
    )
    st.caption(
        "Uygulama yüklenen dosyaları kalıcı diske kaydetmez; analiz oturumu boyunca bellekte işler."
    )
    st.stop()

try:
    selected = identify_uploads(files)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

img_file = selected["img"]
hdr_file = selected["hdr"]
lbl_file = selected["lbl"]

st.sidebar.success("Dosyalar eşleştirildi.")
st.sidebar.caption(f"IMG: {img_file.name}")
st.sidebar.caption(f"HDR: {hdr_file.name}")
st.sidebar.caption(f"LBL: {lbl_file.name if lbl_file else 'Yüklenmedi'}")

# getbuffer() memoryview döndürür; getvalue() gibi büyük bir bytes kopyası oluşturmaz.
try:
    dem, metadata = load_mola_envi(
        img_buffer=img_file.getbuffer(),
        hdr_bytes=hdr_file.getvalue(),
        lbl_bytes=lbl_file.getvalue() if lbl_file else None,
    )
except Exception as exc:
    st.error(f"MOLA verisi okunamadı: {exc}")
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
    })

st.subheader("Yükseklik istatistikleri")
st.caption("Minimum, maksimum ve ortalama tüm geçerli raster piksellerinden; medyan ise örneklenmiş piksellerden hesaplanır.")
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
st.pyplot(fig, use_container_width=True)
plt.close(fig)

if run_slope:
    st.divider()
    st.subheader("Bölgesel eğim analizi")
    with st.spinner("Eğim örnekleri hesaplanıyor..."):
        slope = calculate_sampled_slope(
            dem,
            metadata,
            step=slope_step,
        )
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
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        st.caption(
            f"Türevler orijinal komşu pikseller arasından hesaplandı. Haritada her {slope_step} pikselde bir eğim örneği gösteriliyor. "
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
    - Render ücretsiz servisinde yüklenen veri yalnızca çalışma belleğinde tutulur; yeniden başlatma/uyku durumunda oturum ve yüklemeler kaybolabilir.
    """
)
