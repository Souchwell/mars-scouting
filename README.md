# Mars Base Scout — PDS streaming / Render sürümü

Bu sürüm MOLA MEGDR ENVI rasterlarını PDS arşivinden doğrudan indirir. IMG dosyası tarayıcıdan Render'a yüklenmez; Render sunucusu PDS'den akış hâlinde indirip geçici diske yazar ve NumPy `memmap` ile dosya tabanlı olarak açar.

## Yeni özellikler

- PDS IMG URL'sinden eşleşen `.hdr` ve varsa `.lbl` dosyalarını otomatik bulma
- Yalnızca `pds-geosciences.wustl.edu` alan adından HTTPS indirmeye izin verme
- IMG dosyasını 1 MiB parçalarla geçici diske indirme ve indirme ilerlemesini gösterme
- Rasterı `numpy.memmap` ile açma; IMG'nin tamamını RAM'e kopyalamama
- Yerel geliştirme için eski çoklu dosya yükleme seçeneğini koruma
- Parçalı istatistik, örneklenmiş yükseklik haritası ve isteğe bağlı örneklenmiş eğim analizi
- Render Blueprint yapılandırması
- EMARS NetCDF haritaları: W-E u, N-S v ve toz optik derinliği
- MCD TXT çıktılarındaki sıcaklık katmanlarını ve rüzgâr bileşenlerini ayrıştırma

## Atmosfer verileri

Uygulamadaki **Proje verileri** alanından DEM ve atmosfer dosyaları tek formda
birlikte yüklenebilir: MOLA ENVI `.img` + `.hdr` (isteğe bağlı `.lbl`), tek bantlı
coğrafi GeoTIFF `.tif/.tiff`, EMARS NetCDF `.nc` ve MCD `.txt` çıktıları.
GeoTIFF CRS'i coğrafi enlem-boylam olmalı ve yükseklik birimi metadata içinde
belirtilmelidir. Daha ayrıntılı DEM gridleri desteklenir; rasterın tamamı belleğe
alındığından çok büyük GeoTIFF dosyaları için sunucu belleği dikkate alınmalıdır.
NetCDF yüklemesinde bir zaman indeksi seçilir ve o zamana ait üç değişken haritası sekmelerde gösterilir: W-E rüzgâr bileşeni (u), N-S bileşeni (v) ve toz optik derinliği (tauref/tod/dod). Basınca bağlı alanlarda yüzeye en yakın basınç seviyesi seçilir. MCD TXT dosyalarındaki sıcaklık katmanları ayrı sekmelerde ve değişken adlarıyla gösterilir.

Uygunluk ön değerlendirmesi, eğim, pürüzlülük, yüzey sıcaklığı, rüzgâr/toz ve veri güvenilirliğini tek bir yumuşak maliyet fonksiyonuyla birleştirir. Eksik katmanlar nötr kabul edilir ve güvenilirlik puanını düşürür; aralık sınırları keskin ret üretmez. Başlangıç eşikleri gösterge amaçlıdır ve görev doğrulamalı güvenlik sınırları değildir.

İniş adaylarından araştırma bölgesine rota, çok kaynaklı Dijkstra aramasıyla hesaplanır. Kullanıcının verdiği X/Y değerleri binlik derece olarak yorumlanır; rota hedef koordinatına değil, tanımlı araştırma dikdörtgeninin en düşük maliyetli yaklaşım kenarına bağlanır. Her rotanın uzunluğu, toplam maliyeti ve arazi/atmosfer değişken özetleri listelenir.

## PDS data sources

The PDS URL workflow accepts MOLA ENVI `.img` with a matching `.hdr` (optional `.lbl`), GeoTIFF DEM `.tif/.tiff`, EMARS `.nc`, and MCD `.txt`. Enter each direct HTTPS product URL and choose its type; the app downloads the products and processes the atmosphere files alongside the selected DEM. Downloads are restricted to the configured PDS host and format-specific size limits.

## Yerel çalıştırma

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Yerel kullanımda da PDS URL'sinden indirme seçeneği çalışır. Küçük test verileri için bilgisayardan yükleme modu da korunmuştur.

## GitHub / Render dosya yapısı

Bu klasör mevcut depodaki `mars_base_scout_render` klasörünün içeriğinin yerini alacak şekilde hazırlanmıştır. Depo yapısı:

```text
mars-scouting/
└── mars_base_scout_render/
    ├── app.py
    ├── render.yaml
    ├── requirements.txt
    ├── README.md
    └── src/
        ├── __init__.py
        ├── analysis.py
        ├── data_loader.py
        └── pds_download.py
```

`render.yaml` içindeki `rootDir: mars_base_scout_render` ayarı depo kökündeki Blueprint dosyasından çalışacak şekilde bırakılmıştır. Blueprint Path: `mars_base_scout_render/render.yaml`.

## Güvenlik ve sınırlar

- PDS URL downloads are restricted to direct HTTPS products on `pds-geosciences.wustl.edu`, with supported file extensions and size limits.
- IMG için 350 MiB üst sınır vardır. Metadata dosyaları 5 MiB ile sınırlandırılmıştır.
- Dosyalar Render'ın geçici diskine indirilir. Render Free dosya sistemi kalıcı değildir; servis yeniden başlarsa dosya tekrar indirilmelidir.
- `memmap` tam rasterı belleğe kopyalamaz, ancak istatistik ve çizim sırasında yine de bellek/CPU kullanılır. Render Free üzerindeki her MOLA ürünü için performans garantisi yoktur.
- İndirme başarısız olursa Runtime Logs ve arayüzdeki hata mesajını kontrol et.

## Bilimsel sınırlar

- MEG128 ekvator çevresinde nominal yaklaşık 463 m/piksel çözünürlüktedir.
- Bu ölçek bölgesel topografya ve genel eğim incelemesi içindir; küçük kaya, blok ve metre ölçekli iniş tehlikelerini göstermez.
- Birleşik uygunluk göstergesi keşif amaçlıdır; otomatik iniş alanı seçmez ve görev doğrulamalı iniş güvenliği puanı üretmez.
- NoData pikselleri güvenli arazi olarak yorumlanmaz; analiz dışında tutulur.
- PDS LBL içindeki birim, ölçekleme ve missing-value alanlarını bilimsel sonuçları raporlamadan önce kontrol et.
