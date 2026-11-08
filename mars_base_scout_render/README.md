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

## Varsayılan MOLA ürünü

`https://pds-geosciences.wustl.edu/mgs/urn-nasa-pds-mgs_mola_topography_derived/meg128/megt44n180hb.img`

Arayüze doğrudan `.img` HTTPS bağlantısı gir. Aynı klasördeki `.hdr` zorunludur; `.lbl` varsa otomatik indirilir.

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

- URL alanı yalnızca `https://pds-geosciences.wustl.edu` alan adını ve doğrudan `.img` dosyalarını kabul eder. Bu, uygulamanın rastgele adreslerden dosya indirmesini önler.
- IMG için 350 MiB üst sınır vardır. Metadata dosyaları 5 MiB ile sınırlandırılmıştır.
- Dosyalar Render'ın geçici diskine indirilir. Render Free dosya sistemi kalıcı değildir; servis yeniden başlarsa dosya tekrar indirilmelidir.
- `memmap` tam rasterı belleğe kopyalamaz, ancak istatistik ve çizim sırasında yine de bellek/CPU kullanılır. Render Free üzerindeki her MOLA ürünü için performans garantisi yoktur.
- İndirme başarısız olursa Runtime Logs ve arayüzdeki hata mesajını kontrol et.

## Bilimsel sınırlar

- MEG128 ekvator çevresinde nominal yaklaşık 463 m/piksel çözünürlüktedir.
- Bu ölçek bölgesel topografya ve genel eğim incelemesi içindir; küçük kaya, blok ve metre ölçekli iniş tehlikelerini göstermez.
- Bu sürüm otomatik iniş alanı seçmez ve iniş güvenliği puanı üretmez.
- NoData pikselleri güvenli arazi olarak yorumlanmaz; analiz dışında tutulur.
- PDS LBL içindeki birim, ölçekleme ve missing-value alanlarını bilimsel sonuçları raporlamadan önce kontrol et.
