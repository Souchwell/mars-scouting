# Steel Mars Run

Mars yüzey ve atmosfer verilerini inceleyen Streamlit uygulaması. MOLA/DEM, EMARS NetCDF ve MCD TXT verileriyle Lycus Sulci çevresinde arazi analizi, iniş adayı sıralaması ve araştırma bölgesine rota ön değerlendirmesi sağlar.

## Özellikler

- MOLA ENVI rasterını (`.img` + `.hdr`, isteğe bağlı `.lbl`) doğrudan HTTPS kaynağından indirme ve bellek eşlemeli okuma
- Tek bantlı coğrafi GeoTIFF (`.tif` / `.tiff`) DEM desteği
- EMARS NetCDF (`.nc`) içinde W–E rüzgâr bileşeni (`u`), N–S bileşeni (`v`) ve toz optik derinliği (`tauref` / `tod` / `dod`) haritaları
- MCD TXT çıktılarından sıcaklık katmanlarını değişken adlarıyla haritalama
- Eğim, pürüzlülük, termal koşullar, rüzgâr/toz ve veri güvenilirliğini birleştiren yumuşak maliyetli uygunluk ön değerlendirmesi
- Dijkstra aramasıyla iniş adaylarından araştırma alanının yaklaşım kenarına rota üretimi
- Türkçe ve İngilizce arayüz

## Veri kaynakları

Uygulamada **Proje verileri** bölümünden DEM ve atmosfer dosyaları tek seferde seçilebilir. Desteklenen dosyalar:

| Veri | Biçim | Gerekenler |
| --- | --- | --- |
| MOLA ENVI DEM | `.img` | Aynı klasörde aynı temel ada sahip `.hdr`; `.lbl` isteğe bağlı |
| Coğrafi yükseklik rasterı | `.tif`, `.tiff` | Tek bant; coğrafi koordinat sistemi ve yükseklik birimi metadata içinde bulunmalı |
| EMARS atmosferi | `.nc` | NetCDF; atmosfer değişkenleri ve koordinatları tanınabilir olmalı |
| MCD atmosferi | `.txt` | MCD metin çıktısı; sıcaklık katmanları desteklenir |

PDS arşiv URL'leri ve herkese açık GitHub Release indirme bağlantıları desteklenir. PDS ekranındaki tek kutuya dosya indirme URL'lerini satır satır yapıştırın; uygulama dosya türünü uzantıdan anlar. GitHub Release dosyaları için her dosyanın indirme URL'sini kullanın; uygulama yalnızca GitHub'ın bilinen dosya CDN'lerine giden HTTPS yönlendirmelerini izler. İçerik dönüştürülmeden indirilir; `.img` dosyasının `.hdr` dosyası aynı Release içinde aynı temel adla bulunmalıdır. Başka bir herkese açık dosya sunucusuna izin vermek için Render ortam değişkeni `MARS_DATA_ALLOWED_HOSTS` içine tam alan adını yazın; birden fazla alan adını virgülle ayırın. Protokol ve yol eklemeyin. Bilinmeyen sunuculara yönlendirmeler reddedilir.

URL kullanmak tarayıcıdan uygulamaya dosya yükleme gereğini kaldırır, ancak analiz yine sunucu belleğini ve işlemcisini kullanır. ENVI rasterı bellek eşlemeli açılır; yerel pürüzlülük hesabı ara belleği satır bloklarıyla sınırlar. Büyük GeoTIFF ve NetCDF dosyaları, özellikle 512 MB bellekli servislerde, yine de kaynak sınırına ulaşabilir.

## Analiz ve bilimsel sınırlar

Uygunluk ön değerlendirmesi eğim, pürüzlülük, yüzey sıcaklığı, rüzgâr/toz ve veri güvenilirliğini yumuşak maliyetlerle birleştirir. Eksik katmanlar nötr katkı verirken veri güvenilirliği puanını düşürür. Rota araması Dijkstra algoritmasını kullanır ve tanımlı araştırma dikdörtgeninin en düşük maliyetli yaklaşım kenarına bağlanır.

Bu puanlar ve rotalar keşif amaçlı ön değerlendirmedir; görev doğrulamalı iniş güvenliği ölçütü değildir. MEG128 ürünleri ekvator çevresinde yaklaşık 463 m/piksel nominal çözünürlük sunar ve küçük kaya/blok ya da metre ölçekli iniş tehlikelerini göstermez. NoData hücreleri analiz dışı tutulur. Kullanılan ürünün LBL metadata içindeki birim, ölçekleme ve eksik değer tanımlarını bilimsel sonuçları raporlamadan önce doğrulayın.

## Yerel çalıştırma

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Render dağıtımı

`render.yaml`, deponun kök dizinindedir. Render Blueprint'i bu dosyadan oluşturduğunuzda uygulama depo kökünden kurulur ve başlatılır. Servis geçici disk kullanır; yeniden başlatma sonrasında PDS dosyalarının yeniden indirilmesi gerekir.

`.streamlit/config.toml` dosyası yükleme boyutunu 150 MiB ile sınırlar. URL indirmelerinde desteklenen biçime göre uygulama düzeyinde dosya boyutu sınırları uygulanır. İndirmeler yalnızca izinli HTTPS sunucularını kabul eder; IP adresleri, bilinmeyen sunuculara yönlendirmeler, sorgu parametreleri ve desteklenmeyen uzantılar reddedilir.
