# Mars Base Scout — Render-ready

Mars MOLA MEGDR ENVI rasterlarını web arayüzünden yükleyip ön inceleme yapar.

## Bu sürümde

- `.img`, `.hdr` ve isteğe bağlı `.lbl` dosyalarını tek yükleme alanından birlikte seçme
- Dosya adlarının eşleşmesini doğrulama
- ENVI `samples`, `lines`, `data type`, `byte order`, `header offset`, NoData ve `map info` alanlarını okuma
- Big Endian Int16 dâhil ENVI veri tiplerini destekleme
- Mümkün olduğunda IMG dosyası üzerinde sıfır-kopyalı NumPy görünümü oluşturma
- Büyük raster için parçalı istatistik hesaplama
- Harita önizlemesini örnekleyerek belleği sınırlama
- Orijinal raster komşularından örneklenmiş eğim hesabı
- Render Blueprint (`render.yaml`) ve Streamlit sunucu yapılandırması

## Yerel çalıştırma

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Render üzerinde yayınlama

1. Bu klasörü yeni bir GitHub deposuna gönder.
2. Render Dashboard'da **New → Blueprint** seç.
3. GitHub deposunu bağla. Render kök dizindeki `render.yaml` dosyasını okuyacaktır.
4. Blueprint ile servisi oluştur ve Deploy Logs bölümünü izle.

`render.yaml` şu ayarları içerir:

- Build: `pip install -r requirements.txt`
- Start: Streamlit'i `0.0.0.0` üzerinde ve Render'ın `$PORT` değişkeninde başlatır.
- Health check: `/_stcore/health`
- Plan: `free`
- Region: `frankfurt`

Servis adını `render.yaml` içinde değiştirebilirsin. `mars-base-scout` adı çalışma alanında zaten kullanılıyorsa farklı bir ad belirle.

## Render ücretsiz planı hakkında

Render Free web servisleri 512 MB RAM ve 0.1 CPU kullanır, 15 dakika trafik almadığında uykuya geçebilir ve yeni istekte yaklaşık bir dakika içinde yeniden başlar. Dosya sistemi kalıcı değildir. Bu uygulama yüklenen dosyaları diske kaydetmez; mevcut oturumda bellekte işler.

MOLA MEG128 örneği yaklaşık 129.8 MB IMG dosyasıdır. Kod, IMG verisini mümkün olduğunda kopyalamadan açar; istatistikleri parçalı hesaplar ve haritaları örnekler. Buna rağmen ücretsiz servisin RAM sınırı nedeniyle büyük dosyada bellek yetersizliği yaşanabilir. Böyle olursa önce MEG064 gibi daha düşük çözünürlüklü bir ürünle dene veya daha fazla RAM sunan bir plana geç.

## Bilimsel sınırlar

- MEGDR ürünleri MOLA altimetre ölçümlerinden üretilmiş küresel topografya haritalarıdır. MEG128'in ekvator civarında nominal piksel aralığı yaklaşık 463 m'dir.
- Görüntüleme örneklemesi harita görünümünü küçültür; eğim hesabı yerel türevleri orijinal komşu raster piksellerinden alır, ancak sonuç yine MOLA'nın yaklaşık 463 m/piksel ölçeği ile sınırlıdır.
- Küçük kaya ve metre ölçekli iniş tehlikeleri bu veriyle tespit edilemez.
- LBL içinde ölçekleme alanı bulunmuyorsa varsayılan ölçek 1 ve offset 0 kullanılır. Bilimsel sonuçları raporlamadan önce ilgili PDS etiketindeki birim, ölçekleme ve eksik veri tanımlarını doğrula.
- Bu sürüm coğrafi kırpma, otomatik iniş yeri seçimi veya güvenlik puanlaması yapmaz.
