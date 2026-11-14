"""Small UI translation layer for the Turkish and English app interfaces."""
from __future__ import annotations

from functools import wraps


_EN = {
    'Dil / Language': 'Language / Dil',
    'Veri kaynağı': 'Data source',
    'MOLA verisini nereden alalım?': 'How should MOLA data be loaded?',
    'PDS arşivinden indir (Render için önerilen)': 'Download from PDS archive (recommended for Render)',
    'Bilgisayardan yükle (yerel kullanım)': 'Upload from computer (local use)',
    'Görüntüleme': 'Display',
    'Harita örnekleme aralığı': 'Map sampling interval',
    'Eğim haritası örnekleme aralığı': 'Slope map sampling interval',
    'Bütün raster pikselleri görüntülenir.': 'Display every raster pixel.',
    'Her pikselde eğim hesaplanır.': 'Calculate slope at every pixel.',
    'Mars topografya verisi · DEM görüntüleme · Bölgesel eğim incelemesi': 'Mars topography · DEM viewing · Regional slope analysis',
    "Render kullanımı için önerilen yöntem PDS arşivinden doğrudan indirmedir. IMG dosyası geçici diske parça parça indirilir ve RAM'e bütünüyle kopyalanmadan açılır. Sonuçlar ön inceleme amaçlıdır; tek başına iniş güvenliği kararı üretmez.": 'For Render, downloading directly from the PDS archive is recommended. The IMG is streamed to temporary disk and opened without copying the full raster into RAM. Results are preliminary and do not establish landing safety.',
    'Proje verileri': 'Project data',
    'MOLA/DEM ve atmosfer dosyalarını tek seferde yükleyip işleyebilirsiniz. GeoTIFF ve NetCDF ayrıntılı gridleri desteklenir.': 'Upload and process DEM and atmospheric data together. Detailed GeoTIFF and NetCDF grids are supported.',
    'DEM (.img + .hdr [+ .lbl], .tif/.tiff) ve atmosfer (.nc, .txt) dosyalarını seçin': 'Select DEM (.img + .hdr [+ .lbl], .tif/.tiff) and atmosphere (.nc, .txt) files',
    'Seçilen verileri birlikte yükle ve işle': 'Upload and process selected data',
    'Seçilen proje verileri yüklendi.': 'Project data loaded.',
    'Atmosfer değişken haritaları': 'Atmospheric variable maps',
    'Mars zaman indeksi': 'Mars time index',
    'MCD TXT \u2004 sıcaklık haritaları': 'MCD TXT \u2004 temperature maps',
    'Mars Atmosferi - Veri Katmanları': 'Mars atmosphere - data layers',
    'PDS arşivinden MOLA verisi': 'MOLA data from the PDS archive',
    'MOLA IMG adresi': 'MOLA IMG address',
    'PDS verisini indir ve analize hazırla': 'Download PDS data and prepare analysis',
    'MOLA ENVI Header - Debug': 'MOLA ENVI header - debug',
    'Birim teşhis bilgileri': 'Unit diagnostics',
    'Veri özeti': 'Data summary',
    'Raster metadata bilgileri': 'Raster metadata',
    'Raster okuma ayarları': 'Raster reading settings',
    'Yükseklik istatistikleri': 'Elevation statistics',
    'Minimum, maksimum ve ortalama geçerli raster piksellerinden; medyan ise örneklenmiş piksellerden hesaplanır.': 'Minimum, maximum, and mean use all valid raster pixels; median is estimated from sampled pixels.',
    'Topografya rasterı \u2004 örneklenmiş görünüm': 'Topography raster \u2004 sampled view',
    'Raster sütunu (piksel)': 'Raster column (pixel)',
    'Raster satırı (piksel)': 'Raster row (pixel)',
    'Lycus Sulci \u2004 Üçgen çalışma alanı': 'Lycus Sulci \u2004 triangular study area',
    'Önceden belirlenen üçgen bölge DEM rasterından otomatik olarak kesilir. Yalnızca poligon içerisindeki geçerli pikseller analiz edilir.': 'The preset triangular area is extracted from the DEM. Only valid pixels inside the polygon are analyzed.',
    'Üçgen çalışma alanı DEM rasterından başarıyla çıkarıldı.': 'Triangular study area extracted from the DEM.',
    'Bölgesel arazi pürüzlülüğü': 'Regional terrain roughness',
    'Bölgesel eğim analizi': 'Regional slope analysis',
    'Birleşik bölgesel uygunluk ön değerlendirmesi': 'Combined regional suitability screening',
    'Birleşik ön değerlendirme': 'Combined screening score',
    'Veri güvenilirliği': 'Data reliability',
    'Atmosfer verisi': 'Atmospheric data',
    'Uygun': 'Available',
    'Eksik': 'Missing',
    'Eğim': 'Slope',
    'Pürüzlülük': 'Roughness',
    'Termal koşullar': 'Thermal conditions',
    'Rüzgâr + toz': 'Wind + dust',
    'Eksik bileşenler: ': 'Missing components: ',
    '. Eksiklikler güvenilirlik puanını düşürür.': '. Missing data lowers reliability.',
    'One function combines slope, roughness, surface temperature, wind/dust, and data reliability. Values outside ideal ranges receive a gradual cost; thresholds are initial screening assumptions, not mission safety limits.': 'One function combines slope, roughness, surface temperature, wind/dust, and data reliability. Values outside ideal ranges receive a gradual cost; thresholds are initial screening assumptions, not mission safety limits.',
    'Data reliability details': 'Data reliability details',
    'Layer': 'Layer',
    'Valid / total cells': 'Valid / total cells',
    'Coverage': 'Coverage',
    'Cell count indicates grid detail. Coverage reflects valid data and sample adequacy inside the study triangle; a high count alone does not prove measurement accuracy.': 'Cell count indicates grid detail. Coverage reflects valid data and sample adequacy inside the study area; a high count alone does not prove measurement accuracy.',
    'Önerilen iniş noktaları': 'Recommended landing sites',
    'Hedef nokta bu aşamada kullanılmıyor. Aşağıdaki 10 aday, mevcut çalışma alanı verilerinde yumuşak maliyet ve veri güvenilirliği birlikte değerlendirilerek sıralanır.': 'The destination is a region, not a point. Up to 10 candidates are ranked using soft costs and data reliability.',
    'Yumuşak maliyet ve veri güvenilirliğiyle iniş adayı haritası': 'Landing site suitability and data reliability',
    'Boylam (°)': 'Longitude (°)',
    'Enlem (°)': 'Latitude (°)',
    'Birleşik uygunluk (0–100)': 'Combined suitability (0–100)',
    'Rotaları oluştur': 'Create routes',
    'Hedef bölge koordinatları X/1000 = boylam, Y/1000 = enlem (derece) olarak yorumlanır; turuncu dikdörtgen varış bölgesini gösterir.': 'Target coordinates use X/1000 = longitude and Y/1000 = latitude in degrees; the orange rectangle marks the destination region.',
    'Aday noktalarından araştırma bölgesine geçerli bir Dijkstra rotası bulunamadı.': 'No valid Dijkstra route from the candidate sites to the research region was found.',
    'Dijkstra rotaları oluşturuluyor': 'Generating Dijkstra routes',
    'Olympus Mons çevresel etkilerinin inceleneceği bölgeye Dijkstra rotaları': 'Dijkstra routes to the region for studying environmental effects around Olympus Mons',
    'Araştırma bölgesi': 'Research region',
    'Araştırma hedef bölgesi': 'Research target region',
    'Aday arama alanı': 'Candidate search area',
    'Lycus Sulci çalışma alanı': 'Lycus Sulci study area',
    'En iyi rotaların sıralaması': 'Route ranking',
    "A–J sıralaması Dijkstra'nın toplam risk ağırlıklı yol maliyetine göredir; tabloda rota boyunca değişken ortalamaları, uzunluk ve iniş noktası uygunluğu birlikte sunulur.": 'Routes A–J are ranked by total risk-weighted Dijkstra cost; the table includes route length, environmental averages, and landing-site suitability.',
    'Yıldızlar sıralanmış adaylardır; 4 km yakınlık eşiği kümelenmeyi azaltır. Yüksek çözünürlüklü DEM içinde arama aralığı üçgenin içi, üçgeni kapsamayan bölgesel DEM içinde arama aralığı raster kapsamıdır. Bu sonuçlar görev doğrulamalı iniş emniyeti değildir.': 'Stars mark ranked candidates; a 4 km separation reduces clustering. Search uses the preset triangle when available, otherwise the regional raster coverage. This is not mission-validated landing safety.',
    'İniş noktası aday haritası oluşturulamadı: ': 'Could not create landing-site candidate map: ',
    'Birleşik uygunluk hesabı yapılamadı: ': 'Could not calculate combined suitability: ',
    'Bilimsel ve dağıtım sınırları': 'Scientific and deployment limitations',
    'Tek fonksiyon eğim, pürüzlülük, yüzey sıcaklığı, rüzgâr/toz ve veri güvenilirliğini birleştirir. İdeal aralık dışındaki değerler kademeli maliyet alır; eşikler başlangıç varsayımıdır, görev güvenlik sınırı değildir.': 'One function combines slope, roughness, surface temperature, wind/dust, and data reliability. Values outside ideal ranges receive a gradual cost; values beyond an ideal range are penalized gradually, not rejected outright.',
    'Veri güvenilirliği ayrıntıları': 'Data reliability details',
    'Katman': 'Layer',
    'Geçerli / toplam hücre': 'Valid / total cells',
    'Kapsama': 'Coverage',
    'Hücre sayısı grid ayrıntısını; kapsama ise çalışma alanındaki geçerli veriyi ve örnek yeterliliğini yansıtır. Yüksek hücre sayısı tek başına ölçüm doğruluğunu kanıtlamaz.': 'Cell count indicates grid detail; coverage reflects valid data and sample adequacy inside the study area. A high count alone does not prove measurement accuracy.',
    'Yükseklik': 'Elevation',
    'Ortalama': 'Mean',
    'Medyan': 'Median',
    'Standart sapma': 'Standard deviation',
    'Geçerli veri': 'Valid data',
    'Genişlik': 'Width',
    'Minimum yükseklik': 'Minimum elevation',
    'Maksimum yükseklik': 'Maximum elevation',
    'Ortalama yükseklik': 'Mean elevation',
    'Medyan yükseklik': 'Median elevation',
    'Geçerli piksel': 'Valid pixels',
    'Raster sütunu': 'Raster column',
    'Raster satırı': 'Raster row',
    'Yerel düzlemden RMS sapma (m)': 'RMS deviation from local plane (m)',
    'Eğim (derece)': 'Slope (degrees)',
    'Birleşik uygunluk': 'Combined suitability',
    'Minimum': 'Minimum',
    'Maksimum': 'Maximum',
    'Kaynak değişken: ': 'Source variable: ',
    '; birim: ': '; unit: ',
    '; zaman indeksi: ': '; time index: ',
    'W-E bileşeni (u)': 'W-E component (u)',
    'N-S bileşeni (v)': 'N-S component (v)',
    'Toz optik derinliği': 'Dust optical depth',
    'Rüzgâr ': 'Wind ',
    'Sıcaklık ': 'Temperature ',
    'Zaman indeksi ': 'Time index ',
    'Bir seferde yalnızca bir GeoTIFF seçin.': 'Select only one GeoTIFF at a time.',
    'PDS arşivine bağlanılıyor...': 'Connecting to PDS archive...',
    'Veri türü': 'Data type',
    'Örnek sütun indeksi': 'Sample column index',
    'Örnek satır indeksi': 'Sample row index',
    'Mars yüzey eğimi': 'Mars surface slope',
    'Lycus Sulci \u2004 Üçgen çalışma alanı yükseklik haritası': 'Lycus Sulci \u2004 triangular study area elevation map',
    'Lycus Sulci \u2004 Yerel arazi pürüzlülüğü': 'Lycus Sulci \u2004 local terrain roughness',
    'Pürüzlülük değerleri hesaplanıyor...': 'Calculating roughness values...',
    'Eğim örnekleri hesaplanıyor...': 'Calculating slope samples...',
    'Pürüzlülük hesaplamak için yeterli geçerli komşu piksel bulunamadı.': 'Not enough valid neighboring pixels to calculate roughness.',
    'Eğim hesaplamak için yeterli geçerli komşu piksel bulunamadı.': 'Not enough valid neighboring pixels to calculate slope.',
    'Çalışma alanı koordinatları': 'Study area coordinates',
    'Araştırma bölgesine Dijkstra rotaları': 'Dijkstra routes to the research region',
    'Veriler yüklenemedi veya işlenemedi: ': 'Could not load or process data: ',
    'Atmosfer verileri işlenemedi: ': 'Could not process atmospheric data: ',
    'haritası oluşturulamadı: ': ' map could not be created: ',
    'haritası üretilemedi: ': ' map could not be generated: ',
    'görüntülenemedi: ': ' could not be displayed: ',
    'Çalışma alanı DEM rasterından çıkarıldı: ': 'Study area extracted from DEM: ',
    'Lycus Sulci üçgeni': 'Lycus Sulci triangle',
    'yüklenen raster kapsamı': 'uploaded raster coverage',
    'Rota uzunluğu (km)': 'Route length (km)',
    'Dijkstra maliyeti': 'Dijkstra cost',
    'Hedefe giriş': 'Target entry',
    'İniş adayı': 'Landing candidate',
    'Sıra': 'Rank',
    'Uygunluk': 'Suitability',
    'Veri güvenilirliği (%)': 'Data reliability (%)',
    'Eğim (°)': 'Slope (°)',
    'Pürüzlülük (m)': 'Roughness (m)',
    'Sıcaklık (K)': 'Temperature (K)',
    'Rüzgâr (m/s)': 'Wind (m/s)',
    'Sıcaklık haritası ': 'Temperature map ',
    'Mars Yüzey Sıcaklığı': 'Mars surface temperature',
    'Mars Atmosfer Sıcaklığı': 'Mars atmospheric temperature',
    'Mars Atmosferik Rüzgâr': 'Mars atmospheric wind',
    'Mars Atmosferik Toz Optik Derinliği': 'Mars atmospheric dust optical depth',
    'Sıcaklık (°C)': 'Temperature (°C)',
    'Rüzgâr hızı (m/s)': 'Wind speed (m/s)',
    'Çalışma alanı sınırı': 'Study area boundary',
    "PDS verisi hazır. IMG geçici diskte tutuluyor; raster RAM'e bütünüyle yüklenmiyor.": 'PDS data is ready. The IMG remains on temporary disk; the raster is not fully loaded into RAM.',
    'Analiz için önce PDS verisini indir düğmesine bas.': 'Download the PDS data before analysis.',
    'Geçici raster dosyası artık sunucuda bulunmuyor. Render yeniden başlamış olabilir; dosyayı tekrar indir.': 'The temporary raster is no longer on the server. Render may have restarted; download it again.',
    'PDS dosyaları hazırlanıyor...': 'Preparing PDS files...',
    'HDR/LBL metadata dosyaları alınıyor...': 'Retrieving HDR/LBL metadata files...',
    'PDS LBL metadata dosyası alındı.': 'PDS LBL metadata file retrieved.',
    'Medyan (yaklaşık)': 'Median (approx.)',
    'Raster penceresi: ': 'Raster window: ',
    ' satır, ': ' rows, ',
    ' sütun.': ' columns.',
    'Ortalama pürüzlülük': 'Mean roughness',
    'Medyan pürüzlülük': 'Median roughness',
    '4 sıcaklık katmanı bulundu; en fazla dört harita gösterilir.': '4 temperature layers found; up to four maps are shown.',
    'Mars yılı ': 'Mars year ',
    'Basınca bağlı katmanlarda ': 'For pressure-dependent layers, ',
    'yüzeye en yakın seviye kullanılıyor: ': 'the level nearest the surface is used: ',
    'PDS verisi alınamadı: ': 'Could not retrieve PDS data: ',
    'İndirilen MOLA rasterı açılamadı: ': 'Could not open downloaded MOLA raster: ',
    'Raster istatistikleri hesaplanamadı: ': 'Could not calculate raster statistics: ',
    'Uydu kaynağı': 'Satellite source',
    'Atmosfer verileri': 'Atmospheric data',
    'İndirilen geçici veriyi temizle': 'Clear downloaded temporary data',
    'PDS üzerinden doğrudan indirme': 'Direct download from PDS',
    'bilgisayardan toplu yükleme': 'batch upload from computer',
    'Dosyada ': 'No matching variable in the file for ',
    ' için uygun bir değişken bulunamadı.': '.',
    'Bu zaman ve basınç seviyesinde geçerli hücre yok.': 'No valid cells at this time and pressure level.',
    'Yüklenen DEM varsayılan üçgeni kapsamıyor; adaylar bu veri setinin geçerli raster alanında aranıyor.': 'The uploaded DEM does not cover the preset triangle; candidates are searched across the valid raster coverage.',
    "Varsayılan MOLA MEG128 ürünü ekvator çevresinde yaklaşık 463 m/piksel ölçektedir; GeoTIFF grid çözünürlüğü kaynak metadata'sından alınır.": 'The default MOLA MEG128 product is about 463 m/pixel near the equator; GeoTIFF resolution comes from its source metadata.',
    'Girdi çözünürlüğü küçük kaya, blok ve dar çatlakları göstermeye yetmeyebilir; haritada görülen ayrıntı kaynak gridin ölçeğine bağlıdır.': 'Input resolution may not show small rocks, blocks, or narrow cracks; map detail depends on the source grid scale.',
    'Birleşik uygunluk göstergesi keşif amaçlıdır; görev doğrulamalı iniş güvenliği puanı değildir.': 'The combined suitability indicator is exploratory and is not a mission-validated landing safety score.',
    'NoData pikselleri güvenli arazi olarak yorumlanmaz; analiz dışında tutulur.': 'NoData pixels are excluded from analysis and are not treated as safe terrain.',
    'PDS indirme yöntemi IMG rasterını geçici diske indirir; dosya Render yeniden başlatıldığında kaybolabilir ve yeniden indirilmesi gerekir.': 'The PDS method downloads the IMG raster to temporary disk; it may be lost after a Render restart and need to be downloaded again.',
    'Renk aralığı 0–15° olarak sınırlandırılmıştır; 15° üzerindeki değerler aynı üst renk aralığında gösterilir.': 'The color range is limited to 0–15°; values above 15° use the same upper color range.',
    'Kaynak grid ': 'Source grid ',
    'koordinatlarla eşleşmiyor.': 'does not match the coordinates.',
    'Rota': 'Route',
    'Rotalar sırayla gösteriliyor: ': 'Routes shown in sequence: ',
    'Üçgen çalışma alanı analiz edilemedi: ': 'Could not analyze the triangular study area: ',
    'Pürüzlülük analizi başarısız oldu: ': 'Roughness analysis failed: ',
    'Eğim analizi başarısız oldu: ': 'Slope analysis failed: ',
    'Analiz edilemedi: ': 'Could not analyze: ',
    'indirme başarısız oldu.': 'download failed.',
    'Geçerli hücre yok.': 'No valid cells.',
    'Yeterli geçerli komşu piksel bulunamadı.': 'Not enough valid neighboring pixels.',
    'GeoTIFF DEM tek bantlı olmalı.': 'The GeoTIFF DEM must have a single band.',
    "GeoTIFF CRS'i coğrafi enlem-boylam (Mars) olmalı.": 'The GeoTIFF CRS must be geographic latitude/longitude for Mars.',
    'Döndürülmüş GeoTIFF gridleri henüz desteklenmiyor.': 'Rotated GeoTIFF grids are not supported yet.',
    'GeoTIFF kuzey-yukarı grid yönünde olmalı (boylam sağa, enlem yukarı artar).': 'The GeoTIFF must be north-up (longitude increases rightward and latitude upward).',
    'DEM yükseklik birimi GeoTIFF metadata içinde belirtilmeli.': 'The DEM elevation unit must be specified in GeoTIFF metadata.',
    'GeoTIFF, tanımlı çalışma üçgeninin tamamını kapsamıyor.': 'The GeoTIFF does not cover the full preset study triangle.',
    'GeoTIFF rasterında yeterli geçerli yükseklik verisi yok.': 'The GeoTIFF raster has insufficient valid elevation data.',
    'Araştırma bölgesi mevcut DEM/grid kapsamının dışında kalıyor.': 'The research region lies outside the available DEM/grid coverage.',
    'Araştırma bölgesinin kenarında geçilebilir grid hücresi yok.': 'No traversable grid cell is available along the research region boundary.',
    'Kaynak alan boyutu enlem-boylam koordinatlarıyla eşleşmiyor.': 'Source field dimensions do not match latitude/longitude coordinates.',
    'IMG indiriliyor: ': 'Downloading IMG: ',
    'PDS LBL dosyası bulunamadı veya indirilemedi. İndirme isteğe bağlı metadata olmadan devam ediyor.': 'PDS LBL file was not found or could not be downloaded. Continuing without optional metadata.',
    'Kaynak: ': 'Source: ',
    'Yükseklik birimi:': 'Elevation unit:',
    'LBL mevcut:': 'LBL available:',
    'LBL boyutu:': 'LBL size:',
    'Etiket metadata bilgileri:': 'Label metadata:',
    'Başlıkta bulunamadı': 'Not found in header',
    'En az bir desteklenen dosya seçin.': 'Select at least one supported file.',
    'Veri tipi': 'Data type',
    'Piksel boyutu': 'Pixel size',
    'Ortalama eğim': 'Mean slope',
    'Medyan eğim': 'Median slope',
    '95. yüzdelik': '95th percentile',
    'Kırmızı çizgi çalışma alanının sınırlarını gösterir. Haritadaki yükseklikler seçilen DEM kaynağından alınmıştır. Bu analiz henüz rover geçiş güvenliği değerlendirmesi değildir.': 'The red outline marks the study area. Elevations come from the selected DEM source. This analysis is not a rover traverse safety assessment.',
    'Pürüzlülük, 3×3 piksel pencerelerinde en küçük kareler yöntemiyle uydurulan yerel düzlemden RMS yükseklik sapması olarak hesaplanmıştır. Harita üçgen çevresindeki verileri de gösterir; istatistikler çalışma üçgeninin içinden hesaplanır.': 'Roughness is the RMS elevation deviation from a locally fitted plane in 3×3 pixel windows. The map shows surrounding context; statistics use the study area.',
    'Bu değer bölgesel arazi pürüzlülüğünü temsil eder. DEM çözünürlüğü nedeniyle küçük kaya ve rover tekerleği ölçeğindeki engelleri belirlemek için kullanılamaz.': 'This represents regional terrain roughness. DEM resolution is too coarse to identify small rocks or rover-wheel-scale hazards.',
    'Eğim, orijinal DEM komşu piksellerinden hesaplandı. ': 'Slope is calculated from neighboring pixels in the original DEM. ',
    'Harita her ': 'The map samples every ',
    ' pikselde bir örnek içerir. ': ' pixels. ',
    '15° üzerindeki değerler aynı üst renk aralığında gösterilir.': 'Values above 15° use the same upper color range.',
    'Dosyada ayrıştırılabilir MCD tablo bulunamadı.': 'No parseable MCD table was found in the file.',
    'TXT dosyalarında ayrıştırılabilir MCD tablo bulunamadı.': 'No parseable MCD table was found in the TXT files.',
    'Bir seferde tek EMARS .nc dosyası seçin.': 'Select one EMARS .nc file at a time.',
    'MCD TXT dosyası ': 'MCD TXT file ',
    'MOLA verileri için .img ve .hdr dosyaları zorunludur.': 'MOLA data requires both .img and .hdr files.',
    'Birden fazla .': 'Multiple .',
    ' dosyası seçildi.': ' files selected.',
    'HDR dosyası 5 MiB sınırını aşıyor.': 'The HDR file exceeds the 5 MiB limit.',
    'LBL dosyası 5 MiB sınırını aşıyor.': 'The LBL file exceeds the 5 MiB limit.',
    'IMG dosyası 50 MiB sınırını aşıyor.': 'The IMG file exceeds the 50 MiB limit.',
    'GeoTIFF dosyası 500 MiB sınırını aşıyor.': 'The GeoTIFF file exceeds the 500 MiB limit.',
    'NetCDF dosyası 140 MiB sınırını aşıyor.': 'The NetCDF file exceeds the 140 MiB limit.',
    ' dosyası 20 MiB sınırını aşıyor.': ' file exceeds the 20 MiB limit.',
    '.hdr/.lbl dosyaları tek başına kullanılamaz; .img dosyası da seçin.': '.hdr/.lbl files cannot be used alone; select the matching .img file too.',
    'Aynı yüklemede yalnızca bir DEM kaynağı seçin (.img veya GeoTIFF).': 'Select only one DEM source per upload (.img or GeoTIFF).',
    ' sıcaklık katmanı bulundu; en fazla dört harita gösterilir.': ' temperature layers found; up to four maps are shown.',
    'Renk aralığı 0–15° olarak sınırlandırılmıştır; ': 'The color range is limited to 0–15°; ',
}


_EN.update({
    'Daha fazla URL ekle': 'Add another URL',
    'Yalnızca yapılandırılmış PDS sunucusundaki doğrudan HTTPS dosyaları destekleniyor.': 'Only direct HTTPS files on the configured PDS host are supported.',
    'PDS URL’leri kimlik bilgisi veya standart dışı port içermemeli.': 'PDS URLs must not contain credentials or a nonstandard port.',
    'PDS dosya URL’lerinde sorgu parametresi veya parça bulunmamalı.': 'PDS file URLs must not contain query parameters or fragments.',
    'Doğrudan PDS dosyası için desteklenen uzantılardan birini seçin: ': 'Choose a direct PDS file ending in one of: ',
    'PDS dosyası beklenen boyut sınırını aşıyor: ': 'PDS asset exceeds the ',
    ' MiB indirme sınırı.': ' MiB download limit.',
    'PDS URL veri dosyası yerine HTML sayfası döndürdü.': 'The PDS URL returned an HTML page instead of a data file.',
    'PDS veri dosyası boş geldi.': 'The PDS data file was empty.',
    'PDS veri dosyası indirilemedi: ': 'PDS data file download failed: ',
    'PDS arşiv verileri': 'PDS archive data',
    'Desteklenen PDS ürünlerinin doğrudan HTTPS URL adreslerini girin. ENVI IMG URL’si eşleşen HDR dosyasını ve isteğe bağlı LBL dosyasını da alır.': 'Enter direct HTTPS URLs for supported PDS products. ENVI IMG URLs also fetch the matching HDR and optional LBL files.',
    'MOLA ENVI DEM (.img + .hdr)': 'MOLA ENVI DEM (.img + .hdr)',
    'GeoTIFF DEM (.tif / .tiff)': 'GeoTIFF DEM (.tif / .tiff)',
    'EMARS atmosfer verisi (.nc)': 'EMARS atmospheric data (.nc)',
    'MCD atmosfer verisi (.txt)': 'MCD atmospheric data (.txt)',
    'PDS ürün türü ': 'PDS product type ',
    'PDS dosya URL’si ': 'PDS URL ',
    'PDS verilerini indir ve analize hazırla': 'Download PDS products and prepare analysis',
    'En az bir PDS URL adresi girin.': 'Enter at least one PDS URL.',
    'Her indirmede yalnızca bir EMARS NetCDF dosyası seçin.': 'Select only one EMARS NetCDF file per download.',
    'PDS verileri indiriliyor...': 'Downloading PDS data...',
    'PDS ürünü indiriliyor (': 'Downloading PDS product (',
    ').': ').',
    'Analiz için yükseklik gridini oluşturacak en az bir DEM (.img veya GeoTIFF) ekleyin.': 'Include one or more DEM products (.img or GeoTIFF) so the analysis has an elevation grid.',
    'Aynı anda yalnızca bir GeoTIFF DEM indirin.': 'Download one GeoTIFF DEM at a time.',
    'İndirilen PDS dosyası sayısı: ': 'Downloaded PDS files: ',
    'PDS indirmesi başarısız oldu.': 'PDS download failed.',
    'PDS verileri indirilemedi veya işlenmedi: ': 'Could not download or process PDS data: ',
    'Analizden önce en az bir PDS DEM URL’si indirmeniz gerekiyor.': 'Download at least one PDS DEM URL before analysis.',
    'Analiz edilecek DEM ürünü': 'DEM product to analyze',
    'PDS geçici dosyası bulunamadı. Dosyayı yeniden indirin.': 'The temporary PDS file is missing. Download the product again.',
    'İndirilen MOLA rasterı açılamadı: ': 'Could not open the downloaded MOLA raster: ',
    'İndirilen GeoTIFF DEM açılamadı: ': 'Could not open the downloaded GeoTIFF DEM: ',
    'Kaynak: ': 'Source: ',
    ' | HDR: mevcut | LBL: ': ' | HDR: available | LBL: ',
    'mevcut': 'available',
    'bulunamadı (isteğe bağlı)': 'not found (optional)',
    'PDS doğrudan indirme': 'PDS direct download',
    'İndirilen geçici verileri temizle': 'Clear downloaded temporary data',
    'Olympus Mons etki araştırma bölgesi': 'Olympus Mons impact research area',
    'Doğrudan `.img` dosyalarının HTTPS adreslerini girin. Uygulama eşleşen `.hdr` dosyasını, varsa `.lbl` dosyasını otomatik bulur.': 'Enter HTTPS URLs for the `.img` files. The matching `.hdr` file and, if available, `.lbl` file are detected automatically.',
    'MOLA IMG adresi': 'MOLA IMG URL',
    'Atmosfer verileri': 'Atmospheric data',
    "PDS ile seçilen DEM'e ek olarak EMARS NetCDF veya MCD TXT verisi yükleyebilirsiniz.": 'Upload EMARS NetCDF or MCD TXT data alongside the DEM selected from PDS.',
    'Atmosfer (.nc, .txt) dosyalarını seçin': 'Select atmosphere files (.nc, .txt)',
    'Seçilen atmosfer verilerini yükle ve işle': 'Upload and process selected atmosphere data',
    'Analiz edilecek MOLA ürünü': 'MOLA product to analyze',
    'İndirmek için en az bir IMG adresi girin.': 'Enter at least one IMG URL to download.',
    'Analiz için önce en az bir PDS IMG adresi girip indirin.': 'Enter and download at least one PDS IMG URL before analysis.',
})


_EN.update({
    'Desteklenen verilerin do?rudan HTTPS URL adreslerini girin. ENVI IMG adresi e?le?en HDR dosyas?n? ve iste?e ba?l? LBL dosyas?n? da al?r.': 'Enter direct HTTPS URLs for supported data. ENVI IMG URLs also fetch the matching HDR and optional LBL files.',
    'PDS d??? bir HTTPS alan ad? i?in Render ortam?nda MARS_DATA_ALLOWED_HOSTS de?i?kenini ayarlay?n.': 'For a non-PDS HTTPS host, set MARS_DATA_ALLOWED_HOSTS in the Render environment.',
})

def translate(value, language="tr"):
    if not isinstance(value, str):
        return value
    if language != "en":
        # Some UI labels and table headings are authored in English. Translate
        # those back as well so Turkish mode stays consistent.
        reverse = {english: turkish for turkish, english in _EN.items() if english != turkish}
        if value in reverse:
            return reverse[value]
        result = value
        for english, turkish in sorted(reverse.items(), key=lambda item: len(item[0]), reverse=True):
            if english and english in result:
                result = result.replace(english, turkish)
        return result
    if value in _EN:
        return _EN[value]
    # Dynamic UI messages reuse these phrase fragments (counts, units, filenames).
    phrases = sorted(_EN.items(), key=lambda item: len(item[0]), reverse=True)
    result = value
    for turkish, english in phrases:
        if turkish and turkish in result:
            result = result.replace(turkish, english)
    return result


def _translate_value(value, language):
    if isinstance(value, str):
        return translate(value, language)
    if isinstance(value, tuple):
        return tuple(_translate_value(item, language) for item in value)
    if isinstance(value, list):
        return [_translate_value(item, language) for item in value]
    if isinstance(value, dict):
        return {
            _translate_value(key, language): _translate_value(item, language)
            for key, item in value.items()
        }
    return value


class LocalizedStreamlit:
    """Proxy Streamlit and column/tab containers to translate displayed text."""

    def __init__(self, target, language="tr"):
        self._target = target
        self._language = language

    def __getattr__(self, name):
        value = getattr(self._target, name)
        # Streamlit's session state is a mutable mapping. Wrapping it breaks
        # assignments such as st.session_state["key"] = value.
        if name == "session_state":
            return value
        if callable(value):
            @wraps(value)
            def call(*args, **kwargs):
                translated_args = tuple(_translate_value(arg, self._language) for arg in args)
                translated_kwargs = {key: _translate_value(arg, self._language) for key, arg in kwargs.items()}
                result = value(*translated_args, **translated_kwargs)
                return self._wrap(result)
            return call
        return self._wrap(value)

    def _wrap(self, value):
        if isinstance(value, list):
            return [self._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._wrap(item) for item in value)
        if hasattr(value, "__enter__") and hasattr(value, "__exit__"):
            return _LocalizedContext(value, self._language)
        # Streamlit elements such as columns, placeholders, progress bars and status
        # containers expose rendering methods but are not themselves callable.
        if any(hasattr(value, name) for name in ("metric", "write", "progress", "update", "pyplot", "empty", "caption", "table")):
            return LocalizedStreamlit(value, self._language)
        return value


class _LocalizedContext(LocalizedStreamlit):
    def __enter__(self):
        return self._wrap(self._target.__enter__())

    def __exit__(self, exc_type, exc, tb):
        return self._target.__exit__(exc_type, exc, tb)
