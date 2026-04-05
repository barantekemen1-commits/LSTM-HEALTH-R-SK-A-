# Kardiyovasküler Kriz Öncesi Uyarı Sistemi

**LSTM Tabanlı Kişiselleştirilmiş Pre-Crisis Prediction**  
TÜBİTAK 2242 Üniversite Öğrencileri Araştırma Proje Yarışması — Nisan 2026

---

## Proje Hakkında

Bu proje, giyilebilir bir bilekliğe gömülü sensörlerden (nabız, oksijen satürasyonu, EKG) sürekli veri okuyarak kardiyovasküler krizi başlamadan **10 dakika önce** tespit eden bir yapay zeka sistemi geliştirmeyi amaçlamaktadır.

Mevcut akıllı saat ve bileklik sistemlerinin büyük çoğunluğu anlık eşik değer aşımlarına göre alarm üretir. Bu yaklaşım bireyler arası fizyolojik farklılıkları göz ardı ettiğinden hem yüksek yanlış alarm oranına hem de gerçek risklerin kaçırılmasına yol açmaktadır. Bu çalışmada her kullanıcı için ayrı bir bazal profil oluşturularak kişiselleştirilmiş bir erken uyarı sistemi tasarlanmıştır.

---

## Repoda Neden İki Farklı Model Var?

Bu repoda birbirini tamamlayan iki ayrı çalışma bulunmaktadır:

### 1. `pre_crisis_detection.py` — Ana Model (Simülasyon)

Bu dosya projenin **asıl amacını** gerçekleştiren modeldir. HR + SpO₂ + EKG olmak üzere üç sensör kanalının tamamını kullanır. Gerçek hasta verisine henüz erişim sağlanamadığından (MIMIC-III başvurusu yapıldı, onay bekleniyor) veriler fizyolojik literatüre dayalı olarak simüle edilmiştir. Model krizden **10 dakika önceki prodromal dönemi** yakalamak üzere özellikle tasarlanmıştır.

### 2. `tekemen_baran_v7.py` + `mitbih_loader_v7.py` — Gerçek Veri Doğrulaması

Bu dosyalar sistemin gerçek EKG verisi üzerinde nasıl davrandığını göstermek amacıyla yazılmıştır. PhysioNet MIT-BIH Aritmia Veritabanı (44 kayıt, 26.444 pencere) kullanılmaktadır. Ancak MIT-BIH yalnızca EKG kanalı içerdiğinden HR ve SpO₂ verisi bulunmamaktadır. Bu nedenle bu model sistemin tam kapasitesini değil, yalnızca EKG modalitesindeki performansını ölçmektedir.

### Özet Karşılaştırma

| | `pre_crisis_detection.py` | `tekemen_baran_v7.py` |
|---|---|---|
| Veri türü | Simülasyon | Gerçek EKG (MIT-BIH) |
| Kanallar | HR + SpO₂ + EKG | Yalnızca EKG |
| Pre-crisis tahmini | Var (10 dk önce) | Yok |
| ROC AUC | 0.993 | 0.797 |
| Amaç | Ana sistem | Gerçek veri doğrulaması |

> MIMIC-III erişimi onaylandığında `pre_crisis_detection.py` gerçek HR + SpO₂ + EKG verisiyle yeniden eğitilecek ve iki model birleştirilecektir.

---

## Sistem Nasıl Çalışır?

**1. hafta — Kalibrasyon:**  
Cihaz arka planda çalışarak kullanıcının dinlenme nabzını, oksijen saturasyonunu ve EKG ortalamalarını öğrenir. Bu değerler kişisel referans profili olarak saklanır.

**Sonraki günler — Sürekli İzleme:**  
Her 60 dakikalık sinyal penceresi LSTM modeline verilir. Model kişisel bazaldan sapmaları değerlendirerek bir risk skoru üretir. Risk skoru eşik değerini aşarsa kullanıcı uyarılır, gerekirse otomatik acil çağrısı başlatılır.

**Pre-crisis penceresi:**  
Gerçek hayatta kardiyovasküler olaylar öncesinde nabız yavaşça yükselir, oksijen hafifçe düşer. Model bu prodromal dönemi yakalamak üzere eğitilmiştir — kriz anını değil, krizi öncesini tespit eder.

---

## Sonuçlar

### Simülasyon Modeli — HR + SpO₂ + EKG (3 Kanal)

| Metrik | Değer |
|---|---|
| ROC AUC | **0.993** |
| Duyarlılık (Recall) | **%97.9** |
| Kesinlik (Precision) | **%100.0** |
| Genel Doğruluk | **%99.58** |
| Yanlış Alarm (FP) | **0** |
| Pre-crisis Uyarı Süresi | **10 dakika** |

### MIT-BIH Gerçek EKG Verisi — Yalnızca EKG (1 Kanal)

| Metrik | Değer |
|---|---|
| ROC AUC | 0.797 |
| Duyarlılık (Recall) | 0.852 |
| Kesinlik (Precision) | 0.264 |

> MIT-BIH yalnızca EKG kanalı içerdiğinden bu sonuçlar sistemin tam kapasitesini yansıtmamaktadır. MIMIC-III ile HR + SpO₂ + EKG üçlüsü kullanıldığında simülasyon sonuçlarına yakın performans beklenmektedir.

---

## Dosya Yapısı

```
├── pre_crisis_detection.py           # Ana model — simülasyon, 3 kanal
├── tekemen_baran_v7.py               # MIT-BIH gerçek veri doğrulaması
├── mitbih_loader_v7.py               # MIT-BIH veri yükleme modülü
├── bileklik_pre_crisis_sim_v1.keras  # Simülasyon modeli ağırlıkları
├── bileklik_model_v7.keras           # MIT-BIH modeli ağırlıkları
├── outputs/                          # Görsel çıktılar
│   ├── cm_sim_v1.png
│   ├── roc_sim_v1.png
│   ├── training_sim_v1.png
│   ├── pre_crisis_gorsel_sim_v1.png
│   ├── dashboard_sim_v1_KRIZ.png
│   ├── dashboard_sim_v1_NORMAL.png
│   ├── cm_v7.png
│   ├── roc_v7.png
│   └── training_v7.png
└── README.md
```

---

## Kurulum ve Çalıştırma

Google Colab üzerinde çalıştırmak için:

**Simülasyon modeli:**
```python
!pip install wfdb -q
exec(open('pre_crisis_detection.py').read())
```

**MIT-BIH gerçek veri modeli:**
```python
!pip install wfdb -q
exec(open('tekemen_baran_v7.py').read())
# MIT-BIH verileri PhysioNet'ten otomatik indirilecektir
```

---

## Kullanılan Teknolojiler

- **Python 3.10** — TensorFlow / Keras, NumPy, scikit-learn, matplotlib
- **LSTM** — iki katmanlı, BatchNormalization + Dropout
- **Focal Loss** — sınıf dengesizliğini yönetmek için (γ=2.0, α=0.75)
- **MIT-BIH Aritmia Veritabanı** — PhysioNet açık erişim
- **MIMIC-III** — erişim başvurusu yapıldı, onay bekleniyor

---

## Donanım Hedefi

Yazılım katmanı tamamlandıktan sonra aşağıdaki bileşenlerle fiziksel prototip üretilmesi planlanmaktadır:

- **MAX30102** — PPG sensörü (nabız + SpO₂)
- **AD8232** — tek kanal EKG modülü
- **ESP32** — gömülü mikrodenetleyici
- **GSM modülü** — telefon bağımsız acil çağrı

---

## Yazar

**Baran Tekemen**  
TÜBİTAK 2242 — Sağlık / Biyomedikal Cihaz Teknolojileri  
Nisan 2026
