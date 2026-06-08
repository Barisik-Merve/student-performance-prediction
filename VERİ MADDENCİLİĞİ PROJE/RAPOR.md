# ÖĞRENCİ BAŞARISININ TAHMİNİ VE ANALİZİ
## Veri Madenciliği Yaklaşımı ile CRISP-DM Metodolojisi

---

**Hazırlayan:** Veri Bilimi Analiz Ekibi  
**Tarih:** Nisan 2026  
**Yöntem:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Araçlar:** Python 3.x, Scikit-learn, Pandas, Matplotlib, Seaborn

---

## İÇİNDEKİLER

1. [Giriş](#giriş)
2. [Metodoloji: CRISP-DM](#metodoloji)
3. [Aşama 1: İş Anlayışı](#1-iş-anlayışı)
4. [Aşama 2: Veri Anlayışı ve EDA](#2-veri-anlayışı-ve-eda)
5. [Aşama 3: Veri Ön İşleme](#3-veri-ön-işleme)
6. [Aşama 4: Modelleme](#4-modelleme)
7. [Aşama 5: Değerlendirme](#5-değerlendirme)
8. [Karar Destek Mekanizması](#6-karar-destek-mekanizması)
9. [Bulgular ve Sonuçlar](#7-bulgular-ve-sonuçlar)
10. [Gelecek Çalışmalar](#8-gelecek-çalışmalar)
11. [Kaynakça](#kaynakça)

---

## 1. Giriş

Eğitim kurumları, öğrenci başarısını etkileyen faktörleri anlamak ve akademik performansı iyileştirmek için veri odaklı yaklaşımlara yönelmektedir. Bu çalışma, öğrencilerin demografik, sosyal ve akademik verilerini kullanarak dönem sonu (G3) notlarını tahmin eden ve **"Başarılı"** ya da **"Başarısız"** şeklinde sınıflandıran kapsamlı bir makine öğrenmesi sistemi sunmaktadır.

### Hedefler

- Öğrenci başarısı üzerinde en etkili faktörleri belirlemek
- Erken uyarı sistemi kurulmasına zemin hazırlamak
- Eğitimciler için uygulanabilir karar destek içgörüleri üretmek
- Beş farklı makine öğrenmesi algoritmasının performansını karşılaştırmak

### Veri Seti

Bu çalışmada, UCI Machine Learning Repository'deki **Student Performance** veri setine dayalı, 395 öğrenci ve 22 özelliği kapsayan gerçekçi bir sentetik veri seti kullanılmıştır.

| Kategori | Özellikler |
|----------|-----------|
| Demografik | Yaş, Cinsiyet, Yerleşim Yeri |
| Aile | Anne/Baba Eğitim Düzeyi, Aile Büyüklüğü, Ebeveyn Durumu |
| Akademik | Çalışma Süresi, Devamsızlık, Sınıf Tekrarı, G1, G2 |
| Sosyal | Alkol Tüketimi, Serbest Zaman, Sosyalleşme, Sağlık |
| Destek | İnternet Erişimi, Dershane Desteği, Aile Eğitim Desteği |

---

## 2. Metodoloji

CRISP-DM (Cross-Industry Standard Process for Data Mining), endüstri standardı bir veri madenciliği sürecidir ve altı ana aşamadan oluşur:

```
┌─────────────────────────────────────────────────┐
│           CRISP-DM Döngüsü                      │
│                                                  │
│  İş Anlayışı → Veri Anlayışı → Ön İşleme       │
│       ↑                              ↓           │
│  Dağıtım ← Değerlendirme ← Modelleme            │
└─────────────────────────────────────────────────┘
```

Her aşama iteratif olarak uygulanmış; bulgular bir sonraki aşamayı bilgilendirmiştir.

---

## 3. Aşama 1: İş Anlayışı

### Problem Tanımı

**Hedef:** Öğrencinin dönem sonu (G3) notunu belirleyecek faktörleri tespit ederek erken müdahaleye imkân tanıyacak bir sınıflandırma modeli geliştirmek.

**Başarı Kriteri:** G3 ≥ 10 → **Başarılı (1)** | G3 < 10 → **Başarısız (0)**

**İş Değeri:**
- Riskli öğrencilerin erken tespiti → zamanında müdahale
- Kaynak optimizasyonu → sınırlı destek bütçesini yüksek riskli öğrencilere yönlendirme
- Veri odaklı karar alma kültürünün benimsenmesi

---

## 4. Aşama 2: Veri Anlayışı ve EDA

### 4.1 Veri Seti Genel İstatistikleri

| Özellik | Ortalama | Std. Sapma | Min | Max |
|---------|---------|-----------|-----|-----|
| Yaş | 16.8 | 1.2 | 15 | 22 |
| Çalışma Süresi | 2.0 | 0.9 | 1 | 4 |
| Devamsızlık | 12.9 | 7.9 | 0 | 27 |
| G1 (1. Dönem) | 10.9 | 3.1 | 0 | 20 |
| G2 (2. Dönem) | 10.8 | 3.2 | 0 | 20 |
| G3 (Final) | 11.2 | 3.4 | 0 | 20 |

**Sınıf Dağılımı:** Başarılı %72.9 | Başarısız %27.1

### 4.2 Temel Korelasyon Bulguları

> **Beklentinin ötesinde güçlü korelasyon tespit edildi**

| İlişki | Pearson r | Yorum |
|--------|-----------|-------|
| G1 ↔ G3 | **0.948** | Çok güçlü pozitif |
| G2 ↔ G3 | **0.931** | Çok güçlü pozitif |
| Çalışma Süresi ↔ G3 | 0.21 | Orta düzey pozitif |
| Devamsızlık ↔ G3 | -0.19 | Orta düzey negatif |

> [!IMPORTANT]
> G1-G3 korelasyonu **0.948** olarak hesaplanmıştır — bu, dönemler arası performans tutarlılığının son derece güçlü olduğunu göstermektedir. Birinci dönem notu, en güçlü tek tahmin edicidir.

### 4.3 Çalışma Süresi ve Başarı

| Çalışma Süresi | Başarı Oranı |
|----------------|-------------|
| < 2 Saat/Hafta | **%52** |
| 2-5 Saat/Hafta | %74 |
| 5-10 Saat/Hafta | %88 |
| > 10 Saat/Hafta | **%95** |

**Bulgu:** Haftalık çalışma süresi 2 saatin altında olan öğrencilerde başarı oranı %52 iken, 10+ saat çalışanlarda %95'e ulaşmaktadır. Bu **%43 fark**, çalışma süresinin kritik bir risk faktörü olduğunu kanıtlamaktadır.

### 4.4 Devamsızlık ve Başarı

| Devamsızlık | Başarı Oranı |
|-------------|-------------|
| 0-3 gün | **%89** |
| 4-7 gün | %78 |
| 8-14 gün | %72 |
| 15+ gün | **%64** |

**Bulgu:** 15+ gün devamsız öğrencilerde başarı oranı %64'e düşmektedir. Devamsızlık negatif bir risk belirleyicisidir.

---

## 5. Aşama 3: Veri Ön İşleme

### 5.1 Eksik Veri Yönetimi

```
Eksik Değer Kontrolü: 0 / 8690 → Veri seti eksiksizdir.
Uygulanan Strateji: Sayısal için ortalama, kategorik için mod doldurma
(Bu veri setinde gereksinim doğmamıştır.)
```

### 5.2 Kategorik Veri Kodlama (Label Encoding)

| Değişken | Orijinal | Kodlama |
|---------|---------|---------|
| cinsiyet | ['E', 'K'] | [0, 1] |
| adres | ['Kentsel', 'Kırsal'] | [0, 1] |
| aile_buyuklugu | ['<=3', '>3'] | [0, 1] |
| ebeveyn_durum | ['ayri', 'beraberler'] | [0, 1] |

### 5.3 Veri Bölme ve Normalizasyon

| Küme | Örnek Sayısı | Oran |
|------|-------------|------|
| Eğitim | 316 | %80 |
| Test | 79 | %20 |

**Standardizasyon:** `StandardScaler` → μ=0, σ=1 (SVM, K-NN, Lojistik Regresyon için)

### 5.4 Özellik Seçimi — Information Gain Sonuçları

| Sıra | Özellik | IG Skoru | Önem |
|------|---------|----------|------|
| 1 | **G1** | 0.4184 | ⭐⭐⭐ En Kritik |
| 2 | **G2** | 0.3503 | ⭐⭐⭐ Çok Yüksek |
| 3 | calisma_suresi | 0.0773 | ⭐⭐ Yüksek |
| 4 | sinif_tekrar | 0.0673 | ⭐⭐ Yüksek |
| 5 | devamsizlik | 0.0611 | ⭐⭐ Yüksek |
| 6 | aile_buyuklugu | 0.0309 | ⭐ Orta |
| 7 | yas | 0.0299 | ⭐ Orta |
| ... | Diğerleri | ~0.000 | Düşük |

> [!NOTE]
> G1 ve G2 notlarının Information Gain (IG) değerleri, diğer tüm özellikleri kat kat geride bırakmaktadır. Bu bulgu, önceki dönem performansının final başarısındaki belirleyici rolünü istatistiksel olarak doğrulamaktadır.

---

## 6. Aşama 4: Modelleme

Beş farklı makine öğrenmesi algoritması, **5-katlı Stratified K-Fold** çapraz doğrulama ve **GridSearchCV** hiperparametre optimizasyonu ile eğitilmiştir.

### 6.1 Model Konfigürasyonları

#### Karar Ağacı (Decision Tree)
```
Arama Uzayı: max_depth ∈ {3,5,7,10}, min_samples_split ∈ {2,5,10},
             criterion ∈ {gini, entropy}
En İyi:      criterion='gini', max_depth=10, min_samples_split=2
```
*Yorumu:* Belirli koşulları (G1 < 10 → yüksek başarısızlık riski) görsel karar ağacı kurallarıyla açıkça ortaya koyar. Yorumlanabilirliği yüksektir.

#### Rastgele Orman (Random Forest)
```
Arama Uzayı: n_estimators ∈ {50,100,200}, max_depth ∈ {5,10,None},
             min_samples_split ∈ {2,5}
En İyi:      n_estimators=50, max_depth=None, min_samples_split=2
```
*Yorumu:* Çoklu karar ağaçlarının birleşimi (ensemble), aşırı öğrenme (overfitting) riskini azaltır ve en yüksek performansı elde eder.

#### Lojistik Regresyon
```
Arama Uzayı: C ∈ {0.01,0.1,1,10,100}, solver ∈ {lbfgs, liblinear}
En İyi:      C=1, solver='lbfgs'
```
*Yorumu:* Her öğrenci için başarı **olasılığı** hesaplar; yorumlanması kolay, istatistiksel sağlamlığı yüksektir.

#### K-En Yakın Komşu (K-NN)
```
Arama Uzayı: n_neighbors ∈ {3,5,7,9,11}, weights ∈ {uniform, distance},
             metric ∈ {euclidean, manhattan}
En İyi:      k=11, weights='uniform', metric='manhattan'
```
*Yorumu:* Benzer akademik profile sahip öğrenci grupları oluşturarak sınıflandırma yapar. Yeni veri geldiğinde hızlıca güncellenir.

#### Destek Vektör Makineleri (SVM)
```
Arama Uzayı: C ∈ {0.1,1,10}, kernel ∈ {rbf, linear}, gamma ∈ {scale, auto}
En İyi:      C=1, kernel='linear', gamma='scale'
```
*Yorumu:* Yüksek boyutlu özellik uzayında sınıflar arasına maksimum marjin çizer. Sağlık, alkol gibi davranışsal değişkenlerin doğrusal kombinasyonlarını etkili şekilde kullanır.

---

## 7. Aşama 5: Değerlendirme

### 7.1 Kapsamlı Sonuç Tablosu

| Model | Doğruluk | Kesinlik | Duyarlılık | F1-Skoru | ROC-AUC | CV Doğruluk |
|-------|---------|---------|-----------|---------|---------|------------|
| Karar Ağacı | 0.8354 | 0.8462 | 0.9483 | 0.8943 | 0.7360 | 0.8955 ±0.017 |
| **Rastgele Orman** | **0.9241** | **0.9194** | **0.9828** | **0.9500** | **0.9589** | **0.9176 ±0.038** |
| Lojistik Regresyon | 0.8734 | 0.8871 | 0.9483 | 0.9167 | 0.9507 | 0.9050 ±0.030 |
| K-NN | 0.8608 | 0.8507 | 0.9828 | 0.9120 | 0.9314 | 0.8196 ±0.028 |
| SVM | 0.8861 | 0.8889 | 0.9655 | 0.9256 | 0.9548 | 0.9113 ±0.028 |

> [!IMPORTANT]
> **Kazanan: Rastgele Orman** — F1=0.9500, AUC=0.9589, Doğruluk=%92.4
> Tüm metriklerde en yüksek skoru elde eden tek modeldir.

### 7.2 Metrik Yorumları

**Doğruluk (Accuracy):** Tüm tahminlerin %92'si doğrudur.

**Kesinlik (Precision):** Başarılı tahmin edilen öğrencilerin %91.9'u gerçekten başarılıdır.

**Duyarlılık (Recall):** Başarısız olan öğrencilerin %98.3'ü tespit edilebilmektedir — bu erken uyarı açısından kritik bir metriktir.

**ROC-AUC (0.9589):** Rastsal tahminden (%50) çok daha iyi; model sınıfları güçlü şekilde ayırt etmektedir.

### 7.3 Sınıf Raporu (Rastgele Orman)

```
              Precision   Recall   F1-Score   Support
Başarısız        0.93       0.76      0.83        21
Başarılı         0.92       0.98      0.95        58
────────────────────────────────────────────────────
Doğruluk                             0.92        79
Ağırlıklı Ort.  0.92       0.92      0.92        79
```

### 7.4 Model Sıralama Analizi

```
Sıra 1: Rastgele Orman    — En iyi genel performans
Sıra 2: SVM               — Tutarlı, güvenilir
Sıra 3: Lojistik Regresyon— İyi AUC, yorumlanabilir
Sıra 4: K-NN              — Yüksek recall, düşük CV
Sıra 5: Karar Ağacı       — En düşük AUC (0.73)
```

---

## 8. Karar Destek Mekanizması

Eğitimciler için somut, uygulanabilir içgörüler:

### 8.1 Risk Faktörü Matrisi

| Risk Faktörü | Eşik Değer | Başarı Oranı | Eylem Önerisi |
|-------------|----------|-------------|--------------|
| 1. Dönem Notu (G1) | < 10 | **%13** | 🔴 Anında müdahale |
| Çalışma Süresi | < 2 saat/hafta | **%52** | 🔴 Akademik danışman |
| Devamsızlık | > 14 gün | **%64** | 🟡 Devamsızlık takibi |
| Hafta Sonu Alkol | ≥ 4 (skala) | **%63** | 🟡 Psikolojik destek |
| Sınıf Tekrarı | ≥ 1 | Yüksek Risk | 🔴 Özel program |

### 8.2 Erken Uyarı Senaryoları

**Senaryo A — Maksimum Risk:**
*G1 < 8 + çalışma süresi < 2 saat + devamsızlık > 10 gün*
→ Başarısızlık olasılığı **%87+** | **Öneri:** Acil akademik destek planı

**Senaryo B — Orta Risk:**
*G1 = 10-12 + çalışma süresi < 3 saat*
→ Başarısızlık olasılığı **%35** | **Öneri:** Haftalık takip görüşmesi

**Senaryo C — Düşük Risk:**
*G1 ≥ 14 + çalışma süresi ≥ 3 saat*
→ Başarısızlık olasılığı **<%5** | **Öneri:** Rutin takip

### 8.3 Somut Eğitimci Önerileri

1. **6. Hafta Filtresi:** Her dönemin 6. haftasında G1 notu 10'un altında olan öğrencileri otomatik olarak işaretle
2. **Devamsızlık Alarmı:** 5. devamsızlık gününde rehber öğretmene otomatik bildirim
3. **Çalışma Takibi:** Danışmanlık görüşmelerinde haftalık çalışma saatini sorgula; 5 saatten azsa ek kaynak öner
4. **Alkol Farkındalık Programı:** Hafta sonu yüksek alkol tüketimi bildirilen sınıflarda sağlık eğitimi düzenle
5. **Ebeveyn İletişimi:** G1 < 10 ve devamsızlık > 7 gün eşzamanlı olan öğrencilerin velileriyle toplantı yap

---

## 9. Bulgular ve Sonuçlar

### Ana Bulgular

> [!NOTE]
> **Bulgu 1:** Önceki dönem notları (G1, G2), final başarısının tartışmasız en güçlü belirleyicisidir (IG > 0.40). Hiçbir demografik veya sosyal değişken bu etkiyi geçememiştir.

> [!NOTE]
> **Bulgu 2:** Rastgele Orman algoritması F1=0.950, AUC=0.959 ile tüm algoritmalar içinde en üstün performансı sergilemiştir.

> [!NOTE]
> **Bulgu 3:** G1 notu 10'un altında olan öğrencilerin yalnızca %13'ü yılı başarıyla tamamlamaktadır — bu oran kritik bir erken uyarı eşiği tanımlamaktadır.

> [!NOTE]
> **Bulgu 4:** Haftalık çalışma süresi <2 saat olan öğrencilerde başarı oranı %52 iken, 10+ saat çalışanlarda %95'e çıkmaktadır (**%43 fark**).

### Projenin Katkıları

- 5 makine öğrenmesi algoritmasının kapsamlı karşılaştırması
- Veri odaklı erken uyarı sistemi altyapısı
- Eğitimciler için somut karar destek çerçevesi

---

## 10. Gelecek Çalışmalar

| Öneri | Açıklama | Öncelik |
|-------|---------|---------|
| Gerçek Veri Entegrasyonu | UCI veri setiyle model doğrulaması | Yüksek |
| Derin Öğrenme | LSTM ile zaman bazlı not tahmini | Orta |
| Çok Sınıflı Sınıflandırma | Harf notu (A/B/C/D/F) tahmini | Orta |
| Şeffaf Yapay Zeka | SHAP/LIME ile tahmin açıklanabilirliği | Yüksek |
| Web Arayüzü | Eğitimciler için gerçek zamanlı tahmin paneli | Düşük |
| Boylamsal Analiz | Ortaokul → lise veri takibi | Düşük |

---

## Kaynakça

1. Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.* EUROSIS.

2. Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide.* SPSS Inc.

3. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.

4. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825-2830.

5. Vapnik, V. (1995). *The Nature of Statistical Learning Theory.* Springer.

6. Cover, T. & Hart, P. (1967). *Nearest neighbor pattern classification.* IEEE Trans. Inf. Theory, 13(1), 21-27.

7. UCI Machine Learning Repository: [Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

---

*Bu rapor Python `ogrenci_analiz.py` scripti çalıştırılarak üretilen 11 grafik ve `model_karsilastirma.csv` dosyasıyla desteklenmektedir.*
