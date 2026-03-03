# Banka Müşteri Churn Tahmini

Banka müşterilerinin bankadan ayrılma eğilimini (churn) tahmin etmeye yönelik bir makine öğrenmesi projesidir. Random Forest, Gradient Boosting ve LightGBM modelleri karşılaştırılmış; veri setindeki sınıf dengesizliği SMOTE yöntemiyle giderilmiştir.

---

## İçindekiler

- [Veri Seti](#veri-seti)
- [Yöntem](#yöntem)
- [Sonuçlar](#sonuçlar)
- [Kurulum](#kurulum)
- [Proje Yapısı](#proje-yapısı)
- [Kaynakça](#kaynakça)

---

## Veri Seti

Kaynak: [Kaggle — Churn Prediction for Credit Card Customer](https://www.kaggle.com/datasets/mukeshmanral/churn-prediction-for-credit-card-customer/data)

Veri seti 10.000 banka müşterisine ait demografik ve finansal bilgilerden oluşmaktadır. Hedef değişken `Exited` olup sınıf dağılımı %80 (kaldı) / %20 (ayrıldı) şeklinde dengesizdir.

| Değişken | Açıklama |
|---|---|
| `CreditScore` | Kredi puanı (350–850) |
| `Geography` | Ülke (France, Germany, Spain) |
| `Gender` | Cinsiyet |
| `Age` | Yaş |
| `Tenure` | Bankadaki yıl sayısı |
| `Balance` | Hesap bakiyesi |
| `NumOfProducts` | Sahip olunan ürün sayısı |
| `HasCrCard` | Kredi kartı varlığı |
| `IsActiveMember` | Aktif üyelik durumu |
| `EstimatedSalary` | Tahmini maaş |
| `Exited` | Bankadan ayrılma (hedef değişken) |

---

## Yöntem

### Keşifsel Veri Analizi

Histogram, boxplot ve korelasyon matrisi ile değişkenler incelenmiştir. Öne çıkan bulgular:

- Alman müşterilerin terk oranı %32 ile Fransa (%16) ve İspanya'ya (%17) kıyasla belirgin biçimde yüksektir.
- 3 veya 4 ürüne sahip müşterilerin neredeyse tamamı churn olmuştur (%82–%100).
- Hesap bakiyesi sıfır olan müşteriler görece daha az terk etmiştir.
- Kredi skoru 405'in altında olan 20 müşterinin tamamı churn olmuştur.

### Türetilmiş Değişkenler

| Değişken | Açıklama |
|---|---|
| `credit_score_table` | Kredi skoru, uluslararası derecelendirme tablosuna göre gruplandırıldı |
| `Age` (gruplandırılmış) | 18–30=1, 30–40=2, 40–50=3, 50–60=4, 60–92=5 |
| `retired` | Ülkeye göre emeklilik yaşı bazlı ikili değişken |
| `NOP*` | NumOfProducts'ın churn riskine göre yeniden sıralanmış hali |
| `Balance0` | Hesap bakiyesinin sıfır olup olmadığını gösteren ikili değişken |
| `smallerthan405` | Kredi skoru 405 altında mı |
| `Tenure/NumOfProducts` | Ürün başına bankada geçirilen süre |
| `ES/Age`, `ES/Tenure`, `ES/Score` | Tahmini maaş bazlı oran değişkenleri |
| `Balance/ES` | Bakiye/maaş oranı |

### Ön İşleme

`Geography` ve `Gender` değişkenlerine one-hot encoding uygulanmıştır. Sayısal değişkenler aykırı değerlere karşı dayanıklı olması nedeniyle Robust Scaler ile ölçeklenmiştir. Sınıf dengesizliği SMOTE ile giderilmiştir.

### Modeller

Random Forest, Gradient Boosting ve LightGBM modelleri 10 katlı çapraz doğrulama ile değerlendirilmiştir. LightGBM için hiperparametre arama da yapılmış ancak varsayılan değerlerin üzerine çıkılamamıştır.

---

## Sonuçlar

| Model | CV Doğruluk | Train Doğruluk | Test Doğruluk |
|---|---|---|---|
| Random Forest | — | — | ~%86 |
| Random Forest + SMOTE | %89.24 | %92.61 | %89.39 |
| LightGBM | %85.75 | — | %85.80 |
| LightGBM + SMOTE | %89.47 | %92.61 | %89.45 |

En iyi sonuç LightGBM + SMOTE kombinasyonuyla elde edilmiştir. 3186 tahmin üzerinden %89.42 doğruluk sağlanmış; churn olan müşteriler için precision %92, recall %87, f1-score %89 olarak gerçekleşmiştir.

En belirleyici değişkenler `Age`, `NumOfProducts`, `EstimatedSalary`, `Tenure/Age`, `ES/Tenure` ve `Geography_Germany` olarak öne çıkmıştır.

---

## Kurulum

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm imbalanced-learn
```

Veri setini Kaggle'dan indirip `Churn.csv` adıyla proje klasörüne kaydedin, ardından notebook'u çalıştırın:

```bash
jupyter notebook churn_prediction.ipynb
```

---

## Proje Yapısı

```
├── 1_eda.ipynb               # Keşifsel veri analizi
├── 2_preprocessing.ipynb    # Encoding, feature engineering, ölçekleme
├── 3_modeling.ipynb         # Model eğitimi, SMOTE, değerlendirme
├── Churn.csv
└── README.md
```

---

## Kaynakça

1. [Kaggle — Churn Prediction for Credit Card Customer](https://www.kaggle.com/datasets/mukeshmanral/churn-prediction-for-credit-card-customer/data)
2. [SMOTE for Imbalanced Classification with Python](https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/)
3. [LightGBM — Veri Bilimi Okulu](https://www.veribilimiokulu.com/lightgbm/)
4. [Veri Görselleştirme Kataloğu](https://datavizcatalogue.com/TR/arama.html)
5. [How to Build a Customer Churn Prediction Model in Python?](https://365datascience.com/tutorials/python-tutorials/how-to-build-a-customer-churn-prediction-model-in-python/)
6. [Python — Customer Churn Analysis Prediction](https://www.geeksforgeeks.org/python-customer-churn-analysis-prediction/)
