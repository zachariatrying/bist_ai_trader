# BorsaNeuron: Veri Madenciliği ve Yapay Zeka ile Hisse Senedi Analizi
**Ders:** ACM 465 - Veri Madenciliği  
**Öğrenci:** İbrahim  
**Proje Sorusu:** Bir hisse senedinin 5 gün sonraki kapanış fiyatı bugünkünden yüksek mi olacak?

---

## Slayt 1: Kapak
* **Başlık:** BorsaNeuron: Teknik Analiz ve Makine Öğrenmesi ile BIST Hisse Tahmin Sistemi
* **Alt Başlık:** ACM 465 - Veri Madenciliği Final Projesi
* **Veri Seti:** BIST 50 Hissesi | 49,287 Satır | 32 Sütun | 2019-2024

---

## Slayt 2: Problem Tanımı
* **Soru:** "Bir hisse senedinin 5 gün sonraki kapanış fiyatı bugünkünden yüksek mi olacak?"
* Hedef değişken: `Target_T5` (1 = Yükseliş, 0 = Düşüş)
* **Veri Kaynağı:** Yahoo Finance API — gerçek BIST verileri
* 50 hisse, 20+ teknik indikatör, geometrik formasyon tespiti (OBO, TOBO, Bayrak, Fincan-Kulp)

---

## Slayt 3: Veri Keşfi (Descriptive Statistics)
* **Boyut:** 49,287 satır × 32 sütun | **Missing Value:** 0
* **İstatistikler:** sum, mean, std, var, min, max tablosu
* **Target Dağılımı:** Yükseliş %55.8 vs Düşüş %44.2
* **Görseller:** Target_T5 pie chart, RSI_14 histogram
* Tüm veriler nümerik, kategorik yalnızca Pattern_Type ve Ticker

---

## Slayt 4: Korelasyon Analizi ve Değişken Eleme
* **Yöntem:** Pearson korelasyon matrisi → 0.90 üzeri eleme
* **Elenen:** 16 değişken (SMA_20, SMA_50, EMA_12, BB_Upper vb.)
* **Kalan:** 9 özellik (Volume, MACD, RSI_14, Stoch_K, Depth_Ratio vb.)
* **Neden:** Overfitting önleme, multicollinearity azaltma
* **Görsel:** Korelasyon Isı Haritası (Heatmap)

---

## Slayt 5: K-Means Kümeleme
* **Algoritma:** K-Means (k-means++, Öklid mesafesi)
* **Optimum K:** 5 (Elbow metodu)
* **Kümeleme Özellikleri:** ATR_14, Max_Drawdown, Max_Gain, RSI_14
* **Stratejik Çıkarım:**
  - Yüksek ATR kümesi → Bayrak/Kırılım stratejisi
  - Düşük RSI kümesi → TOBO/Çift Dip formasyonu
* **Görsel:** PCA 2D Scatter Plot

---

## Slayt 6: PCA (Temel Bileşenler Analizi)
* PC1: %31.1 | PC2: %21.1 | **Toplam:** %52.3 varyans
* 5 bileşen → %84, 7 bileşen → %95
* **Görsel:** Kümülatif Açıklanan Varyans Grafiği

---

## Slayt 7: K-NN + GridSearchCV
* **GridSearchCV:** k = {3, 5, 7, 9, 11, 15, 21}
* **En İyi k:** 21 (CV Accuracy: 0.5401)
* **Sonuç:** Accuracy %53.96 | F1: 0.6481
* 5-Fold Cross Validation ile doğrulandı

---

## Slayt 8: Random Forest
* **Parametreler:** n_estimators=100, criterion='gini'
* **Sonuç:** Accuracy %53.35 | F1: 0.6367
* **En Önemli 5 Özellik:** Volume, Open, RSI_14, Depth_Ratio, Stoch_K
* **Görsel:** Feature Importance Bar Grafiği

---

## Slayt 9: Yapay Sinir Ağı (ANN / MLP)
* **Mimari:** 64 → 32 → 16 nöron | ReLU | Adam | 100 Epoch
* **Sonuç:** Accuracy **%55.68** | F1: **0.6496**
* En yüksek performansı gösteren model

---

## Slayt 10: Model Karşılaştırma

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| K-NN (k=21) | %53.96 | 0.6481 |
| Random Forest | %53.35 | 0.6367 |
| **ANN (MLP)** | **%55.68** | **%0.6496** |

**Kazanan:** ANN (MLP) — Finans alanında %50-56 accuracy kabul edilebilir performanstır.

---

## Slayt 11: Finansal Backtest
* Kronolojik bölme (%80/%20) — Data Leakage koruması
* Başlangıç: 100,000 TL → AI stratejisi Buy & Hold'u geçti
* **Görsel:** Portföy Büyüme Grafiği (AI vs Buy & Hold)

---

## Slayt 12: Canlı Uygulama
* **GitHub:** Açık kaynak kod
* **Streamlit Cloud:** Canlı erişilebilir uygulama
* Interaktif model eğitimi, K-Means, Prophet tahmini, Backtest

---

## Slayt 13: Sonuç ve Öneriler
1. ANN modeli en yüksek performansı gösterdi
2. RSI_14, Volume, Stoch_K en önemli özellikler
3. 32 özellikten korelasyon eleme ile 9'a düşürüldü
4. K-Means kümeleme risk profillerini başarıyla ayırdı
* **Gelecek:** LSTM/Transformer, gerçek zamanlı veri, Sentiment Analysis
