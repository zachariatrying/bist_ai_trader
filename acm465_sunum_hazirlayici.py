import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from prophet import Prophet
except ImportError:
    print("Prophet kütüphanesi eksik, zaman serisi tahmini atlanacak (pip install prophet)")
    Prophet = None

# Cıktı klasörü
out_dir = r"C:\Users\ibrah\.gemini\antigravity\scratch\ipo_analyzer\sunum_materyalleri"
os.makedirs(out_dir, exist_ok=True)

# Veriyi Yükle
dataset_path = r"C:\Users\ibrah\.gemini\antigravity\scratch\bist_ai_dataset_real_30cols.csv"
print("Veriseti yükleniyor...")
df = pd.read_csv(dataset_path)

# ==========================================
# BÖLÜM 1: MODEL KARŞILAŞTIRMASI & FEATURE IMPORTANCE
# ==========================================
print("\n--- Bölüm 1: Model Karşılaştırması ve Özellik Önemi ---")
# Feature'ları ayarla (Gereksiz kolonları at)
features = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']
df_ml = df.dropna(subset=features + ['Target_T5']).copy()

X = df_ml[features]
y = df_ml['Target_T5']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)

# YSA (ANN)
ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
ann.fit(X_train_scaled, y_train)
ann_pred = ann.predict(X_test_scaled)
ann_acc = accuracy_score(y_test, ann_pred)

print(f"Random Forest Doğruluğu: %{rf_acc*100:.1f}")
print(f"Yapay Sinir Ağı Doğruluğu: %{ann_acc*100:.1f}")

# Feature Importance Çizimi (Random Forest)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="viridis")
plt.title("Random Forest: İndikatör/Özellik Önem Dereceleri (Feature Importance)")
plt.xlabel("Etki Oranı")
plt.ylabel("Özellik")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "01_Feature_Importance.png"), dpi=300)
plt.close()

# ==========================================
# BÖLÜM 2: HİSSE SEGMENTASYONU (K-MEANS) & PCA
# ==========================================
print("\n--- Bölüm 2: Hisse Segmentasyonu (K-Means) ---")
# Hisseleri grupla (Ortalama Volatilite(ATR), Risk(Drawdown), Getiri(Gain), RSI)
ticker_stats = df.groupby('Ticker')[['ATR_14', 'Max_Drawdown_15D', 'Max_Gain_15D', 'RSI_14']].mean().dropna()

scaler_km = StandardScaler()
scaled_tickers = scaler_km.fit_transform(ticker_stats)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_tickers)
ticker_stats['Cluster'] = clusters

# PCA ile 2 Boyuta indirgeme ve görselleştirme
pca = PCA(n_components=2)
pca_res = pca.fit_transform(scaled_tickers)
ticker_stats['PCA1'] = pca_res[:, 0]
ticker_stats['PCA2'] = pca_res[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=ticker_stats, palette='Set1', s=100, alpha=0.8)
plt.title("Hisse Senedi Kümeleri: PCA Görselleştirmesi (Cluster Distribution)")
plt.xlabel("PCA Bileşeni 1 (Risk/Volatilite Ekseni)")
plt.ylabel("PCA Bileşeni 2 (Momentum Ekseni)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "02_Cluster_PCA.png"), dpi=300)
plt.close()

# Küme Profillerini Yazdır
print("Küme Merkezleri Profili:")
print(ticker_stats.groupby('Cluster')[['ATR_14', 'Max_Drawdown_15D', 'Max_Gain_15D', 'RSI_14']].mean())

# ==========================================
# BÖLÜM 3: ZAMAN SERİSİ TAHMİNİ (PROPHET)
# ==========================================
print("\n--- Bölüm 3: Prophet ile Zaman Serisi Tahmini ---")
if Prophet is not None:
    # THYAO verisini alalım
    df_thy = df[df['Ticker'] == 'THYAO.IS'].copy()
    if not df_thy.empty:
        df_thy['Date'] = pd.to_datetime(df_thy['Date'])
        df_thy = df_thy.sort_values('Date')
        
        # Prophet Formatı
        df_prophet = df_thy[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        
        m = Prophet(daily_seasonality=False, yearly_seasonality=True)
        m.fit(df_prophet)
        
        future = m.make_future_dataframe(periods=60) # 60 gün ileri
        forecast = m.predict(future)
        
        fig1 = m.plot(forecast)
        plt.title("THYAO - Prophet ile Gelecek 60 Günlük Kapanış Fiyatı Tahmini")
        plt.xlabel("Tarih")
        plt.ylabel("Fiyat (TL)")
        plt.savefig(os.path.join(out_dir, "03_Prophet_Forecast.png"), dpi=300)
        plt.close(fig1)
        print("Prophet tahmini başarıyla tamamlandı.")
    else:
        print("THYAO verisi bulunamadı.")
else:
    print("Prophet adımı atlandı.")

print(f"\nİşlem tamam! Bütün grafikler {out_dir} klasörüne kaydedildi.")
