# ==============================================================================
# ACM 465 - VERİ MADENCİLİĞİ VE YAPAY ZEKA PROJESİ
# BIST Hisse Senedi Fiyat Tahmini (Yükseliş / Düşüş Sınıflandırması)
# Veri Seti: bist_ai_dataset_real_30cols.csv (50 BIST hissesi, ~49K satır)
# Soru: "Bir hisse senedinin 5 gün sonraki kapanış fiyatı bugünkünden
#        yüksek mi olacak (1) yoksa düşük/aynı mı kalacak (0)?"
# ==============================================================================

# ENVIRONMENT SET UP
# conda activate tupras_env  (veya kendi ortamınız)
# Gerekli paketler: pandas, numpy, seaborn, matplotlib, scikit-learn, yellowbrick

# ==============================================================================
# 1. BÖLÜM: VERİ KEŞFİ (DATA ANALYSIS / DESCRIPTIVE STATISTICS)
# ==============================================================================

import numpy as np
import pandas as pd

# --- Veri Okuma ---
# NOT: Dosya yolunu kendi bilgisayarınıza göre güncelleyin
df_full = pd.read_csv("C:/Users/ibrah/.gemini/antigravity/scratch/bist_ai_dataset_real_30cols.csv")

print(f"Orijinal veri boyutu: {df_full.shape}")
# Büyük veri setinde hız için %10 alt küme alıyoruz
df = df_full.sample(frac=0.1, random_state=42).reset_index(drop=True)
print(f"Alt küme (subset) boyutu: {df.shape}")

# İlk 5 satır
print("\n--- İlk 5 Satır ---")
print(df.head())

# ==============================================================================
# 1.1 Descriptive Statistics (Tanımlayıcı İstatistikler)
# ==============================================================================

# Sayısal değişkenleri belirliyorum
num_var = [col for col in df.columns if df[col].dtype != 'O']
print(f"\nSayısal değişken sayısı: {len(num_var)}")
print(f"Sayısal değişkenler: {num_var}")

# Temel istatistik fonksiyonları
desc_agg = ['sum', 'mean', 'std', 'var', 'min', 'max']

# Fonksiyonları sayısal değerlere uyguluyorum
desc_agg_dict = {col: desc_agg for col in num_var}
desc_summ = df[num_var].agg(desc_agg_dict)
print("\n--- Descriptive Statistics (sum, mean, std, var, min, max) ---")
print(desc_summ)

# numpy array'e dönüştürme
df_desc_na = np.array(desc_summ)
df_na = np.array(df)
print(f"\nDescriptive stats numpy shape: {df_desc_na.shape}")
print(f"Dataframe numpy shape: {df_na.shape}")

# ==============================================================================
# 1.2 Genel Bakış (Overview)
# ==============================================================================
import seaborn as sns

print(f"\nVeri boyutu: {df.shape}")
print("\n--- Veri Bilgisi (Info) ---")
df.info()

print(f"\nSütunlar: {df.columns.tolist()}")

# Missing value kontrolü
print(f"\nMissing value var mı? {df.isnull().values.any()}")

# Her değişken için descriptive analytics (v2 - transpose)
desc_summv2 = df.describe().T
print("\n--- Describe Transpose ---")
print(desc_summv2)

# ==============================================================================
# 1.3 Hedef Değişken (Target) Tanımlama ve İnceleme
# ==============================================================================

# Veri setimizde Target_T5 zaten mevcut (5 gün sonra fiyat arttıysa 1, azaldıysa 0)
# Bunu 'Target' olarak yeniden adlandırıyoruz
df['Target'] = df['Target_T5']
print("\n--- Target Değişkeni Dağılımı ---")
print(df['Target'].value_counts())

Y = df['Target']

# Ortalamanın üstündeki ve altındaki gözlem sayıları
above_mean = df[df.Target > df.Target.mean()].Target.count()
below_mean = df[df.Target < df.Target.mean()].Target.count()
print(f"\nOrtalamanın üstündeki gözlem sayısı: {above_mean}")
print(f"Ortalamanın altındaki gözlem sayısı: {below_mean}")

# ==============================================================================
# 1.4 Görselleştirme (Visualization)
# ==============================================================================
import matplotlib
matplotlib.use('Agg')  # GUI olmadan grafik kaydetmek için
from matplotlib import pyplot as plt

# Sensör/indikatör verileri (sayısal teknik analiz sütunları)
sensor_cols = ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
               'MACD', 'MACD_Signal', 'RSI_14', 'BB_Upper', 'BB_Middle',
               'BB_Lower', 'ATR_14', 'Stoch_K', 'Stoch_D',
               'Support_Level', 'Resistance_Level', 'Volume_Trend',
               'Depth_Ratio', 'Neckline_Slope']
sensor = df[sensor_cols].copy()

print(f"\nSensör/indikatör sütunları: {sensor.columns.tolist()}")

# Boxplot örneği
sns.boxplot(x=sensor['RSI_14'])
plt.title('RSI_14 Boxplot')
plt.savefig("C:/Users/ibrah/.gemini/antigravity/scratch/boxplot_rsi14.png")
plt.close()
print("\nBoxplot kaydedildi: boxplot_rsi14.png")


def num_summary(data, numerical_col, plot=False):
    """Sayısal değişken için quantile bazlı özet ve histogram."""
    quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(data[numerical_col].describe(quantiles).T)
    if plot:
        data[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.savefig(f"C:/Users/ibrah/.gemini/antigravity/scratch/hist_{numerical_col.replace('/', '_')}.png")
        plt.close()


# Örnek: RSI_14 için özet ve histogram
num_summary(sensor, 'RSI_14', plot=True)

# Tüm değişkenler için grafik (yorumdan çıkararak çalıştırabilirsiniz)
# for col in sensor.columns:
#     num_summary(sensor, col, plot=True)

# ==============================================================================
# 1.5 Target ile Bağımsız Değişken Analizi
# ==============================================================================

print("\n--- Target'a göre RSI_14 Ortalaması ---")
print(df.groupby('Target')['RSI_14'].mean())


def target_summary_with_num(dataframe, target, num_col):
    """Hedef değişkene göre sayısal değişkenin ortalaması."""
    print(dataframe.groupby(target).agg({num_col: 'mean'}), end='\n\n')


target_summary_with_num(df, 'Target', 'RSI_14')

# Tüm sensörler için (yorumdan çıkararak çalıştırabilirsiniz)
# for col in sensor.columns:
#     target_summary_with_num(df, 'Target', col)

# ==============================================================================
# 1.6 Korelasyon Analizi ve Yüksek Korelasyonlu Değişkenlerin Elenmesi
# ==============================================================================

# Sadece sayısal sütunlarla korelasyon
sensor_for_corr = df.drop(columns=['Target', 'Timestamp', 'Attack Type',
                                    'Date', 'Ticker', 'Pattern_Type',
                                    'Target_T3', 'Target_T5', 'Target_T15',
                                    'Max_Drawdown_15D', 'Max_Gain_15D'],
                           errors='ignore')
# Sadece numeric olanları al
sensor_for_corr = sensor_for_corr.select_dtypes(include=[np.number])

corr = sensor_for_corr.corr()

# Korelasyon ısı haritası
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap='RdBu')
plt.title('Korelasyon Isı Haritası')
plt.tight_layout()
plt.savefig("C:/Users/ibrah/.gemini/antigravity/scratch/heatmap_corr.png")
plt.close()
print("\nKorelasyon heatmap kaydedildi: heatmap_corr.png")

# Yüksek korelasyonlu değişkenleri eleme (>0.90)
cor_matrix = sensor_for_corr.corr().abs()
upper_triangle_matrix = cor_matrix.where(
    np.triu(np.ones(cor_matrix.shape), k=1).astype(bool)
)
drop_list = [col for col in upper_triangle_matrix.columns
             if any(upper_triangle_matrix[col] > 0.90)]

print(f"\n0.90 üzeri korelasyonlu (elenecek) değişkenler ({len(drop_list)}):")
print(drop_list)

# Yüksek korelasyonlu sütunları çıkar
df_clean = df.drop(drop_list, axis=1, errors='ignore')
sensor_clean = sensor.drop([c for c in drop_list if c in sensor.columns],
                           axis=1, errors='ignore')

print(f"\nEleme sonrası sensor boyutu: {sensor_clean.shape}")
print(f"Kalan sensör sütunları: {sensor_clean.columns.tolist()}")

# ==============================================================================
# 2. BÖLÜM: MODEL GELİŞTİRME (MODEL DEVELOPMENT)
# ==============================================================================
# Roadmap:
# 1. Modeling (K-Means Kümeleme + K-NN Sınıflandırma)
# 2. Prediction
# 3. Evaluation
# 4. Hyperparameter Optimization (GridSearchCV)
# 5. Finalization

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- Bağımlı ve Bağımsız Değişkenlerin Ayrılması ---
Y = df_clean['Target']
X = df_clean.drop(columns=['Target', 'Target_T3', 'Target_T5', 'Target_T15',
                            'Date', 'Ticker', 'Pattern_Type',
                            'Max_Drawdown_15D', 'Max_Gain_15D'],
                   axis=1, errors='ignore')
# Sadece numeric sütunları al
X = X.select_dtypes(include=[np.number])

print(f"\nX (özellikler) boyutu: {X.shape}")
print(f"Y (hedef) boyutu: {Y.shape}")

# --- Standardizasyon ---
X_scaled = StandardScaler().fit_transform(X)
X_scaled_v1 = pd.DataFrame(X_scaled, columns=X.columns)
print("Özellikler standardize edildi.")

# ==============================================================================
# 2.1 Kümeleme (K-Means Clustering)
# ==============================================================================
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

try:
    from yellowbrick.cluster import KElbowVisualizer

    # Optimum küme sayısını Elbow metodu ile belirliyorum
    kmeans_elbow = KMeans(random_state=42, n_init=10)
    elbow = KElbowVisualizer(kmeans_elbow, k=(2, 15))

    print("\nElbow metodu için model fit ediliyor...")
    elbow.fit(X_scaled_v1)
    elbow.show(outpath="C:/Users/ibrah/.gemini/antigravity/scratch/elbow_grafik.png")
    optimal_k = elbow.elbow_value_
    print(f"Optimum küme sayısı (Elbow): {optimal_k}")
except ImportError:
    print("\nyellowbrick paketi bulunamadı, küme sayısı 5 olarak belirlendi.")
    optimal_k = 5

# K-Means ile kümeleme
if optimal_k is None:
    optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=17, n_init=10).fit(X_scaled_v1)
print(f"\nK-Means parametreleri: {kmeans.get_params()}")
print(f"Küme sayısı: {kmeans.n_clusters}")
print(f"Inertia: {kmeans.inertia_}")

clusters = kmeans.labels_
sensor_clean_copy = sensor_clean.copy()
sensor_clean_copy['cluster'] = clusters + 1  # 1'den başla
print(f"\nKüme dağılımı:")
print(sensor_clean_copy['cluster'].value_counts().sort_index())

# Küme istatistikleri
print("\n--- Küme Bazlı İstatistikler ---")
print(sensor_clean_copy.groupby('cluster').agg(['count', 'mean', 'median']).head())

# Küme sonuçlarını CSV'ye kaydet
sensor_clean_copy.to_csv(
    'C:/Users/ibrah/.gemini/antigravity/scratch/cluster_results.csv', index=False)
print("Küme sonuçları kaydedildi: cluster_results.csv")

# ==============================================================================
# 2.2 PCA (Temel Bileşenler Analizi) ile Kümelerin Görselleştirilmesi
# ==============================================================================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled_v1)
print(f"\nPCA açıklanan varyans oranları: {pca.explained_variance_ratio_}")
print(f"Toplam açıklanan varyans: {np.sum(pca.explained_variance_ratio_):.4f}")

# PCA ile kümeleri 2D scatter plot
plt.figure(figsize=(10, 7))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                      c=clusters, cmap='viridis', alpha=0.5, s=10)
plt.colorbar(scatter, label='Küme')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('K-Means Kümeleri - PCA 2D Görselleştirme')
plt.savefig("C:/Users/ibrah/.gemini/antigravity/scratch/pca_clusters.png", dpi=150)
plt.close()
print("PCA küme grafiği kaydedildi: pca_clusters.png")

# Tam PCA varyans analizi
pca_full = PCA()
pca_full.fit(X_scaled_v1)
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
print(f"\nKümülatif açıklanan varyans (ilk 10 bileşen): {cumsum_var[:10]}")

# ==============================================================================
# 2.3 K-NN Sınıflandırma Modeli
# ==============================================================================

print("\n" + "="*60)
print("K-NN SINIFLANDIRMA MODELİ")
print("="*60)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_v1, Y, test_size=0.20, random_state=42
)
print(f"Train boyutu: {X_train.shape}, Test boyutu: {X_test.shape}")

# İlk model (k=5)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("\n--- K-NN (k=5) Sonuçları ---")
print(classification_report(y_test, y_pred_knn, target_names=['Düşüş(0)', 'Yükseliş(1)']))

# Cross Validation
cv_results = cross_validate(knn_model, X_scaled_v1, Y, cv=5,
                            scoring=['accuracy', 'f1', 'roc_auc'])
print(f"CV Accuracy: {cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std():.4f}")
print(f"CV F1: {cv_results['test_f1'].mean():.4f} +/- {cv_results['test_f1'].std():.4f}")
print(f"CV ROC AUC: {cv_results['test_roc_auc'].mean():.4f} +/- {cv_results['test_roc_auc'].std():.4f}")

# ==============================================================================
# 2.4 Hyperparameter Optimization (GridSearchCV)
# ==============================================================================

print("\n--- GridSearchCV ile Hyperparameter Optimizasyonu ---")
knn_params = {'n_neighbors': [3, 5, 7, 9, 11, 15, 21]}

knn_gs = GridSearchCV(KNeighborsClassifier(),
                      knn_params, cv=5, scoring='accuracy', n_jobs=-1)
knn_gs.fit(X_train, y_train)

best_k = knn_gs.best_params_['n_neighbors']
print(f"En iyi k değeri: {best_k}")
print(f"En iyi CV Accuracy: {knn_gs.best_score_:.4f}")

# En iyi model ile final tahmin
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred_final = knn_final.predict(X_test)

print(f"\n--- K-NN (k={best_k}) Final Sonuçları ---")
print(classification_report(y_test, y_pred_final,
                            target_names=['Düşüş(0)', 'Yükseliş(1)']))

try:
    roc_score = roc_auc_score(y_test, knn_final.predict_proba(X_test)[:, 1])
    print(f"ROC AUC Score: {roc_score:.4f}")
except:
    print("ROC AUC hesaplanamadı.")

# ==============================================================================
# 2.5 Random Forest Modeli (Karşılaştırma için)
# ==============================================================================
from sklearn.ensemble import RandomForestClassifier

print("\n" + "="*60)
print("RANDOM FOREST SINIFLANDIRMA MODELİ")
print("="*60)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Sonuçları ---")
print(classification_report(y_test, y_pred_rf,
                            target_names=['Düşüş(0)', 'Yükseliş(1)']))

# Feature Importance (En önemli özellikler)
feat_imp = pd.Series(rf_model.feature_importances_,
                     index=X.columns).sort_values(ascending=False)
print("\n--- En Önemli 10 Özellik (Feature Importance) ---")
print(feat_imp.head(10))

# Feature importance grafiği
plt.figure(figsize=(10, 6))
feat_imp.head(15).plot(kind='barh')
plt.title('Random Forest - Feature Importance (Top 15)')
plt.xlabel('Önem Derecesi')
plt.tight_layout()
plt.savefig("C:/Users/ibrah/.gemini/antigravity/scratch/feature_importance.png", dpi=150)
plt.close()
print("Feature importance grafiği kaydedildi: feature_importance.png")

# ==============================================================================
# 2.6 Yapay Sinir Ağı (ANN - MLP) Modeli
# ==============================================================================
from sklearn.neural_network import MLPClassifier

print("\n" + "="*60)
print("YAPAY SİNİR AĞI (ANN - MLP) MODELİ")
print("="*60)

ann_model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    max_iter=100,
    batch_size=128,
    random_state=42,
    verbose=False
)
ann_model.fit(X_train, y_train)
y_pred_ann = ann_model.predict(X_test)

print("\n--- ANN (MLP) Sonuçları ---")
print(classification_report(y_test, y_pred_ann,
                            target_names=['Düşüş(0)', 'Yükseliş(1)']))

# ==============================================================================
# 3. BÖLÜM: MODEL KARŞILAŞTIRMA RAPORU
# ==============================================================================
from sklearn.metrics import accuracy_score, f1_score

print("\n" + "="*70)
print("MODEL KARŞILAŞTIRMA RAPORU")
print("="*70)

models = {
    f'K-NN (k={best_k})': y_pred_final,
    'Random Forest': y_pred_rf,
    'ANN (MLP)': y_pred_ann
}

header = f"{'Model':<25} | {'Accuracy':<12} | {'F1-Score':<12}"
print(header)
print("-" * len(header))

for name, preds in models.items():
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"{name:<25} | {acc:<12.4f} | {f1:<12.4f}")

print("="*70)
print("\nPROJE TAMAMLANDI!")
print("Sonuç: BIST hisse senedi 5 günlük fiyat yönü tahmini için")
print("3 farklı yapay zeka algoritması karşılaştırıldı.")
print("En iyi performansı gösteren model seçilerek finalize edildi.")
