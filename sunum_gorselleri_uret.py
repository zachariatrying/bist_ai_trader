"""
ACM 465 Sunum Görselleri Üretici
Gerçek BIST veri setinden tüm slaytlar için PNG grafikler üretir.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings('ignore')

# Çıktı klasörü
OUT = "sunum_gorselleri"
os.makedirs(OUT, exist_ok=True)

# Stil
plt.style.use('dark_background')
COLORS = {'cyan': '#00f2ff', 'green': '#00ff88', 'red': '#ff4444', 'yellow': '#ffbf00', 'bg': '#0e1117'}

# Veri yükle
df = pd.read_csv("C:/Users/ibrah/.gemini/antigravity/scratch/bist_ai_dataset_real_30cols.csv")
print(f"Veri yüklendi: {df.shape}")

# ========== SLAYT 3: Descriptive Stats ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS['bg'])

# Target dağılımı
tc = df['Target_T5'].value_counts()
axes[0].pie(tc.values, labels=['Yükseliş (1)', 'Düşüş (0)'], colors=[COLORS['green'], COLORS['red']],
            autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 13}, startangle=90)
axes[0].set_title('Target_T5 Sınıf Dağılımı', color='white', fontsize=14, fontweight='bold')

# RSI histogram
axes[1].hist(df['RSI_14'].dropna(), bins=50, color=COLORS['cyan'], alpha=0.85, edgecolor='#1a1c23')
axes[1].set_title('RSI_14 Dağılımı', color='white', fontsize=14, fontweight='bold')
axes[1].set_xlabel('RSI_14', color='white')
axes[1].set_ylabel('Frekans', color='white')
axes[1].set_facecolor(COLORS['bg'])

plt.tight_layout()
plt.savefig(f"{OUT}/03_veri_kesfi.png", dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("[OK] 03_veri_kesfi.png")

# ========== SLAYT 4: Korelasyon Heatmap ==========
sensor = df.select_dtypes(include=[np.number]).drop(
    columns=['Target_T3','Target_T5','Target_T15','Max_Drawdown_15D','Max_Gain_15D'], errors='ignore')
corr = sensor.corr()

fig, ax = plt.subplots(figsize=(14, 11), facecolor=COLORS['bg'])
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, ax=ax, linewidths=0.5,
            cbar_kws={'shrink': 0.8}, fmt='.1f', square=True)
ax.set_title('Korelasyon Matrisi (Üst Üçgen Maskelenmiş)', color='white', fontsize=15, fontweight='bold')
ax.set_facecolor(COLORS['bg'])
plt.tight_layout()
plt.savefig(f"{OUT}/04_korelasyon_heatmap.png", dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("[OK] 04_korelasyon_heatmap.png")

# ========== Korelasyon Eleme ==========
cor_abs = sensor.corr().abs()
upper = cor_abs.where(np.triu(np.ones(cor_abs.shape), k=1).astype(bool))
drop_list = [col for col in upper.columns if any(upper[col] > 0.90)]
remaining = [c for c in sensor.columns if c not in drop_list]
print(f"Elenen: {len(drop_list)}, Kalan: {len(remaining)}")

# ========== FEATURE HAZIRLIK ==========
Y = df['Target_T5']
X = df.drop(columns=['Target','Target_T3','Target_T5','Target_T15','Date','Ticker','Pattern_Type',
                      'Max_Drawdown_15D','Max_Gain_15D'] + drop_list, axis=1, errors='ignore')
X = X.select_dtypes(include=[np.number]).dropna(axis=1)
mask = X.notna().all(axis=1) & Y.notna()
X, Y = X[mask], Y[mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# ========== SLAYT 5-6: K-Means & PCA ==========
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_scaled)
pca2 = PCA(n_components=2)
pca_res = pca2.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=COLORS['bg'])

# PCA Scatter
sc = axes[0].scatter(pca_res[:, 0], pca_res[:, 1], c=kmeans.labels_, cmap='turbo', alpha=0.4, s=5)
plt.colorbar(sc, ax=axes[0], label='Küme')
axes[0].set_title('K-Means Kümeleri (PCA 2D)', color='white', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)', color='white')
axes[0].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)', color='white')
axes[0].set_facecolor(COLORS['bg'])

# Kümülatif varyans
pca_full = PCA().fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
axes[1].plot(range(1, len(cumvar)+1), cumvar, 'o-', color=COLORS['cyan'], linewidth=2, markersize=6)
axes[1].axhline(y=0.95, color=COLORS['yellow'], linestyle='--', alpha=0.7, label='%95 eşik')
axes[1].set_title('PCA Kümülatif Açıklanan Varyans', color='white', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Bileşen Sayısı', color='white')
axes[1].set_ylabel('Kümülatif Varyans', color='white')
axes[1].legend()
axes[1].set_facecolor(COLORS['bg'])

plt.tight_layout()
plt.savefig(f"{OUT}/05_kmeans_pca.png", dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("[OK] 05_kmeans_pca.png")

# ========== SLAYT 7-9: Model Eğitimi ==========
# K-NN GridSearch
knn_gs = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3,5,7,9,11,15,21]}, cv=5, scoring='accuracy', n_jobs=-1)
knn_gs.fit(X_train, y_train)
best_k = knn_gs.best_params_['n_neighbors']
knn_pred = knn_gs.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# ANN
ann = MLPClassifier(hidden_layer_sizes=(64,32,16), activation='relu', solver='adam',
                    max_iter=100, batch_size=128, random_state=42, verbose=False)
ann.fit(X_train, y_train)
ann_pred = ann.predict(X_test)
ann_acc = accuracy_score(y_test, ann_pred)
ann_f1 = f1_score(y_test, ann_pred)

print(f"K-NN(k={best_k}): Acc={knn_acc:.4f} F1={knn_f1:.4f}")
print(f"RF: Acc={rf_acc:.4f} F1={rf_f1:.4f}")
print(f"ANN: Acc={ann_acc:.4f} F1={ann_f1:.4f}")

# ========== SLAYT 10: Model Karşılaştırma Bar Chart ==========
models = [f'K-NN\n(k={best_k})', 'Random\nForest', 'ANN\n(MLP)']
accs = [knn_acc, rf_acc, ann_acc]
f1s = [knn_f1, rf_f1, ann_f1]

fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
x = np.arange(len(models))
w = 0.35
bars1 = ax.bar(x - w/2, [a*100 for a in accs], w, label='Accuracy (%)', color=COLORS['cyan'], edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, [f*100 for f in f1s], w, label='F1-Score (%)', color=COLORS['green'], edgecolor='white', linewidth=0.5)
ax.set_ylabel('Skor (%)', color='white', fontsize=13)
ax.set_title('Model Performans Karşılaştırması', color='white', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, color='white', fontsize=12)
ax.legend(fontsize=11)
ax.set_facecolor(COLORS['bg'])
ax.set_ylim(0, 80)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'%{bar.get_height():.1f}',
            ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'%{bar.get_height():.1f}',
            ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT}/10_model_karsilastirma.png", dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("[OK] 10_model_karsilastirma.png")

# ========== SLAYT 8: Feature Importance ==========
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_imp)))
feat_imp.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white', linewidth=0.3)
ax.set_title('Random Forest — Feature Importance', color='white', fontsize=15, fontweight='bold')
ax.set_xlabel('Önem Derecesi', color='white', fontsize=12)
ax.set_facecolor(COLORS['bg'])
plt.tight_layout()
plt.savefig(f"{OUT}/08_feature_importance.png", dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("[OK] 08_feature_importance.png")

# ========== SLAYT 9: ANN Confusion Matrix ==========
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=COLORS['bg'])
for ax, pred, name in [(axes[0], knn_pred, f'K-NN (k={best_k})'),
                        (axes[1], rf_pred, 'Random Forest'),
                        (axes[2], ann_pred, 'ANN (MLP)')]:
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Düşüş', 'Yükseliş'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name, color='white', fontsize=13, fontweight='bold')
    ax.set_facecolor(COLORS['bg'])
plt.suptitle('Confusion Matrix Karşılaştırması', color='white', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/09_confusion_matrix.png", dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("[OK] 09_confusion_matrix.png")

# ========== SLAYT 11: Backtest Senaryosu ==========
df_bt = df.dropna(subset=['RSI_14','MACD','ATR_14','Stoch_K','Volume_Trend',
                           'Depth_Ratio','Neckline_Slope','Expert_Signal',
                           'Target_T5','Max_Gain_15D','Max_Drawdown_15D']).copy()
df_bt['Date'] = pd.to_datetime(df_bt['Date'])
df_bt = df_bt.sort_values('Date')

feats_bt = ['RSI_14','MACD','ATR_14','Stoch_K','Volume_Trend','Depth_Ratio','Neckline_Slope','Expert_Signal']
split = int(len(df_bt)*0.8)
train_bt, test_bt = df_bt.iloc[:split], df_bt.iloc[split:].copy()

sc_bt = StandardScaler()
rf_bt = RandomForestClassifier(n_estimators=50, random_state=42)
rf_bt.fit(sc_bt.fit_transform(train_bt[feats_bt]), train_bt['Target_T5'])
test_bt['AI'] = rf_bt.predict(sc_bt.transform(test_bt[feats_bt]))

daily_gain = test_bt[(test_bt['AI']==1)&(test_bt['Target_T5']==1)].groupby('Date')['Max_Gain_15D'].mean()*0.3
daily_loss = test_bt[(test_bt['AI']==1)&(test_bt['Target_T5']==0)].groupby('Date')['Max_Drawdown_15D'].mean()*0.5
dn = pd.DataFrame({'G': daily_gain, 'L': daily_loss}).fillna(0)
dn['Net'] = dn['G'] + dn['L']
dn['AI_Portfolio'] = 100000 * (1 + dn['Net']/100).cumprod()

mkt = test_bt.groupby('Date')['Max_Gain_15D'].mean()*0.1 - abs(test_bt.groupby('Date')['Max_Drawdown_15D'].mean()*0.1)
dn['BuyHold'] = 100000 * (1 + mkt/100).cumprod()

fig, ax = plt.subplots(figsize=(14, 6), facecolor=COLORS['bg'])
ax.plot(dn.index, dn['AI_Portfolio'], color=COLORS['green'], linewidth=2.5, label='AI Stratejisi (RF)')
ax.plot(dn.index, dn['BuyHold'], color=COLORS['red'], linewidth=2, linestyle='--', label='Buy & Hold')
ax.axhline(y=100000, color='white', linestyle=':', alpha=0.3)
ax.fill_between(dn.index, 100000, dn['AI_Portfolio'], where=dn['AI_Portfolio']>100000,
                color=COLORS['green'], alpha=0.1)

final_ai = dn['AI_Portfolio'].iloc[-1]
final_bh = dn['BuyHold'].iloc[-1]
ax.annotate(f'AI: {final_ai:,.0f} ₺\n(%{(final_ai/100000-1)*100:+.1f})',
            xy=(dn.index[-1], final_ai), fontsize=11, color=COLORS['green'], fontweight='bold',
            xytext=(-120, 20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color=COLORS['green']))

ax.set_title('Yapay Zeka Portföy Büyümesi — Out-of-Sample Backtest', color='white', fontsize=15, fontweight='bold')
ax.set_ylabel('Sermaye (₺)', color='white', fontsize=12)
ax.set_xlabel('Tarih', color='white', fontsize=12)
ax.legend(fontsize=12)
ax.set_facecolor(COLORS['bg'])
plt.tight_layout()
plt.savefig(f"{OUT}/11_backtest.png", dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("[OK] 11_backtest.png")

# ========== SLAYT 7: GridSearchCV Sonuçları ==========
cv_results = knn_gs.cv_results_
ks = [p['n_neighbors'] for p in cv_results['params']]
scores = cv_results['mean_test_score']

fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS['bg'])
ax.plot(ks, scores*100, 'o-', color=COLORS['cyan'], linewidth=2.5, markersize=10)
ax.axvline(x=best_k, color=COLORS['yellow'], linestyle='--', alpha=0.8, label=f'En İyi k={best_k}')
for k, s in zip(ks, scores):
    ax.annotate(f'%{s*100:.1f}', (k, s*100), textcoords='offset points',
                xytext=(0, 12), ha='center', color='white', fontsize=9)
ax.set_title('GridSearchCV — K-NN Hyperparameter Optimizasyonu', color='white', fontsize=14, fontweight='bold')
ax.set_xlabel('k (Komşu Sayısı)', color='white', fontsize=12)
ax.set_ylabel('CV Accuracy (%)', color='white', fontsize=12)
ax.legend(fontsize=11)
ax.set_facecolor(COLORS['bg'])
plt.tight_layout()
plt.savefig(f"{OUT}/07_gridsearch_knn.png", dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("[OK] 07_gridsearch_knn.png")

# ========== SENARYO: THYAO Örnek Tahmin ==========
thyao = df[df['Ticker']=='THYAO.IS'].copy()
thyao['Date'] = pd.to_datetime(thyao['Date'])
thyao = thyao.sort_values('Date').tail(60)

fig, ax = plt.subplots(figsize=(14, 6), facecolor=COLORS['bg'])
ax.plot(thyao['Date'], thyao['Close'], color=COLORS['cyan'], linewidth=2, label='THYAO Kapanış')

# Tahmin noktaları
thyao_feats = thyao[feats_bt].dropna()
if len(thyao_feats) > 0:
    thyao_pred = rf_bt.predict(sc_bt.transform(thyao_feats))
    buy_idx = thyao_feats.index[thyao_pred == 1]
    sell_idx = thyao_feats.index[thyao_pred == 0]
    ax.scatter(thyao.loc[buy_idx, 'Date'], thyao.loc[buy_idx, 'Close'],
               color=COLORS['green'], s=60, zorder=5, label='AI: AL sinyali', marker='^')
    ax.scatter(thyao.loc[sell_idx, 'Date'], thyao.loc[sell_idx, 'Close'],
               color=COLORS['red'], s=40, zorder=5, label='AI: BEKLE', marker='v', alpha=0.5)

ax.set_title('THYAO — Son 60 Gün AI Sinyal Senaryosu', color='white', fontsize=15, fontweight='bold')
ax.set_ylabel('Fiyat (₺)', color='white', fontsize=12)
ax.legend(fontsize=11)
ax.set_facecolor(COLORS['bg'])
plt.tight_layout()
plt.savefig(f"{OUT}/senaryo_thyao.png", dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("[OK] senaryo_thyao.png")

# ========== SENARYO: Küme Profilleri ==========
ticker_stats = df.groupby('Ticker')[['ATR_14','Max_Drawdown_15D','Max_Gain_15D','RSI_14']].mean().dropna()
sc_km = StandardScaler()
km_data = sc_km.fit_transform(ticker_stats)
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(km_data)
ticker_stats['Küme'] = km.labels_

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS['bg'])

for i, (col, title) in enumerate([('ATR_14', 'Volatilite (ATR_14)'), ('Max_Gain_15D', 'Potansiyel Kazanç')]):
    for k in sorted(ticker_stats['Küme'].unique()):
        subset = ticker_stats[ticker_stats['Küme']==k]
        axes[i].barh([f"K{k}: {t}" for t in subset.index[:3]], subset[col].values[:3], alpha=0.8)
    axes[i].set_title(f'Küme Bazlı {title}', color='white', fontsize=13, fontweight='bold')
    axes[i].set_facecolor(COLORS['bg'])

plt.tight_layout()
plt.savefig(f"{OUT}/senaryo_kume_profil.png", dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("[OK] senaryo_kume_profil.png")

print(f"\n[DONE] Toplam {len(os.listdir(OUT))} görsel üretildi: {OUT}/")

