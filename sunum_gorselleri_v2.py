"""
ACM 465 Sunum Gorselleri v2 - XGBoost + Portfolio Simulasyonu + Ek Grafikler
"""
import pandas as pd, numpy as np, os, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

OUT = "sunum_gorselleri"
os.makedirs(OUT, exist_ok=True)
plt.style.use('dark_background')
C = {'cyan':'#00f2ff','green':'#00ff88','red':'#ff4444','yellow':'#ffbf00','bg':'#0e1117','purple':'#a855f7'}

df = pd.read_csv("C:/Users/ibrah/.gemini/antigravity/scratch/bist_ai_dataset_real_30cols.csv")
print(f"Veri: {df.shape}")

# --- KORELASYON ELEME ---
sensor = df.select_dtypes(include=[np.number]).drop(columns=['Target_T3','Target_T5','Target_T15','Max_Drawdown_15D','Max_Gain_15D'], errors='ignore')
cor_abs = sensor.corr().abs()
upper = cor_abs.where(np.triu(np.ones(cor_abs.shape), k=1).astype(bool))
drop_list = [c for c in upper.columns if any(upper[c] > 0.90)]

Y = df['Target_T5']
X = df.drop(columns=['Target','Target_T3','Target_T5','Target_T15','Date','Ticker','Pattern_Type','Max_Drawdown_15D','Max_Gain_15D']+drop_list, axis=1, errors='ignore')
X = X.select_dtypes(include=[np.number]).dropna(axis=1)
m = X.notna().all(axis=1) & Y.notna()
X, Y = X[m], Y[m]

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(Xs, Y, test_size=0.2, random_state=42)

# ===== 4 MODEL EGITIMI =====
knn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors':[3,5,7,9,11,15,21]}, cv=5, scoring='accuracy', n_jobs=-1)
knn.fit(Xtr, ytr); knn_p = knn.predict(Xte)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr); rf_p = rf.predict(Xte)

ann = MLPClassifier(hidden_layer_sizes=(64,32,16), activation='relu', solver='adam', max_iter=100, batch_size=128, random_state=42, verbose=False)
ann.fit(Xtr, ytr); ann_p = ann.predict(Xte)

xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', verbosity=0)
xgb.fit(Xtr, ytr); xgb_p = xgb.predict(Xte)

results = {
    f'K-NN (k={knn.best_params_["n_neighbors"]})': (knn_p, knn),
    'Random Forest': (rf_p, rf),
    'ANN (MLP)': (ann_p, ann),
    'XGBoost': (xgb_p, xgb)
}
for name,(pred,_) in results.items():
    print(f"{name}: Acc={accuracy_score(yte,pred):.4f} F1={f1_score(yte,pred):.4f}")

# ===== GORSEL 1: 4 MODEL BAR CHART =====
fig, ax = plt.subplots(figsize=(12,6), facecolor=C['bg'])
names = list(results.keys())
accs = [accuracy_score(yte,r[0]) for r in results.values()]
f1s = [f1_score(yte,r[0]) for r in results.values()]
x = np.arange(len(names)); w=0.35
b1 = ax.bar(x-w/2, [a*100 for a in accs], w, label='Accuracy (%)', color=C['cyan'], edgecolor='white', lw=0.5)
b2 = ax.bar(x+w/2, [f*100 for f in f1s], w, label='F1-Score (%)', color=C['green'], edgecolor='white', lw=0.5)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'%{b.get_height():.1f}', ha='center', color='white', fontsize=9, fontweight='bold')
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'%{b.get_height():.1f}', ha='center', color='white', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_title('4 Model Performans Karsilastirmasi', fontsize=16, fontweight='bold')
ax.set_ylabel('Skor (%)'); ax.legend(fontsize=11); ax.set_facecolor(C['bg']); ax.set_ylim(0,85)
plt.tight_layout(); plt.savefig(f"{OUT}/model_4_karsilastirma.png", dpi=200, facecolor=C['bg']); plt.close()
print("[OK] model_4_karsilastirma.png")

# ===== GORSEL 2: ROC CURVE - 4 MODEL =====
fig, ax = plt.subplots(figsize=(10,8), facecolor=C['bg'])
colors_roc = [C['cyan'], C['green'], C['yellow'], C['purple']]
for i,(name,(_,model)) in enumerate(results.items()):
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(Xte)[:,1]
    else:
        prob = model.predict_proba(Xte)[:,1]
    fpr, tpr, _ = roc_curve(yte, prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors_roc[i], lw=2.5, label=f'{name} (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1],'--', color='gray', alpha=0.5)
ax.set_title('ROC Curve - 4 Model Karsilastirmasi', fontsize=15, fontweight='bold')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=11); ax.set_facecolor(C['bg'])
plt.tight_layout(); plt.savefig(f"{OUT}/roc_curve_4model.png", dpi=200, facecolor=C['bg']); plt.close()
print("[OK] roc_curve_4model.png")

# ===== GORSEL 3: CONFUSION MATRIX 4 MODEL =====
fig, axes = plt.subplots(1, 4, figsize=(22,5), facecolor=C['bg'])
for ax,(name,(pred,_)) in zip(axes, results.items()):
    cm = confusion_matrix(yte, pred)
    ConfusionMatrixDisplay(cm, display_labels=['Dusus','Yukselis']).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name, fontsize=12, fontweight='bold'); ax.set_facecolor(C['bg'])
plt.suptitle('Confusion Matrix - 4 Model', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_4model.png", dpi=200, facecolor=C['bg'], bbox_inches='tight'); plt.close()
print("[OK] confusion_4model.png")

# ===== GORSEL 4: XGBOOST FEATURE IMPORTANCE =====
xgb_imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10,6), facecolor=C['bg'])
xgb_imp.plot(kind='barh', ax=ax, color=plt.cm.plasma(np.linspace(0.3,0.9,len(xgb_imp))), edgecolor='white', lw=0.3)
ax.set_title('XGBoost - Feature Importance', fontsize=15, fontweight='bold')
ax.set_xlabel('Onem Derecesi'); ax.set_facecolor(C['bg'])
plt.tight_layout(); plt.savefig(f"{OUT}/xgboost_feature_importance.png", dpi=200, facecolor=C['bg']); plt.close()
print("[OK] xgboost_feature_importance.png")

# ===== GORSEL 5: DETAYLI PORTFOLIO SIMULASYONU =====
feats_bt = ['RSI_14','MACD','ATR_14','Stoch_K','Volume_Trend','Depth_Ratio','Neckline_Slope','Expert_Signal']
df_bt = df.dropna(subset=feats_bt+['Target_T5','Max_Gain_15D','Max_Drawdown_15D']).copy()
df_bt['Date'] = pd.to_datetime(df_bt['Date'])
df_bt = df_bt.sort_values('Date')

split = int(len(df_bt)*0.8)
train_bt, test_bt = df_bt.iloc[:split], df_bt.iloc[split:].copy()

sc_bt = StandardScaler()
xgb_bt = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', verbosity=0)
xgb_bt.fit(sc_bt.fit_transform(train_bt[feats_bt]), train_bt['Target_T5'])
test_bt['AI'] = xgb_bt.predict(sc_bt.transform(test_bt[feats_bt]))

# Gunluk portfolio hesapla
BASLANGIC = 100000
portfolio = BASLANGIC
buyhold = BASLANGIC
portfolio_hist = []
buyhold_hist = []
dates = []
islem_sayisi = 0
dogru_islem = 0

for date, group in test_bt.groupby('Date'):
    dates.append(date)
    ai_buys = group[group['AI']==1]
    if len(ai_buys) > 0:
        islem_sayisi += len(ai_buys)
        correct = ai_buys[ai_buys['Target_T5']==1]
        wrong = ai_buys[ai_buys['Target_T5']==0]
        dogru_islem += len(correct)
        gain_pct = correct['Max_Gain_15D'].mean() * 0.3 if len(correct)>0 else 0
        loss_pct = wrong['Max_Drawdown_15D'].mean() * 0.5 if len(wrong)>0 else 0
        net = (gain_pct + loss_pct) / 100
        portfolio *= (1 + net)
    portfolio_hist.append(portfolio)
    mkt_ret = (group['Max_Gain_15D'].mean()*0.1 + group['Max_Drawdown_15D'].mean()*0.1) / 100
    buyhold *= (1 + mkt_ret)
    buyhold_hist.append(buyhold)

win_rate = (dogru_islem/islem_sayisi*100) if islem_sayisi>0 else 0
net_return = (portfolio - BASLANGIC) / BASLANGIC * 100

print(f"\nPORTFOLYO: {BASLANGIC:,} -> {portfolio:,.0f} TL (%{net_return:+.1f})")
print(f"Islem: {islem_sayisi}, Dogru: {dogru_islem}, Win Rate: %{win_rate:.1f}")

# Ana portfolio grafigi
fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=C['bg'])

# 1. Portfolio buyume
ax = axes[0,0]
ax.plot(dates, portfolio_hist, color=C['green'], lw=2.5, label='AI Stratejisi (XGBoost)')
ax.plot(dates, buyhold_hist, color=C['red'], lw=2, ls='--', label='Buy & Hold')
ax.axhline(y=BASLANGIC, color='white', ls=':', alpha=0.3)
ax.fill_between(dates, BASLANGIC, portfolio_hist, where=[p>BASLANGIC for p in portfolio_hist], color=C['green'], alpha=0.1)
ax.fill_between(dates, BASLANGIC, portfolio_hist, where=[p<BASLANGIC for p in portfolio_hist], color=C['red'], alpha=0.1)
ax.set_title(f'100,000 TL Yatirim Simulasyonu\nFinal: {portfolio:,.0f} TL (%{net_return:+.1f})', fontsize=14, fontweight='bold')
ax.set_ylabel('Sermaye (TL)'); ax.legend(fontsize=10); ax.set_facecolor(C['bg'])

# 2. Drawdown
port_arr = np.array(portfolio_hist)
peak = np.maximum.accumulate(port_arr)
drawdown = (port_arr - peak) / peak * 100
ax = axes[0,1]
ax.fill_between(dates, drawdown, 0, color=C['red'], alpha=0.6)
ax.set_title(f'Maksimum Drawdown: %{drawdown.min():.1f}', fontsize=14, fontweight='bold')
ax.set_ylabel('Drawdown (%)'); ax.set_facecolor(C['bg'])

# 3. Aylik getiri
port_df = pd.DataFrame({'Date': dates, 'Value': portfolio_hist})
port_df['Date'] = pd.to_datetime(port_df['Date'])
port_df['Month'] = port_df['Date'].dt.to_period('M')
monthly = port_df.groupby('Month')['Value'].last().pct_change() * 100
ax = axes[1,0]
colors_m = [C['green'] if v>=0 else C['red'] for v in monthly.values]
ax.bar(range(len(monthly)), monthly.values, color=colors_m, alpha=0.8)
ax.axhline(y=0, color='white', ls='-', alpha=0.3)
ax.set_title('Aylik Getiri (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Ay'); ax.set_ylabel('Getiri (%)'); ax.set_facecolor(C['bg'])

# 4. Win rate & metriks
ax = axes[1,1]
ax.axis('off')
metrics_text = f"""
PORTFOLIO OZETI
{'='*35}
Baslangic:     100,000 TL
Bitis:         {portfolio:>12,.0f} TL
Net Getiri:    %{net_return:>+10.1f}
{'='*35}
Toplam Islem:  {islem_sayisi:>12,}
Dogru Islem:   {dogru_islem:>12,}
Win Rate:      %{win_rate:>10.1f}
Max Drawdown:  %{drawdown.min():>10.1f}
{'='*35}
Model: XGBoost (200 trees, depth=6)
Test: Kronolojik %20 Out-of-Sample
"""
ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes, fontsize=13,
        verticalalignment='top', fontfamily='monospace', color=C['cyan'],
        bbox=dict(boxstyle='round', facecolor='#1a1c23', edgecolor=C['cyan']))

plt.tight_layout()
plt.savefig(f"{OUT}/portfolio_detay.png", dpi=200, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("[OK] portfolio_detay.png")

# ===== GORSEL 6: HISSE BAZLI SENARYO (TOP 5) =====
fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor=C['bg'])
top_tickers = ['THYAO.IS', 'SISE.IS', 'EREGL.IS', 'TUPRS.IS', 'ASELS.IS', 'GARAN.IS']

for i, ticker in enumerate(top_tickers):
    ax = axes[i//3, i%3]
    t_df = df[df['Ticker']==ticker].copy()
    t_df['Date'] = pd.to_datetime(t_df['Date'])
    t_df = t_df.sort_values('Date').tail(90)
    ax.plot(t_df['Date'], t_df['Close'], color=C['cyan'], lw=1.5)
    
    t_feats = t_df[feats_bt].dropna()
    if len(t_feats) > 0:
        t_pred = xgb_bt.predict(sc_bt.transform(t_feats))
        buy_i = t_feats.index[t_pred==1]
        sell_i = t_feats.index[t_pred==0]
        ax.scatter(t_df.loc[buy_i,'Date'], t_df.loc[buy_i,'Close'], c=C['green'], s=40, marker='^', zorder=5)
        ax.scatter(t_df.loc[sell_i,'Date'], t_df.loc[sell_i,'Close'], c=C['red'], s=25, marker='v', alpha=0.5, zorder=5)
    
    ax.set_title(ticker.replace('.IS',''), fontsize=13, fontweight='bold')
    ax.set_facecolor(C['bg'])
    ax.tick_params(axis='x', rotation=30)

plt.suptitle('6 Hisse - Son 90 Gun AI Sinyal Senaryosu (Yesil=AL, Kirmizi=BEKLE)', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/senaryo_6hisse.png", dpi=200, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("[OK] senaryo_6hisse.png")

# ===== GORSEL 7: VERI SETI GENEL BAKIS =====
fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=C['bg'])

# Boxplots
for i, col in enumerate(['RSI_14','MACD','Stoch_K','ATR_14','Volume_Trend','Depth_Ratio']):
    ax = axes[i//3, i%3]
    data = df[col].dropna()
    bp = ax.boxplot(data, patch_artist=True, vert=True)
    bp['boxes'][0].set_facecolor(C['cyan'])
    bp['medians'][0].set_color(C['yellow'])
    ax.set_title(col, fontsize=13, fontweight='bold')
    ax.set_facecolor(C['bg'])

plt.suptitle('Temel Indikatorlerin Box Plot Dagilimlari', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/boxplot_indikatorler.png", dpi=200, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print("[OK] boxplot_indikatorler.png")

# ===== GORSEL 8: LEARNING CURVE SIMULE =====
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric='logloss', verbosity=0),
    Xs[:5000], Y.iloc[:5000], cv=5, n_jobs=-1, train_sizes=np.linspace(0.1,1.0,8), scoring='accuracy')

fig, ax = plt.subplots(figsize=(10,6), facecolor=C['bg'])
ax.plot(train_sizes, train_scores.mean(axis=1)*100, 'o-', color=C['cyan'], lw=2.5, label='Train Accuracy')
ax.plot(train_sizes, test_scores.mean(axis=1)*100, 'o-', color=C['yellow'], lw=2.5, label='Validation Accuracy')
ax.fill_between(train_sizes, (train_scores.mean(axis=1)-train_scores.std(axis=1))*100,
                (train_scores.mean(axis=1)+train_scores.std(axis=1))*100, alpha=0.1, color=C['cyan'])
ax.fill_between(train_sizes, (test_scores.mean(axis=1)-test_scores.std(axis=1))*100,
                (test_scores.mean(axis=1)+test_scores.std(axis=1))*100, alpha=0.1, color=C['yellow'])
ax.set_title('XGBoost Learning Curve', fontsize=15, fontweight='bold')
ax.set_xlabel('Egitim Veri Boyutu'); ax.set_ylabel('Accuracy (%)')
ax.legend(fontsize=11); ax.set_facecolor(C['bg'])
plt.tight_layout(); plt.savefig(f"{OUT}/learning_curve_xgb.png", dpi=200, facecolor=C['bg']); plt.close()
print("[OK] learning_curve_xgb.png")

# ===== GORSEL 9: ELBOW METHOD =====
inertias = []
K_range = range(2,11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs[:5000])
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(10,5), facecolor=C['bg'])
ax.plot(K_range, inertias, 'o-', color=C['cyan'], lw=2.5, markersize=10)
ax.axvline(x=5, color=C['yellow'], ls='--', alpha=0.8, label='Secilen k=5')
ax.set_title('Elbow Metodu - Optimum Kume Sayisi', fontsize=15, fontweight='bold')
ax.set_xlabel('k (Kume Sayisi)'); ax.set_ylabel('Inertia')
ax.legend(fontsize=11); ax.set_facecolor(C['bg'])
plt.tight_layout(); plt.savefig(f"{OUT}/elbow_method.png", dpi=200, facecolor=C['bg']); plt.close()
print("[OK] elbow_method.png")

print(f"\n[DONE] Toplam {len(os.listdir(OUT))} gorsel uretildi: {OUT}/")
