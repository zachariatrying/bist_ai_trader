"""
================================================================================
BORSANEURON - TEKNIK ANALIZ TABANLI MODEL EGITIMI
================================================================================
bist_ai_dataset_real_30cols.csv uzerinden egitim:
- 30+ teknik indikator (RSI, MACD, Bollinger, Stochastic, ATR, Support/Resistance)
- Formasyon tanima (TOBO, OBO, Cup_Handle, Flag)
- Expert Signal (AL/SAT/Notr)
- Target: Gelecek 3 gunde %2+ yukselis

Cikti: best_model_acm465.joblib, best_scaler_acm465.joblib, best_features_acm465.joblib
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), 'bist_ai_dataset_real_30cols.csv')

# Alternatif yollar
if not os.path.exists(DATASET_PATH):
    alt_path = r'C:\Users\ibrah\.gemini\antigravity\scratch\bist_ai_dataset_real_30cols.csv'
    if os.path.exists(alt_path):
        DATASET_PATH = alt_path

MODEL_PATH = os.path.join(BASE_DIR, 'best_model_acm465.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'best_scaler_acm465.joblib')
FEATURES_PATH = os.path.join(BASE_DIR, 'best_features_acm465.joblib')

# ============================================================================
# TEKNIK ANALIZ FEATURE SETI (Zengin)
# ============================================================================
TECH_FEATURES = [
    'RSI_14',           # RSI (14 gun)
    'MACD',             # MACD
    'MACD_Signal',      # MACD Signal Line
    'ATR_14',           # Average True Range (Volatilite)          
    'Stoch_K',          # Stochastic K
    'Stoch_D',          # Stochastic D
    'BB_Upper',         # Bollinger Ust
    'BB_Middle',        # Bollinger Orta
    'BB_Lower',         # Bollinger Alt
    'SMA_20',           # 20 gunluk SMA
    'SMA_50',           # 50 gunluk SMA
    'SMA_200',          # 200 gunluk SMA
    'Support_Level',    # Destek seviyesi
    'Resistance_Level', # Direnç seviyesi
    'Volume_Trend',     # Hacim trendi (1/0)
    'Depth_Ratio',      # Derinlik orani
    'Neckline_Slope',   # Boyun cizgisi egimi
    'Expert_Signal',    # Uzman sistemi sinyali (-1, 0, 1)
]

# One-hot encode edilecek kategorik
CATEGORICAL_FEATURE = 'Pattern_Type'

TARGET_COL = 'Target_T3'  # Gelecek 3 gunde yukselis

def main():
    print("=" * 70)
    print("BORSANEURON - TEKNIK ANALIZ MODEL EGITIMI")
    print("=" * 70)
    
    # 1. Veri Yukleme
    print(f"\n[1] Dataset yukleniyor: {os.path.basename(DATASET_PATH)}")
    df = pd.read_csv(DATASET_PATH)
    print(f"    Toplam: {len(df):,} satir, {len(df.columns)} kolon")
    print(f"    Hisse: {df['Ticker'].nunique()} adet")
    print(f"    Tarih: {df['Date'].min()} -> {df['Date'].max()}")
    
    # 2. Pattern_Type one-hot encoding
    print(f"\n[2] Pattern_Type one-hot encoding...")
    pattern_dummies = pd.get_dummies(df[CATEGORICAL_FEATURE], prefix='Pat')
    df = pd.concat([df, pattern_dummies], axis=1)
    
    pattern_cols = [c for c in df.columns if c.startswith('Pat_')]
    print(f"    Formasyon kolonlari: {pattern_cols}")
    
    # 3. Feature listesi
    ALL_FEATURES = TECH_FEATURES + pattern_cols
    
    # NaN/Inf temizligi
    df_clean = df.dropna(subset=ALL_FEATURES + [TARGET_COL]).copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=ALL_FEATURES)
    
    print(f"\n[3] Temiz veri: {len(df_clean):,} satir")
    print(f"    Target dagilimi: {df_clean[TARGET_COL].value_counts().to_dict()}")
    
    # 4. Kronolojik Train/Test Split (%80/%20)
    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(df_clean) * 0.80)
    
    train = df_clean.iloc[:split_idx]
    test = df_clean.iloc[split_idx:]
    
    X_train = train[ALL_FEATURES].values
    y_train = train[TARGET_COL].values
    X_test = test[ALL_FEATURES].values
    y_test = test[TARGET_COL].values
    
    print(f"\n[4] Train: {len(train):,} | Test: {len(test):,}")
    print(f"    Features: {len(ALL_FEATURES)}")
    
    # 5. Olceklendirme
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 6. Model Egitimi
    results = {}
    
    # --- Random Forest ---
    print(f"\n{'='*50}")
    print("MODEL 1: Random Forest (GridSearch)")
    print('='*50)
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    
    print(f"  Accuracy: {rf_acc:.4f}")
    print(f"  F1 Score: {rf_f1:.4f}")
    print(classification_report(y_test, rf_pred, target_names=['ALMA(0)', 'AL(1)'], digits=4))
    
    results['RandomForest'] = {'model': rf, 'accuracy': rf_acc, 'f1': rf_f1, 'predictions': rf_pred}
    
    # --- XGBoost / GradientBoosting ---
    if XGBOOST_AVAILABLE:
        print(f"\n{'='*50}")
        print("MODEL 2: XGBoost")
        print('='*50)
        
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.08,
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model_name = "XGBoost"
    else:
        print(f"\n{'='*50}")
        print("MODEL 2: GradientBoosting")
        print('='*50)
        
        xgb = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.08,
            random_state=42
        )
        model_name = "GradientBoosting"
    
    xgb.fit(X_train_s, y_train)
    xgb_pred = xgb.predict(X_test_s)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)
    
    print(f"  Accuracy: {xgb_acc:.4f}")
    print(f"  F1 Score: {xgb_f1:.4f}")
    print(classification_report(y_test, xgb_pred, target_names=['ALMA(0)', 'AL(1)'], digits=4))
    
    results[model_name] = {'model': xgb, 'accuracy': xgb_acc, 'f1': xgb_f1, 'predictions': xgb_pred}
    
    # 7. En iyi model secimi
    best_name = max(results, key=lambda k: results[k]['f1'])
    best = results[best_name]
    best_model = best['model']
    
    print(f"\n{'='*70}")
    print(f"EN IYI MODEL: {best_name}")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  F1 Score: {best['f1']:.4f}")
    print(f"{'='*70}")
    
    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.DataFrame({'Feature': ALL_FEATURES, 'Importance': best_model.feature_importances_})
        fi = fi.sort_values('Importance', ascending=False)
        
        print("\nOZELLIK ONEMI SIRALAMASI:")
        print("-" * 50)
        for _, row in fi.iterrows():
            bar = "#" * int(row['Importance'] * 100)
            print(f"  {row['Feature']:22s} {row['Importance']:.4f}  {bar}")
    
    # 8. Kaydet
    print(f"\n[KAYIT]")
    joblib.dump(best_model, MODEL_PATH)
    print(f"  Model  -> {MODEL_PATH}")
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Scaler -> {SCALER_PATH}")
    
    joblib.dump(ALL_FEATURES, FEATURES_PATH)
    print(f"  Features ({len(ALL_FEATURES)}) -> {FEATURES_PATH}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best['predictions'])
    print(f"\nConfusion Matrix ({best_name}):")
    print(f"               Tahmin=0  Tahmin=1")
    print(f"  Gercek=0     {cm[0][0]:>7}   {cm[0][1]:>7}")
    print(f"  Gercek=1     {cm[1][0]:>7}   {cm[1][1]:>7}")
    
    print(f"\nEGITIM TAMAMLANDI!")
    print(f"Kayitli dosyalar BorsaNeuron Terminal tarafindan kullanilacak.")


if __name__ == '__main__':
    main()
