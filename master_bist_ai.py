"""
================================================================================
ACM 465 - VİZE + FİNAL BİRLEŞİK DÖNEM PROJESİ
master_bist_ai.py — BIST Borsa Verileri Uzerinde
Makine Ogrenmesi + Derin Ogrenme Pipeline'i
================================================================================
Yazar  : Ibrahim
Tarih  : 2026-04-10
Konu   : 163.000+ satirlik BIST veritabani uzerinde,
         EDA, Feature Engineering, K-Means, PCA, Random Forest,
         LightGBM, Yapay Sinir Agi (ANN) ile "Alim Firsati" tahmini.
================================================================================
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

# --- Sklearn ---
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import joblib

# --- Opsiyonel Kutuphaneler ---
# LightGBM (yoksa GradientBoosting kullanilacak)
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# Keras / TensorFlow (yoksa MLPClassifier kullanilacak)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==============================================================================
# YAPILANDIRMA (Configuration)
# ==============================================================================

# Dosya yollari
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR    = os.path.join(BASE_DIR, 'src', 'market_data_cache')
CSV_FILE     = os.path.join(BASE_DIR, 'tamam_poip_olanlari_cikar_son_listeden.csv')
OUTPUT_CSV   = os.path.join(BASE_DIR, 'master_buy_signals.csv')
MODEL_PATH   = os.path.join(BASE_DIR, 'best_model_acm465.joblib')
SCALER_PATH  = os.path.join(BASE_DIR, 'best_scaler_acm465.joblib')
FEATURES_PATH= os.path.join(BASE_DIR, 'best_features_acm465.joblib')
ANN_PATH     = os.path.join(BASE_DIR, 'best_ann_acm465.h5')

# Sabitler
RANDOM_STATE      = 42
TARGET_HORIZON    = 3     # Gun
TARGET_THRESHOLD  = 0.02  # %2
TRAIN_RATIO       = 0.80
CORR_THRESHOLD    = 0.90  # Multicollinearity siniri
N_CLUSTERS        = 3     # K-Means kume sayisi
PCA_VARIANCE      = 0.95  # Aciklanan varyans orani

# Ozellik listesi
FEATURE_COLS = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Mid', 'BB_Lower', 'BB_Width',
    'SMA_20', 'SMA_50',
    'Daily_Return', 'Volatility_20',
    'Volume_Ratio',
    'is_bull_flag'
]


def p(msg):
    """Unicode-safe print (Windows cp1254 uyumlulugu)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode())


# ==============================================================================
# BOLUM 1: VERI YUKLEME
# ==============================================================================

def load_data():
    """
    Veri kaynagini yukler:
    1. Oncelik: tamam_poip_olanlari_cikar_son_listeden.csv
    2. Yedek: market_data_cache/*.parquet dosyalari
    """
    p("=" * 70)
    p("ASAMA 1: VERI YUKLEME")
    p("=" * 70)

    # --- Yontem 1: CSV dosyasi ---
    if os.path.exists(CSV_FILE):
        p(f"  -> CSV yukleniyor: {os.path.basename(CSV_FILE)}")
        df = pd.read_csv(CSV_FILE)

        # Olasi kolon isim duzenlemeleri
        rename_map = {
            'date': 'Date', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            'ticker': 'Ticker', 'symbol': 'Ticker',
            'Hisse': 'Ticker', 'Tarih': 'Date',
            'Kapanış': 'Close', 'Kapanis': 'Close',
            'Hacim': 'Volume', 'Acilis': 'Open',
            'En Yüksek': 'High', 'En Düşük': 'Low'
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
                  inplace=True)

        # Ticker yoksa dosya adindan cikar
        if 'Ticker' not in df.columns:
            # Eger 'symbol' varsa BIST: on ekini kaldir
            for col in df.columns:
                if df[col].dtype == object and df[col].str.contains('BIST:', na=False).any():
                    df['Ticker'] = df[col].str.replace('BIST:', '', regex=False)
                    break
            if 'Ticker' not in df.columns:
                df['Ticker'] = 'UNKNOWN'

        p(f"  -> Yuklendi: {df.shape[0]:,} satir, {df.shape[1]} kolon")
        return df

    # --- Yontem 2: Parquet cache ---
    p(f"  -> CSV bulunamadi. Parquet dosyalari yukleniyor...")
    pattern = os.path.join(CACHE_DIR, "*_1d_TRY.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"Veri bulunamadi!\n"
            f"  CSV: {CSV_FILE}\n"
            f"  Cache: {CACHE_DIR}"
        )

    p(f"  -> {len(files)} adet parquet dosyasi bulundu.")

    frames = []
    for f in files:
        try:
            temp = pd.read_parquet(f)
            basename = os.path.basename(f)
            ticker = basename.split("_1d")[0]

            if 'symbol' in temp.columns:
                temp['Ticker'] = temp['symbol'].str.replace('BIST:', '', regex=False)
                temp.drop(columns=['symbol'], inplace=True)
            else:
                temp['Ticker'] = ticker

            frames.append(temp)
        except Exception:
            pass

    df = pd.concat(frames, ignore_index=True)

    p(f"  -> Birlestirilen veri: {df.shape[0]:,} satir, {df.shape[1]} kolon")
    p(f"  -> Benzersiz hisse sayisi: {df['Ticker'].nunique()}")
    p(f"  -> Tarih araligi: {df['Date'].min()} - {df['Date'].max()}")

    return df


# ==============================================================================
# BOLUM 2: VERI ON ISLEME ve EDA (Vize Konulari)
# ==============================================================================

def preprocess(df):
    """
    Gelismis veri on isleme:
    - DatetimeIndex donusumu
    - Forward Fill + Backfill
    - Z-Skoru anomali tespiti ve Winsorization
    """
    p("\n" + "=" * 70)
    p("ASAMA 2: VERI ON ISLEME ve EDA")
    p("=" * 70)

    df = df.copy()

    # 2.1 — Zaman serisi formati
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # 2.2 — Eksik deger doldurma (Forward Fill + Backfill)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Olmayan kolonlari atla
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    missing_before = df[numeric_cols].isnull().sum().sum()
    p(f"  -> Eksik deger (once): {missing_before}")

    df[numeric_cols] = df.groupby('Ticker')[numeric_cols].transform(lambda x: x.ffill())
    df[numeric_cols] = df.groupby('Ticker')[numeric_cols].transform(lambda x: x.bfill())

    missing_after = df[numeric_cols].isnull().sum().sum()
    p(f"  -> Eksik deger (sonra): {missing_after}")

    # 2.3 — Z-Skoru anomali tespiti & Winsorization
    anomaly_count = 0
    for col in ['Close', 'Volume']:
        if col not in df.columns:
            continue
        df[f'{col}_z'] = df.groupby('Ticker')[col].transform(
            lambda x: stats.zscore(x, nan_policy='omit') if len(x) > 2 else 0
        )
        mask = df[f'{col}_z'].abs() > 3
        n = mask.sum()
        anomaly_count += n

        if n > 0:
            lo = df.groupby('Ticker')[col].transform(lambda x: x.quantile(0.01))
            hi = df.groupby('Ticker')[col].transform(lambda x: x.quantile(0.99))
            df.loc[mask, col] = df.loc[mask, col].clip(lower=lo[mask], upper=hi[mask])

        df.drop(columns=[f'{col}_z'], inplace=True)

    p(f"  -> {anomaly_count} anomali tespit edildi ve baskilandi (Winsorization)")

    # Gecersiz satirlari cikar
    before = len(df)
    if 'Close' in df.columns:
        df = df[df['Close'] > 0]
    if 'Volume' in df.columns:
        df = df[df['Volume'] >= 0]
    p(f"  -> {before - len(df)} gecersiz satir temizlendi")
    p(f"  -> Son boyut: {df.shape[0]:,} satir")

    # 2.4 — EDA Ozet Istatistikleri
    p("\n  --- EDA OZET ---")
    if 'Close' in df.columns:
        p(f"  Close  -> Ort: {df['Close'].mean():.2f}, Std: {df['Close'].std():.2f}, "
          f"Min: {df['Close'].min():.2f}, Max: {df['Close'].max():.2f}")
    if 'Volume' in df.columns:
        p(f"  Volume -> Ort: {df['Volume'].mean():.0f}, Std: {df['Volume'].std():.0f}")
    p(f"  Hisse Sayisi: {df['Ticker'].nunique()}")

    return df


# ==============================================================================
# BOLUM 3: OZELLIK MUHENDISLIGI (Feature Engineering)
# ==============================================================================

def add_features(df):
    """
    Teknik indikatorler ve formasyon sinyalleri uretir:
    - RSI (14), MACD, MACD_Signal, MACD_Hist
    - Bollinger Bantlari (Upper, Mid, Lower, Width)
    - SMA 20/50, Daily Return, Volatility, Volume Ratio
    - is_bull_flag (Boga Bayragi formasyonu)
    """
    p("\n" + "=" * 70)
    p("ASAMA 3: OZELLIK MUHENDISLIGI")
    p("=" * 70)

    df = df.copy()

    # --- RSI (14 gun) ---
    def calc_rsi(s, period=14):
        d = s.diff()
        g = d.clip(lower=0).rolling(period).mean()
        l = (-d.clip(upper=0)).rolling(period).mean()
        l = l.replace(0, 1e-10)
        return 100 - (100 / (1 + g / l))

    df['RSI'] = df.groupby('Ticker')['Close'].transform(calc_rsi)

    # --- MACD ---
    df['EMA12'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df['EMA26'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df.groupby('Ticker')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df.drop(columns=['EMA12', 'EMA26'], inplace=True)

    # --- Bollinger Bantlari (20 gun) ---
    df['BB_Mid'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).mean())
    bb_std = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).std())
    df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'].replace(0, 1e-10)

    # --- SMA ---
    df['SMA_20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).mean())
    df['SMA_50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(50).mean())

    # --- Getiri ve Volatilite ---
    df['Daily_Return'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change())
    df['Volatility_20'] = df.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(20).std())

    # --- Hacim Orani ---
    df['Volume_SMA_20'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20).mean())
    df['Volume_SMA_20'] = df['Volume_SMA_20'].replace(0, 1e-10)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

    # --- Boga Bayragi (Bull Flag) ---
    df['is_bull_flag'] = 0
    for ticker in df['Ticker'].unique():
        m = df['Ticker'] == ticker
        td = df.loc[m]
        if len(td) < 50:
            continue
        c = td['Close'].values
        h = td['High'].values
        sma = td['SMA_20'].values
        sig = np.zeros(len(td))
        for i in range(40, len(td)):
            wh = h[i-40:i].max()
            wl = c[i-40:i].min()
            if wl <= 0:
                continue
            if (wh - wl) / wl < 0.05:
                continue
            if c[i] > wh * 0.88 and not np.isnan(sma[i]) and c[i] > sma[i]:
                sig[i] = 1
        df.loc[m, 'is_bull_flag'] = sig

    n_flags = int(df['is_bull_flag'].sum())
    p(f"  -> Teknik indikatorler eklendi (RSI, MACD, Bollinger, SMA, ...)")
    p(f"  -> Boga Bayragi (Bull Flag) sinyali: {n_flags:,} adet")

    return df


def create_target(df):
    """
    Hedef degiskeni:
    Gelecek 3 gunde kapanis %2+ yukselmisse Target=1, degilse Target=0.
    """
    df = df.copy()
    df['_fmax'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.shift(-1).rolling(window=TARGET_HORIZON).max()
    )
    df['Target'] = ((df['_fmax'] - df['Close']) / df['Close'] >= TARGET_THRESHOLD).astype(int)
    df.drop(columns=['_fmax'], inplace=True)

    counts = df['Target'].value_counts()
    total = counts.sum()
    p(f"\n  -> Target olusturuldu (Horizon={TARGET_HORIZON}, Threshold=%{TARGET_THRESHOLD*100:.0f})")
    p(f"     Target=0 (Alma): {counts.get(0,0):>8,} ({counts.get(0,0)/total*100:.1f}%)")
    p(f"     Target=1 (Al):   {counts.get(1,0):>8,} ({counts.get(1,0)/total*100:.1f}%)")

    return df


def remove_multicollinear(df, feature_cols, threshold=0.90):
    """
    Birbiriyle %90'dan fazla korele olan ozellikleri otomatik duser.
    Multicollinearity temizligi.
    """
    p(f"\n  --- MULTICOLLINEARITY TEMIZLIGI (esik: {threshold*100:.0f}%) ---")

    temp = df[feature_cols].dropna()
    if temp.empty:
        p("  -> Yeterli veri yok, atlanıyor.")
        return feature_cols

    corr = temp.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        if correlated:
            to_drop.add(col)

    if to_drop:
        p(f"  -> Dusurulecek yuksek korelasyonlu ozellikler: {to_drop}")
    else:
        p(f"  -> Yuksek korelasyonlu ozellik bulunamadi.")

    remaining = [c for c in feature_cols if c not in to_drop]
    p(f"  -> Kalan ozellik sayisi: {len(remaining)} / {len(feature_cols)}")
    return remaining


# ==============================================================================
# BOLUM 4: BOYUT INDIRGEME ve GOZETIMSIZ OGRENME (Final Konulari)
# ==============================================================================

def apply_kmeans(df, feature_cols, n_clusters=3):
    """
    K-Means Clustering ile hisseleri fiyat hareketlerine gore gruplar.
    Her hissenin son indikator degerlerini kullanir.
    """
    p("\n" + "=" * 70)
    p("ASAMA 4a: K-MEANS CLUSTERING")
    p("=" * 70)

    # Her hissenin son satirini al
    latest = df.sort_values('Date').groupby('Ticker').last().reset_index()
    latest_clean = latest.dropna(subset=feature_cols).copy()
    latest_clean = latest_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

    if len(latest_clean) < n_clusters:
        p("  -> Yeterli hisse yok, K-Means atlaniyor.")
        return df

    X_km = StandardScaler().fit_transform(latest_clean[feature_cols].values)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_km)
    latest_clean['Cluster'] = labels

    # Kume dagilimini goster
    p(f"  -> {n_clusters} kume olusturuldu:")
    for c in range(n_clusters):
        count = (labels == c).sum()
        tickers_in = latest_clean[latest_clean['Cluster'] == c]['Ticker'].head(5).tolist()
        p(f"     Kume {c}: {count} hisse (Ornek: {', '.join(tickers_in[:5])})")

    # Ana dataframe'e ekle
    cluster_map = latest_clean.set_index('Ticker')['Cluster'].to_dict()
    df['Cluster'] = df['Ticker'].map(cluster_map).fillna(-1).astype(int)

    return df


def apply_pca(X_train, X_test, feature_names, variance_ratio=0.95):
    """
    PCA (Principal Component Analysis) boyut indirgeme.
    Aciklanan varyans oranini konsola basar.
    """
    p("\n" + "=" * 70)
    p("ASAMA 4b: PCA BOYUT INDIRGEME")
    p("=" * 70)

    pca = PCA(n_components=variance_ratio, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    p(f"  -> Orijinal ozellik sayisi: {X_train.shape[1]}")
    p(f"  -> PCA sonrasi bilesen sayisi: {pca.n_components_}")
    p(f"  -> Aciklanan toplam varyans: %{sum(pca.explained_variance_ratio_)*100:.2f}")

    p("\n  Bilesen bazinda aciklanan varyans:")
    cumulative = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative += var
        bar = "#" * int(var * 100)
        p(f"     PC{i+1:>2}: %{var*100:5.2f}  (Kum: %{cumulative*100:5.2f})  {bar}")
        if i >= 9:  # Ilk 10 bilesen yeterli
            if i + 1 < len(pca.explained_variance_ratio_):
                p(f"     ... (toplam {len(pca.explained_variance_ratio_)} bilesen)")
            break

    return X_train_pca, X_test_pca, pca


# ==============================================================================
# BOLUM 5: MAKINE OGRENMESI MODELLEMESI
# ==============================================================================

def prepare_data(df, feature_cols):
    """Train/Test ayirimi ve olceklendirme."""
    p("\n" + "=" * 70)
    p("ASAMA 5: MODEL EGITIMI")
    p("=" * 70)

    clean = df.dropna(subset=feature_cols + ['Target']).copy()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    clean = clean.sort_values('Date').reset_index(drop=True)

    split = int(len(clean) * TRAIN_RATIO)
    train = clean.iloc[:split]
    test = clean.iloc[split:]

    X_train = train[feature_cols].values
    y_train = train['Target'].values
    X_test = test[feature_cols].values
    y_test = test['Target'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    p(f"  -> Temiz veri: {len(clean):,} satir")
    p(f"  -> Egitim: {X_train_s.shape[0]:,} satir ({TRAIN_RATIO*100:.0f}%)")
    p(f"  -> Test:   {X_test_s.shape[0]:,} satir ({(1-TRAIN_RATIO)*100:.0f}%)")
    p(f"  -> Ozellik: {X_train_s.shape[1]}")
    p(f"  -> Verileri StandardScaler ile olceklendirildi.")

    return X_train_s, X_test_s, y_train, y_test, test, scaler


def train_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest + GridSearchCV hiperparametre optimizasyonu."""
    p("\n" + "-" * 50)
    p("MODEL 1: Random Forest + GridSearchCV")
    p("-" * 50)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 12],
        'min_samples_split': [5, 10]
    }

    base_rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    grid = GridSearchCV(
        base_rf, param_grid,
        cv=3, scoring='f1', n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    pred = best_rf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    p(f"\n  En iyi parametreler: {grid.best_params_}")
    p(f"  GridSearchCV F1 (CV): {grid.best_score_:.4f}")
    p(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    p("\n  Classification Report:")
    p(classification_report(y_test, pred, target_names=['Alma(0)', 'Al(1)'], digits=4))

    return best_rf, pred, acc


def train_lgbm_or_gb(X_train, y_train, X_test, y_test):
    """LightGBM veya GradientBoosting modeli."""
    if LGBM_AVAILABLE:
        name = "LightGBM"
        p("\n" + "-" * 50)
        p("MODEL 2: LightGBM Classifier")
        p("-" * 50)

        scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            scale_pos_weight=scale, random_state=RANDOM_STATE,
            verbose=-1, n_jobs=-1
        )
    else:
        name = "GradientBoosting"
        p("\n" + "-" * 50)
        p("MODEL 2: Gradient Boosting Classifier (LightGBM yok)")
        p("-" * 50)

        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=RANDOM_STATE
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    p(f"\n  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    p("\n  Classification Report:")
    p(classification_report(y_test, pred, target_names=['Alma(0)', 'Al(1)'], digits=4))

    return model, pred, acc, name


# ==============================================================================
# BOLUM 6: DERIN OGRENME (ANN)
# ==============================================================================

def train_ann(X_train, y_train, X_test, y_test, input_dim):
    """
    3 gizli katmanli Yapay Sinir Agi (ANN).
    Keras varsa Keras, yoksa MLPClassifier kullanir.
    Dropout + EarlyStopping ile overfitting engellenir.
    """
    if KERAS_AVAILABLE:
        return _train_keras_ann(X_train, y_train, X_test, y_test, input_dim)
    else:
        return _train_sklearn_ann(X_train, y_train, X_test, y_test)


def _train_keras_ann(X_train, y_train, X_test, y_test, input_dim):
    """Keras Sequential model — 3 gizli katman + Dropout + EarlyStopping."""
    p("\n" + "-" * 50)
    p("MODEL 3: Keras Yapay Sinir Agi (ANN)")
    p("-" * 50)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=0
    )

    history = model.fit(
        X_train, y_train,
        epochs=50, batch_size=256,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=0
    )

    y_prob = model.predict(X_test, verbose=0).flatten()
    pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)

    p(f"\n  Egitim epoch sayisi: {len(history.history['loss'])}")
    p(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    p("\n  Classification Report:")
    p(classification_report(y_test, pred, target_names=['Alma(0)', 'Al(1)'], digits=4))

    # Modeli kaydet
    try:
        model.save(ANN_PATH)
        p(f"  -> Keras model kaydedildi: {ANN_PATH}")
    except Exception:
        pass

    return model, pred, acc, "Keras ANN"


def _train_sklearn_ann(X_train, y_train, X_test, y_test):
    """
    Sklearn MLPClassifier — Keras yoksa kullanilir.
    3 gizli katman (128-64-32) ile ANN simule eder.
    """
    p("\n" + "-" * 50)
    p("MODEL 3: Sklearn MLPClassifier (Keras yok)")
    p("  -> 3 gizli katman: (128, 64, 32)")
    p("  -> EarlyStopping: aktif (validation_fraction=0.15)")
    p("-" * 50)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    p(f"\n  Egitim iterasyon: {model.n_iter_}")
    p(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    p("\n  Classification Report:")
    p(classification_report(y_test, pred, target_names=['Alma(0)', 'Al(1)'], digits=4))

    return model, pred, acc, "MLPClassifier (ANN)"


# ==============================================================================
# BOLUM 7: MODEL DEGERLENDIRME ve CIKTI
# ==============================================================================

def compare_models(results, y_test, feature_names=None):
    """
    Tum modelleri karsilastirir:
    - Accuracy, Precision, Recall, F1
    - Confusion Matrix
    - Feature Importance (en iyi model)
    """
    p("\n" + "=" * 70)
    p("ASAMA 6: MODEL DEGERLENDIRME ve KARSILASTIRMA")
    p("=" * 70)

    # --- Karsilastirma Tablosu ---
    p("\n  +----------------------------+----------+-----------+--------+--------+")
    p("  | Model                      | Accuracy | Precision | Recall |   F1   |")
    p("  +----------------------------+----------+-----------+--------+--------+")

    best_name = None
    best_acc = 0
    best_model = None

    for name, info in results.items():
        pred = info['predictions']
        acc = info['accuracy']
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = info['model']

        p(f"  | {name:<26} | {acc:.4f}   | {prec:.4f}    | {rec:.4f} | {f1:.4f} |")

    p("  +----------------------------+----------+-----------+--------+--------+")
    p(f"\n  EN BASARILI MODEL: {best_name} (Accuracy: {best_acc:.4f})")

    # --- Confusion Matrix (en iyi model) ---
    best_pred = results[best_name]['predictions']
    cm = confusion_matrix(y_test, best_pred)
    p(f"\n  Confusion Matrix ({best_name}):")
    p(f"                 Tahmin=0  Tahmin=1")
    p(f"    Gercek=0     {cm[0][0]:>7}   {cm[0][1]:>7}")
    p(f"    Gercek=1     {cm[1][0]:>7}   {cm[1][1]:>7}")

    # --- Feature Importance ---
    if feature_names and hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        fi = pd.DataFrame({'Ozellik': feature_names, 'Onem': importances})
        fi = fi.sort_values('Onem', ascending=False)

        p(f"\n  OZELLIK ONEMI SIRALAMASI ({best_name}):")
        p("  " + "-" * 45)
        for _, row in fi.iterrows():
            bar = "#" * int(row['Onem'] * 100)
            p(f"  {row['Ozellik']:<18} {row['Onem']:.4f}  {bar}")

        # Bull flag sirasi
        if 'is_bull_flag' in fi['Ozellik'].values:
            rank = fi.reset_index(drop=True)
            idx = rank[rank['Ozellik'] == 'is_bull_flag'].index[0] + 1
            val = fi[fi['Ozellik'] == 'is_bull_flag']['Onem'].values[0]
            p(f"\n  Boga Bayragi formasyonu: Sira #{idx} (Onem: {val:.4f})")

    # En iyi modeli kaydet
    if best_model is not None:
        try:
            joblib.dump(best_model, MODEL_PATH)
            p(f"\n  -> En iyi model kaydedildi: {MODEL_PATH}")
        except Exception as e:
            p(f"  -> Model kayit hatasi: {e}")

    return best_name, best_model


def generate_signals(df, model, scaler, feature_cols):
    """Guncel veride Target=1 sinyali veren hisseleri CSV'ye yazar."""
    p("\n" + "=" * 70)
    p("ASAMA 7: ALIM SINYALLERI (CSV CIKTISI)")
    p("=" * 70)

    latest = df.sort_values('Date').groupby('Ticker').last().reset_index()
    lc = latest.dropna(subset=feature_cols).replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

    if lc.empty:
        p("  -> Son verilerde yeterli veri yok.")
        return

    X = scaler.transform(lc[feature_cols].values)

    preds = model.predict(X)
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X)[:, 1]
    else:
        probas = preds.astype(float)

    lc = lc.copy()
    lc['Prediction'] = preds
    lc['Buy_Probability'] = probas

    buys = lc[lc['Prediction'] == 1].copy()

    if buys.empty:
        p("  -> Alim sinyali bulunamadi. En yuksek olasilikli 10 hisse kaydedilecek.")
        buys = lc.nlargest(10, 'Buy_Probability')

    out_cols = [c for c in ['Ticker', 'Date', 'Close', 'RSI', 'MACD',
                            'BB_Width', 'Volume_Ratio', 'is_bull_flag',
                            'Cluster', 'Buy_Probability'] if c in buys.columns]

    buys = buys[out_cols].sort_values('Buy_Probability', ascending=False)
    buys.to_csv(OUTPUT_CSV, index=False, float_format='%.4f')

    p(f"\n  {len(buys)} adet alim sinyali kaydedildi: {OUTPUT_CSV}")
    p(f"\n  ILK 10 ALIM SINYALI:")
    p("  " + "-" * 70)
    p(buys.head(10).to_string(index=False))


# ==============================================================================
# MAIN: PIPELINE
# ==============================================================================

def main():
    """Tum pipeline'i baslatir."""
    start = datetime.now()

    p("\n" + "#" * 70)
    p("  ACM 465 - MASTER BIST AI PIPELINE")
    p("  Ibrahim - Vize + Final Donem Projesi")
    p("  Tarih: " + start.strftime("%Y-%m-%d %H:%M:%S"))
    p("#" * 70)

    # Kutuphane durumu
    p(f"\n  [Kutuphane Durumu]")
    p(f"    LightGBM:   {'AKTIF' if LGBM_AVAILABLE else 'YOK -> GradientBoosting'}")
    p(f"    Keras/TF:   {'AKTIF' if KERAS_AVAILABLE else 'YOK -> MLPClassifier'}")

    # =====================
    # 1. Veri Yukleme
    # =====================
    raw = load_data()

    # =====================
    # 2. On Isleme
    # =====================
    clean = preprocess(raw)

    # =====================
    # 3. Ozellik Muhendisligi
    # =====================
    featured = add_features(clean)
    featured = create_target(featured)

    # Multicollinearity temizligi
    active_features = remove_multicollinear(featured, FEATURE_COLS, CORR_THRESHOLD)

    # =====================
    # 4. K-Means + PCA
    # =====================
    featured = apply_kmeans(featured, active_features, N_CLUSTERS)

    # =====================
    # 5. Train/Test Ayirimi
    # =====================
    X_train, X_test, y_train, y_test, test_df, scaler = prepare_data(featured, active_features)

    # PCA
    X_train_pca, X_test_pca, pca_obj = apply_pca(X_train, X_test, active_features, PCA_VARIANCE)

    # =====================
    # 6. Model Egitimi
    # =====================
    results = {}

    # Model 1: Random Forest + GridSearchCV
    rf_model, rf_pred, rf_acc = train_random_forest(X_train, y_train, X_test, y_test)
    results['Random Forest'] = {'model': rf_model, 'predictions': rf_pred, 'accuracy': rf_acc}

    # Model 2: LightGBM / GradientBoosting
    lgb_model, lgb_pred, lgb_acc, lgb_name = train_lgbm_or_gb(X_train, y_train, X_test, y_test)
    results[lgb_name] = {'model': lgb_model, 'predictions': lgb_pred, 'accuracy': lgb_acc}

    # Model 3: ANN (Keras veya MLP)
    ann_model, ann_pred, ann_acc, ann_name = train_ann(X_train, y_train, X_test, y_test, X_train.shape[1])
    results[ann_name] = {'model': ann_model, 'predictions': ann_pred, 'accuracy': ann_acc}

    # Bonus: PCA uzerinde RF
    p("\n" + "-" * 50)
    p("MODEL 4: Random Forest + PCA")
    p("-" * 50)
    rf_pca = RandomForestClassifier(n_estimators=200, max_depth=12,
                                     class_weight='balanced',
                                     random_state=RANDOM_STATE, n_jobs=-1)
    rf_pca.fit(X_train_pca, y_train)
    pca_pred = rf_pca.predict(X_test_pca)
    pca_acc = accuracy_score(y_test, pca_pred)
    p(f"\n  Test Accuracy: {pca_acc:.4f} ({pca_acc*100:.2f}%)")
    p("\n  Classification Report:")
    p(classification_report(y_test, pca_pred, target_names=['Alma(0)', 'Al(1)'], digits=4))
    results['RF + PCA'] = {'model': rf_pca, 'predictions': pca_pred, 'accuracy': pca_acc}

    # =====================
    # 7. Degerlendirme
    # =====================
    best_name, best_model = compare_models(results, y_test, active_features)

    # Scaler ve ozellik listesini kaydet (LiveInferenceEngine icin)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(active_features, FEATURES_PATH)
    p(f"  -> Scaler kaydedildi: {SCALER_PATH}")
    p(f"  -> Ozellik listesi kaydedildi: {FEATURES_PATH}")

    # =====================
    # 8. Alim Sinyalleri
    # =====================
    generate_signals(featured, best_model, scaler, active_features)

    # =====================
    # SONUC
    # =====================
    elapsed = (datetime.now() - start).total_seconds()
    p("\n" + "#" * 70)
    p("  PIPELINE TAMAMLANDI!")
    p("#" * 70)
    p(f"  Sure: {elapsed:.1f} saniye")
    p(f"  Kazanan model: {best_name}")
    p(f"  Model dosyasi: {MODEL_PATH}")
    p(f"  Sinyal dosyasi: {OUTPUT_CSV}")
    p("")


if __name__ == "__main__":
    main()
