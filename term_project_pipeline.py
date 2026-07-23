"""
================================================================================
ACM 465 - DÖNEM PROJESİ
BIST Borsa Verileri Üzerinde Makine Öğrenmesi Pipeline'ı
================================================================================
Yazar  : İbrahim (Antigravity Asistan Desteği ile)
Tarih  : 2026-04-10
Konu   : 163.000+ satırlık BIST veritabanı üzerinde,
         Teknik İndikatörler + Formasyon Sinyalleri kullanarak
         Random Forest ve XGBoost ile "Alım Fırsatı" tahmini.
================================================================================
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# XGBoost opsiyonel — yoksa GradientBoosting kullanılır
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost bulunamadı. GradientBoostingClassifier kullanılacak.")

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. BÖLÜM: VERİ YÜKLEME
# ==============================================================================

def load_all_data(cache_dir: str) -> pd.DataFrame:
    """
    market_data_cache klasöründeki tüm *_1d_TRY.parquet dosyalarını
    tek bir DataFrame'de birleştirir.
    
    Returns:
        pd.DataFrame: Tüm hisselerin birleştirilmiş günlük verisi
    """
    print("=" * 70)
    print("AŞAMA 1: VERİ YÜKLEME")
    print("=" * 70)
    
    pattern = os.path.join(cache_dir, "*_1d_TRY.parquet")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"Parquet dosyası bulunamadı: {cache_dir}")
    
    print(f"  → {len(files)} adet parquet dosyası bulundu.")
    
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Dosya adından ticker çıkar (örn: THYAO_1d_TRY.parquet → THYAO)
            basename = os.path.basename(f)
            ticker = basename.split("_1d")[0]
            
            # 'symbol' kolonu varsa kullan, yoksa dosya adından ata
            if 'symbol' in df.columns:
                df['Ticker'] = df['symbol'].str.replace('BIST:', '', regex=False)
            else:
                df['Ticker'] = ticker
                
            frames.append(df)
        except Exception as e:
            pass  # Bozuk dosyaları atla
    
    combined = pd.concat(frames, ignore_index=True)
    
    # Gereksiz kolonları temizle
    if 'symbol' in combined.columns:
        combined.drop(columns=['symbol'], inplace=True)
    
    print(f"  → Birleştirilmiş veri: {combined.shape[0]:,} satır, {combined.shape[1]} kolon")
    print(f"  → Benzersiz hisse sayısı: {combined['Ticker'].nunique()}")
    print(f"  → Tarih aralığı: {combined['Date'].min()} — {combined['Date'].max()}")
    
    return combined


# ==============================================================================
# 2. BÖLÜM: GELİŞMİŞ VERİ ÖN İŞLEME (Advanced Preprocessing)
# ==============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gelişmiş veri ön işleme:
    - DatetimeIndex'e çevirme
    - Forward Fill ile eksik veri doldurma
    - Z-Skoru ile anomali tespiti ve Winsorization
    """
    print("\n" + "=" * 70)
    print("AŞAMA 2: VERİ ÖN İŞLEME")
    print("=" * 70)
    
    df = df.copy()
    
    # 2.1 — Zaman serisi formatına dönüştürme
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # 2.2 — Eksik veri kontrolü ve Forward Fill
    missing_before = df.isnull().sum().sum()
    print(f"  → Eksik değer (önce): {missing_before}")
    
    # Her hisse grubu içinde forward fill uygula
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df.groupby('Ticker')[numeric_cols].transform(
        lambda x: x.ffill()
    )
    # Kalan eksikleri (başlangıçtakiler) backfill ile doldur
    df[numeric_cols] = df.groupby('Ticker')[numeric_cols].transform(
        lambda x: x.bfill()
    )
    
    missing_after = df.isnull().sum().sum()
    print(f"  → Eksik değer (sonra): {missing_after}")
    
    # 2.3 — Z-Skoru ile Anomali Tespiti ve Winsorization
    # Close ve Volume için ayrı ayrı Z-skoru hesapla (her hisse grubu içinde)
    anomaly_count = 0
    
    for col in ['Close', 'Volume']:
        # Grup bazlı Z-skoru hesapla
        df[f'{col}_zscore'] = df.groupby('Ticker')[col].transform(
            lambda x: stats.zscore(x, nan_policy='omit') if len(x) > 2 else 0
        )
        
        # |Z| > 3 olan değerleri tespit et
        anomalies = df[f'{col}_zscore'].abs() > 3
        n_anom = anomalies.sum()
        anomaly_count += n_anom
        
        if n_anom > 0:
            # Winsorization: uç değerleri %1–%99 aralığına çek
            lower = df.groupby('Ticker')[col].transform(lambda x: x.quantile(0.01))
            upper = df.groupby('Ticker')[col].transform(lambda x: x.quantile(0.99))
            df.loc[anomalies, col] = df.loc[anomalies, col].clip(
                lower=lower[anomalies], upper=upper[anomalies]
            )
        
        # Geçici Z-skoru kolonunu sil
        df.drop(columns=[f'{col}_zscore'], inplace=True)
    
    print(f"  → {anomaly_count} adet anomali tespit edildi ve baskılandı (Winsorization)")
    
    # Negatif/sıfır fiyat satırlarını çıkar
    before_clean = len(df)
    df = df[(df['Close'] > 0) & (df['Volume'] >= 0)]
    print(f"  → {before_clean - len(df)} adet geçersiz fiyat satırı temizlendi")
    print(f"  → Son veri boyutu: {df.shape[0]:,} satır")
    
    return df


# ==============================================================================
# 3. BÖLÜM: KAPSAMLI ÖZELLİK MÜHENDİSLİĞİ (Feature Engineering)
# ==============================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her hisse için teknik indikatörleri hesaplar:
    - RSI (14 gün)
    - MACD, MACD Signal
    - Bollinger Bantları (Üst, Orta, Alt)
    - Ek: SMA-20, SMA-50, Getiri (Return), Volatilite
    """
    print("\n" + "=" * 70)
    print("AŞAMA 3: ÖZELLİK MÜHENDİSLİĞİ (Feature Engineering)")
    print("=" * 70)
    
    df = df.copy()
    
    # ---- 3.1: RSI (Relative Strength Index — 14 Günlük) ----
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        loss = loss.replace(0, 1e-10)  # Sıfıra bölmeyi önle
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI'] = df.groupby('Ticker')['Close'].transform(calc_rsi)
    
    # ---- 3.2: MACD (Moving Average Convergence Divergence) ----
    df['EMA_12'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    df['EMA_26'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df.groupby('Ticker')['MACD'].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Geçici EMA kolonlarını sil
    df.drop(columns=['EMA_12', 'EMA_26'], inplace=True)
    
    # ---- 3.3: Bollinger Bantları (20 Günlük) ----
    df['BB_Mid'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=20).mean()
    )
    bb_std = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=20).std()
    )
    df['BB_Upper'] = df['BB_Mid'] + (2 * bb_std)
    df['BB_Lower'] = df['BB_Mid'] - (2 * bb_std)
    
    # Bollinger Band Width (Normalize edilmiş)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    
    # ---- 3.4: Hareketli Ortalamalar ----
    df['SMA_20'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=20).mean()
    )
    df['SMA_50'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=50).mean()
    )
    
    # ---- 3.5: Günlük Getiri ve Volatilite ----
    df['Daily_Return'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.pct_change()
    )
    df['Volatility_20'] = df.groupby('Ticker')['Daily_Return'].transform(
        lambda x: x.rolling(window=20).std()
    )
    
    # ---- 3.6: Hacim Değişimi ----
    df['Volume_SMA_20'] = df.groupby('Ticker')['Volume'].transform(
        lambda x: x.rolling(window=20).mean()
    )
    # Hacim SMA'nın sıfır olmasını önle
    df['Volume_SMA_20'] = df['Volume_SMA_20'].replace(0, 1e-10)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    print("  → Teknik indikatörler eklendi: RSI, MACD, MACD_Signal, MACD_Hist")
    print("  → Bollinger Bantları: BB_Upper, BB_Mid, BB_Lower, BB_Width")
    print("  → Ek özellikler: SMA_20, SMA_50, Daily_Return, Volatility_20, Volume_Ratio")
    
    return df


def add_bull_flag_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Boğa Bayrağı (Bull Flag) formasyonu sinyalini binary olarak üretir.
    
    Mantık:
    1. Son 40 barda güçlü bir yükseliş (direk) olmalı (>%5)
    2. Ardından fiyat zirvenin yakınında konsolide olmalı (bayrak)
    3. Fiyat SMA-20'nin üzerinde olmalı
    
    is_bull_flag = 1 veya 0
    """
    df = df.copy()
    df['is_bull_flag'] = 0
    
    # Her hisse için ayrı hesapla
    for ticker in df['Ticker'].unique():
        mask = df['Ticker'] == ticker
        ticker_df = df.loc[mask].copy()
        
        if len(ticker_df) < 50:
            continue
        
        close = ticker_df['Close'].values
        high = ticker_df['High'].values
        sma20 = ticker_df['SMA_20'].values
        
        signals = np.zeros(len(ticker_df))
        
        for i in range(40, len(ticker_df)):
            # Son 40 bardaki en yüksek ve en düşük
            window_high = high[i-40:i].max()
            window_low = close[i-40:i].min()
            
            # Direk boyu kontrolü (>%5 yükseliş olmuş mu?)
            if window_low <= 0:
                continue
            pole_pct = (window_high - window_low) / window_low
            if pole_pct < 0.05:
                continue
            
            # Konsolidasyon kontrolü: Fiyat zirvenin %88'inin üzerinde mi?
            if close[i] > window_high * 0.88:
                # SMA-20 üzerinde mi?
                if not np.isnan(sma20[i]) and close[i] > sma20[i]:
                    signals[i] = 1
        
        df.loc[mask, 'is_bull_flag'] = signals
    
    n_flags = int(df['is_bull_flag'].sum())
    print(f"  → Boğa Bayrağı (Bull Flag) sinyali eklendi: {n_flags:,} adet tespit")
    
    return df


def create_target(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.02) -> pd.DataFrame:
    """
    Hedef değişkeni oluşturur:
    
    Gelecek `horizon` gün içinde kapanış fiyatı, bugünkü kapanıştan
    `threshold` (varsayılan %2) ve üzeri yükselmiş mi?
    
    Target = 1 → Alım Fırsatı
    Target = 0 → Değil
    """
    df = df.copy()
    
    # Gelecek N günün maksimum kapanış fiyatı
    df['Future_Max_Close'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.shift(-1).rolling(window=horizon).max()
    )
    
    # Target: Gelecek 3 gün içinde %2+ yükselmiş mi?
    df['Target'] = ((df['Future_Max_Close'] - df['Close']) / df['Close'] >= threshold).astype(int)
    
    # Geçici kolonu sil
    df.drop(columns=['Future_Max_Close'], inplace=True)
    
    # Target dağılımı
    target_counts = df['Target'].value_counts()
    total = target_counts.sum()
    print(f"\n  → Hedef (Target) oluşturuldu: (Horizon={horizon} gün, Threshold=%{threshold*100:.0f})")
    print(f"     Target=0 (Alma): {target_counts.get(0, 0):>8,} ({target_counts.get(0, 0)/total*100:.1f}%)")
    print(f"     Target=1 (Al):   {target_counts.get(1, 0):>8,} ({target_counts.get(1, 0)/total*100:.1f}%)")
    
    return df


# ==============================================================================
# 4. BÖLÜM: MODELLEME VE OPTİMİZASYON
# ==============================================================================

# Feature listesi (model eğitiminde kullanılacak kolonlar)
FEATURE_COLS = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Mid', 'BB_Lower', 'BB_Width',
    'SMA_20', 'SMA_50',
    'Daily_Return', 'Volatility_20',
    'Volume_Ratio',
    'is_bull_flag'
]


def prepare_train_test(df: pd.DataFrame):
    """
    Veriyi zamana dayalı (Time-Series Split) olarak ayırır.
    
    ÖNEMLI: Zaman serilerinde rastgele ayırma YANLIŞ olur!
    Geçmiş verilerle gelecek tahmin edilmelidir.
    
    %80 Train (eski veriler) | %20 Test (yeni veriler)
    """
    print("\n" + "=" * 70)
    print("AŞAMA 4: MODEL EĞİTİMİ VE OPTİMİZASYON")
    print("=" * 70)
    
    # NaN içeren satırları çıkar (indikatör hesaplamaları nedeniyle ilk satırlar NaN olur)
    df_clean = df.dropna(subset=FEATURE_COLS + ['Target']).copy()
    
    # Infinite değerleri temizle
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
    
    print(f"  → Temiz veri (NaN/Inf temizliği sonrası): {df_clean.shape[0]:,} satır")
    
    # Zamana göre sırala
    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
    
    # %80 Train / %20 Test — kronolojik sıra ile
    split_idx = int(len(df_clean) * 0.80)
    
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['Target'].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['Target'].values
    
    print(f"  → Eğitim seti: {X_train.shape[0]:,} satır ({X_train.shape[0]/len(df_clean)*100:.1f}%)")
    print(f"  → Test seti:   {X_test.shape[0]:,} satır ({X_test.shape[0]/len(df_clean)*100:.1f}%)")
    print(f"  → Özellik sayısı: {X_train.shape[1]}")
    
    # StandardScaler ile ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("  → Veriler StandardScaler ile ölçeklendirildi.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, test_df, scaler


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    İki model eğitir ve karşılaştırır:
    1. Random Forest Classifier
    2. XGBoost Classifier (veya GradientBoosting)
    
    Her model için:
    - Accuracy
    - Precision, Recall, F1-Score (classification_report)
    """
    results = {}
    
    # ==========================================
    # MODEL 1: Random Forest Classifier
    # ==========================================
    print("\n" + "-" * 50)
    print("MODEL 1: Random Forest Classifier")
    print("-" * 50)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Dengesiz sınıfları dengele
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    print(f"\n  Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_test, rf_pred, 
                                target_names=['Alma (0)', 'Al (1)'],
                                digits=4))
    
    results['Random Forest'] = {
        'model': rf_model,
        'accuracy': rf_acc,
        'predictions': rf_pred
    }
    
    # ==========================================
    # MODEL 2: XGBoost (veya GradientBoosting)
    # ==========================================
    if XGBOOST_AVAILABLE:
        print("-" * 50)
        print("MODEL 2: XGBoost Classifier")
        print("-" * 50)
        
        # Sınıf dengesizlik oranını hesapla
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        model_name = "XGBoost"
    else:
        print("-" * 50)
        print("MODEL 2: Gradient Boosting Classifier")
        print("-" * 50)
        
        xgb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        model_name = "Gradient Boosting"
    
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    print(f"\n  Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_test, xgb_pred, 
                                target_names=['Alma (0)', 'Al (1)'],
                                digits=4))
    
    results[model_name] = {
        'model': xgb_model,
        'accuracy': xgb_acc,
        'predictions': xgb_pred
    }
    
    return results


# ==============================================================================
# 5. BÖLÜM: DERİNLEMESİNE DEĞERLENDİRME
# ==============================================================================

def deep_evaluation(results: dict, feature_names: list):
    """
    En başarılı modeli belirler ve Feature Importance sıralamasını yazdırır.
    """
    print("\n" + "=" * 70)
    print("AŞAMA 5: DERİNLEMESİNE DEĞERLENDİRME")
    print("=" * 70)
    
    # Kazanan modeli bul
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_result = results[best_name]
    
    print(f"\n  🏆 EN BAŞARILI MODEL: {best_name}")
    print(f"     Accuracy: {best_result['accuracy']:.4f}")
    
    # Karşılaştırma tablosu
    print("\n  ┌──────────────────────────┬────────────┐")
    print("  │ Model                    │ Accuracy   │")
    print("  ├──────────────────────────┼────────────┤")
    for name, res in results.items():
        marker = " ← Kazanan" if name == best_name else ""
        print(f"  │ {name:<24} │ {res['accuracy']:.4f}     │{marker}")
    print("  └──────────────────────────┴────────────┘")
    
    # Feature Importance
    best_model = best_result['model']
    importances = best_model.feature_importances_
    
    feat_imp = pd.DataFrame({
        'Özellik': feature_names,
        'Önem': importances
    }).sort_values('Önem', ascending=False)
    
    print(f"\n  📊 ÖZELLİK ÖNEMİ SIRALAMASI ({best_name}):")
    print("  " + "─" * 45)
    for i, row in feat_imp.iterrows():
        bar = "#" * int(row['Önem'] * 100)
        print(f"  {row['Özellik']:<18} {row['Önem']:.4f}  {bar}")
    
    print("\n  💡 YORUM:")
    top3 = feat_imp.head(3)['Özellik'].tolist()
    print(f"     Hedef fiyat hareketini tahmin etmede en kritik 3 özellik:")
    for i, feat in enumerate(top3, 1):
        print(f"     {i}. {feat}")
    
    if 'is_bull_flag' in feat_imp['Özellik'].values:
        flag_rank = feat_imp.reset_index(drop=True)
        flag_idx = flag_rank[flag_rank['Özellik'] == 'is_bull_flag'].index[0] + 1
        flag_imp = feat_imp[feat_imp['Özellik'] == 'is_bull_flag']['Önem'].values[0]
        print(f"\n     Boğa Bayrağı (Bull Flag) formasyonu: Sıra #{flag_idx} (Önem: {flag_imp:.4f})")
    
    return best_name, best_model, feat_imp


# ==============================================================================
# 6. BÖLÜM: ALIM SİNYALİ ÇIKTI DOSYASI
# ==============================================================================

def generate_buy_signals(df: pd.DataFrame, model, scaler, feature_cols: list, output_path: str):
    """
    En son verilere göre model tahminini çalıştırır ve 
    Target=1 (Alım Fırsatı) sinyali üreten hisseleri CSV'ye kaydeder.
    """
    print("\n" + "=" * 70)
    print("AŞAMA 6: ALIM SİNYALLERİ (CSV ÇIKTISI)")
    print("=" * 70)
    
    # Her hissenin en son satırını al
    latest = df.sort_values('Date').groupby('Ticker').last().reset_index()
    
    # Feature'ları hazırla
    latest_clean = latest.dropna(subset=feature_cols).copy()
    
    # Infinite değerleri temizle
    latest_clean = latest_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    
    if latest_clean.empty:
        print("  ⚠️ Son verilerde yeterli veri yok.")
        return
    
    X_latest = scaler.transform(latest_clean[feature_cols].values)
    
    # Tahmin
    predictions = model.predict(X_latest)
    probas = model.predict_proba(X_latest)[:, 1]  # Alım olasılığı
    
    latest_clean['Prediction'] = predictions
    latest_clean['Buy_Probability'] = probas
    
    # Sadece alım sinyali olanları filtrele
    buy_signals = latest_clean[latest_clean['Prediction'] == 1].copy()
    
    if buy_signals.empty:
        print("  ⚠️ Şu an alım sinyali üreten hisse bulunamadı.")
        # Yine de en yüksek olasılıklı 10 hisseyi kaydet
        buy_signals = latest_clean.nlargest(10, 'Buy_Probability').copy()
        print(f"  → Alternatif: En yüksek alım olasılığına sahip {len(buy_signals)} hisse kaydedildi.")
    
    # Çıktı kolonları
    output_cols = [
        'Ticker', 'Date', 'Close', 'RSI', 'MACD', 'BB_Width',
        'Volume_Ratio', 'is_bull_flag', 'Buy_Probability'
    ]
    
    # Mevcut kolonlardan seç
    output_cols = [c for c in output_cols if c in buy_signals.columns]
    
    buy_output = buy_signals[output_cols].sort_values('Buy_Probability', ascending=False)
    
    # CSV'ye kaydet
    buy_output.to_csv(output_path, index=False, float_format='%.4f')
    
    print(f"\n  ✅ {len(buy_output)} adet alım sinyali '{output_path}' dosyasına kaydedildi.")
    print(f"\n  📋 İLK 10 ALIM SİNYALİ:")
    print("  " + "─" * 70)
    print(buy_output.head(10).to_string(index=False))


# ==============================================================================
# MAIN: PIPELINE ÇALIŞTIRMA
# ==============================================================================

def main():
    """
    Tüm pipeline'ı baştan sona çalıştırır.
    """
    print("\n" + "#" * 70)
    print("   ACM 465 - BIST MAKİNE ÖĞRENMESİ PİPELINE'I")
    print("   İbrahim — Dönem Projesi")
    print("#" * 70)
    
    # Veri dizini
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'market_data_cache')
    OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'project_buy_signals.csv')
    
    # ========================
    # AŞAMA 1: Veri Yükleme
    # ========================
    raw_data = load_all_data(CACHE_DIR)
    
    # ========================
    # AŞAMA 2: Ön İşleme
    # ========================
    clean_data = preprocess_data(raw_data)
    
    # ========================
    # AŞAMA 3: Özellik Mühendisliği
    # ========================
    featured_data = add_technical_indicators(clean_data)
    featured_data = add_bull_flag_signal(featured_data)
    featured_data = create_target(featured_data, horizon=3, threshold=0.02)
    
    # ========================
    # AŞAMA 4: Modelleme
    # ========================
    X_train, X_test, y_train, y_test, test_df, scaler = prepare_train_test(featured_data)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # ========================
    # AŞAMA 5: Değerlendirme
    # ========================
    best_name, best_model, feat_imp = deep_evaluation(results, FEATURE_COLS)
    
    # ========================
    # AŞAMA 6: Alım Sinyalleri
    # ========================
    generate_buy_signals(featured_data, best_model, scaler, FEATURE_COLS, OUTPUT_CSV)
    
    print("\n" + "#" * 70)
    print("   ✅ PİPELINE TAMAMLANDI!")
    print("#" * 70)
    print(f"\n  Çıktı dosyası: {OUTPUT_CSV}")
    print(f"  Kazanan model: {best_name}")
    print()


if __name__ == "__main__":
    main()
