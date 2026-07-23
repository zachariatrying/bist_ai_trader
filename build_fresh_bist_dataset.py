"""
==============================================================================
BorsaNeuron - Fresh BIST Dataset Generator, Cleaner & Double Confirmation Trainer
Author: İbrahim Tatar
Description: Downloads current BIST market data, cleans missing values/outliers,
             computes 20 technical indicators, constructs Target_T5 labels,
             and trains serialized ML models (best_model_acm465.joblib).
==============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

FEATURE_NAMES = [
    'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
    'RSI_14', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR_14', 'Stoch_K', 'Stoch_D',
    'Support_Level', 'Resistance_Level', 'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope'
]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def build_and_clean_bist_dataset(tickers=None, period="3y"):
    """
    Downloads BIST equities data, cleans anomalies, and generates 20 technical features.
    """
    if tickers is None:
        tickers = [
            "THYAO.IS", "GARAN.IS", "EREGL.IS", "AKBNK.IS", "SISE.IS", "TUPRS.IS", 
            "KCHOL.IS", "ASELS.IS", "FROTO.IS", "BIMAS.IS", "SAHOL.IS", "SASA.IS", 
            "HEKTS.IS", "YKBNK.IS", "ISCTR.IS", "VAKBN.IS", "PETKM.IS", "KOZAL.IS"
        ]
        
    all_data = []
    print(f"[DATASET BUILDER] Downloading live BIST historical data ({period}) for {len(tickers)} equities...")
    
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval="1d", progress=False)
            if df is None or df.empty or len(df) < 100:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
                
            clean_ticker = t.replace('.IS', '')
            df['Ticker'] = clean_ticker
            df['Date'] = df.index.strftime('%Y-%m-%d')
            
            # --- 1. DATA AUDIT & CLEANING ---
            # Forward fill / backward fill for missing days
            df = df.ffill().bfill()
            
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            # --- 2. FEATURE ENGINEERING (20 FEATURES) ---
            df['SMA_20'] = close.rolling(20).mean()
            df['SMA_50'] = close.rolling(50).mean()
            df['SMA_200'] = close.rolling(200).mean()
            
            df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
            df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            df['RSI_14'] = calculate_rsi(close, 14)
            
            # Bollinger Bands
            std_20 = close.rolling(20).std()
            df['BB_Middle'] = df['SMA_20']
            df['BB_Upper'] = df['BB_Middle'] + (std_20 * 2)
            df['BB_Lower'] = df['BB_Middle'] - (std_20 * 2)
            
            # ATR (Average True Range)
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR_14'] = tr.rolling(14).mean()
            
            # Stochastic Oscillator
            low_14 = low.rolling(14).min()
            high_14 = high.rolling(14).max()
            df['Stoch_K'] = 100 * ((close - low_14) / (high_14 - low_14 + 1e-10))
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
            
            # Support / Resistance & Patterns
            df['Support_Level'] = low.rolling(20).min()
            df['Resistance_Level'] = high.rolling(20).max()
            df['Volume_Trend'] = volume / (volume.rolling(20).mean() + 1e-10)
            df['Depth_Ratio'] = (high - low) / (close + 1e-10)
            df['Neckline_Slope'] = (close - df['Support_Level']) / (df['Resistance_Level'] - df['Support_Level'] + 1e-10)
            
            # --- 3. TARGET CREATION (Target_T5) ---
            # 1 if 5-day forward return > 1.5%, else 0
            df['Target_T5'] = np.where(close.shift(-5) > close * 1.015, 1, 0)
            
            cleaned_df = df.dropna().copy()
            all_data.append(cleaned_df)
            print(f"  └─ Processed {clean_ticker}: {len(cleaned_df)} clean rows.")
        except Exception as e:
            print(f"[WARN] Could not process {t}: {e}")
            
    if not all_data:
        return None
        
    master_df = pd.concat(all_data, ignore_index=True)
    print(f"[DATASET BUILDER SUCCESS] Full Cleaned Dataset Shape: {master_df.shape}")
    return master_df

def retrain_double_confirmation_model(master_df, output_dir="."):
    """
    Trains serialized ML classifiers for Stage-2 AI Double Confirmation.
    """
    X = master_df[FEATURE_NAMES].fillna(0.0)
    y = master_df['Target_T5'].values
    
    # Chronological Train-Test Split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[MODEL TRAINER] Training Random Forest Double Confirmation Model ({len(X_train)} train rows)...")
    model = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"[MODEL EVALUATION] Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")
    
    # Save artifacts
    model_path = os.path.join(output_dir, "best_model_acm465.joblib")
    scaler_path = os.path.join(output_dir, "best_scaler_acm465.joblib")
    features_path = os.path.join(output_dir, "best_features_acm465.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(FEATURE_NAMES, features_path)
    
    print(f"[SUCCESS] Serialized artifacts saved to {model_path}, {scaler_path}, {features_path}.")
    return acc, f1

if __name__ == "__main__":
    df_clean = build_and_clean_bist_dataset()
    if df_clean is not None:
        # Save dataset
        df_clean.to_csv("bist_ai_dataset_real_30cols.csv", index=False)
        print("[DATASET SAVED] bist_ai_dataset_real_30cols.csv successfully updated.")
        retrain_double_confirmation_model(df_clean)
