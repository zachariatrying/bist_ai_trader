"""
==============================================================================
BorsaNeuron - Automated Model Retraining & Dynamic Diagnostic Pipeline
Author: İbrahim Tatar
Description: Downloads updated BIST market data, computes technical indicators,
             trains machine learning classifiers (XGBoost / Random Forest / MLP),
             and updates serialized model & scaler joblib artifacts.
==============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

class BorsaNeuronRetrainPipeline:
    def __init__(self, data_path=None, model_path="best_model_acm465.joblib", scaler_path="best_scaler_acm465.joblib"):
        self.data_path = data_path or "bist_ai_dataset_real_30cols.csv"
        self.model_path = model_path
        self.scaler_path = scaler_path

    def prepare_dataset_from_live_data(self, tickers=None, period="2y"):
        """
        Downloads live yfinance data for target BIST tickers and computes features.
        """
        if tickers is None:
            tickers = ["THYAO.IS", "GARAN.IS", "EREGL.IS", "AKBNK.IS", "SISE.IS", "TUPRS.IS", "KCHOL.IS", "ASELS.IS"]
        
        all_dfs = []
        print(f"[RETRAIN] Downloading live market data for {len(tickers)} tickers...")
        
        for t in tickers:
            try:
                df = yf.download(t, period=period, interval="1d", progress=False)
                if df.empty or len(df) < 60:
                    continue
                    
                # Clean MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                df['Ticker'] = t.replace('.IS', '')
                df['Close_Price'] = df['Close']
                df['Volume_Val'] = df['Volume']
                
                # Technical Indicators
                df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
                df['SMA_10'] = df['Close'].rolling(10).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                df['RSI_14'] = self._calculate_rsi(df['Close'], 14)
                df['Vol_14'] = df['Log_Return'].rolling(14).std()
                
                # Target: 5-day future closing price change
                df['Target_T5'] = np.where(df['Close'].shift(-5) > df['Close'], 1, 0)
                
                all_dfs.append(df.dropna())
            except Exception as e:
                print(f"[WARN] Failed to fetch data for {t}: {e}")
                
        if not all_dfs:
            return None
            
        full_df = pd.concat(all_dfs, ignore_index=True)
        print(f"[RETRAIN] Dataset constructed: {full_df.shape}")
        return full_df

    def train_and_save_model(self, df=None):
        """
        Trains RandomForest/XGBoost classifier and serializes model and scaler.
        """
        if df is None:
            if os.path.exists(self.data_path):
                print(f"[RETRAIN] Loading local dataset from {self.data_path}...")
                df = pd.read_csv(self.data_path)
                if len(df) > 50000:
                    df = df.sample(n=50000, random_state=42)
            else:
                df = self.prepare_dataset_from_live_data()
                
        if df is None or df.empty:
            return False, "Eğitim için geçerli veri seti bulunamadı."
            
        # Select numeric feature columns
        feature_cols = [c for c in df.columns if c not in ['Target_T5', 'Target', 'Ticker', 'Date', 'Symbol'] and df[c].dtype != 'O']
        
        # Determine Target Column
        target_col = 'Target_T5' if 'Target_T5' in df.columns else 'Target'
        
        X = df[feature_cols].fillna(0)
        y = df[target_col].values
        
        # Train / Test Split (Chronological 80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Classifier
        print(f"[RETRAIN] Training Random Forest model on {len(X_train)} samples...")
        model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Predictions & Validation
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"[RETRAIN SUCCESS] Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")
        
        # Save serialized artifacts
        joblib.dump(model, self.model_path)
        joblib.dump(scaler, self.scaler_path)
        
        diagnostics = {
            'accuracy_pct': round(acc * 100, 2),
            'f1_score': round(f1, 4),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(feature_cols),
            'model_saved': self.model_path,
            'scaler_saved': self.scaler_path
        }
        
        return True, diagnostics

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    pipeline = BorsaNeuronRetrainPipeline()
    success, res = pipeline.train_and_save_model()
    print("Retrain Execution Result:", success, res)
