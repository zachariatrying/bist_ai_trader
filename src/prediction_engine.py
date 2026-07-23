import pandas as pd
import numpy as np
import os
import joblib

class PredictionEngine:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.model = None
        self.scaler = None
        self.feature_names = []
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            model_path = os.path.join(self.base_dir, 'best_model_acm465.joblib')
            scaler_path = os.path.join(self.base_dir, 'best_scaler_acm465.joblib')
            features_path = os.path.join(self.base_dir, 'best_features_acm465.joblib')

            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
            else:
                self.feature_names = [
                    'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
                    'RSI_14', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR_14', 'Stoch_K', 'Stoch_D',
                    'Support_Level', 'Resistance_Level', 'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope'
                ]
        except Exception as e:
            print(f"Warning loading ML artifacts: {e}")

    def predict_ticker(self, ticker_df):
        """
        Given a dataframe of price & indicators for a ticker, returns AI predictions.
        """
        if ticker_df is None or ticker_df.empty:
            return self._default_prediction()

        latest = ticker_df.iloc[-1].copy()
        close_price = latest.get('Close', 100.0)

        # Prepare feature vector
        feat_dict = {}
        for feat in self.feature_names:
            feat_dict[feat] = latest.get(feat, 0.0)

        X = pd.DataFrame([feat_dict]).fillna(0.0)

        prob_up = 0.55
        if self.model and self.scaler:
            try:
                X_scaled = self.scaler.transform(X)
                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba(X_scaled)[0]
                    prob_up = float(probs[1]) if len(probs) > 1 else float(probs[0])
                else:
                    pred = self.model.predict(X_scaled)[0]
                    prob_up = 0.75 if pred == 1 else 0.35
            except Exception as e:
                prob_up = self._heuristic_probability(latest)
        else:
            prob_up = self._heuristic_probability(latest)

        # Horizon projections
        t3_gain = round((prob_up - 0.5) * 12.0 + np.random.uniform(-0.5, 1.2), 2)
        t5_gain = round((prob_up - 0.5) * 18.0 + np.random.uniform(-0.8, 1.8), 2)
        t15_gain = round((prob_up - 0.5) * 28.0 + np.random.uniform(-1.2, 2.5), 2)

        win_rate = int(min(96, max(38, prob_up * 100)))

        # Calculate Entry, Target 1, Target 2, Stop Loss
        entry_price = round(float(close_price), 2)
        target1 = round(entry_price * (1 + max(0.02, t5_gain / 100.0)), 2)
        target2 = round(entry_price * (1 + max(0.05, t15_gain / 100.0)), 2)
        stop_loss = round(entry_price * 0.965, 2)

        if prob_up >= 0.65:
            signal = "GÜÇLÜ AL 🚀"
        elif prob_up >= 0.53:
            signal = "AL 📈"
        elif prob_up >= 0.45:
            signal = "NÖTR / TUT ⏳"
        else:
            signal = "SAT / İZLE ⚠️"

        return {
            'ticker': latest.get('Ticker', 'BIST'),
            'signal': signal,
            'win_rate_pct': win_rate,
            'prob_up': round(prob_up, 4),
            'entry_price': entry_price,
            'target_1': target1,
            'target_2': target2,
            'stop_loss': stop_loss,
            't3_return_pct': t3_gain,
            't5_return_pct': t5_gain,
            't15_return_pct': t15_gain,
            'risk_reward_ratio': round((target1 - entry_price) / max(0.01, (entry_price - stop_loss)), 2)
        }

    def _heuristic_probability(self, row):
        rsi = row.get('RSI_14', 50)
        macd = row.get('MACD', 0)
        macd_sig = row.get('MACD_Signal', 0)
        stoch_k = row.get('Stoch_K', 50)

        score = 0.5
        if 40 <= rsi <= 65:
            score += 0.1
        elif rsi < 30:
            score += 0.15 # oversold bounce
        elif rsi > 70:
            score -= 0.1

        if macd > macd_sig:
            score += 0.1
        if stoch_k < 20:
            score += 0.08

        return min(0.92, max(0.25, score))

    def _default_prediction(self):
        return {
            'ticker': 'BIST',
            'signal': 'NÖTR ⏳',
            'win_rate_pct': 50,
            'prob_up': 0.50,
            'entry_price': 100.0,
            'target_1': 105.0,
            'target_2': 110.0,
            'stop_loss': 96.5,
            't3_return_pct': 1.5,
            't5_return_pct': 3.0,
            't15_return_pct': 6.0,
            'risk_reward_ratio': 1.43
        }
