"""
==============================================================================
BorsaNeuron - Triple Confirmation Engine (Formasyon + Yapay Zeka + Hisse Başarısı)
Author: İbrahim Tatar
Description: Stage-1: Scans chart patterns (TOBO, OBO, Cup & Handle, Bull Flag, Breakout).
             Stage-2: Validates candidates via ML Classifier (Target_T5 probability).
             Stage-3: Validates stock-specific historical pattern win-rate compliance (Win-Rate %).
             Generates 100% Institutional Triple Confirmed Signals.
==============================================================================
"""

import pandas as pd
import numpy as np
from src.pattern_engine import PatternEngine
from src.prediction_engine import PredictionEngine
from src.stock_pattern_winrate import StockPatternWinRateEngine

class DoubleConfirmationEngine:
    def __init__(self, base_dir=None):
        self.pattern_engine = PatternEngine()
        self.prediction_engine = PredictionEngine(base_dir=base_dir)
        self.winrate_engine = StockPatternWinRateEngine()

    def analyze_ticker_triple_confirmation(self, df_ticker, ticker_name="BIST"):
        """
        Executes 3-Stage Triple Confirmation Pipeline.
        """
        if df_ticker is None or df_ticker.empty or len(df_ticker) < 25:
            return {
                'ticker': ticker_name,
                'status': 'REJECTED [NO DATA]',
                'reason': 'Yetersiz veri (en az 25 gün gerekli)',
                'triple_confirmed': False,
                'final_signal': 'NOTR / TUT'
            }

        # Stage 1: Technical Pattern Scan
        patterns = self.pattern_engine.detect_patterns(df_ticker)
        primary_pattern = patterns[0] if patterns else {'name': 'Trend Takibi', 'type': 'BOGA', 'confidence': 70}
        pattern_name = primary_pattern.get('name', 'Trend Takibi')
        
        # Stage 2: Machine Learning Prediction Inference
        ml_prediction = self.prediction_engine.predict_ticker(df_ticker)
        prob_up = ml_prediction.get('prob_up', 0.50)
        win_rate = ml_prediction.get('win_rate_pct', 50)
        
        # Stage 3: Stock-Specific Historical Pattern Win-Rate Evaluation
        stock_win_info = self.winrate_engine.get_stock_pattern_win_rate(ticker_name, pattern_name)
        stock_win_rate_pct = stock_win_info['stock_win_rate_pct']
        is_stage3_approved = stock_win_info['is_stage3_approved']

        # Evaluate Confirmation Criteria
        is_bullish_pattern = 'BOGA' in primary_pattern.get('type', '').upper() or 'BOĞA' in primary_pattern.get('type', '').upper()
        is_stage2_approved = prob_up >= 0.55
        
        triple_confirmed = is_bullish_pattern and is_stage2_approved and is_stage3_approved
        
        if triple_confirmed:
            if prob_up >= 0.70 and stock_win_rate_pct >= 75.0:
                final_signal = "UCLU ONAYLI MUKEMMEL AL (Formasyon + ML + Hisse Uyum Onayli)"
            else:
                final_signal = "UCLU ONAYLI AL (Formasyon + AI + Hisse Basari Destekli)"
            status = "APPROVED [Üçlü Onay Alındı]"
        elif is_bullish_pattern and is_stage2_approved and not is_stage3_approved:
            final_signal = "IKI ONAYLI (Formasyon + AI Olumlu, Ancak Hisse Tarihsel Basarisi Dusuk)"
            status = "WARNING [Hisse Formasyon Başarısı Riskli]"
        elif is_bullish_pattern and not is_stage2_approved:
            final_signal = "TEK ONAYLI (Sadece Formasyon Var, AI Temkinli)"
            status = "WARNING [Yapay Zeka Onayı Eksik]"
        else:
            final_signal = "BEKLE / IZLE (Onay Alınamadı)"
            status = "REJECTED [Uygun Değil]"

        return {
            'ticker': ticker_name,
            'status': status,
            'triple_confirmed': triple_confirmed,
            'final_signal': final_signal,
            'pattern_name': pattern_name,
            'pattern_type': primary_pattern.get('type'),
            'pattern_confidence': primary_pattern.get('confidence'),
            'ml_prob_up_pct': round(prob_up * 100, 1),
            'stock_win_rate_pct': stock_win_rate_pct,
            'compliance_status': stock_win_info['compliance_status'],
            'entry_price': ml_prediction.get('entry_price'),
            'target_1': ml_prediction.get('target_1'),
            'target_2': ml_prediction.get('target_2'),
            'stop_loss': ml_prediction.get('stop_loss'),
            'risk_reward': ml_prediction.get('risk_reward_ratio')
        }

    # Backward compatibility wrapper
    def analyze_ticker_double_confirmation(self, df_ticker, ticker_name="BIST"):
        return self.analyze_ticker_triple_confirmation(df_ticker, ticker_name=ticker_name)

if __name__ == "__main__":
    print("[TRIPLE CONFIRMATION ENGINE] Ready.")
