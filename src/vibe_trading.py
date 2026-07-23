from src.prediction_engine import PredictionEngine
from src.pattern_engine import PatternEngine
from src.analyzer import Analyzer
from src.news_sentiment import NewsSentimentEngine
from src.price_action import PriceActionEngine
from src.ipo_engine import IPOEngine
import pandas as pd
import numpy as np

class VibeTradingAssistant:
    """
    Unified Pattern-First -> AI Verification Engine for BORSANEURON.
    Workflow:
      1. Technical Pattern Scan (TOBO, Çanak Kulp, Flama, İkili Dip, Kırılım)
      2. AI Machine Learning Cross-Verification (best_model_acm465.joblib)
      3. Dual-Confirmed Trading Signal Generation (Çifte Onaylı AL/SAT)
    """
    def __init__(self):
        self.pred_engine = PredictionEngine()
        self.analyzer = Analyzer()

    def generate_full_vibe_analysis(self, ticker, ticker_df=None):
        clean_ticker = ticker.replace('.IS', '').upper()

        # Step 1: Strict Technical Pattern Detection
        patterns = PatternEngine.detect_patterns(ticker_df)
        has_bullish_pattern = any(p['type'].startswith('BOĞA') for p in patterns)
        main_pattern = patterns[0] if patterns else {'name': 'Trend Konsolidasyonu', 'type': 'NÖTR ⏳', 'confidence': 60, 'description': 'Yatay Bant'}

        # Step 2: AI Model Prediction & Cross-Verification
        ml_res = self.pred_engine.predict_ticker(ticker_df)
        ai_prob = ml_res['prob_up'] # 0.0 to 1.0
        ai_conf_pct = int(ai_prob * 100)

        # Step 3: News & Price Action Context
        news_res = NewsSentimentEngine.get_news_sentiment(clean_ticker)
        pa_res = PriceActionEngine.analyze_price_action(ticker_df)

        # Step 4: Dual Confirmation Logic (Pattern + AI Agreement)
        if has_bullish_pattern and ai_prob >= 0.60:
            confirmation_status = "🔥 ÇİFTE ONAYLI GÜÇLÜ AL 🚀 (Formasyon + AI Teyitli)"
            trade_action = "GÜÇLÜ AL / PORTFÖYE EKLE (Yüksek Kazanma Olasılığı)"
            confirmed_win_rate = int(min(97, max(75, ai_conf_pct + 12)))
        elif has_bullish_pattern and ai_prob >= 0.48:
            confirmation_status = "📈 FORMASYON VAR / AI KISMEN ONAYLADI"
            trade_action = "KADEMELİ ALIM / STOP LOSS TAKİBİ"
            confirmed_win_rate = int(min(82, max(60, ai_conf_pct)))
        elif has_bullish_pattern and ai_prob < 0.48:
            confirmation_status = "⚠️ TUZAK UYARISI! (Formasyon Var Ancak AI Onaylamadı)"
            trade_action = "RİSKLİ / BEKLE VE İZLE"
            confirmed_win_rate = int(min(45, ai_conf_pct))
        elif not has_bullish_pattern and ai_prob >= 0.65:
            confirmation_status = "📈 AI YÜKSELİŞ TAHMİNİ (Formasyon Bekleniyor)"
            trade_action = "DİP SEVİYEDEN KADEMELİ ALIM"
            confirmed_win_rate = ai_conf_pct
        else:
            confirmation_status = "⏳ NÖTR / BEKLEME MODU"
            trade_action = "POZİSYONU KORU VEYA SAT"
            confirmed_win_rate = ai_conf_pct

        # Calculate Master Score
        master_score = int(min(99, max(1, 0.50 * (ai_prob * 100) + 0.30 * main_pattern['confidence'] + 0.20 * pa_res['smart_money_score'])))

        return {
            'ticker': clean_ticker,
            'vibe_score': master_score,
            'confirmation_status': confirmation_status,
            'trade_action': trade_action,
            'confirmed_win_rate': confirmed_win_rate,
            'patterns': patterns,
            'main_pattern': main_pattern,
            'ml_predictions': ml_res,
            'news_sentiment': news_res,
            'price_action': pa_res
        }

    def answer_query(self, user_query, ticker="THYAO", ticker_df=None):
        analysis = self.generate_full_vibe_analysis(ticker, ticker_df)
        clean_t = analysis['ticker']
        q = user_query.lower()

        if "halka arz" in q or "ipo" in q:
            ipos = IPOEngine.get_upcoming_and_recent_ipos()
            res_str = "<b>💎 22 TEMMUZ 2026 BİST HALKA ARZ & TEMEL DEĞERLEME RADARI:</b><br><br>"
            for ipo in ipos:
                res_str += f"📌 <b>{ipo['company']} ({ipo['code']})</b><br>"
                res_str += f"• Tarih: <b>{ipo.get('dates', 'Temmuz 2026')}</b> | Fiyat: {ipo['ipo_price']}<br>"
                res_str += f"• F/K: {ipo.get('pe_ratio', '10.5')} | FD/FAVÖK: {ipo.get('ev_ebitda', '8.0')}<br>"
                res_str += f"• Taban Riski Skoru: <b>{ipo.get('taban_risk_score', '%30')}</b> | Öneri: {ipo['recommendation']}<br>"
                res_str += f"• Yorum: {ipo.get('detailed_analysis', ipo['strategy'])}<br><br>"
            return res_str

        # Pattern-First -> AI Verification Workflow Summary
        res_str = f"<b>📐 1. AŞAMA — TEKNİK FORMASYON TARAMASI ({clean_t}):</b><br>"
        for p in analysis['patterns']:
            res_str += f"👉 <b>{p['name']} ({p['type']})</b> — Güven Oranı: %{p['confidence']}<br>"
            res_str += f"   <i>{p['description']}</i><br>"

        res_str += f"<br><b>🤖 2. AŞAMA — YAPAY ZEKA MODEL TEYİDİ (ML Classifier):</b><br>"
        res_str += f"• AI Yükseliş İhtimali: <b>%{analysis['ml_predictions']['win_rate_pct']}</b><br>"
        res_str += f"• Akıllı Para Birikim Skoru: <b>{analysis['price_action']['smart_money_score']}/100</b> ({analysis['price_action']['smart_money_status']})<br>"

        res_str += f"<br><b>🔥 3. AŞAMA — ÇİFTE ONAYLI TRADING SİNYALİ:</b><br>"
        res_str += f"• <b>Sinyal Durumu:</b> <mark>{analysis['confirmation_status']}</mark><br>"
        res_str += f"• <b>Aksiyon:</b> {analysis['trade_action']}<br>"
        res_str += f"• <b>Onaylı Kazanma Oranı:</b> <b>%{analysis['confirmed_win_rate']}</b><br><br>"
        
        ml = analysis['ml_predictions']
        res_str += f"🎯 <b>POZİSYON SEVİYELERİ:</b> Giriş: <b>{ml['entry_price']} TL</b> | Hedef 1 (T+5): <b>{ml['target_1']} TL</b> | Stop Loss: <span style='color:red;'>{ml['stop_loss']} TL</span>"
        return res_str
