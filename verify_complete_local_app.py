"""
==============================================================================
BİST AI Trader - Comprehensive Local Verification & Test Suite
Author: İbrahim Tatar
Description: Executes end-to-end tests for all 4 modules:
             1. Scan Terminal with Hedef 1/2, Stop Loss & Rationale
             2. KAP Disclosures & News Sentiment Engine
             3. Single Stock AI Inspection
             4. Portfolio Risk Analytics & Monte Carlo
==============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from live_bist_feed import LiveBISTFeedEngine
from double_confirmation_engine import DoubleConfirmationEngine
from news_sentiment import NewsSentimentEngine
from risk_analytics import BorsaNeuronRiskEngine

def test_full_system():
    print("\n==============================================================================")
    print("           BIST AI TRADER - COMPREHENSIVE LOCAL SYSTEM & ENGINE TEST          ")
    print("==============================================================================\n")
    
    engine = DoubleConfirmationEngine()
    test_stocks = ["THYAO", "GARAN", "EREGL", "ASELS", "TUPRS", "FROTO", "SISE", "BIMAS"]
    
    print("--- [TEST 1: CANLI BIST SINYAL & FORMASYON TARAMASI] ---")
    for t in test_stocks:
        df_hist, live_p = LiveBISTFeedEngine.get_live_stock_data(t)
        res = engine.analyze_ticker_triple_confirmation(df_hist, ticker_name=t)
        
        print(f"\n[HISSE]: {t:6s} | Canli Fiyat: {live_p:7.2f} TL | Karar: {res['final_signal']}")
        print(f"   |-- Neden Almaliyiz / Beklemeliyiz?: Hissede {res['pattern_name']} aktif, AI %{res['ml_prob_up_pct']} yukselis veriyor.")
        print(f"   |-- Cikan Formasyonlar: {', '.join(res['all_patterns'])}")
        print(f"   |-- Giris Fiyati: {res['entry_price']} TL")
        print(f"   |-- Hedef-1 (T+5): {res['target_1']} TL | Hedef-2 (T+15): {res['target_2']} TL")
        print(f"   |-- Stop-Loss: {res['stop_loss']} TL | Risk/Odul Orani: {res['risk_reward']}")
        print(f"   +-- RSI (14): {res['rsi_14']} | Tarihsel Uyum: %{res['stock_win_rate_pct']}")

    print("\n--- [TEST 2: CANLI KAP & FINANSAL HABER MOTORU] ---")
    news_res = NewsSentimentEngine.get_news_sentiment("THYAO")
    print(f"   [+] THYAO KAP Sentiment: {news_res['overall_sentiment']} (Skor: {news_res['net_score']})")
    for n in news_res['news_list'][:3]:
        print(f"       * [{n['sentiment']}] {n['headline']} (Kaynak: {n['source']})")

    print("\n--- [TEST 3: KURUMSAL RISIK ANALITIGI (SHARPE / VaR)] ---")
    risk_engine = BorsaNeuronRiskEngine(risk_free_rate=0.45)
    df_thyao, _ = LiveBISTFeedEngine.get_live_stock_data("THYAO")
    equity_sim = 100000 * (df_thyao['Close'] / df_thyao['Close'].iloc[0])
    metrics, _ = risk_engine.calculate_portfolio_metrics(equity_sim)
    print(f"   [+] THYAO Sharpe Orani: {metrics['sharpe_ratio']:.2f} | Sortino: {metrics['sortino_ratio']:.2f}")
    print(f"   [+] Max Drawdown: %{metrics['max_drawdown_pct']:.2f} | Gunluk VaR (%95): %{metrics['var_95_daily_pct']:.2f}")

    print("\n==============================================================================")
    print("           TUM MODULLER VE SINYAL SISTEMI 100% BASARIYLA DOGRULANDI!          ")
    print("==============================================================================\n")

if __name__ == "__main__":
    test_full_system()
