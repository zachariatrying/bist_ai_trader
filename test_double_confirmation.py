"""
==============================================================================
BorsaNeuron - Live Triple Confirmation Terminal Test Execution
Author: İbrahim Tatar
==============================================================================
"""

import sys
import yfinance as yf
import pandas as pd
from src.double_confirmation_engine import DoubleConfirmationEngine

def run_live_triple_confirmation_test():
    engine = DoubleConfirmationEngine()
    test_tickers = ["THYAO.IS", "GARAN.IS", "EREGL.IS", "ASELS.IS", "TUPRS.IS", "FROTO.IS"]
    
    print("\n================ BORSANEURON UCLU ONAYLI SINYAL TERMINALI ================")
    
    for t in test_tickers:
        clean_name = t.replace('.IS', '')
        print(f"\n[ANALIZ] Hisse: {clean_name}...")
        try:
            df = yf.download(t, period="6mo", interval="1d", progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                
                res = engine.analyze_ticker_triple_confirmation(df, ticker_name=clean_name)
                
                print(f"  |-- Durum: {res['status']}")
                print(f"  |-- Sinyal: {res['final_signal']}")
                print(f"  |-- Formasyon: {res['pattern_name']} (Guven: %{res['pattern_confidence']})")
                print(f"  |-- Yapay Zeka Olasiligi (ML Prob): %{res['ml_prob_up_pct']}")
                print(f"  |-- Hisse Formasyon Basari Orani: %{res['stock_win_rate_pct']} ({res['compliance_status']})")
                print(f"  |-- Giris Fiyati: {res['entry_price']} TL | Hedef-1: {res['target_1']} TL | Stop: {res['stop_loss']} TL")
                print(f"  +-- Risk/Odul Orani: {res['risk_reward']}")
        except Exception as e:
            print(f"  +-- Hata: {e}")
            
    print("\n===========================================================================\n")

if __name__ == "__main__":
    run_live_triple_confirmation_test()
