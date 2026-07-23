"""
==============================================================================
BorsaNeuron - Stock-Specific Historical Pattern Win-Rate Evaluator
Author: İbrahim Tatar
Description: Evaluates stock-specific compliance and historical win-rates
             for detected technical patterns across BIST equities.
==============================================================================
"""

import numpy as np
import pandas as pd

class StockPatternWinRateEngine:
    """
    Computes ticker-specific historical win-rate compliance for given chart patterns.
    """
    # Baseline historical win-rate table per BIST stock & pattern type
    DEFAULT_WIN_RATES = {
        'THYAO': {'TOBO': 74.5, 'Çanak Kulp': 71.0, 'İkili Dip': 68.5, 'Boğa Bayrağı': 78.0, 'OBO': 42.0},
        'GARAN': {'TOBO': 78.0, 'Çanak Kulp': 76.5, 'İkili Dip': 72.0, 'Boğa Bayrağı': 70.0, 'OBO': 38.0},
        'EREGL': {'TOBO': 81.2, 'Çanak Kulp': 79.0, 'İkili Dip': 75.0, 'Boğa Bayrağı': 73.5, 'OBO': 35.0},
        'ASELS': {'TOBO': 76.0, 'Çanak Kulp': 74.0, 'İkili Dip': 70.0, 'Boğa Bayrağı': 80.5, 'OBO': 40.0},
        'TUPRS': {'TOBO': 82.5, 'Çanak Kulp': 77.0, 'İkili Dip': 74.0, 'Boğa Bayrağı': 76.0, 'OBO': 36.0},
        'SISE':  {'TOBO': 72.0, 'Çanak Kulp': 70.5, 'İkili Dip': 69.0, 'Boğa Bayrağı': 71.0, 'OBO': 44.0},
        'BIMAS': {'TOBO': 79.0, 'Çanak Kulp': 80.0, 'İkili Dip': 76.0, 'Boğa Bayrağı': 77.5, 'OBO': 39.0},
        'FROTO': {'TOBO': 83.0, 'Çanak Kulp': 78.5, 'İkili Dip': 75.5, 'Boğa Bayrağı': 82.0, 'OBO': 35.0}
    }

    @staticmethod
    def get_stock_pattern_win_rate(ticker, pattern_name):
        """
        Returns stock-specific historical win-rate (%) for a specific chart pattern.
        """
        clean_ticker = ticker.replace('.IS', '').upper()
        
        # Clean pattern key name
        pat_key = 'TOBO'
        if 'Çanak' in pattern_name or 'Cup' in pattern_name:
            pat_key = 'Çanak Kulp'
        elif 'İkili Dip' in pattern_name or 'W' in pattern_name:
            pat_key = 'İkili Dip'
        elif 'Bayrak' in pattern_name or 'Flag' in pattern_name:
            pat_key = 'Boğa Bayrağı'
        elif 'OBO' in pattern_name and 'TOBO' not in pattern_name:
            pat_key = 'OBO'

        stock_table = StockPatternWinRateEngine.DEFAULT_WIN_RATES.get(clean_ticker)
        
        if stock_table and pat_key in stock_table:
            win_rate = stock_table[pat_key]
        else:
            win_rate = 68.5 if pat_key != 'OBO' else 40.0

        # Compliance status
        if win_rate >= 75.0:
            compliance_status = "YUKSEK UYUM [Tarihsel Basari Yüksek]"
            is_approved = True
        elif win_rate >= 60.0:
            compliance_status = "ORTA UYUM [Kabul Edilebilir Basari]"
            is_approved = True
        else:
            compliance_status = "DUSUK UYUM [Tarihsel Basarisizlik Riski]"
            is_approved = False

        return {
            'ticker': clean_ticker,
            'pattern_key': pat_key,
            'stock_win_rate_pct': float(win_rate),
            'compliance_status': compliance_status,
            'is_stage3_approved': is_approved
        }
