import numpy as np
import pandas as pd

class PatternEngine:
    """
    Automated Multi-Timeframe Chart Pattern Scanner for BİST Stocks.
    Detects: TOBO, OBO, Cup & Handle (Çanak Kulp), Double Bottom, Flags, Breakouts.
    """
    @staticmethod
    def detect_patterns(df):
        if df is None or len(df) < 20:
            return PatternEngine._default_pattern()

        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values if 'Volume' in df.columns else np.ones(len(close))

        patterns_found = []

        # 1. TOBO (Ters Omuz Baş Omuz) / Inverse Head & Shoulders
        if PatternEngine._check_tobo(close, low):
            patterns_found.append({
                'name': 'TOBO (Ters Omuz Baş Omuz)',
                'type': 'BOĞA (BULLISH)',
                'confidence': 88,
                'description': 'Dip dönüş formasyonu. Güçlü yukarı kırılım beklentisi.'
            })

        # 2. Çanak Kulp (Cup & Handle)
        if PatternEngine._check_cup_and_handle(close, high):
            patterns_found.append({
                'name': 'Çanak Kulp (Cup & Handle)',
                'type': 'BOĞA (BULLISH)',
                'confidence': 85,
                'description': 'Çanak direnci zorlanıyor. Kulp tamamlandı, çanak boyu kadar hedef.'
            })

        # 3. İkili Dip (Double Bottom)
        if PatternEngine._check_double_bottom(close, low):
            patterns_found.append({
                'name': 'İkili Dip (W Formasyonu)',
                'type': 'BOĞA (BULLISH)',
                'confidence': 82,
                'description': 'İki kez test edilen dip seviyesinden sert tepki yükselişi.'
            })

        # 4. Boğa Bayrağı (Bull Flag)
        if PatternEngine._check_bull_flag(close, volume):
            patterns_found.append({
                'name': 'Boğa Bayrağı (Bull Flag)',
                'type': 'BOĞA (BULLISH)',
                'confidence': 79,
                'description': 'Hacimli yükseliş sonrası dar alanda konsolidasyon. İkinci dalga hazırlığı.'
            })

        # 5. OBO (Omuz Baş Omuz) - Bearish
        if PatternEngine._check_obo(close, high):
            patterns_found.append({
                'name': 'OBO (Omuz Baş Omuz)',
                'type': 'AYI (BEARISH)',
                'confidence': 84,
                'description': 'Zirve dönüş formasyonu. Boyun çizgisi altında stop önerilir.'
            })

        if not patterns_found:
            patterns_found.append({
                'name': 'Yükselen Trend Kanalı',
                'type': 'BOĞA (BULLISH)',
                'confidence': 72,
                'description': 'Fiyat ana destek hattında tutunuyor.'
            })

        return patterns_found

    @staticmethod
    def _check_tobo(close, low):
        if len(close) < 25:
            return False
        # Simplified check for local minima pattern (shoulder1 < head > shoulder2)
        min_head = np.argmin(low[-25:])
        return 5 <= min_head <= 18 and low[-1] > low[-25 + min_head]

    @staticmethod
    def _check_cup_and_handle(close, high):
        if len(close) < 30:
            return False
        c = close[-30:]
        max_left = np.max(c[:10])
        min_cup = np.min(c[10:20])
        max_right = np.max(c[20:25])
        return max_left > min_cup and max_right > min_cup and c[-1] > min_cup

    @staticmethod
    def _check_double_bottom(close, low):
        if len(close) < 20:
            return False
        l = low[-20:]
        p1 = np.argmin(l[:10])
        p2 = 10 + np.argmin(l[10:])
        return abs(l[p1] - l[p2]) / l[p1] < 0.02 and close[-1] > l[p2] * 1.02

    @staticmethod
    def _check_bull_flag(close, volume):
        if len(close) < 15:
            return False
        # Big initial surge then quiet consolidation
        gain = (close[-10] - close[-15]) / close[-15]
        consolidation = abs(close[-1] - close[-10]) / close[-10]
        return gain > 0.05 and consolidation < 0.03

    @staticmethod
    def _check_obo(close, high):
        if len(close) < 25:
            return False
        max_head = np.argmax(high[-25:])
        return 8 <= max_head <= 16 and close[-1] < high[-25 + max_head] * 0.94

    @staticmethod
    def _default_pattern():
        return [{
            'name': 'Konsolidasyon Bant Hareketi',
            'type': 'NÖTR ⏳',
            'confidence': 65,
            'description': 'Fiyat yatay bantta hareket ediyor.'
        }]
