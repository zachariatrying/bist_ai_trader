"""
================================================================================
ACM 465 - CANLI TAHMIN MOTORU (Live Inference Engine)
live_inference_engine.py
================================================================================
Bu modul, egitilmis ML modelini canli/anlik borsa verisi uzerinde
tahmin yapmak icin kullanir. Herhangi bir projeye import edilebilir
veya FastAPI sunucu olarak calistirilabilir.

Kullanim:
  1. Import:  from live_inference_engine import LiveInferenceEngine
  2. API:     python -m uvicorn live_inference_engine:app --reload --port 8000

API Endpoint:
  POST /predict  ->  {"hisse": "THYAO", "karar": "AL", "guven_orani": 0.88, ...}
================================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Optional, List, Dict, Any
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================================================================
# YAPILANDIRMA
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Varsayilan dosya yollari — istege gore degistirilebilir
DEFAULT_MODEL_PATH    = os.path.join(BASE_DIR, 'best_model_acm465.joblib')
DEFAULT_SCALER_PATH   = os.path.join(BASE_DIR, 'best_scaler_acm465.joblib')
DEFAULT_FEATURES_PATH = os.path.join(BASE_DIR, 'best_features_acm465.joblib')

# Egitimde kullanilan tum ozellikler (30 kolonluk dataset tabanli)
ALL_FEATURES = [
    'RSI_14', 'MACD', 'MACD_Signal',
    'ATR_14', 'Stoch_K', 'Stoch_D',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
    'SMA_20', 'SMA_50', 'SMA_200',
    'Support_Level', 'Resistance_Level',
    'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope',
    'Expert_Signal',
    'Pat_Cup_Handle', 'Pat_Flag', 'Pat_OBO', 'Pat_TOBO', 'Pat_Yok'
]


# ==============================================================================
# CANLI TAHMIN MOTORU (LIVE INFERENCE ENGINE)
# ==============================================================================

class LiveInferenceEngine:
    """
    Egitilmis BIST ML modelini canli veri uzerinde calistiran tahmin motoru.

    Kullanim:
        engine = LiveInferenceEngine()
        engine = LiveInferenceEngine(model_path="custom_model.joblib")

        result = engine.predict_from_ohlcv("THYAO", df_with_ohlcv_columns)
        # -> {"hisse": "THYAO", "karar": "AL", "guven_orani": 0.88, ...}
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
        features_path: str = DEFAULT_FEATURES_PATH,
    ):
        """
        Model, Scaler ve ozellik listesini bellege yukler.

        Args:
            model_path:    Egitilmis model dosyasi (.joblib veya .h5)
            scaler_path:   StandardScaler dosyasi (.joblib)
            features_path: Aktif ozellik listesi (.joblib)
        """
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.model_type = "unknown"
        self._ready = False

        # --- Model Yukleme ---
        try:
            if model_path.endswith('.h5'):
                # Keras modeli
                try:
                    from tensorflow.keras.models import load_model
                    self.model = load_model(model_path)
                    self.model_type = "keras"
                except ImportError:
                    raise ImportError(
                        "Keras/TensorFlow yuklu degil. "
                        ".h5 modeli yuklemek icin 'pip install tensorflow' gerekli."
                    )
            else:
                # Sklearn / joblib modeli
                self.model = joblib.load(model_path)
                self.model_type = "sklearn"
            print(f"[LiveEngine] Model yuklendi: {os.path.basename(model_path)} ({self.model_type})")
        except Exception as e:
            print(f"[LiveEngine] HATA - Model yuklenemedi: {e}")
            return

        # --- Scaler Yukleme ---
        try:
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"[LiveEngine] Scaler yuklendi: {os.path.basename(scaler_path)}")
            else:
                print(f"[LiveEngine] UYARI - Scaler dosyasi bulunamadi: {scaler_path}")
                print(f"[LiveEngine]   -> Yeni StandardScaler olusturulacak (dikkat: uyumsuzluk riski!)")
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
        except Exception as e:
            print(f"[LiveEngine] HATA - Scaler yuklenemedi: {e}")

        # --- Ozellik Listesi Yukleme ---
        try:
            if os.path.exists(features_path):
                self.feature_cols = joblib.load(features_path)
                print(f"[LiveEngine] Ozellik listesi: {self.feature_cols}")
            else:
                # Varsayilan tam listeyi kullan
                self.feature_cols = ALL_FEATURES
                print(f"[LiveEngine] UYARI - Ozellik dosyasi bulunamadi, tam liste kullaniliyor ({len(ALL_FEATURES)} ozellik)")
        except Exception as e:
            self.feature_cols = ALL_FEATURES
            print(f"[LiveEngine] Ozellik yukleme hatasi: {e}")

        self._ready = self.model is not None and self.scaler is not None
        if self._ready:
            print(f"[LiveEngine] Motor HAZIR. Ozellik sayisi: {len(self.feature_cols)}")
        else:
            print(f"[LiveEngine] Motor HAZIR DEGIL!")

    @property
    def is_ready(self) -> bool:
        """Motorun tahmine hazir olup olmadigini dondurur."""
        return self._ready

    # ==========================================================================
    # CANLI OZELLIK MUHENDISLIGI
    # ==========================================================================

    def prepare_live_data(self, raw_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Ham OHLCV verisini modelin anlayacagi formata donusturur.
        30 kolonluk dataset ile egitilmis modele uyumlu 23 feature hesaplar:
        - RSI_14, MACD, MACD_Signal, ATR_14, Stochastic K/D
        - Bollinger Upper/Middle/Lower, SMA 20/50/200
        - Support/Resistance, Depth_Ratio, Neckline_Slope
        - Expert_Signal, Pattern Type dummies
        """
        if not self._ready:
            print("[LiveEngine] Motor hazir degil!")
            return None

        try:
            df = raw_data.copy()

            # --- Kolon kontrolu ---
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"[LiveEngine] Eksik kolonlar: {missing}")
                return None

            if len(df) < 50:
                print(f"[LiveEngine] Yetersiz veri: {len(df)} satir (min 50 gerekli)")
                return None

            # --- RSI (14 gun) ---
            delta = df['Close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            loss = loss.replace(0, 1e-10)
            df['RSI_14'] = 100 - (100 / (1 + gain / loss))

            # --- MACD ---
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # --- ATR (14 gun) ---
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR_14'] = true_range.rolling(14).mean()

            # --- Stochastic Oscillator (14 gun) ---
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14).replace(0, 1e-10)) * 100
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

            # --- Bollinger Bantlari (20 gun) ---
            df['BB_Middle'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
            df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std

            # --- SMA ---
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()

            # --- Support / Resistance Seviyeleri ---
            df['Support_Level'] = df['Low'].rolling(20).min()
            df['Resistance_Level'] = df['High'].rolling(20).max()

            # --- Volume Trend ---
            vol_sma = df['Volume'].rolling(20).mean()
            df['Volume_Trend'] = (df['Volume'] > vol_sma).astype(int)

            # --- Depth Ratio (Fiyatin destek-direnç aralığındaki konumu) ---
            sr_range = (df['Resistance_Level'] - df['Support_Level']).replace(0, 1e-10)
            df['Depth_Ratio'] = (df['Close'] - df['Support_Level']) / sr_range

            # --- Neckline Slope (SMA_20 eğimi) ---
            df['Neckline_Slope'] = df['SMA_20'].diff(5) / df['SMA_20'].shift(5).replace(0, 1e-10)

            # --- Expert Signal (Basit kural tabanli) ---
            df['Expert_Signal'] = 0
            # AL sinyali: RSI < 40 ve Stoch_K < 30 ve Close > SMA_50
            al_mask = (df['RSI_14'] < 40) & (df['Stoch_K'] < 30) & (df['Close'] > df['SMA_50'])
            df.loc[al_mask, 'Expert_Signal'] = 1
            # SAT sinyali: RSI > 70 ve Stoch_K > 80
            sat_mask = (df['RSI_14'] > 70) & (df['Stoch_K'] > 80)
            df.loc[sat_mask, 'Expert_Signal'] = -1

            # --- Pattern Type Detection (Basit) ---
            df['Pat_Cup_Handle'] = 0
            df['Pat_Flag'] = 0
            df['Pat_OBO'] = 0
            df['Pat_TOBO'] = 0
            df['Pat_Yok'] = 1  # Default: formasyon yok

            close = df['Close'].values
            high_vals = df['High'].values
            low_vals = df['Low'].values
            sma20 = df['SMA_20'].values

            for i in range(40, len(df)):
                # Bull Flag detection
                wh = high_vals[i-40:i].max()
                wl = close[i-40:i].min()
                if wl > 0 and (wh - wl) / wl >= 0.05:
                    if close[i] > wh * 0.88 and not np.isnan(sma20[i]) and close[i] > sma20[i]:
                        df.iloc[i, df.columns.get_loc('Pat_Flag')] = 1
                        df.iloc[i, df.columns.get_loc('Pat_Yok')] = 0

                # TOBO detection (basit: son 20 barda V seklinde dip)
                if i >= 20:
                    segment = close[i-20:i+1]
                    mid = len(segment) // 2
                    left_high = segment[:mid].max()
                    dip = segment[mid-3:mid+3].min()
                    right = close[i]
                    if dip < left_high * 0.95 and right > left_high * 0.98:
                        df.iloc[i, df.columns.get_loc('Pat_TOBO')] = 1
                        df.iloc[i, df.columns.get_loc('Pat_Yok')] = 0

            return df

        except Exception as e:
            print(f"[LiveEngine] Ozellik muhendisligi hatasi: {e}")
            return None

    # ==========================================================================
    # TAHMIN
    # ==========================================================================

    def predict_action(self, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Islenilmis veriyi modele sokar ve tahmin + olasilik dondurur.

        Args:
            processed_data: prepare_live_data() ciktisi

        Returns:
            dict: {"karar": "AL"/"SAT", "guven_orani": 0.88, "features_used": [...]}
        """
        if not self._ready:
            return {"hata": "Motor hazir degil"}

        try:
            # Son satiri al (en guncel veri)
            last_row = processed_data.iloc[[-1]]

            # Sadece modelin bekledigini sec
            available = [c for c in self.feature_cols if c in last_row.columns]
            missing_feats = [c for c in self.feature_cols if c not in last_row.columns]

            if missing_feats:
                print(f"[LiveEngine] UYARI - Eksik ozellikler: {missing_feats}")
                # Eksik ozellikleri 0 ile doldur
                for mf in missing_feats:
                    last_row[mf] = 0
                available = self.feature_cols

            X = last_row[self.feature_cols].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Olceklendir
            X_scaled = self.scaler.transform(X)

            # Tahmin
            if self.model_type == "keras":
                prob = float(self.model.predict(X_scaled, verbose=0).flatten()[0])
                karar = 1 if prob >= 0.5 else 0
            else:
                karar = int(self.model.predict(X_scaled)[0])
                if hasattr(self.model, 'predict_proba'):
                    prob = float(self.model.predict_proba(X_scaled)[0][1])
                else:
                    prob = float(karar)

            return {
                "karar_kod": karar,
                "karar": "AL" if karar == 1 else "SAT",
                "guven_orani": round(prob, 4),
                "features_used": self.feature_cols,
            }

        except Exception as e:
            return {"hata": str(e)}

    # ==========================================================================
    # TEK ADIMDA TAHMIN (Convenience Method)
    # ==========================================================================

    def predict_from_ohlcv(self, ticker: str, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ham OHLCV verisinden tek adimda tahmin yapar.

        Args:
            ticker:    Hisse sembolü ("THYAO")
            raw_data:  DataFrame — Open, High, Low, Close, Volume

        Returns:
            dict: Tam tahmin sonucu JSON formati
        """
        # 1. Ozellikleri hesapla
        processed = self.prepare_live_data(raw_data)
        if processed is None:
            return {
                "hisse": ticker,
                "karar": "BELIRSIZ",
                "guven_orani": 0.0,
                "hata": "Veri hazirlanamadi",
                "tetikleyici_nedenler": []
            }

        # 2. Tahmin yap
        result = self.predict_action(processed)

        if "hata" in result:
            return {
                "hisse": ticker,
                "karar": "BELIRSIZ",
                "guven_orani": 0.0,
                "hata": result["hata"],
                "tetikleyici_nedenler": []
            }

        # 3. Tetikleyici nedenleri olustur
        last = processed.iloc[-1]
        nedenler = self._generate_reasons(last, result["karar_kod"])

        return {
            "hisse": ticker,
            "karar": result["karar"],
            "guven_orani": result["guven_orani"],
            "tetikleyici_nedenler": nedenler,
            "detay": {
                "fiyat": round(float(last['Close']), 2),
                "rsi_14": round(float(last.get('RSI_14', 0)), 2),
                "macd": round(float(last.get('MACD', 0)), 4),
                "stoch_k": round(float(last.get('Stoch_K', 0)), 2),
                "stoch_d": round(float(last.get('Stoch_D', 0)), 2),
                "atr_14": round(float(last.get('ATR_14', 0)), 4),
                "depth_ratio": round(float(last.get('Depth_Ratio', 0)), 4),
                "expert_signal": int(last.get('Expert_Signal', 0)),
                "volume_trend": int(last.get('Volume_Trend', 0)),
                "pattern": "Flag" if int(last.get('Pat_Flag', 0)) else
                           "TOBO" if int(last.get('Pat_TOBO', 0)) else
                           "Cup" if int(last.get('Pat_Cup_Handle', 0)) else
                           "OBO" if int(last.get('Pat_OBO', 0)) else "Yok",
            },
            "model_tipi": self.model_type,
            "zaman": datetime.now().isoformat()
        }

    def _generate_reasons(self, row: pd.Series, karar: int) -> List[str]:
        """Tahmin icin insan-okunabilir tetikleyici nedenler uretir."""
        reasons = []
        try:
            rsi = float(row.get('RSI_14', 50))
            macd = float(row.get('MACD', 0))
            stoch_k = float(row.get('Stoch_K', 50))
            stoch_d = float(row.get('Stoch_D', 50))
            atr = float(row.get('ATR_14', 0))
            depth = float(row.get('Depth_Ratio', 0.5))
            expert = int(row.get('Expert_Signal', 0))
            vol_trend = int(row.get('Volume_Trend', 0))
            neckline = float(row.get('Neckline_Slope', 0))
            pat_flag = int(row.get('Pat_Flag', 0))
            pat_tobo = int(row.get('Pat_TOBO', 0))
            pat_cup = int(row.get('Pat_Cup_Handle', 0))

            if karar == 1:  # AL
                if rsi < 30:
                    reasons.append(f"RSI asiri satim bolgesinde ({rsi:.0f})")
                elif rsi < 45:
                    reasons.append(f"RSI dusuk bolgede ({rsi:.0f}) - toparlanma bekleniyor")
                if stoch_k < 20:
                    reasons.append(f"Stochastic asiri satim ({stoch_k:.0f})")
                if macd > 0:
                    reasons.append("MACD pozitif bolgede")
                if pat_flag == 1:
                    reasons.append("Boga Bayragi (Bull Flag) formasyonu aktif")
                if pat_tobo == 1:
                    reasons.append("TOBO formasyonu tespit edildi")
                if pat_cup == 1:
                    reasons.append("Fincan-Kulp formasyonu tespit edildi")
                if vol_trend == 1:
                    reasons.append("Hacim ortalama uzerinde")
                if depth < 0.3:
                    reasons.append(f"Fiyat destege yakin (Derinlik: {depth:.2f})")
                if neckline > 0.01:
                    reasons.append("Yukari yonlu trend egimi")
                if expert == 1:
                    reasons.append("Uzman sistemi AL sinyali verdi")
            else:  # SAT
                if rsi > 70:
                    reasons.append(f"RSI asiri alim bolgesinde ({rsi:.0f})")
                if stoch_k > 80:
                    reasons.append(f"Stochastic asiri alim ({stoch_k:.0f})")
                if macd < 0:
                    reasons.append("MACD negatif bolgede")
                if depth > 0.85:
                    reasons.append(f"Fiyat dirence yakin (Derinlik: {depth:.2f})")
                if neckline < -0.01:
                    reasons.append("Asagi yonlu trend egimi")
                if vol_trend == 0:
                    reasons.append("Hacim ortalamanin altinda")
                if expert == -1:
                    reasons.append("Uzman sistemi SAT sinyali verdi")

            if not reasons:
                reasons.append("Genel teknik gorunum degerlendirmesi")

        except Exception:
            reasons.append("Neden analizi yapilamadi")

        return reasons

    # ==========================================================================
    # TOPLU TAHMIN
    # ==========================================================================

    def predict_batch(self, tickers_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Birden fazla hisse icin toplu tahmin yapar.

        Args:
            tickers_data: {"THYAO": df_thyao, "GARAN": df_garan, ...}

        Returns:
            List[dict]: Her hisse icin tahmin sonucu listesi
        """
        results = []
        for ticker, df in tickers_data.items():
            result = self.predict_from_ohlcv(ticker, df)
            results.append(result)
        return results


# ==============================================================================
# FASTAPI UC NOKTASI (ENDPOINT)
# ==============================================================================

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    # --- Request/Response Modelleri ---
    class OHLCVBar(BaseModel):
        """Tek bir mum verisi."""
        date: str = Field(..., description="Tarih (YYYY-MM-DD)")
        open: float = Field(..., description="Acilis fiyati")
        high: float = Field(..., description="En yuksek")
        low: float = Field(..., description="En dusuk")
        close: float = Field(..., description="Kapanis fiyati")
        volume: float = Field(..., description="Islem hacmi")

    class PredictRequest(BaseModel):
        """Tahmin istegi."""
        hisse: str = Field(..., description="Hisse sembolü (THYAO, GARAN...)")
        veriler: List[OHLCVBar] = Field(..., description="Son 50+ gunluk OHLCV verisi")

    class PredictResponse(BaseModel):
        """Tahmin yaniti."""
        hisse: str
        karar: str
        guven_orani: float
        tetikleyici_nedenler: List[str]
        detay: Optional[Dict[str, Any]] = None
        model_tipi: Optional[str] = None
        zaman: Optional[str] = None
        hata: Optional[str] = None

    # --- FastAPI Uygulamasi ---
    app = FastAPI(
        title="BIST AI - Canli Tahmin Motoru",
        description="ACM 465 egitilmis ML modeli ile canli borsa tahmini",
        version="1.0.0"
    )

    # Motor baslangicta bir kere yuklenir
    _engine: Optional[LiveInferenceEngine] = None

    def _get_engine() -> LiveInferenceEngine:
        global _engine
        if _engine is None:
            _engine = LiveInferenceEngine()
        return _engine

    @app.get("/", tags=["Health"])
    async def root():
        """Sunucu saglik kontrolu."""
        engine = _get_engine()
        return {
            "durum": "aktif" if engine.is_ready else "hazir_degil",
            "model_tipi": engine.model_type,
            "ozellik_sayisi": len(engine.feature_cols) if engine.feature_cols else 0,
            "mesaj": "POST /predict endpoint'ine istek gonderin."
        }

    @app.post("/predict", response_model=PredictResponse, tags=["Tahmin"])
    async def predict(request: PredictRequest):
        """
        Canli OHLCV verisinden AL/SAT tahmini yapar.

        - **hisse**: Hisse sembolü (orn: THYAO)
        - **veriler**: Son 50+ gunun OHLCV verisi (JSON array)
        """
        engine = _get_engine()

        if not engine.is_ready:
            raise HTTPException(status_code=503, detail="Model henuz yuklenmedi")

        if len(request.veriler) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"En az 50 bar gerekli, {len(request.veriler)} bar gonderildi"
            )

        # JSON -> DataFrame
        records = []
        for bar in request.veriler:
            records.append({
                'Date': bar.date,
                'Open': bar.open,
                'High': bar.high,
                'Low': bar.low,
                'Close': bar.close,
                'Volume': bar.volume
            })

        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Tahmin
        result = engine.predict_from_ohlcv(request.hisse, df)

        return PredictResponse(**result)

    @app.post("/predict/batch", tags=["Tahmin"])
    async def predict_batch(requests: List[PredictRequest]):
        """Birden fazla hisse icin toplu tahmin."""
        engine = _get_engine()
        if not engine.is_ready:
            raise HTTPException(status_code=503, detail="Model henuz yuklenmedi")

        results = []
        for req in requests:
            records = [{'Date': b.date, 'Open': b.open, 'High': b.high,
                        'Low': b.low, 'Close': b.close, 'Volume': b.volume}
                       for b in req.veriler]
            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            result = engine.predict_from_ohlcv(req.hisse, df)
            results.append(result)

        return results

    FASTAPI_AVAILABLE = True

except ImportError:
    FASTAPI_AVAILABLE = False
    app = None
    print("[LiveEngine] FastAPI yuklu degil. API devre disi. 'pip install fastapi uvicorn'")


# ==============================================================================
# STANDALONE CALISTIRMA
# ==============================================================================

def _demo():
    """
    Demo: Parquet cache'den bir hisse yukleyip tahmin yapar.
    Motorun calistigini dogrulamak icin kullanilir.
    """
    print("\n" + "=" * 60)
    print("  LIVE INFERENCE ENGINE - DEMO")
    print("=" * 60)

    engine = LiveInferenceEngine()

    if not engine.is_ready:
        print("\nMotor baslatılamadi. Once master_bist_ai.py'yi calistirin.")
        return

    # Demo verisi: Cache'den THYAO yukle
    cache_dir = os.path.join(BASE_DIR, 'src', 'market_data_cache')
    demo_file = os.path.join(cache_dir, 'THYAO_1d_TRY.parquet')

    if os.path.exists(demo_file):
        df = pd.read_parquet(demo_file)
        if 'symbol' in df.columns:
            df.drop(columns=['symbol'], inplace=True)

        print(f"\nDemo hisse: THYAO ({len(df)} bar)")
        result = engine.predict_from_ohlcv("THYAO", df)

        print(f"\n--- TAHMIN SONUCU ---")
        print(f"  Hisse:    {result['hisse']}")
        print(f"  Karar:    {result['karar']}")
        print(f"  Guven:    %{result['guven_orani']*100:.1f}")
        print(f"  Nedenler:")
        for r in result.get('tetikleyici_nedenler', []):
            print(f"    - {r}")
        if 'detay' in result and result['detay']:
            print(f"  Detay:")
            for k, v in result['detay'].items():
                print(f"    {k}: {v}")
    else:
        print(f"\nDemo dosyasi bulunamadi: {demo_file}")
        print("Kendi verinizle test edin: engine.predict_from_ohlcv('HISSE', df)")

    # API durumu
    if FASTAPI_AVAILABLE:
        print(f"\n  FastAPI AKTIF. Baslatmak icin:")
        print(f"    python -m uvicorn live_inference_engine:app --reload --port 8000")
        print(f"    Swagger UI: http://localhost:8000/docs")
    else:
        print(f"\n  FastAPI yuklu degil. API icin: pip install fastapi uvicorn")


if __name__ == "__main__":
    _demo()
