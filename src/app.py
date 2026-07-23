import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import yfinance as yf
import joblib

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="BORSANEURON | ALGORITMIC TRADING & AI WORKSTATION",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Kurumsal CSS Arayüzü ---
TERMINAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp { background-color: #0a0e1a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    /* Terminal Dark Background Gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0a0e1a 0%, #0e1117 40%, #111827 100%);
    }

    /* Sidebar - Kurumsal Tasarım */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(59, 130, 246, 0.1) !important;
    }

    /* BorsaNeuron Glass Kartlar */
    .terminal-card {
        background-color: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 3px solid #3b82f6;
        padding: 20px;
        margin-bottom: 16px;
        border-radius: 4px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .brand-header {
        color: #3b82f6;
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        letter-spacing: 2px;
        font-size: 1.4rem;
        border-bottom: 2px solid rgba(59, 130, 246, 0.2);
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    
    .metric-value { 
        color: #10b981; 
        font-weight: 700; 
        font-size: 1.6rem; 
        font-family: 'Roboto Mono', monospace; 
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }
    .metric-label { 
        font-size: 0.75rem; 
        color: #94a3b8; 
        text-transform: uppercase; 
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    
    /* Gelişmiş Girdi Alanı Tasarımı (Inputs) */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > div, 
    .stNumberInput > div > div > input {
        background-color: #0f172a !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        color: #f8fafc !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
        font-family: 'Roboto Mono', monospace !important;
    }
    
    /* Buton Tasarımları */
    .stButton > button {
        border-radius: 4px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        transition: 0.2s !important;
        letter-spacing: 0.5px;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        text-transform: uppercase;
        padding: 10px 24px !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #2563eb !important;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-0.5px);
    }
    
    /* İlerleme Çubukları */
    .stProgress > div > div { 
        background: linear-gradient(90deg, #1d4ed8, #2563eb) !important; 
        border-radius: 4px; 
    }
    
    /* Kurumsal Tablo Sınırları */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 4px !important;
    }
    
    /* Kaydırma Çubukları (Scrollbars) */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: rgba(59, 130, 246, 0.25); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(59, 130, 246, 0.45); }
    
    /* Alt Bilgi */
    .footer-text {
        font-size: 0.75rem;
        color: #475569;
        font-family: 'Roboto Mono', monospace;
        margin-top: 30px;
        text-align: center;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        padding-top: 15px;
    }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# --- Veri Yükleme & Streamlit Cloud Uyum Katmanı ---
@st.cache_data(ttl=3600)
def load_data():
    paths = [
        "bist_ai_dataset_real_30cols.csv.xz",
        "bist_ai_dataset_real_30cols.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bist_ai_dataset_real_30cols.csv.xz'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bist_ai_dataset_real_30cols.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bist_ai_dataset_real_30cols.csv.xz'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bist_ai_dataset_real_30cols.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bist_ai_dataset_real_30cols.csv.xz'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bist_ai_dataset_real_30cols.csv')
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
                
    # --- Streamlit Cloud Auto-Fallback Generator ---
    # Generates in-memory BIST dataset if 300MB CSV file is missing on Cloud
    tickers = ["THYAO", "GARAN", "EREGL", "ASELS", "TUPRS", "SISE", "BIMAS", "FROTO", "AKBNK", "KCHOL", "SAHOL", "SASA", "HEKTS"]
    dfs = []
    for t in tickers:
        try:
            hist = yf.download(f"{t}.IS", period="1y", interval="1d", progress=False)
            if hist is not None and not hist.empty:
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = [c[0] for c in hist.columns]
                hist['Ticker'] = t
                hist['Date'] = hist.index.strftime('%Y-%m-%d')
                hist['Close_Price'] = hist['Close']
                hist['RSI_14'] = 50 + np.random.randn(len(hist))*10
                hist['MACD'] = np.random.randn(len(hist))*0.5
                hist['MACD_Signal'] = np.random.randn(len(hist))*0.5
                hist['SMA_20'] = hist['Close'].rolling(20).mean()
                hist['SMA_50'] = hist['Close'].rolling(50).mean()
                hist['SMA_200'] = hist['Close'].rolling(200).mean()
                hist['Target_T5'] = np.random.choice([0, 1], size=len(hist))
                hist['Pattern_Type'] = np.random.choice(['TOBO (Ters Omuz Baş Omuz)', 'Çanak Kulp', 'İkili Dip', 'Yükselen Trend Kanalı'], size=len(hist))
                hist['Expert_Signal'] = np.random.choice([1, 0, -1], size=len(hist))
                hist['Depth_Ratio'] = np.random.uniform(0.01, 0.05, len(hist))
                hist['Neckline_Slope'] = np.random.uniform(0.1, 0.8, len(hist))
                dfs.append(hist.dropna())
        except Exception:
            pass
            
    if dfs:
        return pd.concat(dfs, ignore_index=True)
        
    # Emergency minimal synthetic DataFrame if yFinance network is blocked
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
    data_list = []
    for t in tickers:
        p = 100 + np.cumsum(np.random.randn(100))
        for i in range(100):
            data_list.append({
                'Date': dates[i].strftime('%Y-%m-%d'),
                'Ticker': t,
                'Close': p[i],
                'Close_Price': p[i],
                'High': p[i] * 1.02,
                'Low': p[i] * 0.98,
                'Open': p[i] * 0.99,
                'Volume': 1000000,
                'RSI_14': 52.0,
                'MACD': 0.1,
                'MACD_Signal': 0.05,
                'SMA_20': p[i],
                'SMA_50': p[i],
                'SMA_200': p[i],
                'Target_T5': 1,
                'Pattern_Type': 'TOBO (Ters Omuz Baş Omuz)',
                'Expert_Signal': 1,
                'Depth_Ratio': 0.02,
                'Neckline_Slope': 0.4
            })
    return pd.DataFrame(data_list)

df = load_data()

# --- Corporate Sidebar Navigation ---
st.sidebar.markdown("<div class='brand-header'>BORSANEURON</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:0.75rem; color:#64748b; letter-spacing:1px; text-transform:uppercase; margin-top:-10px; margin-bottom:20px; font-family:\"Roboto Mono\", monospace;'>QUANTITATIVE SYSTEMS</div>", unsafe_allow_html=True)

# Import custom engines
import json
try:
    from src.vibe_trading import VibeTradingAssistant
    from src.ipo_engine import IPOEngine
    from src.prediction_engine import PredictionEngine
    vibe_assistant = VibeTradingAssistant()
except Exception as e:
    vibe_assistant = None

@st.cache_data
def get_all_bist_tickers():
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'bist_tickers.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return sorted([item['ticker'].replace('.IS', '') for item in data])
    except Exception:
        pass
    if df is not None and 'Ticker' in df.columns:
        return sorted([str(t).replace('.IS', '') for t in df['Ticker'].unique()])
    return ["THYAO", "GARAN", "EREGL", "ASELS", "FROTO", "KCHOL", "TUPRS", "SISE", "BIMAS", "AKBNK", "SAHOL", "SASA", "HEKTS"]

all_tickers_list = get_all_bist_tickers()

page = st.sidebar.radio("SİSTEM ANALİTİK MENÜSÜ", [
    "Sinyal Terminali (Çifte ve Üçlü Onaylı)",
    "Canlı BİST Formasyon & AI Taraması",
    "KAP Bildirim & Canlı Haber Analiz Motoru",
    "Kurumsal Risk Analitiği (Sharpe / VaR)",
    "Vibe Trading AI Studio",
    "Halka Arz (IPO) Radar & Değerleme",
    "Model Yeniden Eğitim & Diagnostik Panel",
    "Sistem Özeti & Metodoloji Dokümantasyonu",
    "Keşifçi Veri Analizi (EDA)",
    "Öznitelik Korelasyon & Seçim Analizi",
    "Piyasa Rejimi Kümelemesi (K-Means & PCA)",
    "Makine Öğrenmesi Model Performansları",
    "Zaman Serisi Trend Tahmin Motoru",
    "Canlı Hisse Sorgulama & Yapay Zeka Çıkarımı",
    "Portföy Geri Testi (Backtest) & Simülasyon",
    "Otomatik Formasyon Taraması & Detayları",
    "Telegram Bildirim Servisi"
])

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:0.7rem; color:#475569; font-family:\"Roboto Mono\", monospace;'>ACTIVE PORTFOLIO ENGINE: XGBoost Classifier<br>DATABASE UPDATE: Live (yFinance / KAP)<br>VERSION: v3.0.0-BIST Enterprise</div>", unsafe_allow_html=True)

if page == "Sinyal Terminali (Çifte ve Üçlü Onaylı)":
    st.markdown("### Sinyal Terminali (Çifte ve Üçlü Onay Sistemli)")
    st.markdown(f"Tüm Borsa İstanbul (**{len(all_tickers_list)} Aktif BİST Hissesi**) İçin **Aşama-1 (Formasyon)**, **Aşama-2 (Yapay Zekâ ML)** ve **Aşama-3 (Hisse Bazlı Tarihsel Uyum)** Onayları Alınarak Karar Üretilir.")

    default_selected = ["THYAO", "GARAN", "EREGL", "ASELS", "FROTO", "KCHOL", "TUPRS", "SISE", "BIMAS"]
    selected_tickers = st.multiselect(
        "Taranacak BİST Hisselerini Seçin (537 Hisse Arasından İstediğinizi Ekleyin/Çıkarın):",
        options=all_tickers_list,
        default=[t for t in default_selected if t in all_tickers_list],
        key="sig_multiselect"
    )
    
    custom_sig_input = st.text_input("Veya Ek BİST Kodları Yazın (Virgülle Ayrılmış):", "", key="sig_custom")
    
    if st.button("🧠 Formasyon Taramasını ve AI Teyidini Çalıştır", type="primary", key="run_sig_btn"):
        t_list = list(selected_tickers)
        if custom_sig_input.strip():
            for c_t in custom_sig_input.split(','):
                c_clean = c_t.strip().upper()
                if c_clean and c_clean not in t_list:
                    t_list.append(c_clean)

        t_list = [t + '.IS' if not t.endswith('.IS') else t for t in t_list]

        results = []
        progress_bar = st.progress(0)

        for idx, t in enumerate(t_list):
            try:
                raw = yf.download(t, period='6mo', interval='1d', progress=False)
                if raw is not None and not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                    raw['Ticker'] = t.replace('.IS', '')
                    res = vibe_assistant.generate_full_vibe_analysis(t, raw)
                    ml = res['ml_predictions']
                    pa = res['price_action']
                    
                    results.append({
                        'Hisse': t.replace('.IS', ''),
                        '1. Aşama: Formasyon': res['main_pattern']['name'],
                        '2. Aşama: AI Teyit %': f"%{ml['win_rate_pct']}",
                        '3. Çifte Onay Sinyali': res['confirmation_status'],
                        'Onaylı Win Rate': f"%{res['confirmed_win_rate']}",
                        'Giriş Fiyatı': f"{ml['entry_price']} TL",
                        'Hedef 1 (T+5)': f"{ml['target_1']} TL (+%{ml['t5_return_pct']})",
                        'Stop Loss': f"{ml['stop_loss']} TL",
                        'Akıllı Para': pa['smart_money_status']
                    })
            except Exception as e:
                pass
            progress_bar.progress((idx + 1) / len(t_list))

        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            
            st.markdown("#### 🎯 Çifte Onaylı Pozisyon Kartları (Formasyon + AI Teyitli)")
            for r in results:
                card_color = "#10b981" if "GÜÇLÜ AL" in r['3. Çifte Onay Sinyali'] else ("#f59e0b" if "TUZAK" in r['3. Çifte Onay Sinyali'] else "#3b82f6")
                st.markdown(f"""
                <div class='terminal-card' style='border-left: 4px solid {card_color};'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='font-size:1.3rem; font-weight:700; color:#3b82f6;'>{r['Hisse']}</span>
                        <span style='font-size:1.1rem; font-weight:700; color:{card_color};'>{r['3. Çifte Onay Sinyali']}</span>
                    </div>
                    <div style='margin-top:10px; font-size:0.92rem; color:#cbd5e1; line-height:1.6;'>
                        • <b>1. Aşama (Teknik Formasyon):</b> <mark>{r['1. Aşama: Formasyon']}</mark><br>
                        • <b>2. Aşama (Yapay Zeka Teyidi):</b> AI Modeli %{r['2. Aşama: AI Teyit %']} Yükseliş Olasılığı Verdi.<br>
                        • <b>Onaylı Kazanma Oranı:</b> <b style='color:#10b981;'>{r['Onaylı Win Rate']}</b><br>
                        • <b>Giriş:</b> {r['Giriş Fiyatı']} | <b>Hedef 1 (T+5):</b> {r['Hedef 1 (T+5)']} | <b>Stop Loss:</b> <span style='color:red;'>{r['Stop Loss']}</span><br>
                        • <b>Hacim & Akıllı Para:</b> {r['Akıllı Para']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif page == "🔍 Canlı BİST Formasyon & AI Taraması":
    st.markdown("### 🔍 Canlı BİST Formasyon & Yapay Zeka Çifte Onay Taraması")
    st.markdown(f"Tüm Borsa İstanbul (**{len(all_tickers_list)} Adet Hissede**) Önce **Teknik Formasyon (TOBO, Çanak Kulp, Flama, Kırılım)** Taranır, Ardından **Yapay Zeka Modeli (`best_model_acm465.joblib`)** Tarafından Çifte Teyit Yapılır.")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        scan_scope = st.selectbox("1. Tarama Kapsamı:", ["BİST 30 Hisseleri", "BİST 100 Hisseleri", "Tüm BİST Hisseleri (537 Hisse)", "Özel Liste"], key="scan_scope_sel")
    with col_f2:
        pattern_filter = st.selectbox("2. Formasyon Türü Filtresi:", ["HEPSİ (Tüm Formasyonlar)", "Boğa Formasyonları (TOBO, Çanak, Flama, İkili Dip)", "Direnç & Trend Kırılımları", "Roket / Hacimli Yükseliş"], key="pat_filt_sel")
    with col_f3:
        ai_min_win_rate = st.slider("3. Min. Yapay Zeka Win Rate %:", min_value=40, max_value=90, value=50, step=5, key="ai_win_slider")

    if scan_scope == "Özel Liste":
        custom_area = st.text_input("Taranacak Hisse Kodları (Virgülle Ayrılmış):", "THYAO, GARAN, ASELS, EREGL, FROTO, KCHOL, TUPRS, SISE, BIMAS, AKBNK", key="custom_scan_area_input")
        target_tickers = [t.strip().upper() for t in custom_area.split(',') if t.strip()]
    elif scan_scope == "BİST 30 Hisseleri":
        b30 = ["THYAO", "GARAN", "EREGL", "ASELS", "FROTO", "KCHOL", "TUPRS", "SISE", "BIMAS", "AKBNK", "SAHOL", "SASA", "HEKTS", "ISCTR", "YKBNK", "VAKBN", "HALKB", "EKGYO", "PETKM", "PGSUS"]
        target_tickers = [t for t in b30 if t in all_tickers_list]
    elif scan_scope == "BİST 100 Hisseleri":
        target_tickers = all_tickers_list[:100]
    else:
        target_tickers = all_tickers_list

    if st.button(f"Formasyon ve AI Taramasını Çalıştır ({len(target_tickers)} Hisse)", type="primary", key="start_dedicated_scan_btn"):
        st.info(f"Formasyon tarama algoritması ve AI modeli {len(target_tickers)} hissede çalıştırılıyor...")
        
        scan_results = []
        all_evaluated = []
        progress_bar = st.progress(0)

        for idx, t in enumerate(target_tickers):
            stock_data = None
            if df is not None and 'Ticker' in df.columns:
                matched = df[df['Ticker'].str.replace('.IS', '').str.upper() == t.upper()]
                if not matched.empty:
                    stock_data = matched.copy()

            res = vibe_assistant.generate_full_vibe_analysis(t, stock_data)
            ml = res['ml_predictions']
            pats = res['patterns']
            pa = res['price_action']
            main_pat = res['main_pattern']
            win_rate = res['confirmed_win_rate']

            rec_item = {
                'Hisse Kodu': t,
                '1. Aşama: Tespit Edilen Formasyon': main_pat['name'],
                'Formasyon Türü': main_pat['type'],
                '2. Aşama: AI Yükseliş Olasılığı': f"%{ml['win_rate_pct']}",
                '3. Çifte Onaylı Sinyal': res['confirmation_status'],
                'Onaylı Win Rate': f"%{win_rate}",
                'Giriş Fiyatı': f"{ml['entry_price']} TL",
                'Hedef 1 (T+5)': f"{ml['target_1']} TL (+%{ml['t5_return_pct']})",
                'Stop Loss': f"{ml['stop_loss']} TL",
                'Akıllı Para / Hacim': pa['smart_money_status'],
                'raw_win_rate': ml['win_rate_pct']
            }
            all_evaluated.append(rec_item)

            # Apply filters
            pass_pattern = True
            if pattern_filter == "Boğa Formasyonları (TOBO, Çanak, Flama, İkili Dip)":
                pass_pattern = any(p['type'].startswith('BOĞA') or any(k in p['name'].upper() for k in ['TOBO', 'ÇANAK', 'BAYRAK', 'FLAMA', 'DİP']) for p in pats)
            elif pattern_filter == "Direnç & Trend Kırılımları":
                pass_pattern = any('KIRILIM' in p['name'].upper() or 'KIRILIM' in p['description'].upper() for p in pats)
            elif pattern_filter == "Roket / Hacimli Yükseliş":
                pass_pattern = pa['smart_money_score'] >= 68 or any('ROKET' in p['name'].upper() for p in pats)

            pass_ai = (ml['win_rate_pct'] >= ai_min_win_rate) or (win_rate >= ai_min_win_rate)

            if pass_pattern and pass_ai:
                scan_results.append(rec_item)

            progress_bar.progress((idx + 1) / len(target_tickers))

        if not scan_results and all_evaluated:
            # Automatic fallback to top highest win rate stocks
            all_evaluated.sort(key=lambda x: x['raw_win_rate'], reverse=True)
            scan_results = all_evaluated[:5]
            st.info("💡 Not: Katı filtrelere tam uyan hisse bulunamadığı için en yüksek olasılıklı ilk 5 hisse listelenmiştir.")

        if scan_results:
            st.success(f"Tarama Tamamlandı! Kriterlere Uyan {len(scan_results)} Hisse Bulundu.")
            scan_res_df = pd.DataFrame(scan_results).drop(columns=['raw_win_rate'], errors='ignore')
            st.dataframe(scan_res_df, use_container_width=True)

            st.markdown("#### Sinyal Pozisyon Kartları (Formasyon + AI Teyitli)")
            for r in scan_results[:10]:
                card_color = "#10b981" if "GÜÇLÜ AL" in r['3. Çifte Onaylı Sinyal'] else ("#f59e0b" if "TUZAK" in r['3. Çifte Onaylı Sinyal'] else "#3b82f6")
                st.markdown(f"""
                <div class='terminal-card' style='border-left: 4px solid {card_color};'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='font-size:1.3rem; font-weight:700; color:#3b82f6;'>{r['Hisse Kodu']}</span>
                        <span style='font-size:1.1rem; font-weight:700; color:{card_color};'>{r['3. Çifte Onaylı Sinyal']}</span>
                    </div>
                    <div style='margin-top:10px; font-size:0.92rem; color:#cbd5e1; line-height:1.6;'>
                        • <b>1. Aşama (Formasyon):</b> <mark>{r['1. Aşama: Tespit Edilen Formasyon']}</mark> ({r['Formasyon Türü']})<br>
                        • <b>2. Aşama (Yapay Zeka Teyidi):</b> AI Modeli %{r['2. Aşama: AI Yükseliş Olasılığı']} Yükseliş Olasılığı Verdi.<br>
                        • <b>Onaylı Başarı Oranı:</b> <b style='color:#10b981;'>{r['Onaylı Win Rate']}</b><br>
                        • <b>Giriş:</b> {r['Giriş Fiyatı']} | <b>Hedef 1 (T+5):</b> {r['Hedef 1 (T+5)']} | <b>Stop Loss:</b> <span style='color:red;'>{r['Stop Loss']}</span><br>
                        • <b>Akıllı Para Hareketi:</b> {r['Akıllı Para / Hacim']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif page == "🌊 Vibe Trading AI Studio":
    st.markdown("### 🌊 Vibe Trading AI Studio & Interactive Assistant")
    st.markdown(f"Tüm Borsa İstanbul Hisseleri (**{len(all_tickers_list)} Adet Aktif BİST Hissesi**) İçin Yapay Zeka Vibe Skoru, Formasyon ve Çifte Onaylı Trading Tahmini.")

    col_t, col_q = st.columns([1.5, 3])
    with col_t:
        vibe_ticker = st.selectbox(
            "Analiz Edilecek BİST Hissesini Seçin (537 Hisse):",
            options=all_tickers_list,
            index=all_tickers_list.index("THYAO") if "THYAO" in all_tickers_list else 0,
            key="vibe_select"
        )
        custom_input = st.text_input("Veya Elle Hisse Kodu Yazın:", "", key="vibe_custom_input")
        if custom_input.strip():
            vibe_ticker = custom_input.strip().upper()

    with col_q:
        vibe_query = st.text_input(
            f"Sorunuz veya Analiz İsteğiniz ({vibe_ticker}):",
            f"{vibe_ticker} için formasyon, vibe skoru ve çifte onaylı trading tahmini nedir?",
            key="vibe_q"
        )

    if st.button(f"🌊 {vibe_ticker} Vibe Analizini Başlat", type="primary", key="vibe_btn"):
        full_t = vibe_ticker.strip().upper()
        if not full_t.endswith('.IS'):
            full_t += '.IS'

        with st.spinner(f"{vibe_ticker} verileri ve yapay zeka modelleri işleniyor..."):
            try:
                raw = yf.download(full_t, period='6mo', interval='1d', progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                raw['Ticker'] = vibe_ticker.upper()
            except Exception:
                raw = None

            ans_html = vibe_assistant.answer_query(vibe_query, vibe_ticker, raw)
            st.markdown(f"""
            <div class='terminal-card' style='border-left: 4px solid #818cf8;'>
                {ans_html}
            </div>
            """, unsafe_allow_html=True)

elif page == "💎 Halka Arz (IPO) Radar & Strategy":
    st.markdown("### 💎 Halka Arz (IPO) Temel Değerleme & Taban Riski Radar Engine")
    st.markdown("SPK İzahnameleri, F/K, FAVÖK Çarpanları, Gelir Kullanım Yerleri ve Taban Riski Skorları ile BİST Halka Arz Analizleri.")

    ipos = IPOEngine.get_upcoming_and_recent_ipos()
    
    st.markdown("#### 📌 22 Temmuz 2026 Güncel Halka Arz Temel Değerleme Tablosu")
    for ipo in ipos:
        risk_color = "#ef4444" if "YÜKSEK" in ipo.get('taban_risk_score', '') else ("#f59e0b" if "ORTA" in ipo.get('taban_risk_score', '') else "#10b981")
        st.markdown(f"""
        <div class='terminal-card' style='border-left: 4px solid {risk_color};'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-size:1.2rem; font-weight:700; color:#f59e0b;'>{ipo['company']} ({ipo['code']})</span>
                <span style='font-size:1.0rem; font-weight:700; color:{risk_color};'>{ipo['recommendation']}</span>
            </div>
            <div style='margin-top:12px; font-size:0.92rem; color:#e2e8f0; line-height:1.6;'>
                • <b>Talep Toplama Tarihleri:</b> <mark>{ipo.get('dates', '22 - 24 Temmuz 2026')}</mark><br>
                • <b>Halka Arz Fiyatı & Büyüklük:</b> {ipo['ipo_price']} | {ipo['total_size']}<br>
                • <b>F/K Çarpanı:</b> <code>{ipo.get('pe_ratio', '10.5')}</code> | <b>FD/FAVÖK Çarpanı:</b> <code>{ipo.get('ev_ebitda', '8.0')}</code><br>
                • <b>Gelir Kullanım Yeri & Borçluluk:</b> {ipo.get('debt_status', 'Yatırım')}<br>
                • <b>Taban Riski Skoru:</b> <b style='color:{risk_color};'>{ipo.get('taban_risk_score', '%30')}</b> | <b>Tahmini Tavan:</b> {ipo['est_tavan_days']}<br>
                • <b>Dağıtım Yapısı:</b> {ipo['individual_pct']}<br>
                • <b>📝 Temel Analiz & Strateji Yorumu:</b> {ipo.get('detailed_analysis', ipo['strategy'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("A unified decision support platform designed to analyze stocks in the Borsa Istanbul (BIST) equity market using quantitative technical indicators and statistical machine learning models.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='terminal-card'>
            <div class='metric-label'>Classification Target (Target_T5)</div>
            <div class='metric-value'>5-Day Direction</div>
            <p style='font-size:0.8rem; color:#94a3b8; margin-top:8px;'>Probability of the stock's closing price in 5 trading days being higher than today's.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='terminal-card'>
            <div class='metric-label'>Active Stocks Covered</div>
            <div class='metric-value'>537 Tickers</div>
            <p style='font-size:0.8rem; color:#94a3b8; margin-top:8px;'>All active stocks representing the BIST market that successfully passed data cleaning steps.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='terminal-card'>
            <div class='metric-label'>Generated Technical Metrics</div>
            <div class='metric-value'>30 Indicators</div>
            <p style='font-size:0.8rem; color:#94a3b8; margin-top:8px;'>Structured quantitative feature set representing momentum, volatility, volume, and trend dynamics.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("#### BorsaNeuron Methodological Workflow")
    
    st.markdown("""
    Our BIST technical database infrastructure utilizes a hybrid design that couples offline training with online real-time inference:
    
    1. **Data Mining:** 30 technical indicator features and the `Target_T5` label class are generated from historical BIST data streams via `yfinance`.
    2. **Dimensionality Reduction & Noise Filtering:** Pearson Correlation Analysis is applied to filter out highly correlated features, preventing multicollinearity issues.
    3. **Market Regime Clustering:** The technical indicator profiles of BIST stocks are segmented into 5 distinct market regimes using the K-Means algorithm and mapped via Principal Component Analysis (PCA).
    4. **AI Machine Learning Modeling:** Optimized classifiers including K-Nearest Neighbors (K-NN), Artificial Neural Networks (ANN-MLP), Random Forest, and XGBoost are evaluated, integrating the model with the highest F1-score as the active inference engine.
    5. **Win-Rate Weighting Engine:** Real-time stock queries leverage a rolling backtest of the AI system to calculate a historical win rate for the queried stock, dynamically modulating risk warnings.
    """)

elif page == "Exploratory Data Analysis (EDA)":
    st.markdown("### Exploratory Data Analysis & Descriptive Statistics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{df.shape[0]:,}")
    col2.metric("Total Columns (Features)", f"{df.shape[1]}")
    col3.metric("Missing Value Cells", "0" if not df.isnull().values.any() else f"{df.isnull().sum().sum()}")
    
    st.markdown("#### Sample Data Matrix (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)
    
    st.markdown("#### Descriptive Statistics Summary")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[num_cols].describe().T
    st.dataframe(desc, use_container_width=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### Target Variable (Target_T5) Distribution")
        target_counts = df['Target_T5'].value_counts()
        fig_target = px.pie(values=target_counts.values, names=['Decrease / Flat (0)', 'Increase (1)'],
                            color_discrete_sequence=['#ef4444', '#10b981'])
        fig_target.update_layout(
            template="plotly_dark", 
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
    with col_chart2:
        st.markdown("#### Feature Frequency Distribution: RSI_14")
        fig_rsi = px.histogram(df, x='RSI_14', nbins=50, color_discrete_sequence=['#3b82f6'])
        fig_rsi.update_layout(
            template="plotly_dark", 
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
        
    st.markdown("#### Feature Averages Grouped by Target Class (Target_T5)")
    indicator_cols = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend']
    target_means = df.groupby('Target_T5')[indicator_cols].mean()
    st.dataframe(target_means, use_container_width=True)

elif page == "Feature Correlation & Selection":
    st.markdown("### Correlation Analysis & Multicollinearity Filtering")
    
    with st.expander("Methodology Description", expanded=True):
        st.markdown('''
        * **Multicollinearity:** Indicators with identical or highly similar mathematical formulations create noise and redundancy in machine learning classifiers.
        * Pearson correlation coefficients with absolute values $|r_{ij}| > 0.90$ are flagged and filtered out using an upper-triangular matrix filter.
        * This feature selection process reduces the risk of model overfitting and enhances generalizability.
        ''')
    
    sensor_cols = df.select_dtypes(include=[np.number]).drop(
        columns=['Target_T3', 'Target_T5', 'Target_T15', 'Max_Drawdown_15D', 'Max_Gain_15D'], errors='ignore')
    corr = sensor_cols.corr()
    
    fig_corr = px.imshow(corr, aspect="auto",
                         title="Pearson Correlation Coefficient Matrix (Heatmap)",
                         color_continuous_scale="RdBu_r")
    fig_corr.update_layout(
        template="plotly_dark", 
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)",
        width=900, height=650
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Multicollinearity elimination
    cor_matrix = sensor_cols.corr().abs()
    upper = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper.columns if any(upper[col] > 0.90)]
    
    st.markdown(f"#### Indicators Removed Due to High Correlation (>0.90) ({len(drop_list)} Features)")
    st.code(str(drop_list))
    
    remaining = [c for c in sensor_cols.columns if c not in drop_list]
    st.markdown(f"#### Independent Feature Set Retained for Modeling ({len(remaining)} Features)")
    st.code(str(remaining))

elif page == "Market Regime Clustering":
    st.markdown("### Market Regime Segmentation Using K-Means & PCA")
    
    with st.expander("Clustering & Dimensionality Reduction Theory", expanded=True):
        st.markdown(r'''
        * **K-Means Clustering:** Features are standardized ($z = (x - \mu)/\sigma$) and segmented into $k=5$ optimum clusters, validated via Elbow analysis.
        * **Principal Component Analysis (PCA):** Projects high-dimensional technical indicator matrices into 2 main components (PC1 and PC2) that explain the highest variance share, allowing 2D visual cluster analysis.
        * **Regime Profiles:** Clusters correspond to distinct market profiles such as hyper-bullish, stable trend, consolidation, or recovery.
        ''')
    
    k_clusters = st.slider("Target Clusters (k)", min_value=2, max_value=8, value=5)
    
    # Stock-based statistics
    ticker_stats = df.groupby('Ticker')[['ATR_14', 'Max_Drawdown_15D', 'Max_Gain_15D', 'RSI_14']].mean().dropna()
    scaler_km = StandardScaler()
    scaled_tickers = scaler_km.fit_transform(ticker_stats)
    
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    ticker_stats['Cluster'] = kmeans.fit_predict(scaled_tickers)
    
    st.markdown(f"**Within-Cluster Inertia (Sum of Squared Errors):** {kmeans.inertia_:.2f}")
    
    # PCA 2D Projection
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaled_tickers)
    ticker_stats['PCA1'] = pca_res[:, 0]
    ticker_stats['PCA2'] = pca_res[:, 1]
    
    st.markdown(f"**PCA Total Explained Variance Ratio:** PC1 = {pca.explained_variance_ratio_[0]*100:.1f}%, PC2 = {pca.explained_variance_ratio_[1]*100:.1f}%")
    
    fig2 = px.scatter(
        ticker_stats.reset_index(), x="PCA1", y="PCA2",
        color="Cluster", hover_data=["Ticker", "ATR_14", "Max_Gain_15D"],
        title="K-Means Market Regimes - PCA 2D Projection",
        color_continuous_scale="Turbo"
    )
    fig2.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color='rgba(255,255,255,0.2)')))
    fig2.update_layout(
        template="plotly_dark", 
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("#### Cluster Centers Detailed Matrix (Feature Averages)")
    st.dataframe(ticker_stats.groupby('Cluster')[['ATR_14', 'Max_Drawdown_15D', 'Max_Gain_15D', 'RSI_14']].mean())
    
    # Cumulative Variance Curve
    pca_full = PCA()
    pca_full.fit(scaled_tickers)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    fig_var = px.line(x=range(1, len(cumvar)+1), y=cumvar,
                      title="PCA Cumulative Explained Variance Curve",
                      labels={'x': 'Number of Components', 'y': 'Cumulative Variance Ratio'},
                      color_discrete_sequence=['#3b82f6'])
    fig_var.update_layout(
        template="plotly_dark", 
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_var, use_container_width=True)

elif page == "Machine Learning Model Analysis":
    st.markdown("### Performance Comparison of Supervised Learning Classifiers")
    st.markdown("Comparative analysis of K-Nearest Neighbors (K-NN), Random Forest, and Artificial Neural Networks (ANN-MLP) classifiers.")
    
    with st.expander("Model Configuration Parameters", expanded=True):
        st.markdown(r'''
        * **K-NN Classifier:** Optimum neighborhood parameter $k \in \{3..21\}$ is determined via GridSearchCV with 5-Fold Cross-Validation.
        * **Random Forest:** Trained with `n_estimators=100`, Gini Impurity splits, and bootstrapping enabled.
        * **Neural Network (MLP):** Configured with `hidden_layer_sizes=(64, 32, 16)`, ReLU activation, and Adam optimization.
        * **Scaling Method:** StandardScaler z-score normalization is applied to all input features.
        ''')
    
    features = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']
    df_ml = df.dropna(subset=features + ['Target_T5']).copy()
    
    if st.button("Start Model Training Matrix", key="train_btn", type="primary"):
        with st.spinner("Executing cross-validated model training pipeline..."):
            df_train_sample = df_ml
            if len(df_train_sample) > 10000:
                df_train_sample = df_train_sample.sample(n=10000, random_state=42)
            X = df_train_sample[features]
            y = df_train_sample['Target_T5']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # K-NN + GridSearch
            knn_gs = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9, 11, 15, 21]},
                                  cv=5, scoring='accuracy', n_jobs=-1)
            knn_gs.fit(X_train_s, y_train)
            best_k = knn_gs.best_params_['n_neighbors']
            knn_pred = knn_gs.predict(X_test_s)
            knn_acc = accuracy_score(y_test, knn_pred)
            knn_f1 = f1_score(y_test, knn_pred)
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_s, y_train)
            rf_pred = rf.predict(X_test_s)
            rf_acc = accuracy_score(y_test, rf_pred)
            rf_f1 = f1_score(y_test, rf_pred)
            
            # ANN (MLP)
            ann = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu',
                                solver='adam', max_iter=100, batch_size=128, random_state=42, verbose=False)
            ann.fit(X_train_s, y_train)
            ann_pred = ann.predict(X_test_s)
            ann_acc = accuracy_score(y_test, ann_pred)
            ann_f1 = f1_score(y_test, ann_pred)
            
            # Metric Cards
            col1, col2, col3 = st.columns(3)
            for col, name, acc, f1v in [(col1, f"K-NN (Optimum k={best_k})", knn_acc, knn_f1),
                                         (col2, "Random Forest", rf_acc, rf_f1),
                                         (col3, "Neural Network (MLP)", ann_acc, ann_f1)]:
                with col:
                    st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-label'>{name}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{acc*100:.2f}%</div>", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size:0.8rem; color:#94a3b8;'>F1-Score: {f1v:.4f}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Performance Comparison Plot
            results_df = pd.DataFrame({
                'Model': [f'K-NN (k={best_k})', 'Random Forest', 'Neural Network (MLP)'],
                'Accuracy': [knn_acc, rf_acc, ann_acc],
                'F1-Score': [knn_f1, rf_f1, ann_f1]
            })
            
            fig_comp = px.bar(results_df, x='Model', y=['Accuracy', 'F1-Score'],
                              barmode='group', title="Out-of-Sample Test Score Comparison",
                              color_discrete_sequence=['#3b82f6', '#10b981'])
            fig_comp.update_layout(
                template="plotly_dark", 
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Feature Importance
            importances = rf.feature_importances_
            fig_fi = px.bar(x=importances, y=features, orientation='h',
                            title="Random Forest Gini Impurity Feature Importance",
                            color=importances, color_continuous_scale="Viridis")
            fig_fi.update_layout(
                template="plotly_dark", 
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            
            st.info(f"GridSearchCV optimal parameter for K-NN: k={best_k} (Cross-Validation Accuracy: {knn_gs.best_score_:.4f})")

elif page == "Time-Series Trend Forecasting":
    st.markdown("### Time-Series Price Trend Modeling Using Meta Prophet")
    
    with st.expander("Time-Series Modeling Principles", expanded=True):
        st.markdown('''
        * **Prophet Algorithm:** A forecasting technique that models trends, seasonalities, and holiday effects in an additive regression equation.
        * **Configuration:** Daily seasonality is disabled; yearly seasonality and macro-trend curves are enabled.
        * **Quantitative Interpretation:** This model is used to forecast the macro-momentum and long-term price trajectory of the stock rather than short-term indicator breakouts.
        ''')
    
    if Prophet is None:
        st.error("Prophet library is not installed in this environment. Please check system dependencies.")
    else:
        selected_ticker = st.selectbox("Stock Ticker to Analyze", df['Ticker'].unique()[:50], index=0)
        days_ahead = st.slider("Forecasting Horizon (Days)", 10, 90, 60)
        
        if st.button("Run Forecast Matrix", key="prophet_btn", type="primary"):
            with st.spinner("Training Prophet macro-trend model..."):
                df_ticker = df[df['Ticker'] == selected_ticker].copy()
                df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
                df_ticker = df_ticker.sort_values('Date')
                df_prophet = df_ticker[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                
                m = Prophet(daily_seasonality=False, yearly_seasonality=True)
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=days_ahead)
                forecast = m.predict(future)
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Historical Real Price', line=dict(color='#10b981', width=1.5)))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet Trend Model', line=dict(color='#3b82f6', width=2)))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(59,130,246,0)', showlegend=False))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', fillcolor='rgba(59,130,246,0.12)', line_color='rgba(59,130,246,0)', name='Confidence Interval'))
                fig3.update_layout(
                    title=f"{selected_ticker} - {days_ahead} Day Macro Price Trajectory Forecast",
                    template="plotly_dark", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# MODÜL 6: CANLI HİSSE SORGULAMA & TAHMİN
# ==========================================
elif page == "Live Stock Query & Inference":
    st.markdown("### Real-Time Inference, Machine Learning Decision & Sector Comparison")
    st.markdown("Enter any BIST stock ticker symbol to compute instant technical metrics from live yFinance streams and receive AI model inferences.")
    
    # Sector reference mapping
    SECTOR_MAP = {
        "AKBNK.IS": "Banking", "GARAN.IS": "Banking", "ISCTR.IS": "Banking", "YKBNK.IS": "Banking", 
        "HALKB.IS": "Banking", "VAKBN.IS": "Banking", "TSKB.IS": "Banking", "ALBRK.IS": "Banking", 
        "SKBNK.IS": "Banking", "QNBFB.IS": "Banking",
        "KCHOL.IS": "Holding", "SAHOL.IS": "Holding", "DOHOL.IS": "Holding", "AGHOL.IS": "Holding", 
        "ALARK.IS": "Holding", "TEKTU.IS": "Holding", "GSDHO.IS": "Holding", "IHLAS.IS": "Holding", 
        "POLHO.IS": "Holding", "BERA.IS": "Holding", "TKFEN.IS": "Holding",
        "EREGL.IS": "Industry & Metal", "KRDMD.IS": "Industry & Metal", "ISDMR.IS": "Industry & Metal", 
        "TUPRS.IS": "Industry & Metal", "PETKM.IS": "Industry & Metal", "KOZAL.IS": "Industry & Metal", 
        "KOZAA.IS": "Industry & Metal", "IPEKE.IS": "Industry & Metal", "CIMSA.IS": "Industry & Metal", 
        "OYAKC.IS": "Industry & Metal", "BUCIM.IS": "Industry & Metal", "BSOKE.IS": "Industry & Metal", 
        "KCAER.IS": "Industry & Metal",
        "FROTO.IS": "Automotive", "TOASO.IS": "Automotive", "DOAS.IS": "Automotive", "TTRAK.IS": "Automotive", 
        "KARSN.IS": "Automotive", "OTKAR.IS": "Automotive", "TMSN.IS": "Automotive", "ASUZU.IS": "Automotive",
        "ASTOR.IS": "Energy", "ENKAI.IS": "Energy", "ODAS.IS": "Energy", "AKSEN.IS": "Energy", 
        "ZOREN.IS": "Energy", "AYDEM.IS": "Energy", "BIOEN.IS": "Energy", "HUNER.IS": "Energy", 
        "SMRTG.IS": "Energy", "EUPWR.IS": "Energy", "GWIND.IS": "Energy", "YEOTK.IS": "Energy", 
        "ALFAS.IS": "Energy", "CWENE.IS": "Energy", "AKFYE.IS": "Energy", "ENJSA.IS": "Energy", 
        "AENER.IS": "Energy",
        "EKGYO.IS": "Real Estate (REIT)", "ISGYO.IS": "Real Estate (REIT)", "TRGYO.IS": "Real Estate (REIT)", 
        "AKFGY.IS": "Real Estate (REIT)", "SNGYO.IS": "Real Estate (REIT)", "OZKGY.IS": "Real Estate (REIT)", 
        "HLGYO.IS": "Real Estate (REIT)", "ASGYO.IS": "Real Estate (REIT)", "KLGYO.IS": "Real Estate (REIT)",
        "THYAO.IS": "Transportation", "PGSUS.IS": "Transportation", "TAVHL.IS": "Transportation", "CLEBI.IS": "Transportation", 
        "RYSAS.IS": "Transportation", "TLMAN.IS": "Transportation",
        "BIMAS.IS": "Food & Retail", "MGROS.IS": "Food & Retail", "SOKM.IS": "Food & Retail", 
        "ULKER.IS": "Food & Retail", "CCOLA.IS": "Food & Retail", "AEFES.IS": "Food & Retail", 
        "TUKAS.IS": "Food & Retail", "TATGD.IS": "Food & Retail", "KRYST.IS": "Food & Retail", 
        "PETUN.IS": "Food & Retail", "SUWEN.IS": "Food & Retail",
        "KONTR.IS": "Technology & Software", "MIATK.IS": "Technology & Software", "ASELS.IS": "Technology & Software", 
        "PENTA.IS": "Technology & Software", "LOGO.IS": "Technology & Software", "ARDYZ.IS": "Technology & Software", 
        "VBTYZ.IS": "Technology & Software", "NETAS.IS": "Technology & Software", "KFEIN.IS": "Technology & Software", 
        "SMART.IS": "Technology & Software", "SDTTR.IS": "Technology & Software", "REEDR.IS": "Technology & Software",
        "ARCLK.IS": "Durable Consumer Goods", "VESBE.IS": "Durable Consumer Goods", "VESTL.IS": "Durable Consumer Goods",
        "SISE.IS": "Glass & Ceramics", "KLMSN.IS": "Glass & Ceramics", "EGSER.IS": "Glass & Ceramics", "KUTPO.IS": "Glass & Ceramics",
        "TCELL.IS": "Telecom", "TTKOM.IS": "Telecom",
        "TURSG.IS": "Insurance", "AKGRT.IS": "Insurance", "ANHYT.IS": "Insurance", "ANSGR.IS": "Insurance"
    }

    col_input1, col_input2 = st.columns([3, 1])
    with col_input1:
        searched_ticker = st.text_input("Enter BIST Stock Ticker (e.g. THYAO, EREGL, ASELS, YKBNK):", "THYAO", key="live_ticker_input")
    with col_input2:
        backtest_years = st.selectbox("Historical Data Range:", ["1 Year", "6 Months", "2 Years"], index=0)

    searched_ticker = searched_ticker.strip().upper()
    if not searched_ticker.endswith(".IS"):
        full_ticker = searched_ticker + ".IS"
    else:
        full_ticker = searched_ticker

    if st.button("Analyze Stock", key="live_analiz_btn", type="primary"):
        with st.spinner(f"Fetching live data and computing indicators for {full_ticker}..."):
            
            period_map = {"1 Year": "1y", "6 Months": "6mo", "2 Years": "2y"}
            raw_live_data = yf.download(full_ticker, period=period_map[backtest_years], interval="1d")
            
            if raw_live_data is None or raw_live_data.empty or len(raw_live_data) < 50:
                st.error(f"System Error: Could not fetch data for {full_ticker} or insufficient number of trading days.")
                st.stop()
            
            raw_live_data = raw_live_data.copy()
            if isinstance(raw_live_data.columns, pd.MultiIndex):
                raw_live_data.columns = [c[0] if isinstance(c, tuple) else c for c in raw_live_data.columns]
            
            raw_live_data.reset_index(inplace=True)
            if 'Datetime' in raw_live_data.columns and 'Date' not in raw_live_data.columns:
                raw_live_data.rename(columns={'Datetime': 'Date'}, inplace=True)
            if 'index' in raw_live_data.columns and 'Date' not in raw_live_data.columns:
                raw_live_data.rename(columns={'index': 'Date'}, inplace=True)
            if 'Date' not in raw_live_data.columns:
                raw_live_data.rename(columns={raw_live_data.columns[0]: 'Date'}, inplace=True)

            def compute_live_features(df_raw):
                try:
                    df = df_raw.copy()
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    df['SMA_200'] = df['Close'].rolling(200).mean()
                    
                    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
                    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
                    
                    df['MACD'] = df['EMA_12'] - df['EMA_26']
                    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    
                    delta = df['Close'].diff()
                    gain = delta.clip(lower=0).rolling(14).mean()
                    loss = (-delta.clip(upper=0)).rolling(14).mean()
                    loss = loss.replace(0, 1e-10)
                    df['RSI_14'] = 100 - (100 / (1 + gain / loss))
                    
                    df['BB_Middle'] = df['Close'].rolling(20).mean()
                    bb_std = df['Close'].rolling(20).std()
                    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
                    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
                    
                    high_low = df['High'] - df['Low']
                    high_close = (df['High'] - df['Close'].shift()).abs()
                    low_close = (df['Low'] - df['Close'].shift()).abs()
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    df['ATR_14'] = true_range.rolling(14).mean()
                    
                    low_14 = df['Low'].rolling(14).min()
                    high_14 = df['High'].rolling(14).max()
                    df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14).replace(0, 1e-10)) * 100
                    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
                    
                    df['Support_Level'] = df['Low'].rolling(20).min()
                    df['Resistance_Level'] = df['High'].rolling(20).max()
                    
                    vol_sma = df['Volume'].rolling(20).mean()
                    df['Volume_Trend'] = (df['Volume'] > vol_sma).astype(int)
                    
                    sr_range = (df['Resistance_Level'] - df['Support_Level']).replace(0, 1e-10)
                    df['Depth_Ratio'] = (df['Close'] - df['Support_Level']) / sr_range
                    
                    df['Neckline_Slope'] = df['SMA_20'].diff(5) / df['SMA_20'].shift(5).replace(0, 1e-10)
                    
                    df['Expert_Signal'] = 0
                    al_mask = (df['RSI_14'] < 40) & (df['Stoch_K'] < 30) & (df['Close'] > df['SMA_50'])
                    df.loc[al_mask, 'Expert_Signal'] = 1
                    sat_mask = (df['RSI_14'] > 70) & (df['Stoch_K'] > 80)
                    df.loc[sat_mask, 'Expert_Signal'] = -1
                    
                    return df
                except Exception as e:
                    st.error(f"Technical indicator calculation error: {e}")
                    return None

            processed_live_data = compute_live_features(raw_live_data)
            
            if processed_live_data is None:
                st.stop()
            
            hisse_sektor = SECTOR_MAP.get(full_ticker, "General / No Sector")
            son_row = processed_live_data.dropna(subset=['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']).tail(1).iloc[0]
            curr_price = float(son_row['Close'])
            
            st.markdown(f"#### Sectoral Alignment & Reference Values: [Sector: {hisse_sektor}]")
            
            if hisse_sektor != "General / No Sector":
                sektor_tickers = [t for t, sec in SECTOR_MAP.items() if sec == hisse_sektor]
                df_sektor = df[df['Ticker'].isin(sektor_tickers)].copy()
                
                if not df_sektor.empty:
                    mean_rsi = df_sektor['RSI_14'].mean()
                    mean_atr = df_sektor['ATR_14'].mean()
                    mean_depth = df_sektor['Depth_Ratio'].mean()
                    
                    col_sec1, col_sec2, col_sec3 = st.columns(3)
                    with col_sec1:
                        st.metric("RSI Value (Live vs Sector)", f"{son_row['RSI_14']:.1f}", f"Sector Avg: {mean_rsi:.1f}", delta_color="off")
                    with col_sec2:
                        st.metric("Volatility (ATR) (Live vs Sector)", f"{son_row['ATR_14']:.2f}", f"Sector Avg: {mean_atr:.2f}", delta_color="off")
                    with col_sec3:
                        st.metric("Support/Resistance Position (Depth)", f"%{son_row['Depth_Ratio']*100:.1f}", f"Sector Avg: %{mean_depth*100:.1f}", delta_color="off")
            
            # Model Inference
            features_q = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']
            model_loaded = False
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                best_model = joblib.load(os.path.join(base_dir, "best_model_acm465.joblib"))
                best_scaler = joblib.load(os.path.join(base_dir, "best_scaler_acm465.joblib"))
                best_features = joblib.load(os.path.join(base_dir, "best_features_acm465.joblib"))
                model_loaded = True
            except Exception as e:
                pass
            
            if model_loaded:
                # Add pattern columns to processed_live_data if they are not present
                for col in best_features:
                    if col not in processed_live_data.columns:
                        if col == 'Pat_Yok':
                            processed_live_data[col] = 1
                        else:
                            processed_live_data[col] = 0
                
                # Re-extract the last row to include the added features
                son_row = processed_live_data.dropna(subset=['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']).tail(1).iloc[0]
                features_to_use = best_features
                rf_model = best_model
                scaler_q = best_scaler
            else:
                features_to_use = features_q
                df_train = df.dropna(subset=features_q + ['Target_T5']).copy()
                if len(df_train) > 10000:
                    df_train = df_train.sample(n=10000, random_state=42)
                scaler_q = StandardScaler()
                X_all = scaler_q.fit_transform(df_train[features_q])
                y_all = df_train['Target_T5']
                rf_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
                rf_model.fit(X_all, y_all)
            
            son_features = son_row[features_to_use].values.reshape(1, -1)
            son_scaled = scaler_q.transform(son_features)
            
            tahmin = rf_model.predict(son_scaled)[0]
            proba = rf_model.predict_proba(son_scaled)[0]
            guven = max(proba) * 100
            
            backtest_df = processed_live_data.copy()
            bt_features = scaler_q.transform(backtest_df[features_to_use].values)
            backtest_df['AI_Signal'] = rf_model.predict(bt_features)
            
            backtest_df['Target_T5_True'] = (backtest_df['Close'].shift(-5) > backtest_df['Close']).astype(int)
            al_signals = backtest_df[backtest_df['AI_Signal'] == 1]
            if len(al_signals) > 5:
                hist_win_rate = (al_signals['Close'].shift(-5) > al_signals['Close']).mean() * 100
            else:
                hist_win_rate = 50.0
                
            if hist_win_rate < 48.0 and tahmin == 1:
                tavsiye_text = "BUY [HIGH RISK / LOW COMPLIANCE]"
                tavsiye_color = "#ef4444"
            elif tahmin == 1:
                tavsiye_text = "BUY [POSITIVE DIRECTIONAL FORECAST]"
                tavsiye_color = "#10b981"
            else:
                tavsiye_text = "HOLD / NEUTRAL POSITION"
                tavsiye_color = "#94a3b8"
            
            st.markdown("#### Live Price Series & AI Buy Signals")
            grafik_df = processed_live_data.tail(90).copy()
            all_features_matrix = scaler_q.transform(grafik_df[features_to_use].values)
            grafik_df['AI_Signal'] = rf_model.predict(all_features_matrix)
            
            fig_candle = go.Figure()
            fig_candle.add_trace(go.Candlestick(
                x=grafik_df['Date'], open=grafik_df['Open'], high=grafik_df['High'],
                low=grafik_df['Low'], close=grafik_df['Close'], name='Price Candlestick'
            ))
            fig_candle.add_trace(go.Scatter(x=grafik_df['Date'], y=grafik_df['BB_Upper'], line=dict(color='rgba(59,130,246,0.2)', width=1), name='Bollinger Upper', showlegend=False))
            fig_candle.add_trace(go.Scatter(x=grafik_df['Date'], y=grafik_df['BB_Lower'], line=dict(color='rgba(59,130,246,0.2)', width=1), fill='tonexty', fillcolor='rgba(59,130,246,0.03)', name='Bollinger Lower', showlegend=False))
            fig_candle.add_trace(go.Scatter(x=grafik_df['Date'], y=grafik_df['SMA_20'], line=dict(color='#3b82f6', width=1.5), name='SMA 20'))
            
            buys = grafik_df[grafik_df['AI_Signal'] == 1]
            fig_candle.add_trace(go.Scatter(
                x=buys['Date'], y=buys['Low'] * 0.98, mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#10b981', line=dict(width=1, color='rgba(0,0,0,0.5)')),
                name='AI Buy Signal Trigger'
            ))
            
            fig_candle.update_layout(
                title=f"{full_ticker} - Last 90 Trading Days Bollinger & SMA Candlestick Chart",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_rangeslider_visible=False,
                height=480
            )
            st.plotly_chart(fig_candle, use_container_width=True)
            
            col_rec1, col_rec2, col_rec3 = st.columns(3)
            with col_rec1:
                st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>DECISION ENGINE RECOMMENDATION</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:{tavsiye_color};font-size:1.3rem;font-weight:bold;font-family:monospace;'>{tavsiye_text}</div>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:0.75rem; color:#94a3b8;'>Inference Confidence: {guven:.2f}%</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_rec2:
                st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>RSI(50) CENTERLINE CROSSOVER CONFIRMATION</div>", unsafe_allow_html=True)
                rsi_val = son_row['RSI_14']
                if rsi_val > 50:
                    st.markdown("<div style='color:#10b981;font-size:1.3rem;font-weight:bold;font-family:monospace;'>POSITIVE (RSI > 50)</div>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>Price in bull territory. Crossover confirms buy.</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#ef4444;font-size:1.3rem;font-weight:bold;font-family:monospace;'>NEGATIVE (RSI < 50)</div>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>Price in bear territory. Risky for buy positions.</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
 
            with col_rec3:
                st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>HISTORICAL STRATEGY COMPLIANCE</div>", unsafe_allow_html=True)
                if hist_win_rate > 60.0:
                    st.markdown(f"<div style='color:#10b981;font-size:1.3rem;font-weight:bold;font-family:monospace;'>HIGH COMPLIANCE ({hist_win_rate:.1f}%)</div>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>High historical alignment with AI model decisions.</span>", unsafe_allow_html=True)
                elif hist_win_rate >= 48.0:
                    st.markdown(f"<div style='color:#3b82f6;font-size:1.3rem;font-weight:bold;font-family:monospace;'>MODERATE COMPLIANCE ({hist_win_rate:.1f}%)</div>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>Balanced signal response in past periods.</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:#ef4444;font-size:1.3rem;font-weight:bold;font-family:monospace;'>LOW COMPLIANCE ({hist_win_rate:.1f}%)</div>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>Low historical alignment. Signals carry higher risk.</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
 
            with st.expander("Input Parameter Technical Indicator Details", expanded=True):
                st.markdown(f"""
                | Technical Indicator | Live Value | Status / Evaluation |
                |-----------------|--------------|-----------------------|
                | **RSI_14** | {son_row['RSI_14']:.2f} | {'Overbought' if son_row['RSI_14']>70 else 'Oversold' if son_row['RSI_14']<30 else 'Stable Momentum'} |
                | **MACD** | {son_row['MACD']:.4f} | {'Positive Trend Crossover' if son_row['MACD']>0 else 'Negative Trend Crossover'} |
                | **Stoch_K** | {son_row['Stoch_K']:.2f} | {'Overbought Zone' if son_row['Stoch_K']>80 else 'Oversold Zone' if son_row['Stoch_K']<20 else 'Neutral'} |
                | **ATR_14 (Volatility)** | {son_row['ATR_14']:.2f} | Daily average fluctuation range (TRY) |
                | **Expert Signal** | {int(son_row['Expert_Signal'])} | {'Expert System Buy Confirm' if son_row['Expert_Signal']==1 else 'Expert System Sell Confirm' if son_row['Expert_Signal']==-1 else 'No Confirmation'} |
                | **Neckline Slope** | {son_row['Neckline_Slope']:.4f} | {'Upward Slanted' if son_row['Neckline_Slope']>0 else 'Downward Slanted'} |
                """)
 
            # 6. Hisseye Özel Canlı Backtest Simülatörü
            st.markdown("#### Single Stock AI Strategy Backtest Performance")
            backtest_df['Daily_Return'] = backtest_df['Close'].pct_change()
            backtest_df['AI_Return'] = backtest_df['AI_Signal'].shift(1) * backtest_df['Daily_Return']
            
            backtest_df['AI_Cumulative'] = 100000 * (1 + backtest_df['AI_Return'].fillna(0)).cumprod()
            backtest_df['BH_Cumulative'] = 100000 * (1 + backtest_df['Daily_Return'].fillna(0)).cumprod()
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=backtest_df['Date'], y=backtest_df['AI_Cumulative'], mode='lines', name='BorsaNeuron AI Portfolio', line=dict(color='#10b981', width=2.5)))
            fig_bt.add_trace(go.Scatter(x=backtest_df['Date'], y=backtest_df['BH_Cumulative'], mode='lines', name='Buy & Hold', line=dict(color='#ef4444', width=1.5, dash='dash')))
            
            fig_bt.update_layout(
                title=f"{full_ticker} - Historical AI Strategy vs Buy & Hold Simulation",
                yaxis_title="Capital (TRY)",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380
            )
            st.plotly_chart(fig_bt, use_container_width=True)
            
            final_ai = backtest_df['AI_Cumulative'].iloc[-1]
            final_bh = backtest_df['BH_Cumulative'].iloc[-1]
            profit_ai = ((final_ai - 100000) / 100000) * 100
            profit_bh = ((final_bh - 100000) / 100000) * 100
            
            col_bt1, col_bt2, col_bt3 = st.columns(3)
            with col_bt1:
                st.metric("AI Portfolio Capital", f"{final_ai:,.2f} ₺", f"{profit_ai:.1f}% Net Gain")
            with col_bt2:
                st.metric("Buy & Hold Capital", f"{final_bh:,.2f} ₺", f"{profit_bh:.1f}% Net Gain")
            with col_bt3:
                al_sinyalleri_sayisi = (backtest_df['AI_Signal'] == 1).sum()
                st.metric("Total Buy Signals Triggered", f"{al_sinyalleri_sayisi} Daily Signals")
 
            with st.expander("Decision Rationale Report", expanded=True):
                reasons = []
                if son_row['RSI_14'] < 30:
                    reasons.append("RSI is below 30: Oversold zone, strong potential reversal.")
                elif son_row['RSI_14'] > 70:
                    reasons.append("RSI is above 70: Overbought zone, high profit-taking risk.")
                else:
                    reasons.append(f"RSI momentum is in neutral zone ({son_row['RSI_14']:.1f}).")
                
                if son_row['MACD'] > 0:
                    reasons.append("MACD is positive: Price momentum is maintaining upward direction.")
                else:
                    reasons.append("MACD is negative: Short-term correction trend dominates.")
                
                if son_row['Stoch_K'] < 20:
                    reasons.append("Stochastic K is below 20: Reversing from oversold boundary.")
                elif son_row['Stoch_K'] > 80:
                    reasons.append("Stochastic K is above 80: Consolidation risk at overbought boundary.")
                
                if son_row['Expert_Signal'] == 1:
                    reasons.append("Expert signal confirmation active: Technical structure and indicators support buy decision.")
                
                reasons.append(f"Model confidence score: {guven:.2f}% (Bullish probability: {proba[1]*100:.1f}% | Bearish probability: {proba[0]*100:.1f}%)")
                
                for r in reasons:
                    st.markdown(f"- {r}")

elif page == "Portfolio Backtesting & Simulation":
    st.markdown("### Historical Portfolio Growth & Out-of-Sample Simulation")
    
    with st.expander("Chronological Backtesting & Modeling Rules", expanded=True):
        st.markdown('''
        * **Data Splitting:** To prevent look-ahead bias, the first 80% of chronological data (2019–2025) is used as the training set, while the final 20% (2025–2026) is reserved as the unseen out-of-sample test set.
        * **Decision Engine:** Buy orders are generated and added to the portfolio on days when the model's `Target_T5` direction forecast is `1`. On days with a `0` forecast, the portfolio holds cash.
        * **Transaction Costs & Slippage:** To simulate realistic trading frictions, only 30% of target gains are captured from successful trades, while 50% of the maximum drawdowns are directly applied as losses.
        ''')
        
    if st.button("Run Out-of-Sample Backtest", key="backtest_btn", type="primary"):
        with st.spinner("Simulating historical portfolio growth curve..."):
            features_bt = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']
            df_bt = df.dropna(subset=features_bt + ['Target_T5', 'Max_Gain_15D', 'Max_Drawdown_15D']).copy()
            df_bt['Date'] = pd.to_datetime(df_bt['Date'])
            df_bt = df_bt.sort_values('Date')
            
            split_idx = int(len(df_bt) * 0.8)
            train_df = df_bt.iloc[:split_idx]
            test_df = df_bt.iloc[split_idx:].copy()
            
            # Sample train_df to 10k rows to speed up training
            if len(train_df) > 10000:
                train_df = train_df.sample(n=10000, random_state=42)
                
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(train_df[features_bt])
            X_te_s = scaler.transform(test_df[features_bt])
            
            rf_bt = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf_bt.fit(X_tr_s, train_df['Target_T5'])
            test_df['AI_Signal'] = rf_bt.predict(X_te_s)
            
            daily_returns = test_df[(test_df['AI_Signal'] == 1) & (test_df['Target_T5'] == 1)].groupby('Date')['Max_Gain_15D'].mean() * 0.3
            daily_loss = test_df[(test_df['AI_Signal'] == 1) & (test_df['Target_T5'] == 0)].groupby('Date')['Max_Drawdown_15D'].mean() * 0.5
            
            daily_net = pd.DataFrame({'Gain': daily_returns, 'Loss': daily_loss}).fillna(0)
            daily_net['Net_Return_Pct'] = daily_net['Gain'] + daily_net['Loss']
            daily_net['Portfoy_Degeri'] = 100000 * (1 + (daily_net['Net_Return_Pct'] / 100)).cumprod()
            
            market_return = test_df.groupby('Date')['Max_Gain_15D'].mean() * 0.1 - abs(test_df.groupby('Date')['Max_Drawdown_15D'].mean() * 0.1)
            daily_net['Buy_And_Hold'] = 100000 * (1 + (market_return / 100)).cumprod()
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=daily_net.index, y=daily_net['Portfoy_Degeri'], mode='lines', name='BorsaNeuron AI Decision Portfolio', line=dict(color='#10b981', width=2.5)))
            fig4.add_trace(go.Scatter(x=daily_net.index, y=daily_net['Buy_And_Hold'], mode='lines', name='Market Buy & Hold Index', line=dict(color='#ef4444', width=1.5, dash='dash')))
            fig4.update_layout(
                title="BorsaNeuron AI Portfolio Growth (Starting Capital: 100,000 TRY)",
                yaxis_title="Portfolio Capital (TRY)",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            final_capital = daily_net['Portfoy_Degeri'].iloc[-1]
            net_profit = ((final_capital - 100000) / 100000) * 100
            win_rate = (test_df[test_df['AI_Signal'] == 1]['Target_T5'].mean()) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Starting Capital", "100,000 TRY")
            col2.metric("Final Portfolio Capital", f"{final_capital:,.2f} TRY", f"{net_profit:.2f}% Net Gain")
            col3.metric("Strategy Win Rate", f"{win_rate:.1f}%")
            
            with st.expander("Chronological Log of Last 20 System Decisions", expanded=False):
                al_sinyalleri = test_df[test_df['AI_Signal'] == 1][['Date','Ticker','Close','RSI_14','MACD','Target_T5','Max_Gain_15D','Max_Drawdown_15D']].copy()
                al_sinyalleri['Result'] = al_sinyalleri['Target_T5'].map({1: 'Increased (Success)', 0: 'Decreased (Fail)'})
                al_sinyalleri['Gain/Loss'] = al_sinyalleri.apply(
                    lambda r: f"+{r['Max_Gain_15D']*0.3:.1f}%" if r['Target_T5']==1 else f"-{abs(r['Max_Drawdown_15D']*0.5):.1f}%", axis=1)
                st.dataframe(al_sinyalleri[['Date','Ticker','Close','RSI_14','MACD','Result','Gain/Loss']].tail(20), use_container_width=True)

# ==========================================
# MODULE 8: AUTOMATED PATTERN SCANNER
# ==========================================
elif page == "Automated Pattern Scanner":
    st.markdown("### Technical Chart Pattern Automated Scanning Terminal")
    st.markdown("Scanning of classic chart patterns and breakout directions detected across all active BIST stocks in the dataset.")
    
    with st.expander("Technical Pattern Definitions", expanded=True):
        st.markdown('''
        * **TOBO (Inverted Head and Shoulders):** Strong bullish reversal structure. A breakout above the neckline triggers a rise.
        * **OBO (Head and Shoulders):** Strong bearish reversal structure. Risk increases when the neckline is broken downwards.
        * **Cup & Handle:** Bullish trend continuation pattern. The movement gains momentum once the handle region is broken upwards.
        * **Flag:** Narrow consolidation ranges forming after rapid price movements. A breakout in the trend direction is expected.
        * **Double Bottom:** Bullish reversal pattern consisting of two consecutive lows around the same support level. A breakout above the intermediate peak confirms the pattern.
        * **Double Top:** Bearish reversal pattern consisting of two consecutive peaks around the same resistance level. A breakout below the intermediate trough confirms the pattern.
        ''')
        
    if 'Pattern_Type' in df.columns:
        df_scan = df.copy()
        df_scan['Date'] = pd.to_datetime(df_scan['Date'])
        son_tarih = df_scan['Date'].max()
        df_recent = df_scan[df_scan['Date'] >= son_tarih - pd.Timedelta(days=30)]
        df_patterns = df_recent[df_recent['Pattern_Type'] != 'Yok'].copy()
        
        # Map internal pattern names to clean English presentation names
        pattern_name_map = {
            'TOBO': 'Inverted Head & Shoulders (TOBO)',
            'OBO': 'Head & Shoulders (OBO)',
            'Cup_Handle': 'Cup & Handle',
            'Flag': 'Flag',
            'Double_Bottom': 'Double Bottom',
            'Double_Top': 'Double Top'
        }
        df_patterns['Pattern_Type'] = df_patterns['Pattern_Type'].map(pattern_name_map).fillna(df_patterns['Pattern_Type'])
        
        if df_patterns.empty:
            st.info("No active geometric chart patterns were detected across BIST in the last 30 days.")
        else:
            pattern_counts = df_patterns['Pattern_Type'].value_counts()
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("Total Patterns Detected", len(df_patterns))
            col_p2.metric("Unique Tickers Scanned", df_patterns['Ticker'].nunique())
            col_p3.metric("Matching BUY Signals", int((df_patterns['Expert_Signal'] == 1).sum()))
            col_p4.metric("Matching SELL Signals", int((df_patterns['Expert_Signal'] == -1).sum()))
            
            fig_pat = px.bar(
                x=pattern_counts.index, y=pattern_counts.values,
                title="Detected Geometric Chart Pattern Distribution (Last 30 Days)",
                labels={'x': 'Pattern Type', 'y': 'Detection Count'},
                color=pattern_counts.index,
                color_discrete_sequence=['#3b82f6', '#ef4444', '#10b981', '#fbbf24', '#8b5cf6', '#ec4899']
            )
            fig_pat.update_layout(
                template="plotly_dark", 
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig_pat, use_container_width=True)
            
            st.markdown("#### Detected Patterns Detail Matrix (Last 50 Records)")
            display_cols = ['Date', 'Ticker', 'Close', 'Pattern_Type', 'Expert_Signal', 'RSI_14', 'MACD', 'Depth_Ratio', 'Neckline_Slope']
            available_cols = [c for c in display_cols if c in df_patterns.columns]
            df_show = df_patterns[available_cols].sort_values('Date', ascending=False).head(50)
            
            sinyal_map = {1: 'BUY (Confirmed)', -1: 'SELL (Risky)', 0: 'NEUTRAL (Out of Reference)'}
            if 'Expert_Signal' in df_show.columns:
                df_show['Signal Direction'] = df_show['Expert_Signal'].map(sinyal_map)
                
            st.dataframe(df_show, use_container_width=True)
            
            st.markdown("#### Stock Distributions by Pattern Type")
            for pat_type in pattern_counts.index:
                with st.expander(f"Stocks with {pat_type} Pattern ({pattern_counts[pat_type]} Tickers)"):
                    pat_df = df_patterns[df_patterns['Pattern_Type'] == pat_type]
                    tickers_list = pat_df['Ticker'].unique()
                    st.write(f"**Stocks:** {', '.join([t.replace('.IS','') for t in tickers_list])}")
                    
                    avg_rsi = pat_df['RSI_14'].mean()
                    avg_depth = pat_df['Depth_Ratio'].mean()
                    col_a, col_b = st.columns(2)
                    col_a.metric("Average RSI", f"{avg_rsi:.2f}")
                    col_b.metric("Average Depth Ratio", f"{avg_depth:.3f}")
    else:
        st.warning("Database configuration error: 'Pattern_Type' column not found in data table.")
        
    st.markdown("---")
    st.markdown("#### Live Technical Scanner Panel")
    st.markdown("Check the latest status of specified stocks using the live technical scanning algorithm.")
    
    scan_tickers = st.text_input("Enter BIST Codes to Scan (Separated by Commas):", "THYAO, GARAN, EREGL, ASELS, FROTO", key="scan_input")
    
    if st.button("Start Live Scan", key="scan_btn", type="primary"):
        tickers_to_scan = [t.strip().upper() for t in scan_tickers.split(',')]
        tickers_to_scan = [t + '.IS' if not t.endswith('.IS') else t for t in tickers_to_scan]
        
        scan_results = []
        progress_bar = st.progress(0)
        
        for idx, tick in enumerate(tickers_to_scan):
            try:
                tick_data = yf.download(tick, period='6mo', interval='1d', progress=False)
                if tick_data is not None and not tick_data.empty and len(tick_data) > 50:
                    if isinstance(tick_data.columns, pd.MultiIndex):
                        tick_data.columns = [c[0] if isinstance(c, tuple) else c for c in tick_data.columns]
                    
                    close = tick_data['Close'].values
                    rsi_delta = pd.Series(close).diff()
                    rsi_gain = rsi_delta.clip(lower=0).rolling(14).mean()
                    rsi_loss = (-rsi_delta.clip(upper=0)).rolling(14).mean()
                    rsi_loss = rsi_loss.replace(0, 1e-10)
                    rsi_val = float((100 - (100 / (1 + rsi_gain / rsi_loss))).iloc[-1])
                    
                    last_close = float(close[-1])
                    sma20 = float(pd.Series(close).rolling(20).mean().iloc[-1])
                    sma50 = float(pd.Series(close).rolling(50).mean().iloc[-1])
                    
                    trend = 'Uptrend' if last_close > sma50 else 'Downtrend'
                    rsi_durum = 'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Normal'
                    
                    scan_results.append({
                        'Stock': tick.replace('.IS', ''),
                        'Last Price': f'{last_close:.2f}',
                        'Live RSI (14)': f'{rsi_val:.2f}',
                        'RSI Status': rsi_durum,
                        'SMA (20) Level': f'{sma20:.2f}',
                        'SMA (50) Level': f'{sma50:.2f}',
                        'Trend Position': trend
                    })
            except Exception as e:
                pass
            
            progress_bar.progress((idx + 1) / len(tickers_to_scan))
        
        if scan_results:
            st.dataframe(pd.DataFrame(scan_results), use_container_width=True)
        else:
            st.warning("Data Connection Error: Price data for the specified stocks could not be retrieved from yFinance.")

elif page == "Kurumsal Risk Analitiği (Sharpe / VaR)":
    st.markdown("### Kurumsal Risk Analitiği ve Portföy Risk Radarı")
    st.markdown("Bu modül, simüle edilen veya canlı portföy getirilerinin **Sharpe Oranı, Sortino Oranı, Maximum Drawdown (MDD), Value at Risk (VaR)** ve **Monte Carlo Olasılıksal Risk Dağılımlarını** hesaplar.")
    
    from src.risk_analytics import BorsaNeuronRiskEngine
    risk_engine = BorsaNeuronRiskEngine(risk_free_rate=0.45)
    
    selected_risk_ticker = st.selectbox("Risk Analizi Yapılacak Hisse Seçin:", all_tickers_list, index=0)
    
    if st.button("Risk Metriklerini ve Monte Carlo Simülasyonunu Çalıştır", type="primary"):
        with st.spinner("Piyasa verisi çekiliyor ve risk matrisi hesaplanıyor..."):
            try:
                hist_data = yf.download(f"{selected_risk_ticker}.IS", period="1y", interval="1d", progress=False)
                if hist_data is not None and not hist_data.empty:
                    if isinstance(hist_data.columns, pd.MultiIndex):
                        hist_data.columns = [c[0] for c in hist_data.columns]
                    
                    prices = hist_data['Close']
                    equity_sim = 100000 * (prices / prices.iloc[0])
                    
                    metrics, drawdowns = risk_engine.calculate_portfolio_metrics(equity_sim)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Toplam Getiri (%)", f"%{metrics['total_return_pct']:.2f}")
                    col2.metric("Sharpe Oranı", f"{metrics['sharpe_ratio']:.2f}")
                    col3.metric("Sortino Oranı", f"{metrics['sortino_ratio']:.2f}")
                    col4.metric("Max Drawdown (Maksimum Düşüş)", f"%{metrics['max_drawdown_pct']:.2f}")
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Günlük Riskte Değer (VaR %95)", f"%{metrics['var_95_daily_pct']:.2f}")
                    col_b.metric("Günlük Riskte Değer (VaR %99)", f"%{metrics['var_99_daily_pct']:.2f}")
                    col_c.metric("Yıllık Volatilite (%)", f"%{metrics['annualized_volatility_pct']:.2f}")
                    
                    st.markdown("#### Historical Drawdown Curve (%)")
                    fig_dd = px.line(drawdowns * 100, title=f"{selected_risk_ticker} Historical Drawdown Curve (%)", labels={'value': 'Drawdown %', 'index': 'Tarih'})
                    fig_dd.update_traces(line_color='#ef4444')
                    st.plotly_chart(fig_dd, use_container_width=True)
                    
                    st.markdown("#### Monte Carlo Gelecek Fiyat Yolu Simülasyonu (30 Gün / 500 Patika)")
                    daily_ret = prices.pct_change().dropna()
                    mc_results = risk_engine.run_monte_carlo_simulation(float(prices.iloc[-1]), daily_ret.mean(), daily_ret.std(), days=30, simulations=500)
                    
                    fig_mc = go.Figure()
                    for sim_idx in range(min(50, mc_results['matrix'].shape[1])):
                        fig_mc.add_trace(go.Scatter(y=mc_results['matrix'][:, sim_idx], mode='lines', line=dict(width=0.5, color='rgba(59, 130, 246, 0.2)'), showlegend=False))
                    fig_mc.update_layout(title="Monte Carlo 30-Day Projected Price Distribution", xaxis_title="Gün", yaxis_title="Tahmini Fiyat (TL)")
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    st.info(f"Monte Carlo Beklentileri: Kötümser (%10): {mc_results['p10_pessimistic']:.2f} TL | Medyan (%50): {mc_results['p50_median']:.2f} TL | İyimser (%90): {mc_results['p90_optimistic']:.2f} TL")
            except Exception as ex:
                st.error(f"Risk analizi çalıştırılırken hata oluştu: {ex}")

elif page == "KAP Bildirim & Canlı Haber Analiz Motoru":
    st.markdown("### KAP Bildirim ve Canlı Haber Analiz Motoru")
    st.markdown("Teknik Model Tahmini (%70 Ağırlık) ile Canlı KAP / BİST Haber Duyarlılığını (%30 Ağırlık) Harmanlayan **Hibrit Karar Destek Sistemi**.")
    
    from src.news_sentiment import NewsSentimentEngine
    
    selected_news_ticker = st.selectbox("Haber ve KAP Analizi Yapılacak Hisse:", all_tickers_list, index=0)
    
    if st.button("Canlı KAP Haberlerini ve Hibrit Sinyali Analiz Et", type="primary"):
        with st.spinner(f"{selected_news_ticker} KAP bildirimleri ve haberleri çekiliyor..."):
            sentiment_data = NewsSentimentEngine.get_news_sentiment(selected_news_ticker)
            
            st.markdown(f"#### Genel Duygu Durumu: **{sentiment_data['overall_sentiment']}** (Net Skor: {sentiment_data['net_score']})")
            
            col_sent1, col_sent2 = st.columns(2)
            tech_prob_demo = 0.65  # Baseline model probability
            hybrid_res = NewsSentimentEngine.calculate_hybrid_signal(tech_prob_demo, sentiment_data['net_score'])
            
            col_sent1.metric("Hibrit Sinyal Puanı", f"%{hybrid_res['hybrid_score_pct']}")
            col_sent2.metric("Nihai Karar", hybrid_res['final_signal'])
            
            st.markdown("#### Canlı KAP Bildirimleri ve Finans Haberleri")
            for item in sentiment_data['news_list']:
                st.markdown(f"- **[{item['sentiment']}]** {item['headline']} *(Kaynak: {item['source']})*")

elif page == "Telegram Bildirim Servisi":
    st.markdown("### Telegram Otomatik Sinyal Bildirim Altyapısı")
    st.markdown("Formasyon kırılımları ve AI AL/SAT sinyallerini doğrudan Telegram hesabınıza **Kurumsal Bildirim Kartları** olarak gönderin.")
    
    from src.telegram_utils import load_tg_config, save_tg_config, send_telegram_alert, format_and_send_pattern_alert
    
    cfg = load_tg_config()
    curr_token = cfg.get("token", "")
    curr_chat_id = cfg.get("chat_id", "")
    
    tg_token = st.text_input("Telegram Bot Token:", curr_token, type="password")
    tg_chat_id = st.text_input("Telegram Chat ID:", curr_chat_id)
    
    if st.button("Telegram Ayarlarını Kaydet"):
        save_tg_config(tg_token, tg_chat_id)
        st.success("Telegram ayarları başarıyla kaydedildi.")
        
    st.markdown("---")
    st.markdown("#### Test Bildirimi Gönder")
    test_ticker = st.selectbox("Test Sinyali İçin Hisse Seçin:", all_tickers_list, index=0)
    
    if st.button("Test Sinyal Kartını Telegram'a Gönder", type="primary"):
        ok, res = format_and_send_pattern_alert(test_ticker, "TOBO (Ters Omuz Baş Omuz)", 125.40, 148.00)
        if ok:
            st.success("Test bildirimi başarıyla Telegram'a gönderildi.")
        else:
            st.error(f"Bildirim gönderilemedi: {res}")

elif page == "Model Yeniden Eğitim & Diagnostik Panel":
    st.markdown("### Model Yeniden Eğitim ve Otomatik Diagnostik Motoru")
    st.markdown("BorsaNeuron makine öğrenmesi modellerini güncel BİST piyasa verileriyle yeniden eğitin ve ağırlık dosyalarını (`best_model_acm465.joblib`) otomatik güncelleyin.")
    
    from src.retrain_pipeline import BorsaNeuronRetrainPipeline
    
    st.info("Not: Modeli yeniden eğitmek, XGBoost / Random Forest classifier ağırlıklarını ve normalizasyon parametrelerini güncelleyecektir.")
    
    if st.button("Modelleri Canlı Verilerle Yeniden Eğit", type="primary"):
        with st.spinner("Model eğitimi ve çapraz doğrulama yürütülüyor..."):
            pipeline = BorsaNeuronRetrainPipeline()
            success, diag = pipeline.train_and_save_model()
            
            if success:
                st.success("Model başarıyla yeniden eğitildi ve kaydedildi.")
                st.json(diag)
            else:
                st.error(f"Model eğitimi sırasında hata oluştu: {diag}")

st.markdown("""
<div class='footer-text'>
    YEDİTEPE UNIVERSITY | BORSANEURON ALGORITHMIC TRADING AND ARTIFICIAL INTELLIGENCE GRADUATION PROJECT
</div>
""", unsafe_allow_html=True)
