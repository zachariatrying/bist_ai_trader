import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import json
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="BİST AI Trader - Kurumsal Trading Terminali",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
TERMINAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Roboto+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .main { background-color: #0b0f19; }
    
    .stAppHeader { background-color: rgba(11, 15, 25, 0.9); }
    
    /* Terminal Card */
    .terminal-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.4rem !important;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .brand-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 1px;
    }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# Cache & Dataset Setup
@st.cache_data(ttl=3600)
def load_bist_tickers():
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'bist_tickers.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return sorted([item['ticker'].replace('.IS', '') for item in data if 'ticker' in item])
    except Exception:
        pass
    return ["THYAO", "GARAN", "EREGL", "ASELS", "FROTO", "KCHOL", "TUPRS", "SISE", "BIMAS", "AKBNK", "SAHOL", "SASA", "HEKTS", "PETKM", "PGSUS", "ISCTR", "YKBNK", "VAKBN", "HALKB", "EKGYO"]

all_tickers_list = load_bist_tickers()

# Import Custom Engines
try:
    from src.double_confirmation_engine import DoubleConfirmationEngine
    from src.kap_news_scraper import KAPNewsScraper
    from src.news_sentiment import NewsSentimentEngine
    from src.risk_analytics import BorsaNeuronRiskEngine
    from src.vibe_trading import VibeTradingAssistant
    from src.live_bist_feed import LiveBISTFeedEngine
    confirmation_engine = DoubleConfirmationEngine()
    vibe_assistant = VibeTradingAssistant()
except Exception as e:
    confirmation_engine = None
    vibe_assistant = None

# Sidebar Navigation (4 SADE & NET İŞ İSTASYONU)
st.sidebar.markdown("<div class='brand-header'>BİST AI TRADER</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:0.75rem; color:#64748b; margin-top:-5px; margin-bottom:15px;'>KURUMSAL TRADİNG TERMİNALİ</div>", unsafe_allow_html=True)

page = st.sidebar.radio("SİSTEM ANALİTİK MENÜSÜ", [
    "🚀 Canlı BİST Sinyal & Formasyon Taraması",
    "📰 Canlı KAP & Finansal Haber Analizi",
    "🔍 Tek Hisseli Yapay Zekâ Analizi",
    "💼 Portföy Risk & Backtest Radarı"
])

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:0.7rem; color:#475569; font-family:sans-serif;'>ENGINE: XGBoost + Pattern + Live BIST Feed<br>DATA: Live Real-Time<br>VERSION: v3.5-BIST Live</div>", unsafe_allow_html=True)


# ==============================================================================
# MODÜL 1: CANLI BİST SİNYAL & FORMASYON TARAMASI (100% LIVE REAL-TIME FEED)
# ==============================================================================
if page == "🚀 Canlı BİST Sinyal & Formasyon Taraması":
    st.markdown("### 🚀 Canlı BİST Sinyal & Formasyon Taraması")
    st.markdown("Borsa İstanbul Hisselerinde **1. Aşama (Teknik Formasyon)**, **2. Aşama (Yapay Zekâ ML)** ve **3. Aşama (Hisse Başarı Uyum)** Analizi ile Sinyal Üretir.")

    # Filtreler Barı
    col1, col2, col3 = st.columns(3)
    with col1:
        scan_scope = st.selectbox("1. Tarama Kapsamı:", ["BİST 30 Hisseleri", "Özel Hisse Seçimi", "BİST 100 Hisseleri", "Tüm BİST (537 Hisse)"], key="scope_sel")
    with col2:
        conf_filter = st.selectbox("2. Onay Seviyesi Filtresi:", [
            "HEPSİ (Tüm Sinyaller)",
            "🚀 ÜÇLÜ ONAYLI (En Güvenli - Formasyon + AI + Hisse Başarı)",
            "📈 ÇİFTE ONAYLI (Formasyon + AI Teyitli)",
            "⚠️ TEK ONAYLI (Sadece Formasyon / Sadece AI)"
        ], key="conf_sel")
    with col3:
        pattern_type_filter = st.selectbox("3. Formasyon Türü:", [
            "HEPSİ (Tüm Formasyonlar)",
            "Boğa Formasyonları (TOBO, Çanak, Flama, Dip)",
            "Kırılım & Trend Takibi"
        ], key="pat_sel")

    # Kapsama Göre Hisse Listesi Oluşturma
    if scan_scope == "BİST 30 Hisseleri":
        b30 = ["THYAO", "GARAN", "EREGL", "ASELS", "FROTO", "KCHOL", "TUPRS", "SISE", "BIMAS", "AKBNK", "SAHOL", "SASA", "HEKTS", "ISCTR", "YKBNK", "PETKM", "PGSUS"]
        target_tickers = [t for t in b30 if t in all_tickers_list]
    elif scan_scope == "Özel Hisse Seçimi":
        selected_custom = st.multiselect("Taranacak Hisseleri Seçin:", all_tickers_list, default=["THYAO", "GARAN", "EREGL", "ASELS", "TUPRS", "SISE"], key="custom_tickers_multisel")
        target_tickers = selected_custom if selected_custom else ["THYAO", "GARAN", "EREGL"]
    elif scan_scope == "BİST 100 Hisseleri":
        target_tickers = all_tickers_list[:100] if len(all_tickers_list) >= 100 else all_tickers_list
    else:
        target_tickers = all_tickers_list[:50]

    # Session State Persistence
    if 'scan_results' not in st.session_state:
        st.session_state['scan_results'] = []

    col_btn, col_info = st.columns([1.5, 3])
    with col_btn:
        run_button = st.button(f"🧠 {len(target_tickers)} Hissede Taramayı Başlat", type="primary")

    if run_button or not st.session_state['scan_results']:
        with st.spinner("Canlı BİST verileri ve 3 Aşamalı Yapay Zeka Sinyalleri işleniyor..."):
            results = []
            all_eval = []
            progress_bar = st.progress(0)
            
            for idx, t in enumerate(target_tickers):
                try:
                    df_hist, live_p = LiveBISTFeedEngine.get_live_stock_data(t)
                    if df_hist is not None and not df_hist.empty:
                        res = confirmation_engine.analyze_ticker_triple_confirmation(df_hist, ticker_name=t)
                        item = {
                            'Hisse': t,
                            'Durum': res['status'],
                            'Nihai Sinyal': res['final_signal'],
                            '1. Aşama Formasyon': f"{res['pattern_name']} (%{res['pattern_confidence']})",
                            '2. Aşama AI Olasılık': f"%{res['ml_prob_up_pct']}",
                            '3. Aşama Hisse Uyum': f"%{res['stock_win_rate_pct']} ({res['compliance_status']})",
                            'Canlı Fiyat': f"{live_p:.2f} TL",
                            'Hedef-1': f"{res['target_1']} TL",
                            'Stop-Loss': f"{res['stop_loss']} TL",
                            'Risk/Ödül': res['risk_reward'],
                            'raw_prob': res['ml_prob_up_pct'],
                            'triple_confirmed': res['triple_confirmed'],
                            'double_confirmed': res['double_confirmed']
                        }
                        all_eval.append(item)

                        # Apply Filters
                        pass_conf = True
                        if conf_filter == "🚀 ÜÇLÜ ONAYLI (En Güvenli - Formasyon + AI + Hisse Başarı)":
                            pass_conf = res['triple_confirmed']
                        elif conf_filter == "📈 ÇİFTE ONAYLI (Formasyon + AI Teyitli)":
                            pass_conf = res['double_confirmed'] or res['triple_confirmed']
                        elif conf_filter == "⚠️ TEK ONAYLI (Sadece Formasyon / Sadece AI)":
                            pass_conf = not res['triple_confirmed']

                        pass_pat = True
                        if pattern_type_filter == "Boğa Formasyonları (TOBO, Çanak, Flama, Dip)":
                            pass_pat = any(k in res['pattern_name'].upper() for k in ['TOBO', 'ÇANAK', 'DİP', 'BAYRAK', 'FLAMA', 'YÜKSELEN'])
                        elif pattern_type_filter == "Kırılım & Trend Takibi":
                            pass_pat = any(k in res['pattern_name'].upper() for k in ['KIRILIM', 'TREND', 'KANAL'])

                        if pass_conf and pass_pat:
                            results.append(item)
                except Exception:
                    pass
                progress_bar.progress((idx + 1) / len(target_tickers))

            if not results and all_eval:
                all_eval.sort(key=lambda x: x['raw_prob'], reverse=True)
                results = all_eval[:5]

            st.session_state['scan_results'] = results

    # Render Results permanently from Session State
    scan_results = st.session_state.get('scan_results', [])
    if scan_results:
        st.success(f"✅ Sinyal Tablosu Gösteriliyor ({len(scan_results)} Hisse)")
        display_df = pd.DataFrame(scan_results).drop(columns=['raw_prob', 'triple_confirmed', 'double_confirmed'], errors='ignore')
        st.dataframe(display_df, use_container_width=True)

        st.markdown("#### 🎯 Sinyal Pozisyon Kartları")
        for r in scan_results[:10]:
            card_color = "#10b981" if "ÜÇLÜ ONAYLI" in r['Nihai Sinyal'] or "ÇİFTE ONAYLI" in r['Nihai Sinyal'] else "#f59e0b"
            st.markdown(f"""
            <div class='terminal-card' style='border-left: 4px solid {card_color};'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-size:1.3rem; font-weight:700; color:#38bdf8;'>{r['Hisse']} ({r['Canlı Fiyat']})</span>
                    <span style='font-size:1.1rem; font-weight:700; color:{card_color};'>{r['Nihai Sinyal']}</span>
                </div>
                <div style='margin-top:10px; font-size:0.92rem; color:#cbd5e1; line-height:1.6;'>
                    • <b>1. Aşama (Formasyon):</b> {r['1. Aşama Formasyon']}<br>
                    • <b>2. Aşama (Yapay Zekâ Tahmini):</b> {r['2. Aşama AI Olasılık']} Yükseliş Olasılığı<br>
                    • <b>3. Aşama (Hisse Başarım Uyum):</b> {r['3. Aşama Hisse Uyum']}<br>
                    • <b>Canlı Fiyat:</b> {r['Canlı Fiyat']} | <b>Hedef-1:</b> {r['Hedef-1']} | <b>Stop Loss:</b> <span style='color:#ef4444;'>{r['Stop-Loss']}</span> | <b>Risk/Ödül:</b> {r['Risk/Ödül']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# MODÜL 2: CANLI KAP & FİNANSAL HABER ANALİZİ
# ==============================================================================
elif page == "📰 Canlı KAP & Finansal Haber Analizi":
    st.markdown("### 📰 Canlı KAP Bildirimleri ve Haber Analiz Motoru")
    st.markdown("Aratılan BİST hissesi için **Kamuyu Aydınlatma Platformu (KAP)** bildirimlerini ve canlı finans haberlerini tarayıp duygu skoru hesaplar.")

    selected_news_ticker = st.selectbox("Haber & KAP Bildirimi Taranacak Hisse:", all_tickers_list, index=0)
    
    if st.button("📰 Canlı KAP Haberlerini Taramasını Çalıştır", type="primary"):
        with st.spinner(f"{selected_news_ticker} canlı KAP ve haber başlıkları çekiliyor..."):
            news_data = NewsSentimentEngine.get_news_sentiment(selected_news_ticker)
            
            st.markdown(f"#### Genel Duygu Durumu: **{news_data['overall_sentiment']}** (Net Skor: {news_data['net_score']})")
            
            st.markdown("#### Canlı Bildirim ve Haber Başlıkları")
            for item in news_data['news_list']:
                tag_color = "#10b981" if item['sentiment'] == "POZİTİF" else ("#ef4444" if item['sentiment'] == "NEGATİF" else "#94a3b8")
                st.markdown(f"""
                <div class='terminal-card' style='border-left: 3px solid {tag_color};'>
                    <b>[{item['sentiment']}]</b> {item['headline']}<br>
                    <small style='color:#94a3b8;'>Kaynak: {item['source']}</small>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# MODÜL 3: TEK HİSSELİ YAPAY ZEKÂ ANALİZİ
# ==============================================================================
elif page == "🔍 Tek Hisseli Yapay Zekâ Analizi":
    st.markdown("### 🔍 Tek Hisseli Detaylı Yapay Zekâ Analizi")
    st.markdown("Seçilen BİST hissesinin grafik formasyonlarını, teknik göstergelerini ve AI tahmin parametrelerini detaylı inceler.")

    col1, col2 = st.columns([2, 1])
    with col1:
        target_stock = st.selectbox("Hisse Seçin:", all_tickers_list, index=0)
    with col2:
        period_choice = st.selectbox("Grafik Periyodu:", ["3mo", "6mo", "1y", "2y"], index=1)

    if st.button("🔍 Detaylı Analizi Başlat", type="primary"):
        with st.spinner(f"{target_stock} detaylı verileri işleniyor..."):
            try:
                df_stock, live_p = LiveBISTFeedEngine.get_live_stock_data(target_stock)
                if df_stock is not None and not df_stock.empty:
                    res = confirmation_engine.analyze_ticker_triple_confirmation(df_stock, ticker_name=target_stock)
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Nihai Karar", res['status'])
                    col_b.metric("Canlı Fiyat", f"{live_p:.2f} TL")
                    col_c.metric("Yapay Zekâ Yükseliş Prob.", f"%{res['ml_prob_up_pct']}")
                    col_d.metric("Hisse Formasyon Başarısı", f"%{res['stock_win_rate_pct']}")
                    
                    # Chart
                    fig = px.line(df_stock, y='Close', title=f"{target_stock} Canlı Fiyat Grafiği ve Kapanış Trendi", labels={'value': 'Fiyat (TL)', 'Date': 'Tarih'})
                    fig.update_traces(line_color='#38bdf8')
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as ex:
                st.error(f"Hisse verisi işlenirken hata oluştu: {ex}")


# ==============================================================================
# MODÜL 4: PORTFÖY RİSK & BACKTEST RADARI
# ==============================================================================
elif page == "💼 Portföy Risk & Backtest Radarı":
    st.markdown("### 💼 Portföy Risk Analitiği & Monte Carlo Simülasyonu")
    st.markdown("Portföyün **Sharpe Oranı, Sortino Oranı, Maximum Drawdown (MDD), Value at Risk (VaR)** ve **Monte Carlo Gelecek Fiyat Simülasyonunu** hesaplar.")

    risk_engine = BorsaNeuronRiskEngine(risk_free_rate=0.45)
    selected_risk_stock = st.selectbox("Risk Analizi Yapılacak Hisse:", all_tickers_list, index=0)

    if st.button("📊 Risk ve Simülasyonu Çalıştır", type="primary"):
        with st.spinner(f"{selected_risk_stock} risk metrikleri hesaplanıyor..."):
            try:
                hist_data, _ = LiveBISTFeedEngine.get_live_stock_data(selected_risk_stock)
                if hist_data is not None and not hist_data.empty:
                    prices = hist_data['Close']
                    equity_sim = 100000 * (prices / prices.iloc[0])
                    metrics, drawdowns = risk_engine.calculate_portfolio_metrics(equity_sim)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Sharpe Oranı", f"{metrics['sharpe_ratio']:.2f}")
                    c2.metric("Sortino Oranı", f"{metrics['sortino_ratio']:.2f}")
                    c3.metric("Max Drawdown", f"%{metrics['max_drawdown_pct']:.2f}")
                    c4.metric("Günlük VaR (%95)", f"%{metrics['var_95_daily_pct']:.2f}")
                    
                    # Drawdown Chart
                    fig_dd = px.line(drawdowns * 100, title=f"{selected_risk_stock} Historical Drawdown Curve (%)")
                    fig_dd.update_traces(line_color='#ef4444')
                    st.plotly_chart(fig_dd, use_container_width=True)
            except Exception as ex:
                st.error(f"Risk analizi hatası: {ex}")

st.markdown("---")
st.markdown("<div style='text-align:center; font-size:0.8rem; color:#64748b;'>BİST AI TRADER QUANTITATIVE SYSTEMS</div>", unsafe_allow_html=True)
