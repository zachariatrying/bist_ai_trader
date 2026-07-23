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
    
    .terminal-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    
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

# Sidebar Navigation
st.sidebar.markdown("<div class='brand-header'>BİST AI TRADER</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:0.75rem; color:#64748b; margin-top:-5px; margin-bottom:15px;'>KURUMSAL TRADİNG TERMİNALİ</div>", unsafe_allow_html=True)

page = st.sidebar.radio("SİSTEM ANALİTİK MENÜSÜ", [
    "🚀 Canlı BİST Sinyal & Formasyon Taraması",
    "📰 Canlı KAP & Finansal Haber Analizi",
    "🔍 Tek Hisseli Yapay Zekâ Analizi",
    "💼 Portföy Risk & Backtest Radarı"
])

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:0.7rem; color:#475569; font-family:sans-serif;'>ENGINE: XGBoost + Pattern + KAP Scraper<br>DATA: Live Real-Time<br>VERSION: v3.7-BIST Live Fix</div>", unsafe_allow_html=True)


# ==============================================================================
# MODÜL 1: CANLI BİST SİNYAL & FORMASYON TARAMASI (ALWAYS RENDERING FIX)
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
        target_tickers = all_tickers_list

    # Session State Setup
    if 'scan_data' not in st.session_state:
        st.session_state['scan_data'] = []

    col_btn, col_info = st.columns([1.5, 3])
    with col_btn:
        run_button = st.button(f"🚀 {len(target_tickers)} Hissede Taramayı Başlat", type="primary")

    if run_button or len(st.session_state['scan_data']) == 0:
        with st.spinner("Canlı BİST verileri ve 3 Aşamalı Yapay Zeka Sinyalleri işleniyor..."):
            all_eval = []
            progress_bar = st.progress(0)
            
            for idx, t in enumerate(target_tickers):
                try:
                    df_hist, live_p = LiveBISTFeedEngine.get_live_stock_data(t)
                    if df_hist is not None and not df_hist.empty:
                        res = confirmation_engine.analyze_ticker_triple_confirmation(df_hist, ticker_name=t)
                        
                        # Strategic rationale
                        if res['triple_confirmed']:
                            rationale = f"Hissede {res['pattern_name']} formasyonu doğrulandı. AI %{res['ml_prob_up_pct']} yükseliş veriyor. Tarihsel uyum %{res['stock_win_rate_pct']} yüksek."
                        elif res['double_confirmed']:
                            rationale = f"Formasyon ({res['pattern_name']}) ve Yapay Zekâ olumlu (%{res['ml_prob_up_pct']})."
                        else:
                            rationale = f"Formasyon ayı veya momentum zayıf (RSI: {res['rsi_14']}). Temkinli olunmalı."

                        item = {
                            'Hisse Kodu': t,
                            'Canlı Fiyat': f"{live_p:.2f} TL",
                            'Nihai Karar': res['final_signal'],
                            'Neden Almalıyız / Beklemeliyiz?': rationale,
                            '1. Aşama Formasyon': f"{res['pattern_name']} (%{res['pattern_confidence']})",
                            '2. Aşama AI Olasılık': f"%{res['ml_prob_up_pct']}",
                            '3. Aşama Hisse Uyum': f"%{res['stock_win_rate_pct']} ({res['compliance_status']})",
                            'Giriş Fiyatı': f"{res['entry_price']} TL",
                            'Hedef-1 (Kısa Vade T+5)': f"{res['target_1']} TL",
                            'Hedef-2 (Orta Vade T+15)': f"{res['target_2']} TL",
                            'Stop-Loss (Zarar Kes)': f"{res['stop_loss']} TL",
                            'Risk/Ödül Oranı': res['risk_reward'],
                            'raw_prob': res['ml_prob_up_pct'],
                            'triple_confirmed': res['triple_confirmed'],
                            'double_confirmed': res['double_confirmed'],
                            'pattern_name': res['pattern_name']
                        }
                        all_eval.append(item)
                except Exception:
                    pass
                progress_bar.progress((idx + 1) / len(target_tickers))

            st.session_state['scan_data'] = all_eval

    # Filter stored scan data dynamically for immediate UI responsiveness
    raw_results = st.session_state.get('scan_data', [])
    filtered_results = []

    for r in raw_results:
        pass_conf = True
        if conf_filter == "🚀 ÜÇLÜ ONAYLI (En Güvenli - Formasyon + AI + Hisse Başarı)":
            pass_conf = r['triple_confirmed']
        elif conf_filter == "📈 ÇİFTE ONAYLI (Formasyon + AI Teyitli)":
            pass_conf = r['double_confirmed'] or r['triple_confirmed']
        elif conf_filter == "⚠️ TEK ONAYLI (Sadece Formasyon / Sadece AI)":
            pass_conf = not r['triple_confirmed']

        pass_pat = True
        if pattern_type_filter == "Boğa Formasyonları (TOBO, Çanak, Flama, Dip)":
            pass_pat = any(k in r['pattern_name'].upper() for k in ['TOBO', 'ÇANAK', 'DİP', 'BAYRAK', 'FLAMA', 'YÜKSELEN', 'TREND'])
        elif pattern_type_filter == "Kırılım & Trend Takibi":
            pass_pat = any(k in r['pattern_name'].upper() for k in ['KIRILIM', 'TREND', 'KANAL'])

        if pass_conf and pass_pat:
            filtered_results.append(r)

    # Always fallback to raw results if strict filter yields 0
    display_results = filtered_results if filtered_results else raw_results

    if display_results:
        if not filtered_results and raw_results:
            st.warning("⚠️ Seçtiğiniz dar filtreye birebir uyan hisse bulunamadı. Genel taranan hisse listesi gösteriliyor.")
        else:
            st.success(f"✅ Sinyal Tablosu Gösteriliyor ({len(display_results)} Hisse)")

        display_df = pd.DataFrame(display_results).drop(columns=['raw_prob', 'triple_confirmed', 'double_confirmed', 'pattern_name'], errors='ignore')
        st.dataframe(display_df, use_container_width=True)

        st.markdown("#### 🎯 Pozisyon Kartları & Hedef/Stop Analizi")
        for r in display_results[:10]:
            card_color = "#10b981" if "ÜÇLÜ ONAYLI" in r['Nihai Karar'] or "ÇİFTE ONAYLI" in r['Nihai Karar'] else "#f59e0b"
            st.markdown(f"""
            <div class='terminal-card' style='border-left: 4px solid {card_color};'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-size:1.3rem; font-weight:700; color:#38bdf8;'>{r['Hisse Kodu']} ({r['Canlı Fiyat']})</span>
                    <span style='font-size:1.1rem; font-weight:700; color:{card_color};'>{r['Nihai Karar']}</span>
                </div>
                <div style='margin-top:10px; font-size:0.95rem; color:#cbd5e1; line-height:1.7;'>
                    • <b>💡 Neden Almalıyız / Beklemeliyiz?:</b> <mark>{r['Neden Almalıyız / Beklemeliyiz?']}</mark><br>
                    • <b>1. Aşama Formasyon:</b> {r['1. Aşama Formasyon']}<br>
                    • <b>2. Aşama AI Yükseliş Olasılığı:</b> {r['2. Aşama AI Olasılık']}<br>
                    • <b>3. Aşama Hisse Uyum:</b> {r['3. Aşama Hisse Uyum']}<br>
                    • <b>📍 Giriş Fiyatı:</b> <span style='color:#38bdf8; font-weight:bold;'>{r['Giriş Fiyatı']}</span><br>
                    • <b>🎯 Hedef 1 (Kısa Vade T+5):</b> <span style='color:#10b981; font-weight:bold;'>{r['Hedef-1 (Kısa Vade T+5)']}</span> | <b>🎯 Hedef 2 (Orta Vade T+15):</b> <span style='color:#10b981; font-weight:bold;'>{r['Hedef-2 (Orta Vade T+15)']}</span><br>
                    • <b>🛑 Stop-Loss (Zarar Kes):</b> <span style='color:#ef4444; font-weight:bold;'>{r['Stop-Loss (Zarar Kes)']}</span> | <b>⚖️ Risk/Ödül Oranı:</b> {r['Risk/Ödül Oranı']}
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
    st.markdown("Seçilen BİST hissesinin **formasyonlarını, hedeflerini, stop seviyelerini ve neden almamız gerektiğini** detaylı açıklar.")

    col1, col2 = st.columns([2, 1])
    with col1:
        target_stock = st.selectbox("Analiz Edilecek Hissedarı Seçin:", all_tickers_list, index=0)
    with col2:
        period_choice = st.selectbox("Grafik Periyodu:", ["3mo", "6mo", "1y", "2y"], index=1)

    if st.button("🔍 Detaylı Analizi Başlat", type="primary"):
        with st.spinner(f"{target_stock} detaylı verileri işleniyor..."):
            try:
                df_stock, live_p = LiveBISTFeedEngine.get_live_stock_data(target_stock)
                if df_stock is not None and not df_stock.empty:
                    res = confirmation_engine.analyze_ticker_triple_confirmation(df_stock, ticker_name=target_stock)
                    
                    # Core Executive Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Nihai Karar", res['status'])
                    c2.metric("Canlı Fiyat", f"{live_p:.2f} TL")
                    c3.metric("Yapay Zekâ Prob.", f"%{res['ml_prob_up_pct']}")
                    c4.metric("Hisse Formasyon Uyum", f"%{res['stock_win_rate_pct']}")

                    st.markdown("---")
                    
                    # Detailed Trading Targets & Stop Loss Card
                    st.markdown("#### 🎯 Alım-Satım Seviyeleri ve Stratejik Gerekçe")
                    st.markdown(f"""
                    <div class='terminal-card' style='border-left: 5px solid #38bdf8;'>
                        <h4 style='color:#38bdf8; margin-top:0;'>💡 Neden Almalıyız / Beklemeliyiz?</h4>
                        <p style='font-size:1.05rem; line-height:1.7; color:#f1f5f9;'>
                            • <b>Tespit Edilen Formasyon:</b> <mark>{res['pattern_name']}</mark> ({res['pattern_type']} - Güven: %{res['pattern_confidence']})<br>
                            • <b>Tüm Çıkan Formasyonlar:</b> {', '.join(res['all_patterns'])}<br>
                            • <b>RSI (14) Göstergesi:</b> {res['rsi_14']} (Momentum Durumu)<br>
                            • <b>Tarihsel Uyum:</b> {res['compliance_status']} (%{res['stock_win_rate_pct']})<br>
                            • <b>Stratejik Yorum:</b> Hissede {res['pattern_name']} formasyonu aktif olup yapay zekâ %{res['ml_prob_up_pct']} yükseliş olasılığı öngörmektedir.
                        </p>
                        <hr style='border-color:#334155;'>
                        <div style='display:grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size:1rem;'>
                            <div><b>📍 Giriş Fiyatı:</b><br><span style='color:#38bdf8; font-size:1.2rem; font-weight:bold;'>{res['entry_price']} TL</span></div>
                            <div><b>🎯 Hedef 1 (T+5):</b><br><span style='color:#10b981; font-size:1.2rem; font-weight:bold;'>{res['target_1']} TL</span></div>
                            <div><b>🎯 Hedef 2 (T+15):</b><br><span style='color:#10b981; font-size:1.2rem; font-weight:bold;'>{res['target_2']} TL</span></div>
                            <div><b>🛑 Stop-Loss:</b><br><span style='color:#ef4444; font-size:1.2rem; font-weight:bold;'>{res['stop_loss']} TL</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
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
