CSS_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp { font-family: 'Inter', sans-serif; }

    /* Terminal Dark Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0a0e1a 0%, #0e1117 40%, #111827 100%);
    }

    /* Sidebar - Dark Terminal */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 26, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0,242,255,0.08);
    }

    /* Terminal Brand Header */
    .brand-header {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00f2ff;
        text-shadow: 0 0 20px rgba(0,242,255,0.3);
        letter-spacing: 3px;
        text-transform: uppercase;
        border-bottom: 2px solid rgba(0,242,255,0.3);
        padding-bottom: 12px;
        margin-bottom: 20px;
    }

    /* Page Sub-Header */
    .page-subtitle {
        color: #4a5568;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* Terminal Glass Cards */
    .glass-card {
        background: rgba(14, 17, 23, 0.8);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 4px;
        border: 1px solid rgba(0,242,255,0.08);
        border-left: 3px solid rgba(0,242,255,0.3);
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .glass-card:hover {
        border-left-color: #00f2ff;
        box-shadow: 0 4px 30px rgba(0,242,255,0.1);
    }

    /* Metric Cards - Terminal Style */
    .metric-card {
        background: rgba(14, 17, 23, 0.9);
        border-radius: 4px;
        border: 1px solid rgba(0,242,255,0.12);
        padding: 16px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #00f2ff, transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .metric-card:hover::before { opacity: 1; }
    .metric-card:hover { 
        border-color: rgba(0,242,255,0.3);
        box-shadow: 0 0 20px rgba(0,242,255,0.08);
    }

    /* Metric Values - Cyan Glow */
    .metric-value {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.8rem; font-weight: 700;
        line-height: 1.2;
        margin: 6px 0;
        color: #00f2ff;
        text-shadow: 0 0 15px rgba(0,242,255,0.3);
    }

    .metric-label {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.7rem; color: #4a5568;
        text-transform: uppercase; letter-spacing: 2px; font-weight: 600;
    }

    .metric-delta-up { color: #00ff88; font-weight: 600; font-size: 0.9rem; font-family: 'Roboto Mono', monospace; text-shadow: 0 0 8px rgba(0,255,136,0.3); }
    .metric-delta-down { color: #ff4444; font-weight: 600; font-size: 0.9rem; font-family: 'Roboto Mono', monospace; text-shadow: 0 0 8px rgba(255,68,68,0.3); }
    
    /* Native Metrics override */
    div[data-testid="stMetricValue"] { color: #00f2ff; font-family: 'Roboto Mono', monospace; font-weight: 700; }
    
    /* Streamlit Native Inputs - Terminal Style */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > div, 
    .stNumberInput > div > div > input {
        background-color: rgba(10, 14, 26, 0.8) !important;
        border: 1px solid rgba(0,242,255,0.12) !important;
        color: #e2e8f0 !important;
        border-radius: 2px !important;
        padding: 10px 14px !important;
        font-family: 'Roboto Mono', monospace !important;
    }
    .stTextInput > div > div > input:focus, 
    .stSelectbox > div > div > div:focus {
        border-color: #00f2ff !important;
        box-shadow: 0 0 0 1px rgba(0,242,255,0.3) !important;
    }

    /* Primary Buttons - Terminal Cyan */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d4e6, #00f2ff) !important;
        border: none !important; 
        border-radius: 2px !important;
        font-family: 'Roboto Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 12px 28px !important;
        transition: 0.3s !important;
        color: #0e1117 !important;
        box-shadow: 0 0 20px rgba(0,242,255,0.2) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 35px rgba(0,242,255,0.4) !important;
        transform: translateY(-1px);
    }

    /* Secondary Buttons */
    .stButton > button {
        border-radius: 2px !important;
        font-family: 'Roboto Mono', monospace !important;
        letter-spacing: 1px;
    }

    /* Progress bars - Cyan */
    .stProgress > div > div { background: linear-gradient(90deg, #00d4e6, #00f2ff) !important; border-radius: 2px; }

    hr { border-color: rgba(0,242,255,0.08) !important; margin: 1.5rem 0 !important; }
    
    /* Terminal Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: rgba(0,242,255,0.3); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,242,255,0.5); }

    /* Headings */
    h1, h2, h3 { 
        color: #e2e8f0 !important; 
        font-family: 'Roboto Mono', monospace !important; 
    }

    /* Pattern Badges */
    .pattern-title {
        color: #00f2ff;
        font-family: 'Roboto Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .pattern-type-bullish { color: #00ff88; font-weight: 600; font-size: 0.8rem; }
    .pattern-type-bearish { color: #ff4444; font-weight: 600; font-size: 0.8rem; }
    .pattern-type-neutral { color: #fbbf24; font-weight: 600; font-size: 0.8rem; }

    /* Status Messages */
    .status-msg {
        background: rgba(0,242,255,0.05);
        border-left: 3px solid #00f2ff;
        padding: 12px 20px;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.85rem;
        color: #00f2ff;
        letter-spacing: 1px;
        margin: 10px 0;
    }

    /* News Cards */
    .news-card {
        background: rgba(14, 17, 23, 0.6);
        border: 1px solid rgba(0,242,255,0.06);
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 8px;
        transition: border-color 0.3s;
    }
    .news-card:hover { border-color: rgba(0,242,255,0.2); }

    /* Sentiment */
    .sentiment-positive { color: #00ff88; font-weight: 700; }
    .sentiment-negative { color: #ff4444; font-weight: 700; }
    .sentiment-neutral { color: #94a3b8; font-weight: 600; }

    /* Alert Cards */
    .alert-triggered {
        background: rgba(0,255,136,0.06);
        border: 1px solid rgba(0,255,136,0.2);
        border-left: 3px solid #00ff88;
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .alert-waiting {
        background: rgba(0,242,255,0.04);
        border: 1px solid rgba(0,242,255,0.1);
        border-left: 3px solid rgba(0,242,255,0.3);
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }

    /* Step Cards */
    .step-card {
        background: rgba(0,242,255,0.03);
        border-left: 2px solid rgba(0,242,255,0.2);
        padding: 10px 16px;
        margin-bottom: 8px;
        border-radius: 0 4px 4px 0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0px; }
    .stTabs [data-baseweb="tab"] { 
        font-family: 'Roboto Mono', monospace !important;
        letter-spacing: 1px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #00f2ff !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(0,242,255,0.08) !important;
        border-radius: 2px !important;
    }
</style>
"""
