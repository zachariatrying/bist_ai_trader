# BorsaNeuron: Algorithmic Stock Price Forecasting & Automated Technical Pattern Scanner Platform

Welcome to the official repository of **BorsaNeuron**, a comprehensive quantitative decision support and analytics workstation designed for the Borsa Istanbul (BIST) equity market. This platform integrates statistical machine learning classifiers, unsupervised regime clustering, macroeconomic indicators, and dynamic chart pattern scanners into a unified interactive business intelligence terminal.

Developed as a Graduation Thesis project at **Yeditepe University**, Department of Management Information Systems, BorsaNeuron bridges the gap between high-dimensional statistical modeling and active portfolio execution.

---

## Key Platform Features

### 1. Quantitative Inference Engine & Live Forecasting
* **Real-Time Data Streaming:** Leverages yFinance API streams to download live BIST daily price matrices dynamically.
* **Algorithmic Inference:** Standardizes real-time indicators and runs online inferences using optimized supervised machine learning models to forecast 5-day future closing price directions (`Target_T5`).
* **Core Model:** Implements serialized high-performance XGBoost and Random Forest classifiers.

### 2. Unsupervised Market Regime Segmentation
* **K-Means Clustering:** Segments technical indicator profiles into 5 unique market regimes (breakout, taban, consolidation, recovery, bull run) optimized via Elbow analysis.
* **Dimensionality Reduction:** Projects high-dimension indicators into 2D feature space using Principal Component Analysis (PCA) for visual cluster separation and regime profile analysis.

### 3. Out-of-Sample Portfolio Backtest Simulator
* **Chronological Simulation:** Simulates an algorithmic strategy on out-of-sample testing data (2023–2024), starting with 100,000 TL capital.
* **Friction and slippage modeling:** Incorporates transaction costs, commission rates, and slippage margins by capturing 30% of target gains and applying 50% of downside risks.
* **Index Comparison:** Renders active growth curves comparing the BorsaNeuron AI Strategy directly against a baseline Buy & Hold index.

### 4. Dynamic Win-Rate Weighting Engine
* **Historical Compliance:** Performs a 1-year rolling historical backtest for the queried BIST stock under the AI strategy to calculate its unique win rate.
* **Karar Düzeltme (Risk Modulator):** Automatically adjusts recommendation alerts (AL, AL Yüksek Risk, BEKLE) based on the stock's compliance rate, issuing warning flags if historical compliance falls below 48%.

### 5. Automated Technical Pattern Scanner
* **Geometrical Scanning:** Automates the scanning of historical BIST datasets to flag geometric chart formations such as Cup & Handle, Head and Shoulders (OBO), Inverse Head and Shoulders (TOBO), and Flags.
* **BIST 537 Comprehensive Scan:** Analyzes all 537 active BIST stocks, grouping active formations dynamically and charting indicator averages.

---

## Software Requirements and Installation

### 1. Requirements
* **Python v3.9 or v3.11**
* **Git** (for version control)
* **Playwright** (for automated dashboard screenshot capture)

### 2. Setup Guide
Clone the repository and install all required python libraries:

```bash
# Clone the repository
git clone https://github.com/zachariatrying/borsaneuron.git
cd borsaneuron

# Install python dependencies
pip install -r requirements.txt
```

### 3. Running the Dashboard
To start the BorsaNeuron interactive web dashboard locally, execute:

```bash
streamlit run src/app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## File Hierarchy

The project repository is structured logically to segregate offline data science scripts, serializations, live Streamlit frontends, and graduation thesis documentation:

```text
borsaneuron/
├── .streamlit/
│   └── config.toml                  # Streamlit interface configurations
├── bist_ai_dataset_real_30cols.csv   # Compressed BIST historical dataset (79.18 MB)
├── best_scaler_acm465.joblib         # Serialized StandardScaler weights
├── best_model_acm465.joblib          # Serialized MLP/Random Forest model
├── acm465_proje.py                   # Offline Data Science and Modeling Pipeline
├── requirements.txt                  # Python package dependency list
├── Dockerfile                        # Production container blueprint
├── capture_ui.py                     # Automated Playwright screenshot script
├── src/
│   ├── app.py                        # Streamlit web application entrypoint
│   ├── theme.py                      # UI Color theme and stylesheet guide
│   ├── data_manager.py               # Data loaders and yfinance API client
│   ├── earnings_data.py              # Macro-corporate data structures
│   ├── macro_data.py                 # Macroeconomic data parsers
│   └── verify_tobo_strict.py         # TOBO and Cup-Handle pattern scanners
└── graduation_thesis/
    ├── Tez_Metni_Final.md            # Complete Thesis Markdown draft
    ├── BorsaNeuron_Graduation_Thesis.docx # Compiled MS Word Thesis (Academic Standard)
    ├── BorsaNeuron_Graduation_Thesis.pdf  # Compiled PDF Thesis
    ├── convert_to_docx.py            # Custom Markdown -> DOCX compiler
    ├── convert_to_pdf.py             # Custom Markdown -> PDF compiler
    └── images/                       # Screen captures and figures
```

---

*BorsaNeuron — Designed and Developed by İbrahim Tatar (MIS Department, Yeditepe University)*
