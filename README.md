# Multi-Source Financial News & Filings "Anomaly & Sentiment Divergence" Engine

An end-to-end quantitative NLP, data engineering, and asset pricing pipeline that quantifies narrative shifts in SEC disclosures, benchmarks sentiment against post-filing market equity movements, and generates backtested trading signals based on market mispricings.

---

## 📌 Executive Summary
Financial markets frequently misprice qualitative risk factors disclosed in SEC filings due to processing delays in unstructured text. This engine bridges narrative text and asset pricing by isolating **Item 2: Management's Discussion and Analysis (MD&A)** in SEC Form 10-Q filings, applying dual NLP sentiment extraction (FinBERT Transformer Deep Learning vs. Loughran-McDonald Financial Lexicon), and measuring historical divergence against firm-specific **Cumulative Abnormal Returns (CAR)**.

By standardizing market surprise signals into standardized Z-scores (`D_t`), the engine identifies statistical anomalies where equity market price action diverges significantly from underlying narrative disclosure sentiment.

---

## 🛠️ System Architecture & Workflow

```text
[ SEC EDGAR / Yahoo Finance ]
              │
              ▼
    1. Ingestion Pipeline (step1_ingestion.py)
              │
              ├──► Preprocessed Market Prices & Disclosures
              │
              ▼
    2. Dual NLP Engine (step2_nlp_sentiment.py & advanced_mda_parser.py)
              │   ├── Regular Expression HTML Cleaning (Item 2 MD&A Isolation)
              │   ├── FinBERT Neural Sentiment Model (Probabilistic Classification)
              │   └── Loughran-McDonald Lexicon Analysis (Uncertainty, Litigious, Net Sentiment)
              │
              ▼
    3. Quantitative Asset Pricing Model (step3_car_calculation.py)
              │   └── 250-Day Rolling OLS Linear Regression vs. Benchmark (SPY)
              │   └── 3-Day Event Window Cumulative Abnormal Returns (CAR)
              │
              ▼
    4. Anomaly & Signal Generation Engine (step4_divergence_engine.py)
              │   └── Divergence Metric: D_t = Z(Sentiment) - Z(CAR)
              │
              ▼
    5. Backtest Strategy & Visualization (backtest_strategy.py & plot_divergence.py)
              └── Long/Short Equity Backtest Execution & Diagnostic Plots

## 🔬 Core Methodology

- **MD&A Narrative Extraction:** Regex-based HTML parsing to isolate Item 2 (MD&A) from SEC 10-Q filings.
- **Dual Sentiment Engine:** Benchmarks FinBERT Transformer probabilities against Loughran-McDonald financial dictionary counts.
- **Cumulative Abnormal Returns (CAR):** 250-day rolling OLS regression against SPY over a 3-day event window (`[-1, +1]`).
- **Anomaly Detection Signal:** Standardized Z-score divergence metric (`D_t = Z_Sentiment - Z_CAR`) to identify mispriced disclosures.

## 📁 Repository Structure

financial-sentiment-engine/
├── config.py                 # Central parameters & ticker configurations
├── step1_ingestion.py        # SEC EDGAR & market data ingestion
├── step2_nlp_sentiment.py    # FinBERT inference engine
├── step3_car_calculation.py  # OLS regression & CAR asset pricing
├── step4_divergence_engine.py# Z-score normalization & anomaly flagger
├── advanced_mda_parser.py   # Regex parser & Loughran-McDonald lexicon
├── backtest_strategy.py     # Long/Short systematic backtest
├── plot_divergence.py        # Diagnostic plot generator
├── requirements.txt          # Dependencies
├── .gitignore                # Version control exclusions
└── tests/                    # Automated test suite
    └── test_pipeline.py      # Pipeline & schema integrity tests

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9+
- macOS / Linux / Windows WSL

---

### Option A: For Setup After Cloning

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/riddhiverma04/Financial-Sentiment-Engine.git](https://github.com/riddhiverma04/Financial-Sentiment-Engine.git)
   cd Financial-Sentiment-Engine

2. **Create and activate virtual environment:**
   '''bash
   python3 -m venv venv
   source venv/bin/activate

3. **Install project dependencies:**
   '''bash
   pip install -r requirements.txt

### Option B: For Running Pipeline Commands

1. **Activate virtual environment:**
   '''bash
   source venv/bin/activate

2. **Execute the End-to-End Pipeline:**
   '''bash
   # Step 1: Ingest SEC disclosures & market equity prices
   python step1_ingestion.py

   # Step 2: Compute FinBERT sentiment probabilities
   python step2_nlp_sentiment.py

   # Step 3: Compute OLS market beta & Cumulative Abnormal Returns (CAR)
   python step3_car_calculation.py

   # Step 4: Generate divergence metrics & flag anomalies
   python step4_divergence_engine.py

   # Additional Analysis: Run MD&A Loughran-McDonald parser & backtest
   python advanced_mda_parser.py
   python backtest_strategy.py

3. **Run Automated Test Suite:**
   '''bash
   purest tests/

