# Financial Sentiment Engine

**A Python-based financial NLP and quantitative research pipeline for analyzing SEC filings, financial sentiment, and market reactions.**

The Financial Sentiment Engine investigates how sentiment in corporate disclosures relates to subsequent market performance. It combines SEC Form 10-Q filing ingestion, FinBERT sentiment scoring, financial lexicon analysis, benchmark-based return modeling, and a sentiment–market divergence metric.

**Focus areas:** Financial Analysis · Python · Natural Language Processing · Quantitative Finance · Sentiment Analysis · Statistical Modeling

## Project Overview

Corporate filings contain information about business performance, risks, uncertainty, and management's outlook. This project explores how that information can be analyzed systematically and compared with market movements around filing events.

The current implementation focuses on NVIDIA (NVDA) quarterly filings and uses SPY as a market benchmark.

The pipeline is designed to:

- Retrieve SEC Form 10-Q filings and market-price data.
- Extract and score financial text using FinBERT.
- Calculate additional financial-language measures using a custom Loughran–McDonald-inspired lexicon.
- Estimate cumulative abnormal returns (CAR) around filing events.
- Compare standardized sentiment and CAR values to flag potential divergence.
- Export analysis results for further investigation and visualization.

**Important:** Divergence flags are exploratory research indicators. They do not establish market mispricing, causality, or trading profitability.

## How It Works

### 1. Data Ingestion

`step1_ingestion.py`

Retrieves quarterly SEC filings using `sec-edgar-downloader` and downloads historical market prices using `yfinance`.

The current script is configured for:

- **Company:** NVIDIA (NVDA)
- **Filing type:** Form 10-Q
- **Filings requested:** Four recent quarterly filings
- **Market benchmark:** SPY
- **Configured market-data period:** January 1, 2023 – January 1, 2026

The script calculates daily percentage returns for the selected stock and benchmark.

### 2. Financial Text and FinBERT Sentiment

`step2_nlp_sentiment.py`

Processes downloaded filing text by removing HTML markup, normalizing whitespace, and splitting long documents into manageable text chunks.

The pipeline uses the `ProsusAI/finbert` model through Hugging Face Transformers and PyTorch.

For each chunk, it calculates:

**Net sentiment = P(positive) − P(negative)**

The chunk-level scores are averaged to create a document sentiment score ranging from -1 to +1.

- Negative values indicate relatively more negative model-assigned sentiment.
- Positive values indicate relatively more positive model-assigned sentiment.
- Values near zero indicate a more balanced score.

The script saves filing-level sentiment results to a processed CSV file.

### 3. Financial Lexicon Analysis

`advanced_mda_parser.py`

Provides an additional dictionary-based analysis of financial language in the MD&A section of SEC filings.

The parser uses BeautifulSoup, regular expressions, and a custom set of financial terms inspired by the Loughran–McDonald financial dictionary.

It calculates word counts and frequency percentages for categories including:

- Positive language
- Negative language
- Uncertainty
- Litigious language
- Constraining language

It also calculates a net sentiment measure from positive and negative word frequencies.

**Methodology note:** The current implementation uses a custom keyword list inspired by the Loughran–McDonald approach. It should not be described as a complete implementation of the official dictionary unless the full dictionary is incorporated.

### 4. Cumulative Abnormal Returns (CAR)

`step3_car_calculation.py`

Combines filing sentiment with market-return data to examine market performance around filing events.

The project uses a benchmark-based regression approach to estimate expected returns and compare them with observed returns.

Cumulative abnormal returns aggregate abnormal returns over the selected event window. The current README specifies a three-day event window, `[-1, +1]`; confirm that the calculation and filing-date alignment in the implementation match this specification before interpreting results.

CAR estimates are sensitive to the benchmark, estimation period, event dates, and other information released around the same time.

### 5. Sentiment–Market Divergence

`step4_divergence_engine.py`

Combines sentiment and CAR into a standardized divergence measure.

The implementation standardizes both columns and calculates:

\[
D_t = Z(\text{Sentiment}_t) - Z(\text{CAR}_t)
\]

where:

- \(Z(\text{Sentiment}_t)\) is the standardized sentiment score.
- \(Z(\text{CAR}_t)\) is the standardized cumulative abnormal return.
- \(D_t\) is the difference between the standardized measures.

The current implementation uses an absolute divergence threshold of 1.0 standard deviation to flag observations.

These flags indicate differences between the two measures under the chosen standardization. They are not independently validated buy or sell recommendations.

### 6. Visualization and Backtesting

The repository includes:

- `plot_divergence.py` — creates a visualization of sentiment and CAR-related results.
- `backtest_strategy.py` — explores a strategy based on divergence signals.

Backtest performance should be interpreted only after checking event timing, signal construction, transaction costs, benchmark comparisons, and the risk of look-ahead bias. No profitability claim is made by this README.

## Pipeline Architecture

```text
SEC EDGAR filings + market-price data
                 |
                 v
       1. Data Ingestion
          step1_ingestion.py
                 |
                 v
       2. Financial Text Analysis
          step2_nlp_sentiment.py
          FinBERT sentiment scores
                 |
                 +----------------------+
                 |                      |
                 v                      v
       3. CAR Calculation       Financial Lexicon Analysis
          step3_car_calculation.py     advanced_mda_parser.py
                 |
                 v
       4. Sentiment–Market Divergence
          step4_divergence_engine.py
                 |
                 v
       Processed CSV outputs
                 |
                 v
       Visualization / Backtesting
       plot_divergence.py
       backtest_strategy.py
```

## Technology Stack

| Area | Tools |
|---|---|
| Programming language | Python |
| Data processing | pandas, NumPy |
| Financial data | yfinance |
| SEC filing ingestion | sec-edgar-downloader |
| NLP model | FinBERT via Hugging Face Transformers |
| Model inference | PyTorch |
| Text parsing | BeautifulSoup, regular expressions, lxml |
| Statistical analysis | statsmodels and regression-based calculations |
| Visualization | Matplotlib, Seaborn |
| Testing | pytest |

## Repository Structure

```text
Financial-Sentiment-Engine/
├── data/
├── reports/
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py
├── step1_ingestion.py
├── step2_nlp_sentiment.py
├── step3_car_calculation.py
├── step4_divergence_engine.py
├── advanced_mda_parser.py
├── backtest_strategy.py
└── plot_divergence.py
```

The `data/`, `reports/`, and `tests/` directories contain supporting project assets. Consult the repository for the current contents of each directory.

## Getting Started

### Prerequisites

- Python compatible with the pinned dependencies in `requirements.txt`
- Git
- Internet access for downloading filings, market data, and the FinBERT model on first use

### 1. Clone the repository

```bash
git clone https://github.com/riddhiverma04/Financial-Sentiment-Engine.git
cd Financial-Sentiment-Engine
```

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first installation may take time because PyTorch and Transformers are among the dependencies.

### 4. Configure SEC contact information

The ingestion script currently contains a placeholder email address. Before running it, replace the placeholder with your own valid contact email in accordance with SEC fair-access guidance.

Also review the ticker, date range, filing count, and output paths in the scripts if you want to analyze a different company or period.

### 5. Run the main pipeline

Run each stage from the repository root in the following order:

```bash
python step1_ingestion.py
python step2_nlp_sentiment.py
python step3_car_calculation.py
python step4_divergence_engine.py
```

Run the additional analysis scripts when their expected input files have been created:

```bash
python advanced_mda_parser.py
python plot_divergence.py
python backtest_strategy.py
```

Some scripts currently use hard-coded ticker values and file paths. If you change the ticker or data period, make sure the configuration is consistent across all stages.

### 6. Run tests

The project includes a `tests/` directory and lists pytest in its dependencies. Run:

```bash
python -m pytest tests/
```

Check the test results and resolve any missing-input or environment issues before treating the pipeline as reproducible.

## Limitations and Research Considerations

- **Company coverage:** The current scripts are configured primarily for NVDA rather than a fully generalized multi-company system.
- **Text extraction:** HTML and regular-expression parsing may not isolate every MD&A section correctly across different filing formats.
- **Sentiment scoring:** FinBERT results depend on the text provided, chunking, model behavior, and filing content.
- **Lexicon analysis:** The custom keyword list is not equivalent to the full official Loughran–McDonald dictionary.
- **Event studies:** CAR estimates depend on event-date accuracy, the benchmark, regression assumptions, and the chosen event window.
- **Divergence scoring:** Standardization over a small number of filing observations can be unstable.
- **Backtesting:** Results can be affected by look-ahead bias, transaction costs, event selection, and limited observations.

This project is for educational and research purposes. It does not constitute investment advice.

## Future Improvements

Potential next steps include:

- Generalize ticker, date range, and filing-type configuration across the entire pipeline.
- Improve MD&A extraction and add tests for different filing formats.
- Add a reproducible comparison between FinBERT and the financial lexicon.
- Validate filing dates and event windows systematically.
- Add baseline comparisons, transaction costs, drawdown, and out-of-sample backtesting.
- Include representative output tables and visualizations with documented data periods.
- Improve error handling, logging, and automated tests.

## Author

**Riddhi Verma**

Independent project exploring financial analysis, natural language processing, and quantitative research using Python.

**GitHub repository:** [Financial Sentiment Engine](https://github.com/riddhiverma04/Financial-Sentiment-Engine)

