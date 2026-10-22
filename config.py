import os

# Target Tickers & Benchmarks
PRIMARY_TICKER = "NVDA"
BENCHMARK_TICKER = "SPY"

# Time Horizon & Event Windows
CAR_EVENT_WINDOW_DAYS = 3  # Event window [-1, +1] around 10-Q filing
OLS_LOOKBACK_TRADING_DAYS = 250  # Estimation window for market beta (β)

# Statistical Thresholds
ANOMALY_Z_THRESHOLD = 1.0  # Divergence threshold for market surprise signals

# Directories
DATA_DIR = "./data"
PROCESSED_DATA_DIR = "./data/processed"
REPORTS_DIR = "./reports"
FIGURES_DIR = "./reports/figures"

# Ensure output directories exist automatically
for folder in [DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(folder, exist_ok=True)