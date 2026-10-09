import os
import warnings
import pandas as pd
import yfinance as yf
from sec_edgar_downloader import Downloader

# Suppress LibreSSL warning on macOS
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# ==============================================================================
# SUB-STEP 1A: SEC EDGAR FILINGS DOWNLOADER
# ==============================================================================
def download_sec_filings(
    ticker: str, 
    filing_type: str = "10-Q", 
    limit: int = 4, 
    download_dir: str = "./data",
    user_email: str = "student@university.edu"
) -> str:
    """
    Downloads raw SEC quarterly (10-Q) filings directly from SEC EDGAR.
    """
    print(f"\n[SEC EDGAR] Initiating download of {limit} recent {filing_type} filings for '{ticker}'...")
    
    dl = Downloader(
        company_name="UniversityCapstoneProject", 
        email_address=user_email, 
        download_folder=download_dir
    )
    
    dl.get(filing_type, ticker, limit=limit)
    
    target_path = os.path.abspath(f"{download_dir}/sec-edgar-filings/{ticker}/{filing_type}")
    print(f"[SEC EDGAR] ✓ Success! Filings saved to local path:\n  └─ {target_path}")
    return target_path


# ==============================================================================
# SUB-STEP 1B: FINANCIAL MARKET PRICE & RETURN INGESTION
# ==============================================================================
def fetch_market_returns(
    ticker: str, 
    start_date: str, 
    end_date: str,
    benchmark_ticker: str = "SPY"
) -> pd.DataFrame:
    """
    Pulls daily closing price series for target ticker and market benchmark index (SPY).
    Handles MultiIndex column structures cleanly across all yfinance versions.
    """
    print(f"\n[yFinance] Fetching price series for '{ticker}' and '{benchmark_ticker}' from {start_date} to {end_date}...")
    
    tickers_to_pull = [ticker, benchmark_ticker]
    
    # Download raw daily data
    raw_data = yf.download(tickers_to_pull, start=start_date, end=end_date)
    
    # Extract price column (fallback to 'Close' if 'Adj Close' isn't top-level)
    if "Adj Close" in raw_data.columns.levels[0]:
        price_data = raw_data["Adj Close"]
    elif "Close" in raw_data.columns.levels[0]:
        price_data = raw_data["Close"]
    else:
        # If single-level columns or auto-flattened
        price_data = raw_data
        
    # Ensure correct column selection & ordering
    price_data = price_data[[ticker, benchmark_ticker]]
    
    # Calculate daily percentage returns: R_t = (P_t - P_{t-1}) / P_{t-1}
    returns_df = price_data.pct_change().dropna()
    returns_df.columns = [f"{ticker}_Return", f"{benchmark_ticker}_Return"]
    
    print(f"[yFinance] ✓ Success! Retrieved {len(returns_df)} daily return trading periods.")
    return returns_df


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    TARGET_TICKER = "NVDA"
    START_DATE = "2023-01-01"
    END_DATE = "2026-01-01"
    STUDENT_EMAIL = "student@university.edu"
    
    print("=========================================================")
    print("      STEP 1: MULTI-SOURCE DATA INGESTION ENGINE         ")
    print("=========================================================")
    
    # 1. Download SEC 10-Q Filings
    filings_folder = download_sec_filings(
        ticker=TARGET_TICKER, 
        filing_type="10-Q", 
        limit=4, 
        user_email=STUDENT_EMAIL
    )
    
    # 2. Extract Stock and Market Returns
    market_returns = fetch_market_returns(
        ticker=TARGET_TICKER, 
        start_date=START_DATE, 
        end_date=END_DATE
    )
    
    # 3. Save Market Data to CSV for downstream steps
    os.makedirs("./data/processed", exist_ok=True)
    csv_output_path = f"./data/processed/{TARGET_TICKER}_market_returns.csv"
    market_returns.to_csv(csv_output_path)
    print(f"\n[Data Pipeline] Processed return data saved to:\n  └─ {csv_output_path}")
    
    print("\n--- SAMPLE DAILY RETURNS DATA FRAME ---")
    print(market_returns.head(5))
    print("=========================================================")