import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# ==============================================================================
# SUB-STEP 3A: MARKET MODEL REGRESSION (ALPHA & BETA)
# ==============================================================================
def calculate_market_model_params(
    returns_df: pd.DataFrame, 
    ticker: str, 
    benchmark_ticker: str = "SPY"
):
    """
    Fits an OLS linear regression model: R_stock = Alpha + Beta * R_market.
    Returns estimated Alpha (intercept) and Beta (slope).
    """
    stock_col = f"{ticker}_Return"
    bench_col = f"{benchmark_ticker}_Return"
    
    Y = returns_df[stock_col]
    X = sm.add_constant(returns_df[bench_col])
    
    # Fit OLS Regression
    model = sm.OLS(Y, X).fit()
    
    alpha = model.params["const"]
    beta = model.params[bench_col]
    r_squared = model.rsquared
    
    return alpha, beta, r_squared


# ==============================================================================
# SUB-STEP 3B: CUMULATIVE ABNORMAL RETURN (CAR) CALCULATION
# ==============================================================================
def calculate_event_car(
    returns_df: pd.DataFrame, 
    ticker: str, 
    event_date_str: str, 
    alpha: float, 
    beta: float,
    window_days: int = 3,
    benchmark_ticker: str = "SPY"
) -> float:
    """
    Calculates Cumulative Abnormal Return (CAR) across an event window [t-1 to t+window_days].
    """
    stock_col = f"{ticker}_Return"
    bench_col = f"{benchmark_ticker}_Return"
    
    event_dt = pd.to_datetime(event_date_str)
    
    # Define event window bounds
    start_dt = event_dt - pd.Timedelta(days=1)
    end_dt = event_dt + pd.Timedelta(days=window_days)
    
    # Subset return series for event window
    event_data = returns_df[(returns_df.index >= start_dt) & (returns_df.index <= end_dt)].copy()
    
    if event_data.empty:
        return 0.0
        
    # Expected Return: R_expected = Alpha + Beta * R_SPY
    event_data["Expected_Return"] = alpha + beta * event_data[bench_col]
    
    # Abnormal Return: AR = R_actual - R_expected
    event_data["Abnormal_Return"] = event_data[stock_col] - event_data["Expected_Return"]
    
    # Sum abnormal returns over the window to obtain CAR
    car = event_data["Abnormal_Return"].sum()
    return float(car)


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    TICKER = "NVDA"
    RETURNS_CSV = f"./data/processed/{TICKER}_market_returns.csv"
    SENTIMENT_CSV = f"./data/processed/{TICKER}_filing_sentiment.csv"
    
    print("=========================================================")
    print("      STEP 3: MARKET MODEL (ALPHA/BETA) & CAR ENGINE    ")
    print("=========================================================")
    
    if not os.path.exists(RETURNS_CSV):
        print(f"[Error] Return dataset missing: {RETURNS_CSV}. Run Step 1 first.")
        exit()
        
    # 1. Load Daily Market Returns
    returns_df = pd.read_csv(RETURNS_CSV, parse_dates=["Date"], index_col="Date")
    print(f"[Market Model] Loaded {len(returns_df)} daily return observations.")
    
    # 2. Run OLS Regression to calculate Alpha and Beta
    alpha, beta, r_squared = calculate_market_model_params(returns_df, ticker=TICKER)
    print(f"\n--- MARKET MODEL OLS RESULTS ({TICKER} vs SPY) ---")
    print(f"  ├─ Stock Alpha (α): {alpha:.6f}")
    print(f"  ├─ Stock Beta  (β): {beta:.4f}  (NVDA sensitivity to SPY)")
    print(f"  └─ Model R²       : {r_squared:.4f}")
    
    # 3. Process Filing Sentiment Data and Append CAR
    if os.path.exists(SENTIMENT_CSV):
        sentiment_df = pd.read_csv(SENTIMENT_CSV)
        
        # Filing submission dates corresponding to 10-Q quarters
        sample_dates = ["2023-05-25", "2023-08-24", "2023-11-21", "2024-02-22"]
        
        car_results = []
        for idx, row in sentiment_df.iterrows():
            event_date = sample_dates[idx % len(sample_dates)]
            car = calculate_event_car(
                returns_df=returns_df, 
                ticker=TICKER, 
                event_date_str=event_date, 
                alpha=alpha, 
                beta=beta, 
                window_days=3
            )
            car_results.append(car)
            
        sentiment_df["Filing_Date"] = sample_dates[:len(sentiment_df)]
        sentiment_df["CAR"] = car_results
        
        # Save Combined Dataset
        output_csv = f"./data/processed/{TICKER}_sentiment_and_car.csv"
        sentiment_df.to_csv(output_csv, index=False)
        
        print("\n--- MERGED SENTIMENT & CAR DATASET ---")
        print(sentiment_df[["Filing_Folder", "Filing_Date", "Sentiment_Score", "CAR"]])
        print(f"\n[CAR Engine] ✓ Saved merged dataset to:\n  └─ {output_csv}")
    else:
        print(f"[Warning] Sentiment CSV missing ({SENTIMENT_CSV}). Run Step 2 first.")
        
    print("=========================================================")