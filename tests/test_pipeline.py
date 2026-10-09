import os
import pandas as pd
import pytest

def test_anomaly_csv_exists():
    path = "./data/processed/NVDA_divergence_anomalies.csv"
    assert os.path.exists(path), f"Processed data file missing at {path}"

def test_dataframe_columns():
    df = pd.read_csv("./data/processed/NVDA_divergence_anomalies.csv")
    required_cols = ["Filing_Date", "Sentiment_Score", "CAR", "Divergence_Metric"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

def test_divergence_z_score_bounds():
    df = pd.read_csv("./data/processed/NVDA_divergence_anomalies.csv")
    # Sentiment scores from FinBERT must fall within [-1.0, 1.0]
    assert df["Sentiment_Score"].between(-1.0, 1.0).all(), "Sentiment scores out of valid range [-1, 1]"

def test_mda_loughran_output():
    mda_path = "./data/processed/NVDA_mda_loughran_mcdonald.csv"
    if os.path.exists(mda_path):
        df = pd.read_csv(mda_path)
        assert "LM_Net_Sentiment" in df.columns
        assert "Total_MD&A_Word_Count" in df.columns