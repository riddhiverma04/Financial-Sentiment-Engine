import os
import pandas as pd
import numpy as np

# ==============================================================================
# SUB-STEP 4A: Z-SCORE NORMALIZATION & DIVERGENCE METRIC CALCULATION
# ==============================================================================
def compute_divergence_metrics(
    df: pd.DataFrame, 
    threshold: float = 1.0
) -> pd.DataFrame:
    """
    Standardizes Sentiment Scores and CAR values into Z-scores,
    calculates the Divergence Metric (D_t = Z_Sentiment - Z_CAR),
    and flags sentiment-market anomalies based on standard deviation thresholds.
    """
    data = df.copy()
    
    # Calculate means and standard deviations
    mean_s, std_s = data["Sentiment_Score"].mean(), data["Sentiment_Score"].std()
    mean_car, std_car = data["CAR"].mean(), data["CAR"].std()
    
    # Avoid division by zero if std is zero
    std_s = std_s if std_s != 0 else 1.0
    std_car = std_car if std_car != 0 else 1.0
    
    # 1. Z-Score Normalization
    data["Z_Sentiment"] = (data["Sentiment_Score"] - mean_s) / std_s
    data["Z_CAR"] = (data["CAR"] - mean_car) / std_car
    
    # 2. Divergence Metric: D_t = Z_Sentiment - Z_CAR
    data["Divergence_Metric"] = data["Z_Sentiment"] - data["Z_CAR"]
    
    # 3. Anomaly Classification
    def classify_anomaly(div_val):
        if div_val > threshold:
            return "Bullish Divergence (Positive Tone, Negative Market)"
        elif div_val < -threshold:
            return "Bearish Divergence (Negative Tone, Positive Market)"
        else:
            return "Aligned (Normal Market Processing)"
            
    data["Anomaly_Flag"] = data["Divergence_Metric"].apply(classify_anomaly)
    data["Is_Anomaly"] = data["Divergence_Metric"].abs() > threshold
    
    return data


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    TICKER = "NVDA"
    INPUT_CSV = f"./data/processed/{TICKER}_sentiment_and_car.csv"
    OUTPUT_CSV = f"./data/processed/{TICKER}_divergence_anomalies.csv"
    
    print("=========================================================")
    print("   STEP 4: DIVERGENCE SCORE ENGINE & ANOMALY FLAGGING    ")
    print("=========================================================")
    
    if not os.path.exists(INPUT_CSV):
        print(f"[Error] Input dataset missing: {INPUT_CSV}. Run Step 3 first.")
        exit()
        
    # Load combined Sentiment and CAR dataset
    df = pd.read_csv(INPUT_CSV)
    print(f"[Divergence Engine] Loaded {len(df)} filing event records.")
    
    # Compute Divergence and Flags (Threshold set to 1.0 standard deviations)
    results_df = compute_divergence_metrics(df, threshold=1.0)
    
    # Save Final Anomaly Dataset
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n--- FINAL DIVERGENCE & ANOMALY MATRIX ---")
    display_cols = ["Filing_Date", "Sentiment_Score", "CAR", "Z_Sentiment", "Z_CAR", "Divergence_Metric", "Anomaly_Flag"]
    print(results_df[display_cols].to_string(index=False))
    
    print("\n=========================================================")
    print(f"[Divergence Engine] ✓ Pipeline complete! Final output saved to:\n  └─ {OUTPUT_CSV}")
    print("=========================================================")