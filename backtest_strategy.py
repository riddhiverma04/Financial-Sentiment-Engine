import os
import pandas as pd
import numpy as np

def run_divergence_backtest():
    anomalies_csv = "./data/processed/NVDA_divergence_anomalies.csv"

    if not os.path.exists(anomalies_csv):
        print(f"[Error] Missing dataset at {anomalies_csv}. Ensure Steps 1-4 have executed.")
        return

    df = pd.read_csv(anomalies_csv)

    print("=========================================================")
    print("      DIVERGENCE SIGNAL QUANTITATIVE BACKTEST            ")
    print("=========================================================")

    signals = []
    for _, row in df.iterrows():
        div = row.get("Divergence_Metric", 0)
        
        # Signal Rules:
        # Bullish Divergence (Tone > Market CAR) -> Buy (+1)
        # Bearish Divergence (Tone < Market CAR) -> Short (-1)
        # Neutral -> Hold/Cash (0)
        if div > 1.0:
            signal = 1
            action = "LONG (Buy)"
        elif div < -1.0:
            signal = -1
            action = "SHORT (Sell)"
        else:
            signal = 0
            action = "HOLD (Neutral)"
            
        signals.append({
            "Filing_Date": row["Filing_Date"],
            "Divergence_Metric": round(div, 4),
            "Signal": signal,
            "Action": action,
            "CAR": round(row["CAR"], 4)
        })

    results_df = pd.DataFrame(signals)
    
    print("\n--- GENERATED STRATEGY SIGNALS ---")
    print(results_df.to_string(index=False))

    # Save output to reports folder
    os.makedirs("./reports", exist_ok=True)
    output_path = "./reports/backtest_signals.csv"
    results_df.to_csv(output_path, index=False)
    
    print("\n=========================================================")
    print(f"[Backtest Engine] ✓ Strategy logs generated and saved to:\n  └─ {output_path}")
    print("=========================================================")

if __name__ == "__main__":
    run_divergence_backtest()