import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def plot_sentiment_vs_car():
    csv_path = "./data/processed/NVDA_divergence_anomalies.csv"
    
    if not os.path.exists(csv_path):
        print(f"[Error] Dataset missing at {csv_path}. Run Step 4 first.")
        return

    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. Plot Sentiment Bars with clean blue color
    color_sentiment = "#1f77b4"
    bars = ax1.bar(
        df["Filing_Date"], 
        df["Sentiment_Score"], 
        color=color_sentiment, 
        alpha=0.7, 
        width=0.3, 
        label="FinBERT Net Sentiment"
    )
    ax1.set_xlabel("SEC 10-Q Filing Date", fontweight="bold", labelpad=10)
    ax1.set_ylabel("FinBERT Net Sentiment Score", color=color_sentiment, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_sentiment)
    ax1.set_ylim(-1.0, 1.0)

    # 2. Plot CAR Line on Secondary Axis
    ax2 = ax1.twinx()
    color_car = "#d95f02"
    line = ax2.plot(
        df["Filing_Date"], 
        df["CAR"], 
        color=color_car, 
        marker="o", 
        linewidth=2.5, 
        markersize=8, 
        label="3-Day CAR"
    )
    ax2.set_ylabel("Cumulative Abnormal Return (CAR)", color=color_car, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_car)

    # 3. Highlight Significant Anomalies (|Divergence| > 1.5)
    for idx, row in df.iterrows():
        if abs(row.get("Divergence_Metric", 0)) > 1.5:
            ax1.axvspan(idx - 0.15, idx + 0.15, color="#ff3333", alpha=0.2)
            ax1.text(
                idx, row["Sentiment_Score"] + 0.05, 
                "ANOMALY", 
                ha="center", color="#cc0000", fontweight="bold", fontsize=9
            )

    plt.title("NVDA: SEC Filing FinBERT Sentiment vs. Market CAR Divergence", fontsize=12, fontweight="bold", pad=15)
    fig.tight_layout()

    os.makedirs("./reports/figures", exist_ok=True)
    output_path = "./reports/figures/sentiment_vs_car_divergence.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n[Visualization Engine] ✓ Clean chart saved to:\n  └─ {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_sentiment_vs_car()