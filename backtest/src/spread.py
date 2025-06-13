import json
import pandas as pd

# Load data
with open("./data/btc_1m_185days.json") as f:
    data = json.load(f)

df = pd.DataFrame(data["candles"])

# Convert to float
df["fillHigh"] = df["fillHigh"].astype(float)
df["fillLow"] = df["fillLow"].astype(float)
df["oracleClose"] = df["oracleClose"].astype(float)

# Method 1: Using midpoint as fair value
df["mid_price"] = (df["fillHigh"] + df["fillLow"]) / 2
df["spread_abs"] = df["fillHigh"] - df["fillLow"]
df["spread_bps_mid"] = (df["spread_abs"] / df["mid_price"]) * 10000

# Method 2: Using oracle as fair value
df["spread_bps_oracle"] = (df["spread_abs"] / df["oracleClose"]) * 10000

# Buy/sell costs relative to oracle
df["buy_cost_oracle"] = (
    (df["fillHigh"] - df["oracleClose"]) / df["oracleClose"]
) * 10000
df["sell_cost_oracle"] = (
    (df["oracleClose"] - df["fillLow"]) / df["oracleClose"]
) * 10000

# Buy/sell costs relative to mid
df["buy_cost_mid"] = ((df["fillHigh"] - df["mid_price"]) / df["mid_price"]) * 10000
df["sell_cost_mid"] = ((df["mid_price"] - df["fillLow"]) / df["mid_price"]) * 10000

print("BTC Spread Analysis: Midpoint vs Oracle\n")

print("METHOD 1: MIDPOINT AS FAIR VALUE")
print(f"Candles: {len(df):,}")
print(f"Mean spread: {df['spread_bps_mid'].mean():.2f} bps")
print(f"Median spread: {df['spread_bps_mid'].median():.2f} bps")
print(f"90th percentile: {df['spread_bps_mid'].quantile(0.90):.2f} bps")
print(f"99th percentile: {df['spread_bps_mid'].quantile(0.99):.2f} bps")
print(f"Buy cost (avg): {df['buy_cost_mid'].mean():.2f} bps")
print(f"Sell cost (avg): {df['sell_cost_mid'].mean():.2f} bps")
print(
    f"Round trip cost: {df['buy_cost_mid'].mean() + df['sell_cost_mid'].mean():.2f} bps"
)

print("\nMETHOD 2: ORACLE AS FAIR VALUE")
print(f"Candles: {len(df):,}")
print(f"Mean spread: {df['spread_bps_oracle'].mean():.2f} bps")
print(f"Median spread: {df['spread_bps_oracle'].median():.2f} bps")
print(f"90th percentile: {df['spread_bps_oracle'].quantile(0.90):.2f} bps")
print(f"99th percentile: {df['spread_bps_oracle'].quantile(0.99):.2f} bps")
print(f"Buy cost (avg): {df['buy_cost_oracle'].mean():.2f} bps")
print(f"Sell cost (avg): {df['sell_cost_oracle'].mean():.2f} bps")
print(
    f"Round trip cost: {df['buy_cost_oracle'].mean() + df['sell_cost_oracle'].mean():.2f} bps"
)

print("\nORACLE vs MIDPOINT COMPARISON")
print(
    f"Oracle vs Mid price diff (avg): {(df['oracleClose'] - df['mid_price']).mean():.2f}"
)
print(
    f"Oracle vs Mid price diff (std): {(df['oracleClose'] - df['mid_price']).std():.2f}"
)
print(f"Oracle vs Mid correlation: {df['oracleClose'].corr(df['mid_price']):.6f}")
