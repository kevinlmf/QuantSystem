import yfinance as yf
import os

def download_yahoo_data(symbol="SPY", start="2023-01-01", end="2024-01-01"):
    df = yf.download(symbol, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    

    os.makedirs("data", exist_ok=True)

    filepath = f"data/{symbol}_1d.csv"
    df.to_csv(filepath)
    print(f"✅ Saved data to: {filepath}")

if __name__ == "__main__":
    download_yahoo_data(symbol="SPY", start="2023-01-01", end="2024-01-01")
