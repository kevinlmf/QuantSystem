import pandas as pd

def load_csv_data(path):
    """
    Load market data from a CSV file and preprocess it for TradingEnv.

    Args:
        path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing OHLCV data
    """
    df = pd.read_csv(path, skiprows=3, header=None)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Drop the Date column and keep only OHLCV numerical values
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    return df






