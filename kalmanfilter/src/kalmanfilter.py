import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_nifty_data():
    """Locates and loads all NIFTY historical CSV files from the parallel data directory."""
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / 'data'

    # Match NIFTY CSV files in the data directory
    csv_pattern = str(data_dir / 'NIFTY*.csv')
    files = glob.glob(csv_pattern)

    if not files:
        # Fallback to any CSV if specific naming varies
        csv_pattern = str(data_dir / '*.csv')
        files = glob.glob(csv_pattern)

    if not files:
        raise FileNotFoundError(f"No CSV data files found in directory: {data_dir}")

    df_list = [pd.read_csv(file) for file in files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def parse_and_resample(df):
    """Parses raw data, cleans columns, and resamples to weekly closing prices with increases/decreases."""
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify date/time and close price columns flexibly
    date_col = next((c for c in df.columns if 'date' in c or 'time' in c), 'date')
    close_col = next((c for c in df.columns if 'close' in c), 'close')

    # Parse dates and set index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    df.set_index(date_col, inplace=True)

    # Resample to weekly frequency (Friday close)
    weekly_df = df[close_col].resample('W-FRI').last().dropna().to_frame(name='close')

    # Calculate weekly price changes and directional indicators
    weekly_df['price_change'] = weekly_df['close'].diff()
    weekly_df['weekly_return'] = weekly_df['close'].pct_change()
    weekly_df['increase'] = weekly_df['price_change'] > 0
    weekly_df['decrease'] = weekly_df['price_change'] < 0

    return weekly_df


def apply_kalman_filter(prices, Q=1e-5, R=1e-2):
    """
    Implements a 1D linear Kalman Filter to track the underlying trend price.
    - Q: Process noise covariance (trust in the model dynamics)
    - R: Measurement noise covariance (trust in the observed prices)
    """
    n = len(prices)
    filtered_state_means = np.zeros(n)

    # Initialize state estimate and error covariance
    xhat = prices.iloc[0]
    P = 1.0

    for i in range(n):
        z = prices.iloc[i]

        # 1. Prediction Step
        xhat_minus = xhat
        P_minus = P + Q

        # 2. Update Step
        K = P_minus / (P_minus + R)  # Kalman gain
        xhat = xhat_minus + K * (z - xhat_minus)
        P = (1 - K) * P_minus

        filtered_state_means[i] = xhat

    return pd.Series(filtered_state_means, index=prices.index)


def generate_signals(weekly_df):
    """Applies Kalman filter and creates trading signals based on price crossovers."""
    weekly_df['kalman_trend'] = apply_kalman_filter(weekly_df['close'])

    # Determine position: +1 when price is above the Kalman trend, -1 when below
    weekly_df['position'] = 0
    weekly_df.loc[weekly_df['close'] > weekly_df['kalman_trend'], 'position'] = 1
    weekly_df.loc[weekly_df['close'] < weekly_df['kalman_trend'], 'position'] = -1

    # Identify trade execution points where position changes
    weekly_df['trade_signal'] = weekly_df['position'].diff()

    return weekly_df


def plot_results(weekly_df):
    """Plots the NIFTY weekly close prices, Kalman filter trend, and buy/sell signals."""
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_df.index, weekly_df['close'], label='Weekly Close Price', color='dodgerblue', alpha=0.5)
    plt.plot(weekly_df.index, weekly_df['kalman_trend'], label='Kalman Filter Trend', color='darkorange', linewidth=2)

    # Buy signals (transition upward)
    buy_signals = weekly_df[weekly_df['trade_signal'] > 0]
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=120, label='Buy Signal', zorder=5)

    # Sell signals (transition downward)
    sell_signals = weekly_df[weekly_df['trade_signal'] < 0]
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=120, label='Sell Signal',
                zorder=5)

    plt.title('NIFTY Weekly Trend & Kalman Filter Trading Signals', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def main():
    print("Loading NIFTY historical data files...")
    df = load_nifty_data()
    print(f"Loaded raw records: {len(df)}")

    print("Parsing data and aggregating weekly metrics...")
    weekly_df = parse_and_resample(df)

    print("Constructing Kalman Filter and generating signals...")
    weekly_df = generate_signals(weekly_df)

    # Display summary of recent weekly parsing & signals
    print("\nRecent Weekly Summary (Last 5 weeks):")
    print(weekly_df[['close', 'price_change', 'increase', 'decrease', 'kalman_trend', 'position']].tail())

    print("\nPlotting visual graph...")
    plot_results(weekly_df)


if __name__ == "__main__":
    main()