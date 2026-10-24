import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_local_nifty_data(path_pattern="../data/NIFTY*.csv"):
    """
    Finds and loads Nifty CSV files from the local directory.
    Assumes standard columns: Date, Open, High, Low, Close (case-insensitive).
    """
    all_files = glob.glob(path_pattern)
    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern: {path_pattern}")

    # Load the first matching file (or combine them if multiple files split by year)
    df_list = []
    for filename in all_files:
        temp_df = pd.read_csv(filename)
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    # Standardize column names (strip spaces, lowercase/titlecase match)
    df.columns = [c.strip().capitalize() for c in df.columns]

    # Ensure correct sorting by date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df.set_index('Date', inplace=True)

    return df


def calculate_mama(df, fast_limit=0.5, slow_limit=0.05):
    """
    Calculates MAMA and FAMA using John Ehlers' Hilbert Transform DSP method.
    """
    # Use Typical Price or (High + Low) / 2
    if 'High' in df.columns and 'Low' in df.columns:
        price = (df['High'] + df['Low']) / 2.0
    else:
        price = df['Close']

    n = len(price)

    smooth = np.zeros(n)
    detrender = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    jI = np.zeros(n)
    jQ = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    period = np.zeros(n)
    smooth_period = np.zeros(n)
    phase = np.zeros(n)
    mama = np.zeros(n)
    fama = np.zeros(n)

    mama[:] = price.values
    fama[:] = price.values
    p = price.values

    for i in range(6, n):
        # 1. 4-bar WMA Smoothing
        smooth[i] = (4 * p[i] + 3 * p[i - 1] + 2 * p[i - 2] + p[i - 3]) / 10.0

        # 2. Hilbert Transform Detrender
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
                        - 0.5769 * smooth[i - 4] - 0.0962 * smooth[i - 6]) * (0.075 * period[i - 1] + 0.54)

        # 3. Compute In-Phase (I) and Quadrature (Q) components
        i1[i] = detrender[i - 3]
        q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i - 2]
                 - 0.5769 * detrender[i - 4] - 0.0962 * detrender[i - 6]) * (0.075 * period[i - 1] + 0.54)

        # 4. Advance phase by 90 degrees
        jI[i] = (0.0962 * i1[i] + 0.5769 * i1[i - 2] - 0.5769 * i1[i - 4] - 0.0962 * i1[i - 6]) * (
                    0.075 * period[i - 1] + 0.54)
        jQ[i] = (0.0962 * q1[i] + 0.5769 * q1[i - 2] - 0.5769 * q1[i - 4] - 0.0962 * q1[i - 6]) * (
                    0.075 * period[i - 1] + 0.54)

        # 5. Phasor addition
        i2[i] = i1[i] - jQ[i]
        q2[i] = q1[i] + jI[i]

        # Smooth components
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1]

        # 6. Homodyne Discriminator for Period
        re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]

        re[i] = 0.2 * re[i] + 0.8 * re[i - 1]
        im[i] = 0.2 * im[i] + 0.8 * im[i - 1]

        if im[i] != 0 and re[i] != 0:
            period[i] = 360.0 / (np.arctan(im[i] / re[i]) * (180.0 / np.pi))
        if period[i] > 1.5 * period[i - 1]:
            period[i] = 1.5 * period[i - 1]
        if period[i] < 0.67 * period[i - 1]:
            period[i] = 0.67 * period[i - 1]
        if period[i] < 6:
            period[i] = 6
        if period[i] > 50:
            period[i] = 50

        period[i] = 0.2 * period[i] + 0.8 * period[i - 1]
        smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1]

        # 7. Phase & Phase Rate
        if i1[i] != 0:
            phase[i] = np.arctan(q1[i] / i1[i]) * (180.0 / np.pi)

        delta_phase = phase[i - 1] - phase[i]
        if delta_phase < 1:
            delta_phase = 1

        # 8. Adaptive Alpha
        alpha = fast_limit / delta_phase
        if alpha < slow_limit:
            alpha = slow_limit
        if alpha > fast_limit:
            alpha = fast_limit

        # 9. MAMA & FAMA update
        mama[i] = alpha * p[i] + (1 - alpha) * mama[i - 1]
        fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * fama[i - 1]

    df['MAMA'] = mama
    df['FAMA'] = fama
    return df


def run_backtest():
    # Load from local folder pattern
    df = load_local_nifty_data("../data/NIFTY*.csv")

    # Calculate MAMA indicators
    df = calculate_mama(df)

    # Strategy Crossover Signal (Shifted to prevent look-ahead bias)
    df['Signal'] = np.where(df['MAMA'] > df['FAMA'], 1, 0)
    df['Position'] = df['Signal'].shift(1)

    # Performance metrics
    df['Market_Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Market_Returns']

    df['Cumulative_Market'] = (1 + df['Market_Returns'].fillna(0)).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns'].fillna(0)).cumprod()

    # Compute stats
    total_days = len(df)
    cagr_strat = (df['Cumulative_Strategy'].iloc[-1]) ** (252 / total_days) - 1
    cagr_mkt = (df['Cumulative_Market'].iloc[-1]) ** (252 / total_days) - 1
    sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252)

    print("=== Backtest Results (Local Nifty Data) ===")
    print(f"Strategy CAGR: {cagr_strat * 100:.2f}%")
    print(f"Benchmark CAGR: {cagr_mkt * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")

    # Plot performance curve
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative_Market'], label='Nifty Buy & Hold', color='gray', alpha=0.7)
    plt.plot(df.index, df['Cumulative_Strategy'], label='MAMA Strategy', color='blue', linewidth=1.5)
    plt.title("MAMA Filter Strategy vs. Nifty Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Growth")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_backtest()