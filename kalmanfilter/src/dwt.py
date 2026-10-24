import glob
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt


def load_local_nifty_data(path_pattern="../data/NIFTY*.csv"):
    """
    Finds and loads Nifty CSV files from the local directory.
    Assumes standard columns: Date, Open, High, Low, Close (case-insensitive).
    """
    all_files = glob.glob(path_pattern)
    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern: {path_pattern}")

    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # Standardize column names
    df.columns = [c.strip().capitalize() for c in df.columns]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df.set_index('Date', inplace=True)

    return df


def dwt_denoise(data, wavelet='db4', level=3):
    """
    Applies Discrete Wavelet Transform (DWT) multi-level decomposition
    and universal soft thresholding to denoise price series.
    """
    # 1. Wavelet Decomposition
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # 2. Estimate noise variance using Median Absolute Deviation (MAD) of the finest detail coefficients
    mad = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))
    sigma = mad / 0.6745

    # 3. Universal threshold (VisuShrink)
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # 4. Threshold detail coefficients (keep approximation coefficients intact)
    thresholded_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        thresholded_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))

    # 5. Reconstruct denoised signal via Inverse DWT
    denoised = pywt.waverec(thresholded_coeffs, wavelet)

    # Align lengths in case reconstruction padding causes minor mismatch
    if len(denoised) > len(data):
        denoised = denoised[:len(data)]
    elif len(denoised) < len(data):
        denoised = np.pad(denoised, (0, len(data) - len(denoised)), 'edge')

    return denoised


def run_dwt_backtest():
    # 1. Load local data
    df = load_local_nifty_data("../data/NIFTY*.csv")

    # 2. Apply DWT Denoising to Close Prices
    close_prices = df['Close'].values
    df['DWT_Trend'] = dwt_denoise(close_prices, wavelet='db4', level=3)

    # 3. Strategy Logic: Long when price is above the DWT trend line, Cash otherwise
    df['Signal'] = np.where(df['Close'] > df['DWT_Trend'], 1, 0)
    df['Position'] = df['Signal'].shift(1)  # Shift to avoid look-ahead bias

    # 4. Returns & Performance Metrics
    df['Market_Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Market_Returns']

    df['Cumulative_Market'] = (1 + df['Market_Returns'].fillna(0)).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns'].fillna(0)).cumprod()

    total_days = len(df)
    cagr_strat = (df['Cumulative_Strategy'].iloc[-1]) ** (252 / total_days) - 1
    cagr_mkt = (df['Cumulative_Market'].iloc[-1]) ** (252 / total_days) - 1
    sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252)

    print("=== DWT Denoising Strategy Results ===")
    print(f"Strategy CAGR: {cagr_strat * 100:.2f}%")
    print(f"Benchmark CAGR: {cagr_mkt * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")

    # 5. Plot Performance
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative_Market'], label='Nifty Buy & Hold', color='gray', alpha=0.7)
    plt.plot(df.index, df['Cumulative_Strategy'], label='DWT Strategy', color='purple', linewidth=1.5)
    plt.title("Discrete Wavelet Transform (DWT) Denoising Strategy vs. Nifty Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Growth")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_dwt_backtest()