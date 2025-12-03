import matplotlib.pyplot as plt
import pandas as pd
import os
def _save_plot(default_name: str, out_path: str | None):
    """
    Helper to save the current Matplotlib figure.
    If out_path is None, saves to charts/<default_name>.
    Ensures the directory exists.
    """
    if out_path is None:
        out_path = os.path.join("charts", default_name)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved → {out_path}")

def plot_sleep_index(df_daily: pd.DataFrame, out_path: str | None = None):
    plt.figure()
    df_daily["sleep_index"].plot()
    plt.axhline(0, linestyle="--")
    plt.title("Daily Sleep Index")
    plt.ylabel("SleepIndex (hours vs target)")
    plt.xlabel("Date")
    plt.tight_layout()
    _save_plot("plot_sleep_index.png", out_path)


def plot_pnl_curve(trades: pd.DataFrame, out_path: str | None = None):
    plt.figure()
    trades["cum_pnl"].plot()
    plt.title("Strategy Cumulative PnL")
    plt.ylabel("PnL ($)")
    plt.xlabel("Date (contract expiry)")
    plt.tight_layout()
    _save_plot("plot_pnl_curve.png", out_path)

def plot_sleep_volatility(df_daily: pd.DataFrame, out_path: str | None = None):
    """
    Sleep 'volatility' = rolling 7d std of SleepIndex.
    """
    vol = df_daily["sleep_index"].rolling(7).std().dropna()
    plt.figure()
    plt.hist(vol, bins=20)
    plt.title("Distribution of 7-Day SleepIndex Volatility")
    plt.xlabel("Std Dev of SleepIndex")
    plt.ylabel("Count")
    plt.tight_layout()
    _save_plot("plot_sleep_volatility.png", out_path)

