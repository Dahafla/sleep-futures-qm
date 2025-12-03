import matplotlib.pyplot as plt
import pandas as pd


def plot_sleep_index(df_daily: pd.DataFrame, out_path: str | None = None):
    plt.figure()
    df_daily["sleep_index"].plot()
    plt.axhline(0, linestyle="--")
    plt.title("Daily Sleep Index")
    plt.ylabel("SleepIndex (hours vs target)")
    plt.xlabel("Date")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()


def plot_forecast_vs_actual(
    y_test: pd.Series,
    y_pred: pd.Series,
    out_path: str | None = None,
):
    plt.figure()
    y_test.plot(label="Actual SleepIndex (t+1)")
    y_pred.plot(label="Predicted SleepIndex (t+1)")
    plt.axhline(0, linestyle="--", alpha=0.5)
    plt.legend()
    plt.title("Forecast vs Actual Next-Day SleepIndex")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()


def plot_pnl_curve(trades: pd.DataFrame, out_path: str | None = None):
    plt.figure()
    trades["cum_pnl"].plot()
    plt.title("Strategy Cumulative PnL")
    plt.ylabel("PnL ($)")
    plt.xlabel("Date (contract expiry)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()


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
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()
