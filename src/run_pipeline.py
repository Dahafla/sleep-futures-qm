from pathlib import Path

from .config import DATA_PROCESSED
from .data_loader import load_raw_sleep_data
from .features import engineer_features
from .model import train_sleep_model
from .backtest import compute_strategy_pnl
from .visualize import (
    plot_sleep_index,
    plot_pnl_curve,
    plot_sleep_volatility,
)


def run_pipeline():
    # 1) Load raw data
    df_daily = load_raw_sleep_data()

    # Save processed base time-series
    Path(DATA_PROCESSED).parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(DATA_PROCESSED)

    # 2) Features
    df_features, feature_cols = engineer_features(df_daily)

    # 3) Train classifier
    results = train_sleep_model(df_features, feature_cols)

    print("=== Model Metrics (Classification) ===")
    print(f"Overall accuracy: {results.accuracy:.3%}")
    print(
        f"Directional accuracy (ignoring neutral): "
        f"{results.directional_accuracy:.3%}"
    )

    # 4) Backtest using predicted direction as position
    bt = compute_strategy_pnl(results.y_test_reg, results.y_pred_cls)

    print("\n=== Backtest Results (Direction Strategy) ===")
    print(f"Total PnL: ${bt.total_pnl:.2f}")
    print(f"Sharpe ratio: {bt.sharpe:.3f}")
    print(f"Max drawdown: ${bt.max_drawdown:.2f}")

    # 5) Visualizations (we keep the same ones)
    plot_sleep_index(df_daily)
    plot_pnl_curve(bt.trades)
    plot_sleep_volatility(df_daily)


if __name__ == "__main__":
    run_pipeline()
