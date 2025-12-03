from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONTRACT_MULTIPLIER


@dataclass
class BacktestResults:
    trades: pd.DataFrame
    sharpe: float
    max_drawdown: float
    total_pnl: float


def compute_strategy_pnl(
    y_true: pd.Series,       # numeric SleepIndex (t+1)
    y_pred_cls: pd.Series,   # predicted direction: -1 (short), 1 (long)
) -> BacktestResults:
    """
    Strategy:
        position_t = y_pred_cls_t
        payoff_t   = CONTRACT_MULTIPLIER * position_t * y_true_t
    """
    trades = pd.DataFrame(
        {"y_true": y_true, "y_pred_cls": y_pred_cls}
    ).dropna()

    trades["position"] = trades["y_pred_cls"]

    trades["pnl"] = CONTRACT_MULTIPLIER * trades["position"] * trades["y_true"]
    trades["cum_pnl"] = trades["pnl"].cumsum()

    pnl = trades["pnl"]
    if pnl.std(ddof=1) == 0:
        sharpe = np.nan
    else:
        sharpe = float(np.sqrt(252) * pnl.mean() / pnl.std(ddof=1))

    equity = trades["cum_pnl"]
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = float(drawdown.min())

    total_pnl = float(trades["pnl"].sum())

    return BacktestResults(
        trades=trades,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        total_pnl=total_pnl,
    )
