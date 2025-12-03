import pandas as pd
import streamlit as st

from src.config import DATA_PROCESSED
from src.features import engineer_features
from src.model import train_sleep_model
from src.backtest import compute_strategy_pnl


st.set_page_config(page_title="Sleep Futures Dashboard", layout="wide")
st.title("🛌 Sleep Futures: Synthetic Derivatives on My Sleep")


@st.cache_data
def load_data():
    # DATA_PROCESSED is a Path object from config.py; pandas can read it directly
    df_daily = pd.read_csv(DATA_PROCESSED, parse_dates=["date"]).set_index("date")
    df_features, feature_cols = engineer_features(df_daily)
    results = train_sleep_model(df_features, feature_cols)
    bt = compute_strategy_pnl(results.y_test_reg, results.y_pred_cls)
    return df_daily, df_features, feature_cols, results, bt


df_daily, df_features, feature_cols, results, bt = load_data()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Sleep Index", "Direction Model", "Strategy PnL", "Sleep Volatility"]
)

with tab1:
    st.subheader("Daily Sleep Index")
    st.line_chart(df_daily["sleep_index"])
    st.write("SleepIndex = hours_slept − target (7.5 hours).")

with tab2:
    st.subheader("Next-Day Sleep Direction (Classification)")
    chart_df = pd.DataFrame(
        {
            "SleepIndex (t+1)": results.y_test_reg,
            "Predicted Direction": results.y_pred_cls,
        }
    ).dropna()
    st.line_chart(chart_df)
    st.write(
        f"Accuracy: **{results.accuracy:.1%}** · "
        f"Directional accuracy (non-neutral): **{results.directional_accuracy:.1%}**"
    )

with tab3:
    st.subheader("Strategy Cumulative PnL")
    st.line_chart(bt.trades["cum_pnl"])
    st.write(f"Total PnL: **${bt.total_pnl:.2f}**")
    st.write(f"Sharpe: **{bt.sharpe:.3f}** · Max drawdown: **${bt.max_drawdown:.2f}**")

with tab4:
    st.subheader("Sleep Volatility (7-day rolling std of SleepIndex)")
    vol = df_daily["sleep_index"].rolling(7).std().dropna()
    st.line_chart(vol)

st.sidebar.header("Model / Data Info")
st.sidebar.write(f"Observations: {len(df_daily)} days")
st.sidebar.write(f"Features used: {len(feature_cols)}")
