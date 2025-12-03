# Sleep Futures Quant Research Project

**Time-Series Modeling • CatBoost • Synthetic Derivatives • Backtesting • Streamlit Dashboard**

Sleep Futures is a **behavioral–finance quant research project** that transforms my daily sleep patterns into a *tradable synthetic futures contract*.
Using engineered features, nonlinear classification modeling, and a custom long/short backtesting engine, the system forecasts the direction of tomorrow’s Sleep Index and trades on it.

The result: a unique project that blends **quant research, machine learning, and personal behavioral data** into a fully simulated alpha-generating strategy.
This makes it a unique **behavioral-finance + time-series modeling project with real forecasting, alpha signals, and a fully implemented trading simulation.**

# Project Summary

This project converts my nightly sleep behavior into a **synthetic futures market** by defining a *Sleep Index* and *forecasting its next-day direction*. A CatBoostClassifier generates *long/short signals*, which feed into a backtesting engine with **PnL, Sharpe, and drawdown tracking**.

**The strategy achieves:**
- PnL: +$76.83
- Sharpe Ratio: 9.287
- Max Drawdown: –$13.17

# Core Concept

## Sleep Index Definition:

**A positive index** = slept more than target.
**A negative index** = slept less.

$$
SleepIndex_t = hours\!sleept_t - 7.5
$$

## Synthetic Futures Contract:

$$
Payoff_{t+1} = ContractMultiplier \times SleepIndex_{t+1}
$$

**Trading Logic:**

- Predict ↑ → LONG
- Predict ↓ → SHORT
- Predict → → FLAT


# Example Performance

**Classification Metrics:**

- Accuracy: 66.67%
- Directional Accuracy (ex-neutral): 66.67%
- Backtest (Directional Strategy)
- Total PnL: $76.83
- Sharpe Ratio: 9.287
- Max Drawdown: –$13.17

**These values show strong signal-to-noise (high Sharpe) and effective directional prediction.**

# Visualizations & Interpretation

## 1. Daily Sleep Index

**Shows how many hours above/below target slept each day.**

![Daily SLeep Index](charts/plot_sleep_index.png)

**Interpretation:**

- Positive spikes = overslept
- Negative dips = underslept
- Volatility reflects behavioral inconsistency
- This time series is the core forecasting target

## 2. Sleep Volatility Histogram

**Shows the 7-day rolling standard deviation of SleepIndex.**

![Sleep Volatility](charts/plot_sleep_volatility.png)

**Interpretation:**

- Most 7-day volatility is between 1.0 and 2.5
- High volatility means irregular sleep patterns
- Helps explain why forecasting sleep is harder during chaotic periods


## 3. Strategy PnL Curve

**Displays cumulative profit of the long/short strategy.**

![Cumulative PnL](charts/plot_pnl_curve.png)

**Interpretation:**

- The model produced positive cumulative PnL
- Sharpe ratio > 9 indicates very high signal-to-noise
- Max drawdown shows the worst peak-to-bottom decline
- This demonstrates alpha generation from behavioral patterns

# Features

**Includes both statistical and behavioral features:**

- Rolling means (3, 7, 14 days)
- Sleep deficit
- Circadian drift (bedtime deviation)
- Sleep efficiency
- Day-of-week one-hots
- Weekend vs weekday
- Next-day directional target (↑, ↓, →)

# Modeling

- CatBoostClassifier (best-in-class for nonlinear tabular TS data)
- Chronological train/test split
- Early stopping
- Multi-class direction prediction

# Trading Strategy & Backtest

- Custom long/short engine
- Daily returns calculation
- Cumulative PnL
- Sharpe ratio
- Max drawdown tracking
- Performance visualization


# Why This Project Matters

This pipeline demonstrates real quant concepts:

- Signal generation from behavioral time-series
- Feature engineering with rolling windows & volatility
- ML forecasting using nonlinear classifiers
- Derivative payoff construction
- Backtesting + risk metrics used in hedge funds
- Visualization tooling for explaining and monitoring signals

# Project Architecture

```bash
.
├── data/
│   ├── raw/                      # Input sleep CSV
├── src/
│   ├── data_loader.py            # Load + clean raw sleep data
│   ├── features.py               # Feature engineering
│   ├── model.py                  # CatBoost classification model
│   ├── backtest.py               # Synthetic futures backtesting
│   ├── visualize.py              # Matplotlib visualizations
│   └── run_pipeline.py           # Full pipeline runner
├── app.py                        # Streamlit dashboard
├── charts/                       # Saved PNG plots
├── requirements.txt
└── README.md
```

# Installation & Usage

```bash
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/sleep-futures.git
cd sleep-futures-qm

2. Create virtual env
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Add sleep.csv

data/raw/sleep.csv

5. Run the full pipeline
python3 -m src.run_pipeline

6. Run the Streamlit dashboard

streamlit run app.py
```