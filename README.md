# Sleep Futures Quant Research Project

Built with **Python, CatBoost, Pandas, Matplotlib, and Streamlit**

# Project Overview

Sleep Futures is a **behavioral-finance quant research** project that transforms daily sleep behavior into a **synthetic futures contract** and evaluates whether sleep patterns contain **predictive directional information.**

The project defines a *Sleep Index*, forecasts its **next-day direction** using a nonlinear classifier, and simulates a **long/short trading strategy** with full backtesting, risk metrics, and visualization.

This work is designed as a research and signal-evaluation framework, demonstrating how nontraditional behavioral data can be structured, modeled, and evaluated using real quant methodologies.

# Core Concept

## Sleep Index Definition:

The Sleep Index measures deviation from a target sleep duration:

$$
SleepIndex_t = hours\!sleept_t - 7.5
$$

**Positive** = overslept
**Negative** = underslept


## Synthetic Futures Contract:

A synthetic derivative is defined on the Sleep Index:

$$
Payoff_{t+1} = ContractMultiplier \times SleepIndex_{t+1}
$$

Directional exposure is taken based on the model’s forecast.

**Trading Logic:**

- Predict ↑ → LONG
- Predict ↓ → SHORT
- Predict → → FLAT

# Methodology

## 1. Feature Engineering

- Features combine statistical structure and behavioral context:
- Rolling means (3, 7, 14 days)
- Rolling volatility
- Sleep deficit
- Circadian drift (bedtime deviation)
- Sleep efficiency
- Day-of-week one-hot encoding
- Weekend vs weekday indicators

## 2. Modeling

- CatBoostClassifier (nonlinear, tabular-friendly)
- Multi-class direction prediction (↑ / ↓ / →)
- Chronological train/test split
- Early stopping to reduce overfitting
- CatBoost was selected for its robustness on small, noisy, nonstationary datasets.

## 3. Backtesting & Evaluation

- Custom long/short backtesting engine
- Daily PnL calculation
- Cumulative returns
- Sharpe ratio
- Maximum drawdown
- Performance visualization

# Results

**Directoin Strategy Performance:**
- Total PnL: $76.83
- Sharpe Ratio: 9.287
- Max Drawdown: –$13.17

**Classification Metrics:**

- Accuracy: 66.67%
- Directional Accuracy (ex-neutral): 66.67%

These results indicate **strong signal-to-noise within sample**, while acknowledging limitations related to dataset size and non-independence

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

- Most 7-day volatility is between 1.0 and 2.5 hours
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