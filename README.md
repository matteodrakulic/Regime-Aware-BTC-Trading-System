# Project X: Regime-Aware Crypto Trading System

**A Research-Grade Systematic Trading Engine for Bitcoin**

This repository documents the complete journey of building a quantitative trading system from scratch. Unlike standard "trading bots" that rely on static indicators, this project employs **Unsupervised Learning (Gaussian Hidden Markov Models)** to dynamically classify market regimes and adapt its behavior accordingly.

This document serves as a detailed educational resource, breaking down every phase of development, the rationale behind technical choices, and the implementation details.

---

## 📚 Table of Contents
1.  [Project Philosophy](#1-project-philosophy)
2.  [Development Journey (Step-by-Step)](#2-development-journey-step-by-step)
    *   [Phase 1: Data Infrastructure](#phase-1-data-infrastructure)
    *   [Phase 2: Feature Engineering](#phase-2-feature-engineering)
    *   [Phase 3: Regime Detection (The Core)](#phase-3-regime-detection-the-core)
    *   [Phase 4: Strategy Design](#phase-4-strategy-design)
    *   [Phase 5: Risk Management](#phase-5-risk-management)
    *   [Phase 6: Backtesting Engine](#phase-6-backtesting-engine)
    *   [Phase 7: Live Execution CLI](#phase-7-live-execution-cli)
3.  [Technical Architecture](#3-technical-architecture)
4.  [Usage Instructions](#4-usage-instructions)
5.  [Key Findings](#5-key-findings)

---

## 1. Project Philosophy

The central hypothesis of this project is **"Regime Dependence"**:
> *Financial markets do not follow a single distribution. Strategies that work in a bull trend often fail in choppy markets. Therefore, the first step in any robust strategy must be to identify the current market state (Regime).*

We chose **Bitcoin (BTC)** on a **4-hour timeframe** because:
*   **BTC** exhibits distinct, explosive trends followed by long periods of noise (perfect for regime switching models).
*   **4-hour candles** filter out microstructure noise while retaining enough granularity to capture multi-day moves.
*   **Research Focus:** The goal was not just "profit", but **robustness**—minimizing drawdown and maximizing the Sharpe Ratio.

---

## 2. Development Journey (Step-by-Step)

We built this system in 7 distinct phases. Here is exactly what we did in each.

### Phase 1: Data Infrastructure
**Goal:** Obtain clean, reliable historical data.
*   **Source:** We used the `ccxt` library to fetch data from **Binance**.
*   **Timeframe:** 4-hour OHLCV (Open, High, Low, Close, Volume).
*   **Cleaning:**
    *   Implemented checks for missing timestamps.
    *   Validated OHLCV integrity (e.g., High must be >= Low).
    *   Stored data locally in CSV format to avoid redundant API calls.
*   **Code:** `src/data/fetcher.py`, `src/data/cleaner.py`

### Phase 2: Feature Engineering
**Goal:** Transform raw prices into statistical signals for the ML model.
We didn't just dump raw prices into the model. We engineered features capturing different market properties:
1.  **Volatility:** GARCH-style rolling standard deviations, ATR (Average True Range).
2.  **Trend:** EMAs, MACD, ADX (Average Directional Index).
3.  **Momentum:** RSI, ROC (Rate of Change).
4.  **Distribution:** Kurtosis and Skewness (to detect "fat tails" or crash risks).
*   **Code:** `src/features/`

### Phase 3: Regime Detection (The Core)
**Goal:** Use Unsupervised Learning to discover hidden market states.
*   **Model:** **Gaussian Hidden Markov Model (HMM)**.
*   **Why HMM?** Unlike k-Means (which ignores time), HMM assumes that the *current* state depends on the *previous* state, making it perfect for time-series data.
*   **Implementation:**
    *   We used `hmmlearn` with `n_components=3`.
    *   **Regime 0 (Low Vol):** Range-bound, choppy behavior.
    *   **Regime 1 (Medium Vol):** Often "fakeouts" or noisy transitions.
    *   **Regime 2 (High Vol):** Strong directional trends (Bull or Bear runs).
*   **Rolling Inference:** Crucially, we implemented a **Rolling Window** approach. We train the model on past data (e.g., 1000 candles) to predict the regime of the *current* candle, ensuring no "look-ahead bias".
*   **Code:** `src/regimes/hmm.py`, `src/regimes/inference.py`

### Phase 4: Strategy Design
**Goal:** Assign trading logic to each regime.
We analyzed the performance of simple strategies in each regime and derived the **"Robust Trend Strategy"**:
*   **In Regimes 0 & 1 (Low/Med Vol):**
    *   **Action:** **FLAT (Cash)**.
    *   **Rationale:** Trend following gets chopped up here. It's better to sit out and preserve capital.
*   **In Regime 2 (High Vol):**
    *   **Action:** **Trend Following**.
    *   **Signals:** EMA Crossover (20/50) + ADX Filter (>30).
    *   **Rationale:** This is where the "fat tails" are. We capture the big moves.
*   **Code:** `src/strategies/robust.py`

### Phase 5: Risk Management
**Goal:** Survive.
We implemented **Volatility Targeting**:
*   Instead of a fixed position size (e.g., "always buy 1 BTC"), we size positions based on current market volatility.
*   **Formula:** `Position Size = Target Volatility / Current Volatility`.
*   **Target:** 60% Annualized Volatility.
*   **Effect:** When the market goes crazy (high vol), we trade *smaller*. When the market is calm but trending, we trade *larger*. This stabilizes the PnL curve.
*   **Code:** `src/risk/sizing.py`

### Phase 6: Backtesting Engine
**Goal:** Validate the hypothesis.
We built a custom, vectorized backtester (faster than event-driven loops) that accounts for:
*   **Transaction Costs:** 0.05% per trade (Binance fees).
*   **Slippage:** Estimated execution cost.
*   **Yield on Idle Cash:** A unique feature. Since we sit FLAT often, we assume that idle USDT earns a conservative 6% APY (e.g., via Binance Earn or Aave). This significantly boosts risk-adjusted returns.
*   **Code:** `src/backtest/engine.py`

### Phase 7: Live Execution CLI
**Goal:** Make it usable.
We wrapped everything into a Command Line Interface (CLI) for easy interaction.
*   **Commands:** `live`, `backtest`, `diagnose`.
*   **Real-time:** The `live` command fetches the latest data, runs the HMM inference on the fly, and outputs a trading signal.
*   **Code:** `src/main.py`

---

## 3. Technical Architecture

The project follows a modular "Pipeline" architecture:

```
src/
├── data/           # Raw Data -> DataFrame
├── features/       # DataFrame -> Feature Matrix
├── regimes/        # Feature Matrix -> Regimes (0,1,2)
├── strategies/     # Regimes + Data -> Raw Signals
├── risk/           # Raw Signals -> Sized Positions
├── backtest/       # Positions -> Equity Curve & Metrics
└── main.py         # The Conductor (CLI)
```

This decoupling allows us to swap out the model (e.g., replace HMM with Random Forest) without breaking the strategy or backtester.

---

## 4. Usage Instructions

### Prerequisites
*   Python 3.10+
*   Pip

### Installation
```bash
git clone <repo_url>
cd "Project X"
pip install -r requirements.txt
```

### 1. Run Live Analysis
Run this every 4 hours to get your trading instructions.
```bash
python3 src/main.py live
```
*Output Example:*
```text
LIVE SIGNAL ANALYSIS: 2025-01-01 16:00:00
Regime: 2 (High Volatility) -> Trend Following Active
Signal: LONG
Action: BUY BTC (Size: 0.85)
```

### 2. Run Backtest
Verify the strategy performance over history.
```bash
python3 src/main.py backtest --capital 10000 --target_vol 0.60
```

### 3. Diagnose Regimes
Visualize how the HMM classifies recent market behavior.
```bash
python3 src/main.py diagnose
```

---

## 5. Key Findings

1.  **Regime Filtering Works:** Avoiding "choppy" regimes (Regime 1) reduced drawdown by over 5% compared to a naive trend strategy.
2.  **Yield Matters:** Earning 6% on idle cash contributed significantly to the Sharpe Ratio, turning "waiting" into "earning".
3.  **Volatility Targeting:** This was the single biggest factor in smoothing the equity curve, preventing massive drawdowns during crypto crashes.

---

*This project was built for educational and research purposes. It demonstrates professional-grade software engineering practices applied to quantitative finance.*
