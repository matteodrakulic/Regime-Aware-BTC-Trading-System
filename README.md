# Project X: Regime-Aware Quantitative Trading System

A systematic trading engine for Bitcoin that uses Unsupervised Learning (Gaussian Hidden Markov Models) to adapt to changing market conditions.

## Project Overview

Financial markets do not follow a single stationary distribution. Strategies that perform well during trends often fail during mean-reverting or choppy periods. This project addresses that problem by explicitly modeling the "Market Regime" before applying any trading logic.

The system operates on 4-hour Bitcoin (BTC) data. It uses a Gaussian HMM to classify the market into three distinct states (Low, Medium, and High Volatility) and only deploys capital when the market conditions favor the underlying trend-following strategy.

**Key Features:**
*   **Dynamic Regime Detection:** Uses a Rolling Window HMM to avoid look-ahead bias.
*   **Volatility Targeting:** Adjusts position size based on realized volatility (Target: 60% annualized).
*   **Capital Efficiency:** Idle capital (during unfavorable regimes) is assumed to generate yield (e.g., via stablecoin lending).
*   **Modular Architecture:** clear separation between data, feature engineering, inference, and execution.

For a detailed breakdown of the trading logic and performance metrics, see [STRATEGY.md](STRATEGY.md).

## Technical Architecture

The codebase is organized as a data processing pipeline:

```
src/
├── data/           # Data ingestion and cleaning (OHLCV)
├── features/       # Feature engineering (Trend, Volatility, Momentum)
├── regimes/        # HMM Inference and Regime Detection
├── strategies/     # Signal generation logic
├── risk/           # Position sizing and risk limits
├── backtest/       # Vectorized backtesting engine
└── main.py         # CLI entry point
```

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd project-x
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The system is controlled via a Command Line Interface (CLI) in `src/main.py`.

### 1. Live Trading (Human-in-the-Loop)

This is the primary mode for execution. The script analyzes the latest data, determines the current regime, and outputs a trading instruction. You do not need an API connection; the system prints the decision, and you execute it manually on your exchange.

**Run the command:**
```bash
python3 src/main.py live
```

**Example Output:**
```text
LIVE SIGNAL ANALYSIS: 2025-01-01 16:00:00
Latest Close:   $98,242.89
Regime:         2 (High Volatility)
Action:         LONG BTC (85.0% of Capital)
```

**Interpretation:**
*   **LONG BTC:** Open a long position.
*   **SHORT BTC:** Open a short position (via perpetual futures).
*   **FLAT / CASH:** Close all positions and hold stablecoins (earning yield).
*   **% of Capital:** The recommended position size based on volatility targeting.

### 2. Backtesting

To verify the strategy performance over historical data:

```bash
python3 src/main.py backtest --capital 10000 --target_vol 0.60
```

This will run the simulation over the full dataset (defaulting to `data/raw/btc_4h.csv`) and print performance metrics (Sharpe, CAGR, Drawdown).

### 3. Diagnostics

To visualize how the HMM is currently classifying market states:

```bash
python3 src/main.py diagnose
```

This calculates the regime distribution and transition probabilities for the dataset.

## Extensions

The current setup is designed for research and semi-automated trading. Future extensions could include:
*   **Automated Execution:** Integrating `ccxt` or exchange-specific SDKs to place orders automatically.
*   **Multi-Asset Support:** Expanding the feature set to trade ETH or SOL.
*   **Alternative Models:** Swapping the HMM for Isolation Forests or LSTM autoencoders for anomaly detection.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Quantitative trading involves significant risks, including the potential loss of all capital.
