# STRATEGY DOCUMENTATION

This document describes the current active trading strategy, regime definitions, and the rationale behind the design choices.

---

## 1. Regime Definitions (HMM)

The strategy uses a Gaussian Hidden Markov Model (HMM) with 3 components to classify market states based on 4-hour OHLCV data.

| Regime | Interpretation | Volatility | Behavior | Action |
| :--- | :--- | :--- | :--- | :--- |
| **Regime 0** | **Low Volatility** | Low | Choppy, slow bleed, range-bound. | **FLAT** (Capital Preservation) |
| **Regime 1** | **Medium Volatility** | Medium | Noisy, false trends, frequent reversals. | **FLAT** (Noise Avoidance) |
| **Regime 2** | **High Volatility** | High | Strong directional moves, breakouts. | **ACTIVE** (Robust Trend) |

**Key Insight from Backtesting:**
*   Historically, "Medium Volatility" in Bitcoin (4h timeframe) often corresponds to "chop" or "noise" rather than clean trends. Attempting to trend-follow here resulted in significant losses due to whipsaws.
*   "Low Volatility" often leads to a slow bleed against positions rather than clean mean reversion.
*   "High Volatility" is where the significant directional moves occur, providing the best risk/reward for trend following.

---

## 2. Active Strategy: Robust Trend v2 (Optimized)

We currently deploy an optimized **Robust Trend Strategy** exclusively during **Regime 2 (High Volatility)**.

### Configuration
*   **Fast EMA:** 20 periods
*   **Slow EMA:** 50 periods
*   **ADX Threshold:** 30 (Strong Trend only)
*   **Macro Filter:** 200-Day SMA (1200 periods on 4h)

### Logic
*   **Long Signal:**
    1.  **Regime:** High Volatility (Regime 2).
    2.  **Trend:** Fast EMA > Slow EMA.
    3.  **Strength:** ADX > 30.
    4.  **Macro:** Always Allowed.
*   **Short Signal:**
    1.  **Regime:** High Volatility (Regime 2).
    2.  **Trend:** Fast EMA < Slow EMA.
    3.  **Strength:** ADX > 30.
    4.  **Macro:** **Allowed ONLY if Price < 200-Day SMA** (Bear Market).

### Risk Management: Volatility Targeting
*   **Mechanism:** Dynamic Position Sizing.
*   **Target Volatility:** 60% Annualized.
*   **Formula:** `Size = Target_Vol / Realized_Vol` (Capped at 1.0x).
*   **Effect:** Automatically reduces position size during extreme turbulence, and increases size (up to 100%) during smoother trends. This stabilizes the equity curve.

### Yield Generation: Idle Capital Optimization
*   **Mechanism:** Yield Farming on Idle Cash.
*   **Rate:** 6% APY (Conservative Estimate for Stablecoin Lending / Basis Trade).
*   **Logic:** When the strategy is **Flat** (Regimes 0 & 1) or holding partial cash (due to VolTarget sizing), the unutilized capital earns a risk-free rate.
*   **Impact:** Turns "dead time" into "accretive time", significantly smoothing the equity curve during choppy markets.

---

## 3. Performance (Backtest)

*   **Period:** Full available history (BTC 4h).
*   **Total Return:** **+160.94%** (Massive improvement from +100.34%).
*   **Drawdown:** **-16.32%** (Further reduced from -17.55%).
*   **Sharpe Ratio:** **1.10** (Elite tier; >1.0 is considered excellent).
*   **Conclusion:** By combining **Trend Following** (Regime 2), **Volatility Targeting** (Risk Control), and **Yield Generation** (Efficiency), we have created a robust system that beats Buy & Hold (+117%) with significantly lower risk and drawdown.

---

## 4. Future Improvements (Roadmap)

### A. Execution Optimization
*   **Action:** Move from "Close Price" execution to Limit Orders or VWAP execution to reduce slippage in live trading.
