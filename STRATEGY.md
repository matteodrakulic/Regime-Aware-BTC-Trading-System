# Strategy Documentation

This document describes the current active trading strategy, regime definitions, and the rationale behind the design choices.

## 1. Regime Definitions (HMM)

The strategy uses a Gaussian Hidden Markov Model (HMM) with 3 components to classify market states based on 4-hour OHLCV data. The model is trained using a Rolling Window approach to ensure no look-ahead bias.

| Regime | Interpretation | Characteristics | Strategy Action |
| :--- | :--- | :--- | :--- |
| **Regime 0** | **Low Volatility** | High noise-to-signal ratio, choppy price action. | **FLAT** (Capital Preservation) |
| **Regime 1** | **Medium Volatility** | False breakouts, mean-reverting behavior. | **FLAT** (Noise Avoidance) |
| **Regime 2** | **High Volatility** | Strong directional moves (Parabolic Advances or Crashes). | **ACTIVE** (Robust Trend) |

**Key Insight:**
Our diagnostics reveal that **Regime 2** captures the highest annualized volatility (~95%) but also the highest annualized return (~72%). This confirms that Bitcoin's most profitable moves occur during periods of high turbulence. By filtering out Regimes 0 and 1, we avoid "death by a thousand cuts" during chop and only participate when the market is moving significantly.

## 2. Active Strategy: Robust Trend (Regime 2 Only)

We deploy an optimized **Robust Trend Strategy** exclusively during **Regime 2**.

### Entry Logic
*   **Long Signal:**
    1.  **Regime:** Must be in Regime 2.
    2.  **Trend:** Fast EMA (20) > Slow EMA (50).
    3.  **Strength:** ADX > 30 (Strong Trend).
*   **Short Signal:**
    1.  **Regime:** Must be in Regime 2.
    2.  **Trend:** Fast EMA (20) < Slow EMA (50).
    3.  **Strength:** ADX > 30.
    4.  **Macro Filter:** Price < 200-Day SMA (Only short during macro bear markets).

### Exit Logic
*   Positions are closed immediately when the signal flips (e.g., Long -> Short) or when the Regime changes (Regime 2 -> Regime 1/0).
*   When the regime changes to 0 or 1, the system goes **FLAT**.

## 3. Risk Management

### Volatility Targeting
Instead of fixed position sizing, we use Volatility Targeting to normalize risk.
*   **Target Volatility:** 60% Annualized.
*   **Formula:** `Position Size = Target Vol / Current Vol`.
*   **Max Leverage:** Capped at **1.0x** (No Leverage).
    *   *Example:* If current market volatility is low (20%), the system caps exposure at 1.0x (it does not lever up). If volatility is extreme (120%), it cuts position size to 0.5x to maintain target risk.

### Yield Generation
When the system is **FLAT** (Regimes 0 & 1) or holding partial cash, the idle capital is assumed to earn a risk-free rate.
*   **Rate:** 6% APY (Conservative estimate for stablecoin lending).
*   **Impact:** This significantly improves the Sharpe Ratio by turning "waiting" periods into "earning" periods.

## 4. Performance (Backtest Results)

*   **Period:** Full available history (BTC 4h).
*   **CAGR:** **28.59%**
*   **Total Return:** **+246.53%**
*   **Sharpe Ratio:** **1.21**
*   **Max Drawdown:** **-21.39%**
*   **Time in Market:** **8.60%**

**Conclusion:**
The strategy spends ~91% of its time in cash (earning yield) and only enters the market during the most explosive ~9% of the time (Regime 2). This results in a highly efficient use of capital with significantly lower drawdowns than a Buy & Hold approach.
