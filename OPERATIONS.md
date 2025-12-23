# Operational Guide: Live Trading & Execution

This document explains how to transition this system from a "Research Backtester" to a "Live Trading Engine".

## 1. How it works in practice

You do **not** necessarily need to write complex code to integrate with an exchange. There are two ways to run this:

### A. The "Human-in-the-Loop" Approach (Recommended for starting)
You run the script manually (or set it to run automatically) once every 4 hours. It tells you what to do.

1.  **Script runs:** `python src/main.py live`
2.  **Output:** 
    ```text
    LIVE SIGNAL ANALYSIS: 2025-01-01 16:00:00
    Latest Close:   $88,242.89
    Regime:         2 (High Volatility)
    Action:         LONG BTC (100.0% of Capital)
    ```
3.  **Your Action:** You open your phone/exchange and click Buy.

### B. The "Fully Automated" Approach
The script connects to the Exchange API (e.g., Binance, Bybit) and executes orders for you.

1.  **Script runs:** Automatically via a server (cron job) every 4 hours.
2.  **Output:** Logs to a file.
3.  **Action:** The script sends a REST API request to the exchange to place the order.

---

## 2. Understanding the "Yield Generation"

You asked: *"Will I have to integrate the program... so these actions are performed automatically?"*

**Good News:** For the Yield Generation strategy, **you usually do NOT need to do anything extra.**

### How it works structurally:
1.  **"Going Flat":** When the strategy says "Flat" (Regime 0 or 1), it means **Sell BTC -> Hold USDT**.
2.  **Automatic Yield:** Most modern crypto exchanges (Bybit, Binance, etc.) have "Flexible Savings" or "Earn" accounts.
    *   You can set your USDT to **"Auto-Subscribe"**.
    *   This means any USDT sitting in your account *automatically* earns interest (APY) every day.
    *   You don't need to "send" it anywhere. You just sell BTC, and the resulting USDT starts earning interest immediately.
    *   When the strategy says "Buy BTC", the exchange automatically pulls from your savings balance (on some exchanges) or you redeem it instantly.

### The 6% Assumption
*   **Is it fixed?** No. It fluctuates daily based on market demand for leverage.
*   **Is it realistic?**
    *   **Bull Market:** USDT borrowing rates often hit 10-20%+.
    *   **Bear Market:** Rates can drop to 2-4%.
    *   **Average:** 6% is a reasonable long-term average for stablecoin yields in crypto.
*   **Implementation:** In the backtest, we used a fixed 6% for simplicity. In reality, your returns will vary slightly day-to-day, but the *mechanic* (earning interest on idle cash) is solid.

---

## 3. Deployment Checklist

To go live, we would need to build a `src/execution` module.

### Phase 1: Signal Generator (Current State)
- [x] Download Data
- [x] Calculate Features
- [x] Determine Regime
- [x] Output Ideal Position

### Phase 2: Live Connector (Future)
- [ ] **Exchange Client:** Connect to (e.g.) Bybit API using `pybit` or `ccxt`.
- [ ] **Wallet Manager:** Check current balance (USDT + BTC).
- [ ] **Order Executor:** Calculate difference between *Current Position* and *Ideal Position*, and send buy/sell orders.
- [ ] **Notifier:** Send Telegram/Discord message on execution.
