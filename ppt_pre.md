---
marp: true
theme: default
paginate: true
title: Market Regime-Aware Prediction of Cross-Sectional Stock Returns
---

# Market Regime-Aware Prediction of Cross-Sectional Stock Returns

## SDSC4009 Project Presentation

- Goal: predict next-month stock returns using technical, fundamental, and market-regime information
- Main question: when does regime information actually help?
- Main answer: technical signals favored LSTM, while fundamental signals benefited most from regime-aware tree models

---

# Presentation Roadmap

1. Problem and motivation
2. Data pipeline and feature engineering
3. Market regime detection with HMM
4. Feature diagnostics and cleaner feature sets
5. Baseline LightGBM comparison
6. Model family comparison: LightGBM vs XGBoost vs LSTM
7. LSTM tuning and XGBoost regime-aware extension
8. Diagnostics, interpretation, and final conclusion

**How to present this**

Follow the workflow in the notebook: data -> regime -> features -> models -> diagnostics -> conclusion.

---

# 1. Motivation

- Stock return prediction is difficult because relationships change across time
- The same signal may work differently in calm and stressed markets
- We wanted to test two ideas at the same time:
  - whether market regime information improves prediction
  - whether different model families are better suited for different signal types

**Project hypothesis**

- Technical signals may benefit from sequence models
- Fundamental signals may benefit more from regime-aware tree models

**Key message**

Regime matters, but not equally for every type of information.

---

# 2. What We Actually Built

The notebook was organized into 10 parts:

1. Data loading and feature engineering
2. Market regime detection with HMM
3. Helper functions and model builders
4. Monthly dataset assembly and feature diagnostics
5. Baseline LightGBM on cleaner feature sets
6. Model family comparison
7. Report charts, LSTM tuning, and regime-aware extension
8. Model diagnostics and feature importance
9. Fama-MacBeth extension
10. Conclusion

**Visual to use**

Use a simple workflow diagram with these 10 stages.

---

# 3. Data and Sample

- Universe: 40 large-cap US stocks
- Stock inputs: daily OHLCV, technical indicators, fundamentals
- Market inputs: NASDAQ-100 and VIX
- Forecast target: next-month return, using a 21-trading-day horizon
- Fundamental lag: 3 months to reduce look-ahead bias

| Dataset | Shape before cleaning | Final usable rows |
|---|---:|---:|
| Training panel | 100,320 x 86 | 89,400 |
| Testing panel | 25,120 x 86 | 14,200 |

| Monthly sample | Shape |
|---|---:|
| Monthly train panel | 4,240 x 119 |
| Monthly test panel | 680 x 119 |

| Sample | Period |
|---|---|
| Train | 2011-01-31 to 2019-10-31 |
| Test | 2020-12-31 to 2022-04-29 |

---

# 4. Data Pipeline

For each stock, we did the following:

1. Load technical and fundamental files
2. Merge them into one daily panel
3. Lag fundamental variables
4. Engineer return and technical features
5. Create next-month return target
6. Handle missing values
7. Convert daily panel into month-end cross-sectional panel

**Important design choice**

We modeled month-end cross sections rather than raw daily predictions, because the project target was next-month stock return.

**Visual to use**

Use a process chart: daily stock data -> lag fundamentals -> feature engineering -> monthly panel -> models.

---

# 5. What the Monthly Forecast Actually Means

Our final prediction is not based on a monthly average row.

Instead, for each stock we do this:

1. Take the last trading day of month $t$
2. Use that month-end feature snapshot as the input
3. Predict the return from month-end $t$ to month-end $t+1$

So the interpretation is:

- rebalance date = month-end
- input = month-end stock snapshot
- target = next month-end return

**Presentation line**

This is a monthly cross-sectional stock-ranking problem, not a daily forecasting problem.

---

# 6. Why Month-End Is Still Reasonable

At first glance, it looks like the days before month-end are being thrown away, but that is not really what happens.

Those daily observations are already embedded in the features:

- `return_1m` uses the prior 21 trading days
- `return_3m` and `momentum_12m` summarize longer histories
- `MACDh`, `ATR`, `stdev`, and `obv` are built from daily paths
- the month-end close anchors the future month-end return target

So we keep one row per stock per month, but that row still contains information accumulated over the month.

**Why this is a good compromise**

- cleaner alignment with quarterly fundamentals
- less repeated noise than daily rows
- easy to explain as a monthly rebalancing strategy

---

# 7. Market Regime Detection with HMM

- Model: 2-state Gaussian Hidden Markov Model
- Inputs:
  - NASDAQ-100 log return
  - 20-day realized volatility
  - VIX level
  - VIX change
- 20 random restarts for stability

| State | Mean Return | Mean Volatility | Mean VIX | Interpretation |
|---|---:|---:|---:|---|
| 0 | 0.000034 | 0.017699 | 27.279 | High-volatility or stressed regime |
| 1 | 0.001032 | 0.008284 | 15.006 | Low-volatility or calm regime |

| State | Count |
|---|---:|
| 0 | 1,276 |
| 1 | 2,228 |

**Visual to use**

Use the HMM regime plot with return shading and the VIX panel.

---

# 8. Why the HMM Step Matters

- The states were not arbitrary clusters
- State 0 had higher VIX and higher market volatility
- State 1 had lower VIX, lower volatility, and higher average return
- This gave us an economically meaningful state variable to merge into the stock panel

**Interpretation**

- State 0 = stressed market
- State 1 = calmer market

**Presentation point**

This is the bridge between market-level information and stock-level prediction.

---

# 9. Feature Diagnostics

Before modeling, we expanded the candidate feature pool and checked:

- feature availability
- correlation structure
- VIF
- univariate monthly IC

Main redundancy flags:

- `volatility_1m` and `stdev_feature` were near duplicates
- `price_vs_sma20` strongly overlapped with `return_1m`
- `return_on_assets` strongly overlapped with `return_on_equity`
- `cash_to_liabilities` strongly overlapped with `current_ratio`

**Visual to use**

Use the expanded feature correlation heatmap.

**Why this matters**

We did not want to throw every engineered variable into the models. We wanted cleaner, more defensible signal sets.

---

# 10. Cleaner Feature Sets

## Cleaner technical-only set

- `return_1m`
- `momentum_12m`
- `MACDh`
- `adx`
- `ATR`
- `obv`

## Cleaner fundamental-only set

- `pfcf_ratio`
- `return_on_assets`
- `ebitda_margin`
- `gross_margin`
- `quick_ratio`
- `debt_to_equity`
- `cash_to_liabilities`

## Cleaner all-features set

- technical-only + fundamental-only combined

**Extra step**

All features were converted into monthly cross-sectional quantile scores so stocks were compared relative to peers each month.

---

# 11. Baseline Experiment: LightGBM Only

We first fixed the model family and only changed:

- feature set
- regime design

Three LightGBM designs:

1. Pure baseline
2. State as feature
3. Regime-aware

This stage answered one narrow question first:

**If we stay within one strong tree model, does regime information help?**

---

# 12. LightGBM Results

| Feature Set | Pure IC | State as Feature IC | Regime-Aware IC |
|---|---:|---:|---:|
| Cleaner all features | 0.0208 | 0.0503 | 0.0234 |
| Cleaner technical-only | 0.0848 | 0.0288 | 0.0643 |
| Cleaner fundamental-only | 0.0582 | 0.1083 | 0.0870 |

| Feature Set | Best Test R2_oos |
|---|---:|
| Cleaner all features | -0.0137 |
| Cleaner technical-only | -0.0171 |
| Cleaner fundamental-only | 0.0167 |

**What we learned**

- Regime helped fundamentals much more than technicals in LightGBM
- The best LightGBM result was fundamental-only with state as feature
- This gave Test IC = 0.1083 and Test R2_oos = 0.0167

---

# 13. Full Model Family Comparison

We then compared three model families on the cleaner feature sets:

- LightGBM
- XGBoost
- LSTM (PyTorch)

This stage tested whether the best model depends on the type of signal.

**Visual to use**

Use the grouped bar charts for Test IC and Test R2_oos by model family.

---

# 14. Model Family Results

| Feature Set | LightGBM IC | XGBoost IC | LSTM IC | Winner |
|---|---:|---:|---:|---|
| Cleaner all features | 0.0503 | 0.0599 | 0.0230 | XGBoost |
| Cleaner technical-only | 0.0288 | 0.0579 | 0.1242 | LSTM |
| Cleaner fundamental-only | 0.1083 | 0.0797 | 0.0333 | LightGBM |

| Feature Set | LightGBM R2_oos | XGBoost R2_oos | LSTM R2_oos |
|---|---:|---:|---:|
| Cleaner all features | -0.0137 | -0.0054 | -0.0100 |
| Cleaner technical-only | -0.0314 | -0.0139 | 0.0079 |
| Cleaner fundamental-only | 0.0167 | 0.0158 | -0.0279 |

**Main insight**

There was no single universally best model. The winner depended on the feature family.

---

# 15. Technical Route: Why LSTM Won

Best technical result:

- Model: LSTM on cleaner technical-only features
- Test IC = 0.1242
- Test R2_oos = 0.0079

Why this likely happened:

- technical indicators are built from price dynamics
- sequence models can capture temporal structure better than static trees
- the LSTM outperformed both LightGBM and XGBoost on the same technical feature set

**Interpretation**

If the task is stock ranking using technical signals, the LSTM route is the strongest result in the project.

---

# 16. LSTM Tuning: What We Tested

We ran a focused tuning grid on the cleaner technical-only LSTM:

- lookback: 3, 6, 9
- hidden size: 16, 32, 64
- learning rate: 0.001, 0.0005
- batch size: 16, 32

Best validation config:

- lookback = 3
- hidden size = 32
- dense size = 16
- learning rate = 0.0005
- batch size = 16

Validation result:

- Validation IC = 0.0514
- Validation R2_oos = 0.0025

---

# 17. LSTM Tuning: Final Takeaway

| Model | Test IC | Test R2_oos |
|---|---:|---:|
| Original LSTM technical-only | 0.1242 | 0.0079 |
| Tuned LSTM technical-only | 0.0983 | 0.0032 |

**Interpretation**

- More tuning did not improve the final test result
- The original technical LSTM was already strong
- This is useful in the presentation because it shows the result was not just tuning luck

**Visual to use**

Use the small comparison table rather than all eight tuning rows.

---

# 18. Fundamental Route: XGBoost Regime Extension

After the LightGBM results, we focused on the fundamental-only route and compared two XGBoost designs:

1. State as feature
2. Regime-aware

This was the key follow-up question:

**For fundamental variables, is it enough to add the state as one more feature, or is it better to let the model behave differently across regimes?**

---

# 19. XGBoost Design Results

| Fundamental-only design | Test IC | Test R2_oos |
|---|---:|---:|
| XGBoost state as feature | 0.0797 | 0.0158 |
| XGBoost regime-aware | 0.1211 | 0.0281 |

For reference:

| Comparator | Test IC | Test R2_oos |
|---|---:|---:|
| LightGBM state as feature | 0.1083 | 0.0167 |

**Main conclusion**

The regime-aware XGBoost model became the best overall fundamental route and the best benchmark-relative model in the project.

**Visual to use**

Use the XGBoost design comparison bar charts.

---

# 20. Regime-Aware LSTM: Why It Did Not Help

We also tested a regime-aware LSTM by training separate LSTM models by state.

Main result:

- it did not improve on the original LSTM
- on cleaner technical-only features, IC fell from 0.1242 to 0.0200
- on cleaner fundamental-only features, both IC and R2_oos became worse
- only the all-features case showed a small IC improvement, but R2_oos stayed negative

**Interpretation**

The LSTM seems to work better when regime is used as context inside one shared sequence model, rather than by splitting the sequence data into smaller state-specific samples.

---

# 21. Why the Fundamental Route Is Strong for Presentation

- It uses interpretable financial variables
- It directly answers the regime question
- It gives strong prediction metrics
- It is supported by diagnostics and econometric evidence

Top importance variables in the strong fundamental model:

- `gross_margin`
- `ebitda_margin`
- `pfcf_ratio`
- `debt_to_equity`
- `cash_to_liabilities`
- `return_on_assets`

**Visual to use**

Use the fundamental feature-importance bar chart.

---

# 22. Diagnostics: Best Cleaner LightGBM Fundamental Model

Model: cleaner fundamental-only LightGBM with state as feature

| Metric | Value |
|---|---:|
| Test MSE | 0.005649 |
| Test IC | 0.108331 |
| Test R2_oos | 0.016734 |

By state:

| State | Test IC | Test R2_oos |
|---|---:|---:|
| 0 | 0.1185 | 0.0048 |
| 1 | 0.0762 | 0.0426 |

**Interpretation**

- better ranking in the stressed state
- better benchmark-relative fit in the calm state

**Visual to use**

Use the actual-vs-predicted scatter and the monthly average actual-vs-predicted chart.

---

# 23. Diagnostics: Regime-Aware XGBoost by State

| State | Test IC | Test R2_oos |
|---|---:|---:|
| 0 | 0.0562 | -0.0545 |
| 1 | 0.1492 | 0.0526 |

**What this tells us**

- The regime-aware XGBoost model worked much better in State 1
- In the calm regime, the model had both strong ranking and positive benchmark-relative fit
- This supports the idea that fundamental information is priced differently across regimes

**Visual to use**

Use the regime-aware scatter-by-state plots and the by-state importance charts.

---

# 24. Economic Interpretation of By-State Importance

The regime-aware fundamental model shifted emphasis across states.

State 0 placed more weight on:

- `gross_margin`
- `ebitda_margin`
- `pfcf_ratio`
- `return_on_assets`

State 1 placed relatively more weight on:

- `cash_to_liabilities`
- `quick_ratio`
- `debt_to_equity`

**Interpretation**

The model appears to move between profitability and valuation signals on one side, and liquidity and balance-sheet resilience on the other.

---

# 25. Fama-MacBeth Cross-Check

We also ran a cleaner fundamental-only Fama-MacBeth regression with regime interactions.

| Term | Mean Beta | t-stat | p-value |
|---|---:|---:|---:|
| `state_indicator` | 0.005268 | 4.769 | 0.000005 |
| `cash_to_liabilities x state1` | 0.002518 | 2.534 | 0.0125 |
| `debt_to_equity` | -0.002465 | -2.901 | 0.0044 |

**Interpretation**

- Regime shifts mattered statistically, not just in machine learning
- Liquidity mattered more in State 1
- Higher leverage was associated with lower future return in the baseline regime

---

# 26. Interpretation: Why the Results Split

The key interpretation is that regime information did not help every model in the same way.

- For technical signals, the main challenge is extracting time structure from price-based indicators
- For fundamental signals, the main challenge is that the pricing of those variables changes across market states
- That is why the best technical route was the LSTM, while the best fundamental route was regime-aware XGBoost

**Main insight**

The value of regime information depends on the type of signal being modeled, not just on whether a regime variable is included.

---

# 27. Insights for the Discussion

- The technical route gave the strongest ranking result, which matters for stock selection
- The fundamental regime-aware route gave the strongest benchmark-relative result, which is stronger for claiming real forecasting value
- The fundamental route is also easier to explain in finance terms because the drivers are profitability, valuation, liquidity, and leverage
- The Fama-MacBeth results support the same story, so the machine-learning result is not purely a black-box finding

**How to present this**

For the written report and final takeaway, emphasize the regime-aware fundamental route as the main finance result, and present the technical LSTM as the strongest alternative route for ranking.

---

# 28. What We Learned Overall

1. The HMM produced meaningful calm and stressed market states
2. Feature diagnostics improved the quality of the final model sets
3. Technical and fundamental signals should not be treated the same way
4. LSTM was strongest for technical signals
5. Regime-aware XGBoost was strongest for fundamental signals

**Core insight**

The usefulness of regime information depends on the type of signal and on the model architecture.

---

# 29. Final Conclusion

## Best technical route

- LSTM on cleaner technical-only features
- Test IC = 0.1242
- Best pure ranking result

## Best overall presentation route

- Regime-aware XGBoost on cleaner fundamental-only features
- Test IC = 0.1211
- Test R2_oos = 0.0281
- Stronger finance interpretation and stronger benchmark-relative performance

**Recommended final message**

Lead with the regime-aware fundamental result as the main story, and present the technical LSTM as the best alternative route.

---

# 30. Final One-Slide Summary

- We built a full monthly stock-return prediction pipeline from daily stock and market data
- We detected calm and stressed market states using a 2-state HMM
- We screened features and created cleaner technical and fundamental sets
- LSTM won on technical signals
- Regime-aware XGBoost won on fundamental signals
- Best finance takeaway: cleaner fundamentals become more predictive when the model is allowed to vary across market regimes

## Closing line

Market regime information is not universally useful, but it becomes valuable when paired with cleaner fundamental variables and an appropriate model design.
