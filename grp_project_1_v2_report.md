# Market Regime-Aware Prediction of Cross-Sectional Stock Returns

## Abstract

This study examines whether next-month stock returns can be predicted more effectively by combining stock-level technical and fundamental characteristics with market regime information. Using a panel of 40 large-cap US equities, a monthly cross-sectional forecasting pipeline was constructed from daily price, volume, and accounting data. Market regimes were identified through a two-state Gaussian Hidden Markov Model estimated on NASDAQ-100 returns, realised volatility, and the VIX. Three model families were evaluated: LightGBM, XGBoost, and a PyTorch-based LSTM. The results indicate that predictive performance depends on the interaction between signal type and modelling framework. Technical variables were most effectively captured by the LSTM, which achieved the highest ranking performance on the cleaner technical feature set with Test IC = 0.1242 and Test $R^2_{oos}$ = 0.0079. Fundamental variables responded more strongly to regime conditioning, with the regime-aware XGBoost model on the cleaner fundamental feature set producing the strongest benchmark-relative result, with Test IC = 0.1211 and Test $R^2_{oos}$ = 0.0281. A Fama-MacBeth extension with regime interactions provided supporting evidence that the pricing of selected fundamentals changes across market states. Overall, the findings suggest that regime information is most valuable when paired with economically interpretable fundamental variables, whereas technical signals are more effectively exploited through sequential modelling.

## Introduction

Forecasting stock returns remains difficult because return-generating processes are unstable over time and depend on both firm-specific information and broader market conditions. In practice, technical indicators, accounting ratios, and macro-financial risk measures may contain useful predictive information, but their relevance is unlikely to remain constant across different market environments. This makes regime dependence an important issue in return prediction.

The present study investigates whether cross-sectional stock-return forecasts improve when market state information is incorporated explicitly into the modelling framework. The project is motivated by three questions. First, can a cleaner monthly prediction pipeline be constructed from raw daily stock-panel data? Second, does market regime information improve predictive performance once technical and fundamental features have been screened for redundancy? Third, which empirical route is more defensible in a final finance report: a technical sequence-model approach or a regime-aware fundamental approach?

To address these questions, the analysis begins with a daily data-processing pipeline that creates monthly stock-level features and next-month return targets. Market regimes are then extracted using a Hidden Markov Model estimated on market-wide indicators. After feature diagnostics and screening, three model families are compared: LightGBM, XGBoost, and LSTM. The empirical results reveal a clear split: technical signals are most successfully modelled by the LSTM, whereas fundamental variables gain more from regime-aware tree-based models.

## Methods

### Data

The stock panel was constructed from a daily US equity dataset covering 40 large-cap firms, including AAPL, MSFT, GOOGL, AMZN, JNJ, XOM, NVDA, and other major names. Market-state variables were obtained from NASDAQ-100 prices and the VIX. The raw daily stock pipeline implemented the following sequence:

1. Daily OHLCV and accounting variables were loaded for each stock.
2. Fundamental variables were lagged to reduce look-ahead bias.
3. Technical and accounting-based predictors were engineered.
4. The target variable was defined as next-month stock return.
5. Missing values were handled and the cleaned train and test panels were saved.

The resulting sample sizes were as follows.

| Dataset | Shape before cleaning | Final usable rows |
|---|---:|---:|
| Training panel | 100,320 x 86 | 89,400 |
| Testing panel | 25,120 x 86 | 14,200 |

Monthly prediction panels were formed from month-end observations.

| Monthly sample | Shape |
|---|---:|
| Monthly train panel | 4,240 x 119 |
| Monthly test panel | 680 x 119 |

The monthly train period spans 2011-01-31 to 2019-10-31, and the monthly test period spans 2020-12-31 to 2022-04-29.

The month-end design is intentional rather than incidental. For each stock, the model uses the last trading day of month $t$ as the decision date and predicts the return from month-end $t$ to month-end $t+1$. This is a natural setup for a monthly rebalancing strategy and aligns well with the frequency at which many accounting variables evolve. Although the final modelling panel keeps only one row per stock per month, the information from earlier trading days is not discarded in an economic sense. Those earlier observations are embedded in the engineered predictors themselves, including one-month and three-month returns, twelve-month momentum, volatility, MACD, ATR, OBV, and the month-end price level from which the next-month return is measured. The design therefore reduces repeated daily observations while still retaining information accumulated over the month.

This monthly framework is also especially defensible for fundamentals. Since accounting variables are typically updated quarterly and were additionally lagged in this project to reduce look-ahead bias, a daily prediction horizon would create many observations with nearly unchanged fundamental inputs. Sampling at month-end provides a cleaner alignment between information arrival, portfolio decision timing, and forecast horizon.

### Market Regime Identification

Market regimes were estimated using a two-state Gaussian Hidden Markov Model. The model was fitted on four market-wide variables:

- NASDAQ-100 log returns
- 20-day realised volatility
- VIX level
- VIX daily change

Twenty random initialisations were used in order to improve stability. The final state classification was economically interpretable.

| State | Mean Return | Return Std | Mean Volatility | Mean VIX | Interpretation |
|---|---:|---:|---:|---:|---|
| 0 | 0.000034 | 0.019093 | 0.017699 | 27.279 | High-volatility regime |
| 1 | 0.001032 | 0.008252 | 0.008284 | 15.006 | Low-volatility regime |

The state counts were 1,276 observations in State 0 and 2,228 observations in State 1. Accordingly, State 0 was interpreted as a stressed, high-volatility regime, while State 1 was interpreted as a calmer, lower-volatility regime.

### Feature Engineering and Screening

The initial feature pool combined technical and accounting-based measures. The notebook then performed diagnostic screening using redundancy checks, correlations, variance inflation factors, and univariate monthly information coefficients.

Several overlaps were identified. `volatility_1m` and `stdev_feature` were effectively duplicates. `price_vs_sma20` was highly correlated with `return_1m`. `return_on_assets` and `return_on_equity` were strongly correlated, as were `cash_to_liabilities` and `current_ratio`. These findings motivated the use of a smaller and more defensible set of predictors in the final modelling stage.

Three cleaner feature sets were retained.

Technical-only feature set:

- `return_1m`
- `momentum_12m`
- `MACDh`
- `adx`
- `ATR`
- `obv`

Fundamental-only feature set:

- `pfcf_ratio`
- `return_on_assets`
- `ebitda_margin`
- `gross_margin`
- `quick_ratio`
- `debt_to_equity`
- `cash_to_liabilities`

All-features set:

- Combined technical-only and fundamental-only predictors

All monthly predictors were transformed using cross-sectional quantile scaling so that each stock’s monthly feature values were expressed relative to peers in the same month.

### Modelling Strategy

Three predictive model families were evaluated:

- LightGBM
- XGBoost
- LSTM implemented in PyTorch

For the tree-based models, three regime treatments were considered:

1. Pure baseline, with no regime information
2. State as feature, where the HMM state was included as an additional predictor
3. Regime-aware design, where modelling or prediction was conditioned on the inferred state

Model evaluation focused on both forecast accuracy and cross-sectional ranking quality. The metrics reported were MSE, RMSE, MAE, $R^2$, out-of-sample $R^2$ relative to a historical-mean benchmark, denoted $R^2_{oos}$, and the Spearman Information Coefficient (IC). Because the project’s objective is predictive ranking and benchmark-relative performance, Test IC and Test $R^2_{oos}$ were treated as the primary summary measures.

The resulting interpretation is cross-sectional and monthly: at each month-end, the model observes a stock-specific feature snapshot, ranks stocks or predicts their next-month return, and is then evaluated on the return realised by the next month-end. This is different from averaging all trading days within the month. Instead, the procedure uses month-end sampling after daily features have already accumulated within-month information.

### Supplementary Econometric Analysis

To complement the machine-learning results, a Fama-MacBeth regression was estimated using the cleaner fundamental feature set and regime interaction terms. This extension was intended to assess whether the pricing of selected accounting characteristics varies across market states in a more interpretable cross-sectional framework.

## Results

### LightGBM Baseline Comparison

The first stage fixed the model family at LightGBM and examined whether the inclusion of regime information improved performance across the three cleaner feature sets.

| Feature Set | Model Design | Test IC | Test $R^2_{oos}$ |
|---|---|---:|---:|
| Cleaner all features | Pure baseline | 0.0208 | -0.0240 |
| Cleaner all features | State as feature | 0.0503 | -0.0137 |
| Cleaner all features | Regime-aware | 0.0234 | -0.0541 |
| Cleaner technical-only | Pure baseline | 0.0848 | -0.0171 |
| Cleaner technical-only | State as feature | 0.0288 | -0.0314 |
| Cleaner technical-only | Regime-aware | 0.0643 | -0.0814 |
| Cleaner fundamental-only | Pure baseline | 0.0582 | -0.0015 |
| Cleaner fundamental-only | State as feature | 0.1083 | 0.0167 |
| Cleaner fundamental-only | Regime-aware | 0.0870 | -0.0208 |

Within the LightGBM family, the strongest cleaner specification was the fundamental-only model with state included as a feature. This model achieved Test IC = 0.1083 and Test $R^2_{oos}$ = 0.0167. By contrast, the technical-only specification produced acceptable ranking performance in the pure baseline case, but its benchmark-relative $R^2_{oos}$ remained negative. These results suggested that regime information was more useful in the interpretation of fundamental variables than in the interpretation of technical indicators within LightGBM.

### Model Family Comparison

The second stage compared LightGBM, XGBoost, and LSTM across the three cleaner feature sets.

| Feature Set | LightGBM Test IC | XGBoost Test IC | LSTM Test IC |
|---|---:|---:|---:|
| Cleaner all features | 0.0503 | 0.0599 | 0.0230 |
| Cleaner technical-only | 0.0288 | 0.0579 | 0.1242 |
| Cleaner fundamental-only | 0.1083 | 0.0797 | 0.0333 |

| Feature Set | LightGBM Test $R^2_{oos}$ | XGBoost Test $R^2_{oos}$ | LSTM Test $R^2_{oos}$ |
|---|---:|---:|---:|
| Cleaner all features | -0.0137 | -0.0054 | -0.0100 |
| Cleaner technical-only | -0.0314 | -0.0139 | 0.0079 |
| Cleaner fundamental-only | 0.0167 | 0.0158 | -0.0279 |

This comparison revealed a strong division between signal families. The LSTM delivered the best technical result, with Test IC = 0.1242 and Test $R^2_{oos}$ = 0.0079 on the cleaner technical feature set. In contrast, the strongest fundamental result within this comparison remained the LightGBM state-as-feature model, with Test IC = 0.1083 and Test $R^2_{oos}$ = 0.0167. No model using the combined all-features set outperformed the best specialised technical or fundamental routes.

### LSTM Tuning

An eight-configuration tuning exercise was conducted for the cleaner technical-only LSTM, varying lookback length, hidden size, learning rate, and batch size. The best validation specification used lookback = 3, hidden size = 32, dense size = 16, learning rate = 0.0005, and batch size = 16. This configuration achieved Validation IC = 0.0514 and Validation $R^2_{oos}$ = 0.0025.

However, the tuned model did not exceed the original technical LSTM on the final test set.

| Model | Test IC | Test $R^2_{oos}$ |
|---|---:|---:|
| LSTM (original technical-only) | 0.1242 | 0.0079 |
| LSTM (tuned technical-only) | 0.0983 | 0.0032 |

This result indicates that limited tuning did not materially improve on the original specification and that the baseline technical LSTM was already close to the local optimum within the tested grid.

### Regime-Aware XGBoost Extension

Given the earlier evidence that regime information was most valuable for fundamentals, the analysis next compared two XGBoost designs: state as feature and regime-aware.

| Feature Set | XGBoost State-as-Feature IC | XGBoost Regime-Aware IC | State-as-Feature $R^2_{oos}$ | Regime-Aware $R^2_{oos}$ |
|---|---:|---:|---:|---:|
| Cleaner all features | 0.0599 | 0.0408 | -0.0054 | -0.0223 |
| Cleaner technical-only | 0.0579 | 0.0422 | -0.0139 | -0.0569 |
| Cleaner fundamental-only | 0.0797 | 0.1211 | 0.0158 | 0.0281 |

The regime-aware design improved performance only in the fundamental-only specification, but the improvement there was economically and empirically meaningful. The regime-aware XGBoost model on the cleaner fundamental feature set became the strongest overall fundamental route, achieving Test IC = 0.1211 and Test $R^2_{oos}$ = 0.0281. It therefore exceeded the performance of the LightGBM state-as-feature model and provided the best benchmark-relative result in the project.

### Model Diagnostics

The best cleaner LightGBM model, namely the fundamental-only state-as-feature specification, produced Test MSE = 0.005649, Test IC = 0.108331, and Test $R^2_{oos}$ = 0.016734. Its feature importance ranking is reported below.

| Feature | Importance Share |
|---|---:|
| `gross_margin` | 15.29% |
| `ebitda_margin` | 15.04% |
| `pfcf_ratio` | 14.77% |
| `debt_to_equity` | 14.18% |
| `cash_to_liabilities` | 13.97% |
| `return_on_assets` | 13.41% |
| `quick_ratio` | 11.46% |
| `state` | 1.89% |

Although the state variable itself carried modest direct importance, its inclusion improved predictive performance. This suggests that regime information functions primarily as a conditioning variable that alters the interpretation of other predictors rather than as a dominant independent driver.

By test-state subsample, the same LightGBM model achieved the following results.

| State | Test IC | Test $R^2_{oos}$ |
|---|---:|---:|
| 0 | 0.1185 | 0.0048 |
| 1 | 0.0762 | 0.0426 |

The model ranked stocks more effectively in the stressed state, but it generated stronger benchmark-relative fit in the calmer state.

The regime-aware XGBoost fundamental model exhibited even clearer state dependence.

| State | Test IC | Test $R^2_{oos}$ |
|---|---:|---:|
| 0 | 0.0562 | -0.0545 |
| 1 | 0.1492 | 0.0526 |

This pattern indicates that the regime-aware XGBoost specification was particularly effective in the calmer regime. The by-state feature importance profiles were also economically coherent. In State 0, the model placed relatively greater emphasis on profitability and valuation variables such as `gross_margin`, `ebitda_margin`, `pfcf_ratio`, and `return_on_assets`. In State 1, relatively more weight was placed on liquidity and balance-sheet resilience measures such as `cash_to_liabilities`, `quick_ratio`, and `debt_to_equity`.

### Fama-MacBeth Evidence

The Fama-MacBeth extension supported the interpretation that the pricing of fundamentals varies with regime.

The strongest interaction terms were:

| Term | Mean Beta | t-stat | p-value |
|---|---:|---:|---:|
| `state_indicator` | 0.005268 | 4.769 | 0.000005 |
| `cash_to_liabilities × state1` | 0.002518 | 2.534 | 0.0125 |

The positive and statistically significant state indicator suggests an upward intercept shift in State 1. In addition, the positive interaction on `cash_to_liabilities` implies that liquidity carries a stronger positive pricing effect in the calmer regime.

Among the regime-0 main effects, the strongest result was a negative coefficient on `debt_to_equity`.

| Factor | Mean Beta | t-stat | p-value |
|---|---:|---:|---:|
| `debt_to_equity` | -0.002465 | -2.901 | 0.0044 |

This indicates that higher leverage was associated with lower subsequent returns in the baseline regime. Taken together, the Fama-MacBeth evidence is consistent with the machine-learning results: the cross-sectional payoff to selected accounting characteristics changes across market states.

## Discussion

The central contribution of this study is to show that the usefulness of market regime information depends on the type of signal being modelled. Regime conditioning was not uniformly beneficial across all specifications. Instead, its value emerged most clearly when applied to a screened set of fundamental variables. In that setting, the regime-aware XGBoost model produced the best overall benchmark-relative performance, suggesting that the cross-sectional pricing of balance-sheet strength, profitability, and valuation characteristics is state dependent.

By contrast, the strongest technical result came from the LSTM applied to the cleaner technical feature set. This outcome is consistent with the view that technical signals contain temporal structure that is not fully captured by static tree models. The LSTM’s ability to outperform tree-based alternatives on technical indicators suggests that sequence modelling is appropriate when the predictors themselves are constructed from dynamic price patterns.

An additional regime-aware LSTM extension did not strengthen this conclusion. When separate LSTM models were trained by regime, performance deteriorated on the cleaner technical and cleaner fundamental feature sets, and only the combined all-features case showed a modest IC improvement while benchmark-relative fit remained negative. This suggests that the LSTM benefits more from using regime as contextual information within a shared sequence-learning problem than from splitting the time series into smaller regime-specific training samples. In other words, for sequence models, explicit regime partitioning appears to reduce sample efficiency more than it improves state-specific pattern extraction.

The contrast between the two best-performing routes is analytically useful. The technical route provides the strongest ranking result, while the fundamental regime-aware route provides the strongest benchmark-relative result and the most economically interpretable story. For a finance-oriented written report, the latter route is arguably more compelling because it combines improved predictive performance with a clear mechanism and supporting econometric evidence.

Several limitations remain. The universe is restricted to a relatively small set of large-cap firms, the test window is limited, and the regime structure is reduced to two states. Moreover, hyperparameter tuning was intentionally modest, which preserves interpretability but may leave some predictive gains unexplored. These limitations do not overturn the main result, but they suggest that future work could examine broader universes, longer horizons, alternative regime specifications, and more systematic tuning procedures.

In conclusion, the evidence indicates that market regime information is most valuable when paired with cleaner fundamental variables, whereas technical indicators are more effectively exploited through sequence models. The project therefore supports two defensible empirical conclusions: a technical conclusion centred on the LSTM’s superior ranking performance, and a broader finance conclusion centred on the regime-aware XGBoost model’s stronger benchmark-relative accuracy and clearer economic interpretation.