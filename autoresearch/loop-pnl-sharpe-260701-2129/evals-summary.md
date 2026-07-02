# PnL + Sharpe Autoresearch

- **Metric:** composite = mean_total_return + 0.05×min_sharpe + 0.02×min_window_return
- **Sharpe floor:** 0.3
- **Best composite:** `raw_vol s20 cost5` (score=0.2670, coverage=100.0%)
- **Best min Sharpe:** 0.5563
- **Best mean return:** 0.2366
- **Highest PnL (eligible):** `raw_vol s20 cost5` (mean_ret=0.2366, min_sharpe=0.5563, coverage=100.0%)

## Recommendation

For **max PnL with Sharpe discipline**, prefer full-coverage configs with min_sharpe ≥ 0.3 rather than selective high-Sharpe/low-coverage gates.
