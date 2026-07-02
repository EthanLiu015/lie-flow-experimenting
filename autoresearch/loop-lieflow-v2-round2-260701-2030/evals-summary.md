# LieFlow v2 Round-2 Autoresearch

**Metric:** min Sharpe across 6 windows @ 10bps, n=60 universe  
**Status:** Partial convergence — 4 configs all-windows-positive, no pure canonical LieFlow

## Winners (all 6 windows Sharpe > 0)

| Config | min Sharpe | mean Sharpe | Coverage |
|--------|-----------|-------------|----------|
| raw_vol smooth=20 | 0.47 | 0.91 | 100% days |
| mom_minus_vol flip s25 | 0.03 | 0.38 | 100% days (pure LieFlow) |
| hybrid vol conc1.0 | 0.30 | 1.14 | ~64% days |
| hybrid vol conc1.2 w0.5 | 1.09 | 1.94 | ~25% days |

## Key takeaways

1. **Pure LieFlow alpha is weak** — `mom_minus_vol` with sign flip is the only full-coverage LieFlow strategy that clears zero on all windows, and only barely (wf_2022_2024 Sharpe = 0.03).
2. **Raw vol dominates** — same universe/costs, min Sharpe 0.47 with positive IC in 5/6 windows.
3. **LieFlow adds value as a gate** — hybrid vol uses LieFlow concentration to filter raw-vol days; conc1.0 is the defensible middle ground.
4. **Delta/residual strategies failed** — delta_momentum, delta_radial, conc_scaled_momentum all deeply negative on multiple windows.

## Recommended configs

- **Research / full coverage:** `mom_minus_vol --signal-sign -1 --signal-smoothing 25`
- **Production / robust:** `--raw-vol-baseline --signal-smoothing 20`
- **LieFlow-enhanced:** `--hybrid-vol --min-concentration-ratio 1.0 --signal-smoothing 20`
