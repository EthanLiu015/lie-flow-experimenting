# Hybrid Fallback Autoresearch

- Honest min/mean Sharpe across all 6 windows
- ML test-only LieFlow; IS dates use raw-vol fallback
- Cost: 10bps
- Eligible (min_sharpe≥0.3, lieflow_influence≥0.05): 0
- Best: `hybrid fb c0.00 s16 w0.2 e0.50-1.00` composite=-100.4618 minS=-0.462 meanAnn=3.06% influence=39.7%
- Raw vol baseline: minS=0.451 meanAnn=6.99%
