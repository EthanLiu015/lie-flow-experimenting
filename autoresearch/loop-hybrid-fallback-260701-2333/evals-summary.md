# Hybrid Fallback Autoresearch

- Honest min/mean Sharpe across all 6 windows
- ML test-only LieFlow; IS dates use raw-vol fallback
- Cost: 10bps
- Eligible (min_sharpe≥0.3, lieflow_influence≥0.05): 46
- Best: `hybrid fb c0.50 s16 w0.2 e0.75-1.00` composite=0.5300 minS=0.460 meanAnn=7.10% influence=39.7%
- Raw vol baseline: minS=0.451 meanAnn=6.99%
