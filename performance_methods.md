# Performance Measurement Methodology

| Metric | Definition | Data Source |
|--------|------------|-------------|
| Handshake Time | Median ± MAD over all runs | all_runs_merged.csv: handshake_ms column |
| Decrypt Time | 50th percentile | all_runs_merged.csv: p50_decrypt_ms column |
| Delivery Ratio | Mean with Wilson 95% CI | all_runs_merged.csv: delivery_ratio column |
| Safe Area | Fraction of (L,p) grid with D≥0.95 | Computed from delivery_ratio grouped by latency_ms, loss_pct |

All measurements from commit: $(git rev-parse HEAD 2>/dev/null || echo "pending")
