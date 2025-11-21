# Figure Interpretation Guide

## Figure 1: Handshake Time CDFs
**What to read**: Find your SLO on x-axis, read CDF value on y-axis
**Operator takeaway**: Higher CDF at your SLO = better; ML-KEM-512 reaches 100% by 50ms
**Key metric**: P(T_h ≤ 100ms) for each KEM

## Figure 2: Delivery vs Latency
**What to read**: Slope indicates sensitivity to network delay
**Operator takeaway**: Flatter curves are more resilient; all KEMs maintain >95% until 100ms
**Key metric**: Delivery ratio at your typical latency

## Figure 3: Delivery Heatmap
**What to read**: Green = operational (D≥0.95), Red = failing
**Operator takeaway**: Find your (latency, loss) point; green means KEM will work
**Key metric**: Safe area percentage (green fraction)

## Figure 4: Decrypt Time Distribution
**What to read**: Peak location = typical decrypt time, width = variability
**Operator takeaway**: Narrow peaks are predictable; Classic-McEliece 10x slower
**Key metric**: Median and 95th percentile decrypt times

## Figure 5: Goodput vs Loss
**What to read**: Slope shows throughput sensitivity to packet loss
**Operator takeaway**: Flatter = more stable; all KEMs degrade similarly
**Key metric**: Goodput at 5% loss (typical wireless)
