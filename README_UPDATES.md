# README Updates Required

## Change "first study" to:
"To our knowledge, this is a comprehensive study of PQC KEMs under realistic IoT network conditions"

## Update security claims to:
"Uses liboqs C implementations for cryptographic operations. Python orchestration is for research only and not timing-safe."

## Add limitations section:
### Limitations
- Bernoulli loss model (not bursty)
- Single broker instance (not distributed)
- Python overhead not suitable for production
- No formal security proofs provided

## Add measurement details:
"All metrics use: Median ± MAD for timing, Mean with Wilson CI for ratios. Source: all_runs_merged.csv"
