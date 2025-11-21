# Security Notes

## Cryptographic Implementation
- **Constant-time operations**: We use liboqs C implementations which provide constant-time primitives
- **Python orchestration**: The Python wrapper is NOT constant-time and is for research purposes only
- **Side-channel resistance**: Limited to the underlying liboqs library guarantees
- **Production readiness**: This is a research prototype, not production-ready code

## Limitations
- Python timing variations may leak information
- No formal security proofs provided
- Testing done on standard OS without isolation
