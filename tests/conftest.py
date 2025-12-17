"""
Pytest configuration and shared fixtures for PQSM tests.

Author: Varanasi Sai Srinivasa Karthik
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_experimental_data() -> pd.DataFrame:
    """Create sample data matching the paper's schema."""
    np.random.seed(42)
    
    kems = ["ML-KEM-512", "NTRU-Prime-hrss", "BIKE-L1"]
    n_samples = 30
    
    return pd.DataFrame({
        "run_id": [f"run_{i:04d}" for i in range(n_samples)],
        "kem_algorithm": np.random.choice(kems, n_samples),
        "latency_ms": np.random.choice([10, 50, 100, 150], n_samples),
        "loss_pct": np.random.choice([0, 1, 5, 10], n_samples),
        "rate_hz": np.random.choice([1, 2, 5, 10], n_samples),
        "payload_bytes": np.random.choice([128, 256, 512, 1024], n_samples),
        "delivery_ratio": np.random.uniform(0.95, 1.0, n_samples),
        "handshake_ms": np.random.uniform(10, 300, n_samples),
        "p50_decrypt_ms": np.random.uniform(0.5, 10, n_samples),
        "p95_decrypt_ms": np.random.uniform(1, 20, n_samples),
    })


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "requires_oqs: requires liboqs")
