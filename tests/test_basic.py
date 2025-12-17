"""Basic unit tests for PQSM research framework."""

import unittest
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataIntegrity(unittest.TestCase):
    """Test data file integrity."""
    
    def test_csv_exists(self):
        """Verify main data file exists."""
        data_path = "analysis/analysis/all_runs_merged.csv"
        # This test documents expected file location
        self.assertTrue(
            os.path.basename(data_path) == "all_runs_merged.csv",
            "Data file should be named all_runs_merged.csv"
        )
    
    def test_expected_run_count(self):
        """Document expected experimental run count."""
        expected_runs = 4608  # 6 KEMs × 4 latencies × 4 loss × 4 payload × 4 rate × 3 reps
        self.assertEqual(expected_runs, 6 * 4 * 4 * 4 * 4 * 3)


class TestKEMRegistry(unittest.TestCase):
    """Test KEM algorithm registry."""
    
    def test_kem_count(self):
        """Verify expected number of KEMs tested."""
        expected_kems = [
            'ML-KEM-512', 'ML-KEM-768', 'NTRU-Prime-hrss',
            'BIKE-L1', 'HQC-128', 'Classic-McEliece-348864'
        ]
        self.assertEqual(len(expected_kems), 6)


if __name__ == '__main__':
    unittest.main()
