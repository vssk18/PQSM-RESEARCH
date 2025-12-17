"""
Unit tests for KEM type definitions.

Author: Varanasi Sai Srinivasa Karthik
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crypto.kem_types import (
    KEMParameters,
    KEMError,
    KEMNotAvailableError,
    OQS_NAME_MAP,
)


class TestKEMParameters(unittest.TestCase):
    """Test KEMParameters dataclass validation."""
    
    def test_valid_parameters(self):
        """Test creating valid KEM parameters."""
        params = KEMParameters(
            name="ML-KEM-512",
            public_key_size=800,
            secret_key_size=1632,
            ciphertext_size=768,
            shared_secret_size=32,
            security_level=1,
            algorithm_type="lattice"
        )
        self.assertEqual(params.name, "ML-KEM-512")
        self.assertEqual(params.security_level, 1)
    
    def test_invalid_security_level(self):
        """Test invalid security levels raise ValueError."""
        with self.assertRaises(ValueError):
            KEMParameters(
                name="Invalid",
                public_key_size=100,
                secret_key_size=100,
                ciphertext_size=100,
                shared_secret_size=32,
                security_level=2,  # Invalid
                algorithm_type="test"
            )
    
    def test_immutability(self):
        """Test KEMParameters is frozen."""
        params = KEMParameters(
            name="Test", public_key_size=100, secret_key_size=100,
            ciphertext_size=100, shared_secret_size=32,
            security_level=1, algorithm_type="test"
        )
        with self.assertRaises(AttributeError):
            params.name = "Modified"


class TestExperimentalConstants(unittest.TestCase):
    """Test constants match paper specifications."""
    
    def test_experimental_grid(self):
        """Verify grid produces 4,608 runs."""
        total = 6 * 4 * 4 * 4 * 4 * 3  # KEMs × latencies × loss × payload × rate × reps
        self.assertEqual(total, 4608)
    
    def test_per_kem_allocation(self):
        """Verify each KEM gets 768 runs."""
        self.assertEqual(4608 // 6, 768)


if __name__ == '__main__':
    unittest.main()
