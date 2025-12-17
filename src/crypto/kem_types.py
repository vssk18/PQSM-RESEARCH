"""
Type definitions for KEM implementations.

This module provides type-safe interfaces and custom exceptions
for the PQSM cryptographic framework.

Author: Varanasi Sai Srinivasa Karthik
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable


# Custom Exceptions
class KEMError(Exception):
    """Base exception for KEM operations."""
    pass


class KEMNotAvailableError(KEMError):
    """Raised when a KEM algorithm is not available."""
    def __init__(self, algorithm: str, reason: str = ""):
        self.algorithm = algorithm
        self.reason = reason
        message = f"KEM algorithm '{algorithm}' is not available"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class KEMKeyGenerationError(KEMError):
    """Raised when key generation fails."""
    pass


class KEMEncapsulationError(KEMError):
    """Raised when encapsulation fails."""
    pass


class KEMDecapsulationError(KEMError):
    """Raised when decapsulation fails."""
    pass


class SharedSecretMismatchError(KEMError):
    """Raised when shared secrets don't match (verification failure)."""
    pass


@runtime_checkable
class KEMProtocol(Protocol):
    """
    Protocol defining the interface for KEM implementations.
    """
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate (public_key, secret_key)."""
        ...
    
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Return (ciphertext, shared_secret)."""
        ...
    
    def decapsulate(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """Return shared_secret."""
        ...


@dataclass(frozen=True)
class KEMParameters:
    """Immutable parameters for a KEM algorithm."""
    name: str
    public_key_size: int
    secret_key_size: int
    ciphertext_size: int
    shared_secret_size: int
    security_level: int
    algorithm_type: str
    
    keygen_cycles: Optional[int] = None
    encaps_cycles: Optional[int] = None
    decaps_cycles: Optional[int] = None
    stack_bytes: Optional[int] = None
    heap_bytes: Optional[int] = None
    constant_time: bool = True
    hardware_accelerated: bool = False
    side_channel_resistant: bool = True
    
    def __post_init__(self) -> None:
        if self.security_level not in (0, 1, 3, 5):
            raise ValueError(f"Invalid security level: {self.security_level}")
        if self.public_key_size <= 0:
            raise ValueError("public_key_size must be positive")
        if self.secret_key_size <= 0:
            raise ValueError("secret_key_size must be positive")


# Algorithm name mappings (OQS library names)
OQS_NAME_MAP: Dict[str, str] = {
    "ML-KEM-512": "Kyber512",
    "ML-KEM-768": "Kyber768",
    "ML-KEM-1024": "Kyber1024",
    "BIKE-L1": "BIKE-L1",
    "BIKE-L3": "BIKE-L3",
    "HQC-128": "HQC-128",
    "HQC-192": "HQC-192",
    "HQC-256": "HQC-256",
    "Classic-McEliece-348864": "Classic-McEliece-348864",
    "NTRU-HPS-2048-509": "NTRU-HPS-2048-509",
    "NTRU-Prime-sntrup761": "sntrup761",
    "NTRU-Prime-hrss": "ntruhps2048677",
}


__all__ = [
    "KEMError",
    "KEMNotAvailableError",
    "KEMKeyGenerationError",
    "KEMEncapsulationError",
    "KEMDecapsulationError",
    "SharedSecretMismatchError",
    "KEMProtocol",
    "KEMParameters",
    "OQS_NAME_MAP",
]
