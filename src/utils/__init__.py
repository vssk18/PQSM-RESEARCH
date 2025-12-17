"""
Shared utilities for PQSM Research framework.

Provides common functionality including:
- Logging configuration
- Reproducibility helpers
- Configuration management
"""

import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def setup_logging(
    name: str = "pqsm",
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Configure logging for PQSM modules.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_experiment_config(
    seed: int,
    output_dir: Path,
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Log experiment configuration for reproducibility.
    
    Args:
        seed: Random seed used for experiment
        output_dir: Directory to save config file
        additional_params: Extra parameters to log
        
    Returns:
        Short hash identifying this configuration
    """
    config: Dict[str, Any] = {
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    
    if additional_params:
        config["parameters"] = additional_params
    
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = output_dir / f"experiment_config_{config_hash}.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    return config_hash


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    name: str = "DataFrame"
) -> None:
    """
    Validate that a DataFrame has required columns.
    
    Raises:
        ValueError: If required columns are missing
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


__all__ = [
    "setup_logging",
    "log_experiment_config",
    "validate_dataframe",
]
