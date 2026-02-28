"""Utility functions for the OrganoChron pipeline.

Provides configuration loading, logging setup, checkpoint management,
random seed control, and common data I/O helpers.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    logger.info(f"Loaded configuration from {config_path}")
    return cfg


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "results/logs", level: str = "INFO") -> None:
    """Configure loguru sinks for console and file output.

    Parameters
    ----------
    log_dir : str
        Directory for log files.
    level : str
        Minimum log level.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "organochron_{time}.log",
        rotation="50 MB",
        retention="7 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
    )
    logger.info("Logging initialised")


class Timer:
    """Simple context-manager timer that logs elapsed time.

    Example
    -------
    >>> with Timer("my_task"):
    ...     heavy_computation()
    """

    def __init__(self, label: str = "Task") -> None:
        self.label = label
        self.start: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        logger.info(f"[START] {self.label}")
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.perf_counter() - self.start
        logger.info(f"[DONE]  {self.label} — {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for numpy and Python's random module.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    logger.debug(f"Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Checkpoint / persistence
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist.

    Parameters
    ----------
    path : str or Path
        Directory path.

    Returns
    -------
    Path
        The same path as a ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(obj: Any, name: str, directory: str = "results/checkpoints") -> Path:
    """Persist an arbitrary Python object via pickle.

    Parameters
    ----------
    obj : Any
        Object to save.
    name : str
        Base filename (without extension).
    directory : str
        Target directory.

    Returns
    -------
    Path
        Path to the saved file.
    """
    d = ensure_dir(directory)
    path = d / f"{name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(name: str, directory: str = "results/checkpoints") -> Any:
    """Load a previously saved checkpoint.

    Parameters
    ----------
    name : str
        Base filename (without extension).
    directory : str
        Directory containing checkpoints.

    Returns
    -------
    Any
        The unpickled object.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    """
    path = Path(directory) / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    logger.info(f"Checkpoint loaded: {path}")
    return obj


def checkpoint_exists(name: str, directory: str = "results/checkpoints") -> bool:
    """Check whether a checkpoint file exists.

    Parameters
    ----------
    name : str
        Base filename (without extension).
    directory : str
        Directory containing checkpoints.

    Returns
    -------
    bool
    """
    return (Path(directory) / f"{name}.pkl").exists()


# ---------------------------------------------------------------------------
# DataFrame I/O helpers
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to Parquet format.

    Parameters
    ----------
    df : pd.DataFrame
    path : str or Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    logger.info(f"Saved parquet: {path}  ({len(df)} rows)")


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from Parquet.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_parquet(Path(path), engine="pyarrow")
    logger.info(f"Loaded parquet: {path}  ({len(df)} rows)")
    return df


def save_json(obj: Any, path: str | Path) -> None:
    """Save an object as JSON.

    Parameters
    ----------
    obj : Any
        JSON-serialisable object.
    path : str or Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class _Enc(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=_Enc)
    logger.info(f"Saved JSON: {path}")


def load_json(path: str | Path) -> Any:
    """Load a JSON file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    Any
    """
    with open(Path(path), "r") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def fdr_correction(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : np.ndarray
        Array of raw p-values.
    alpha : float
        Significance level (used only for the boolean mask).

    Returns
    -------
    rejected : np.ndarray of bool
        Which hypotheses are rejected at *alpha*.
    pvals_corrected : np.ndarray
        Adjusted p-values.
    """
    from statsmodels.stats.multitest import multipletests
    rejected, pvals_corrected, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    return rejected, pvals_corrected


def age_midpoint(age_bracket: str, mapping: dict[str, int] | None = None) -> float:
    """Convert a GTEx age bracket string to a numeric midpoint.

    Parameters
    ----------
    age_bracket : str
        E.g. ``"20-29"`` or ``"60-69"``.
    mapping : dict, optional
        Override mapping; if *None* the default GTEx decades are used.

    Returns
    -------
    float
        Midpoint value.

    Example
    -------
    >>> age_midpoint("40-49")
    45.0
    """
    if mapping is None:
        mapping = {"20-29": 25, "30-39": 35, "40-49": 45,
                   "50-59": 55, "60-69": 65, "70-79": 75}
    return float(mapping.get(age_bracket, np.nan))


# ---------------------------------------------------------------------------
# Tissue classification for colouring in figures
# ---------------------------------------------------------------------------

TISSUE_SYSTEM: dict[str, str] = {
    "Adipose - Subcutaneous": "metabolic",
    "Adipose - Visceral (Omentum)": "metabolic",
    "Adrenal Gland": "endocrine",
    "Artery - Aorta": "cardiovascular",
    "Artery - Coronary": "cardiovascular",
    "Artery - Tibial": "cardiovascular",
    "Brain - Cortex": "nervous",
    "Brain - Cerebellum": "nervous",
    "Colon - Sigmoid": "digestive",
    "Colon - Transverse": "digestive",
    "Esophagus - Mucosa": "digestive",
    "Heart - Atrial Appendage": "cardiovascular",
    "Heart - Left Ventricle": "cardiovascular",
    "Kidney - Cortex": "urinary",
    "Liver": "metabolic",
    "Lung": "respiratory",
    "Muscle - Skeletal": "musculoskeletal",
    "Nerve - Tibial": "nervous",
    "Pancreas": "digestive",
    "Pituitary": "endocrine",
    "Skin - Not Sun Exposed (Suprapubic)": "integumentary",
    "Skin - Sun Exposed (Lower leg)": "integumentary",
    "Small Intestine - Terminal Ileum": "digestive",
    "Spleen": "immune",
    "Stomach": "digestive",
    "Thyroid": "endocrine",
    "Whole Blood": "immune",
}

SYSTEM_COLORS: dict[str, str] = {
    "cardiovascular": "#E24A33",
    "nervous": "#7A68A6",
    "digestive": "#348ABD",
    "metabolic": "#188487",
    "endocrine": "#A60628",
    "immune": "#CF4457",
    "respiratory": "#467821",
    "musculoskeletal": "#D4A017",
    "integumentary": "#E5AE38",
    "urinary": "#56B4E9",
}
