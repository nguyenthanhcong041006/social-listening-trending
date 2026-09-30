import os
import json
import pickle
import yaml
import pandas as pd
from typing import Any, Dict
from loguru import logger

def ensure_dir(file_path: str):
    """Creates the parent directory if it does not already exist."""
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], file_path: str):
    """Saves data to a YAML file."""
    ensure_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Loads a DataFrame from CSV, Parquet, or JSON format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
        return pd.read_json(file_path, lines=file_path.endswith(".jsonl"))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_dataframe(df: pd.DataFrame, file_path: str):
    """Saves a DataFrame to CSV or Parquet format."""
    ensure_dir(file_path)
    if file_path.endswith(".csv"):
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    elif file_path.endswith(".parquet"):
        df.to_parquet(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)
    logger.info(f"Saved {len(df)} rows to: {file_path}")

def save_pickle(obj: Any, file_path: str):
    """Serializes a Python object to a pickle file."""
    ensure_dir(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(file_path: str) -> Any:
    """Deserializes a Python object from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)
