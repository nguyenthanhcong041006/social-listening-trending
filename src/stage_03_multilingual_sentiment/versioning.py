import os
import re
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger
from src.utils.file_io import resolve_path, resolve_output_path

DEFAULT_EMBEDDING_BASE_DIR = "models/xlm_roberta/embedding"
DEFAULT_SENTIMENT_BASE_DIR = "models/xlm_roberta/sentiment"
DEFAULT_FALLBACK_EMBEDDING = "xlm-roberta-base"
DEFAULT_FALLBACK_SENTIMENT = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

VERSION_PATTERN = re.compile(r"^v(\d+)$", re.IGNORECASE)


def get_version_number(folder_name: str) -> Optional[int]:
    """Extracts integer version number from directory name (e.g., 'v1' -> 1, 'v2' -> 2)."""
    match = VERSION_PATTERN.match(folder_name)
    return int(match.group(1)) if match else None


def get_all_versions(base_dir: str) -> List[Tuple[int, str]]:
    """Returns sorted list of existing versions: [(1, 'v1'), (2, 'v2')]."""
    base_dir = resolve_path(base_dir)
    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        return []

    versions = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            v_num = get_version_number(item)
            if v_num is not None:
                versions.append((v_num, item))

    versions.sort(key=lambda x: x[0])
    return versions


def get_next_version(base_dir: str) -> str:
    """Computes the next version identifier (e.g. 'v1' if none exist, 'v2' if 'v1' exists, ...)."""
    versions = get_all_versions(base_dir)
    if not versions:
        return "v1"
    latest_num = max(v[0] for v in versions)
    return f"v{latest_num + 1}"


def is_valid_model_dir(dir_path: str) -> bool:
    """Validates whether a directory contains valid config.json or model weight files."""
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        return False
    files = set(os.listdir(dir_path))
    has_config = "config.json" in files
    has_weights = any(f.endswith((".safetensors", ".bin", ".pt", ".pth")) for f in files)
    return has_config or has_weights


def get_latest_version_dir(base_dir: str) -> Optional[str]:
    """Finds the highest numbered version directory containing valid model checkpoint files."""
    base_dir = resolve_path(base_dir)
    versions = get_all_versions(base_dir)
    if not versions:
        return None

    for _, folder_name in reversed(versions):
        candidate_path = os.path.join(base_dir, folder_name)
        if is_valid_model_dir(candidate_path):
            return candidate_path

    latest_folder = os.path.join(base_dir, versions[-1][1])
    return latest_folder if os.path.exists(latest_folder) else None


def resolve_model_path(
    model_identifier: Optional[str],
    default_fallback: str,
    task: Optional[str] = None
) -> str:
    """
    Intelligently resolves a model path:
    1. If pointing to a base directory (models/xlm_roberta/embedding or sentiment):
       - Automatically detects the latest valid version (v2, v1,...).
       - If no versions exist yet: gracefully falls back to default_fallback from Hugging Face.
    2. If pointing to a specific version (e.g., models/xlm_roberta/sentiment/v1): uses that exact version.
    3. If a Hugging Face Hub model identifier (e.g., xlm-roberta-base): downloads directly from HF.
    """
    if not model_identifier:
        return default_fallback

    norm_id = model_identifier.replace("\\", "/").rstrip("/")
    resolved_id = resolve_path(norm_id)

    if os.path.exists(resolved_id) and os.path.isdir(resolved_id):
        # Case 1: Directory itself contains model checkpoint
        if is_valid_model_dir(resolved_id):
            return norm_id

        # Case 2: Base directory containing versioned subfolders (v1, v2,...)
        latest_dir = get_latest_version_dir(resolved_id)
        if latest_dir and is_valid_model_dir(latest_dir):
            rel_path = os.path.relpath(latest_dir, os.getcwd()).replace("\\", "/")
            logger.info(f"[{task or 'Model'}] Automatically resolved to latest version: '{rel_path}'")
            return rel_path

        # Case 3: Empty directory (or only contains .gitkeep) -> Safe fallback
        logger.warning(
            f"[{task or 'Model'}] Directory '{norm_id}' has no saved checkpoints yet (v1, v2,...). "
            f"Falling back to default base model: '{default_fallback}'."
        )
        return default_fallback

    return model_identifier


def save_fine_tuned_checkpoint(
    model: Any,
    tokenizer: Any,
    task: str,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Saves a fine-tuned model checkpoint under the next auto-incremented version folder:
    - Embedding -> models/xlm_roberta/embedding/v{N}
    - Sentiment -> models/xlm_roberta/sentiment/v{N}
    """
    task = task.lower()
    if base_dir is None:
        if "embed" in task:
            base_dir = DEFAULT_EMBEDDING_BASE_DIR
        elif "sent" in task:
            base_dir = DEFAULT_SENTIMENT_BASE_DIR
        else:
            base_dir = f"models/xlm_roberta/{task}"

    base_dir = resolve_output_path(base_dir)
    base_dir = os.path.abspath(os.path.normpath(base_dir))
    try:
        os.makedirs(base_dir, exist_ok=True)
    except OSError:
        from pathlib import Path
        Path(base_dir).mkdir(parents=True, exist_ok=True)

    # 1. Determine next version
    next_ver = get_next_version(base_dir)
    version_dir = os.path.abspath(os.path.normpath(os.path.join(base_dir, next_ver)))
    try:
        os.makedirs(version_dir, exist_ok=True)
    except OSError:
        from pathlib import Path
        Path(version_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving fine-tuned {task} model to new version: {version_dir}")

    # 2. Save weights and tokenizer
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(version_dir)
    else:
        import torch
        torch.save(model.state_dict(), os.path.join(version_dir, "pytorch_model.bin"))

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(version_dir)

    # 3. Write version_info.json
    now_str = datetime.now().isoformat()
    version_info = {
        "version": next_ver,
        "task": task,
        "saved_at": now_str,
        "saved_path": version_dir.replace("\\", "/"),
        "metadata": metadata or {}
    }
    with open(os.path.join(version_dir, "version_info.json"), "w", encoding="utf-8") as f:
        json.dump(version_info, f, indent=2, ensure_ascii=False)

    # 4. Update latest_version.json pointer
    latest_pointer = {
        "latest_version": next_ver,
        "latest_dir": version_dir.replace("\\", "/"),
        "updated_at": now_str,
        "task": task
    }
    with open(os.path.join(base_dir, "latest_version.json"), "w", encoding="utf-8") as f:
        json.dump(latest_pointer, f, indent=2, ensure_ascii=False)

    rel_path = os.path.relpath(version_dir, os.getcwd()).replace("\\", "/")
    logger.info(f"==> Successfully saved version [{next_ver}] to: '{rel_path}'")
    return rel_path


def list_available_versions(base_dir: str) -> List[Dict[str, Any]]:
    """Lists all saved versions along with their metadata."""
    base_dir = resolve_path(base_dir)
    versions = get_all_versions(base_dir)
    summary = []
    for v_num, v_name in versions:
        v_path = os.path.join(base_dir, v_name)
        info_file = os.path.join(v_path, "version_info.json")
        info = {"version": v_name, "path": v_path.replace("\\", "/"), "is_valid": is_valid_model_dir(v_path)}
        if os.path.exists(info_file):
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    info.update(meta)
            except Exception:
                pass
        summary.append(info)
    return summary
