"""SROIE data loading utilities used by the benchmark."""

import json
from pathlib import Path
from typing import Dict, List


def load_sroie_gt_kvp(
    kvp_dir: Path,
) -> Dict[str, List[Dict[str, str]]]:
    """Load SROIE document-level KVP ground truth.

    Each JSON file may contain either:
      1. {"kvp": [{"key": "...", "value": "..."}, ...]}
      2. [{"key": "...", "value": "..."}, ...]
    """

    kvp_dir = Path(kvp_dir)

    gt: Dict[str, List[Dict[str, str]]] = {}

    files = sorted(
        path
        for path in kvp_dir.iterdir()
        if path.suffix == ".json"
    )

    print(
        f"Found {len(files)} annotation files "
        f"in {kvp_dir}"
    )

    for path in files:
        doc_id = path.stem

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "kvp" in data:
            kvps = data["kvp"]
        elif isinstance(data, list):
            kvps = data
        else:
            print(
                f"[WARN] Unknown SROIE annotation format "
                f"for {path.name}, skipping."
            )
            continue

        clean_list: List[Dict[str, str]] = []

        for item in kvps:
            if not isinstance(item, dict):
                continue

            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()

            if key == "" and value == "":
                continue

            clean_list.append(
                {
                    "key": key,
                    "value": value,
                }
            )

        gt[doc_id] = clean_list

    print(
        f"Loaded KVP ground truth for "
        f"{len(gt)} SROIE documents."
    )

    return gt


def load_doc_texts(
    text_dir: Path,
) -> Dict[str, str]:
    """Load SROIE text files keyed by document ID."""

    text_dir = Path(text_dir)

    texts: Dict[str, str] = {}

    files = sorted(
        path
        for path in text_dir.iterdir()
        if path.suffix == ".txt"
    )

    print(
        f"Found {len(files)} text files "
        f"in {text_dir}"
    )

    for path in files:
        doc_id = path.stem
        doc_text = path.read_text(encoding="utf-8")
        texts[doc_id] = doc_text

    return texts
