"""CORD data loading utilities used by the benchmark."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_head(
    path: Path,
    n: int = 200,
) -> str:
    """Read the first n characters of a text file."""

    with Path(path).open("r", encoding="utf-8") as f:
        return f.read(n)


def _coerce_kvp_list(
    data: Any,
) -> Optional[List[Dict[str, str]]]:
    """Coerce supported CORD annotation formats to KVP dictionaries."""

    if (
        isinstance(data, dict)
        and "kvp" in data
        and isinstance(data["kvp"], list)
    ):
        kvps = data["kvp"]

    elif isinstance(data, list):
        kvps = data

    else:
        return None

    clean: List[Dict[str, str]] = []

    for item in kvps:
        if not isinstance(item, dict):
            continue

        key = item.get(
            "key",
            item.get(
                "Key",
                item.get("k", ""),
            ),
        )

        value = item.get(
            "value",
            item.get(
                "Value",
                item.get("v", ""),
            ),
        )

        key = str(key).strip()
        value = str(value).strip()

        if key == "" and value == "":
            continue

        clean.append(
            {
                "key": key,
                "value": value,
            }
        )

    return clean


def load_cord_gt_kvp(
    kvp_dir: Path,
    sanity_samples: int = 3,
) -> Dict[str, List[Dict[str, str]]]:
    """Load CORD document-level KVP ground truth."""

    kvp_dir = Path(kvp_dir)

    gt: Dict[str, List[Dict[str, str]]] = {}

    files = sorted(
        path
        for path in kvp_dir.iterdir()
        if path.suffix == ".json"
    )

    print(
        f"[SANITY] KVP_DIR: {kvp_dir}  "
        f"(files: {len(files)})"
    )

    bad = 0
    shown = 0

    for path in files:
        doc_id = path.stem

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            bad += 1
            continue

        kvps = _coerce_kvp_list(data)

        if kvps is None:
            bad += 1
            continue

        if shown < sanity_samples:
            shown += 1

            raw_head = _read_head(
                path,
                200,
            ).replace("\n", "\\n")

            print(
                f"\n[SANITY] GT SAMPLE FILE: "
                f"{path.name}"
            )
            print(
                "[SANITY] RAW_HEAD:",
                raw_head,
            )
            print(
                "[SANITY] PARSED_KVP_LEN:",
                len(kvps),
            )

        gt[doc_id] = kvps

    print(
        f"Loaded KVP ground truth for "
        f"{len(gt)} CORD documents. "
        f"Bad files skipped: {bad}"
    )

    if len(gt) == 0:
        raise RuntimeError(
            "Loaded 0 GT documents from KVP_DIR.\n"
            "This usually means you're pointing to "
            "the wrong folder.\n"
            f"KVP_DIR: {kvp_dir}"
        )

    return gt


def load_doc_texts(
    text_dir: Path,
) -> Dict[str, str]:
    """Load CORD text files keyed by document ID."""

    text_dir = Path(text_dir)

    texts: Dict[str, str] = {}

    files = sorted(
        path
        for path in text_dir.iterdir()
        if path.suffix == ".txt"
    )

    print(
        f"[SANITY] TEXT_DIR: {text_dir}  "
        f"(files: {len(files)})"
    )

    for path in files:
        doc_id = path.stem
        doc_text = path.read_text(encoding="utf-8")
        texts[doc_id] = doc_text

    return texts
