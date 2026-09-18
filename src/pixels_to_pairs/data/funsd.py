"""FUNSD data loading utilities used by the benchmark."""

import json
from pathlib import Path
from typing import Any, Dict, List


def kvp_from_funsd_form(
    data: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Extract linked question-answer KVPs from FUNSD annotations."""

    form = data.get("form", [])
    id2ent = {
        ent["id"]: ent
        for ent in form
    }

    kvps = []

    for ent in form:
        if ent.get("label", "").lower() != "question":
            continue

        qid = ent["id"]
        key_text = ent.get("text", "").strip()

        if not key_text:
            continue

        for link in ent.get("linking", []):
            if not isinstance(link, list) or len(link) != 2:
                continue

            q_link, a_link = link

            if q_link != qid:
                continue

            ans_ent = id2ent.get(a_link)

            if ans_ent is None:
                continue

            val_text = ans_ent.get("text", "").strip()

            if not val_text:
                continue

            kvps.append(
                {
                    "key": key_text,
                    "value": val_text,
                }
            )

    return kvps


def load_funsd_gt_kvp(
    annotation_dir: Path,
) -> Dict[str, List[Dict[str, str]]]:
    """Load FUNSD document-level KVP ground truth."""

    annotation_dir = Path(annotation_dir)

    gt: Dict[str, List[Dict[str, str]]] = {}

    files = sorted(
        path
        for path in annotation_dir.iterdir()
        if path.suffix == ".json"
    )

    print(
        f"Found {len(files)} annotation files "
        f"in {annotation_dir}"
    )

    for path in files:
        doc_id = path.stem

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if (
            isinstance(data, list)
            and data
            and isinstance(data[0], dict)
            and "key" in data[0]
        ):
            kvps = [
                {
                    "key": str(item["key"]),
                    "value": str(item.get("value", "")),
                }
                for item in data
            ]

        elif isinstance(data, dict) and "form" in data:
            kvps = kvp_from_funsd_form(data)

        else:
            print(
                f"[WARN] Unknown annotation format for "
                f"{path.name}, skipping."
            )
            continue

        gt[doc_id] = kvps

    print(
        f"Loaded KVP ground truth for "
        f"{len(gt)} documents."
    )

    return gt


def load_doc_texts(
    text_dir: Path,
) -> Dict[str, str]:
    """Load document text files keyed by document ID."""

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
