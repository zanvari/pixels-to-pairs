"""SROIE prompt construction used by the benchmark."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


SYSTEM_MESSAGE = (
    "You extract key–value pairs from OCR text of receipts and invoices. "
    "Return ONLY a JSON object with the exact schema "
    '{"kvp": [{"key":..., "value":...}, ...]}. '
    "Do not output any extra text."
)


USER_INSTRUCTIONS_BASE = """
Task: From the OCR text of a single receipt or invoice, extract the following fields if present:

- company  : the merchant/store name
- address  : the full business address
- date     : the transaction date
- total    : the final total amount paid (prefer the grand total / inclusive-of-tax amount)

Output rules (strict):
- Output ONLY valid JSON (no markdown, no code fences, no explanations).
- JSON must be exactly: {"kvp": [{"key": "...", "value": "..."}, ...]}
- Keys should be lower-case strings from this set whenever possible:
    ["company", "address", "date", "total"]
- Values should be the best string span for that field (do NOT invent values).
- If a field is missing, simply omit it from "kvp".
- If nothing is extractable: {"kvp": []}
""".strip()


USER_INSTRUCTIONS_FEWSHOT = (
    USER_INSTRUCTIONS_BASE
    + """

CRITICAL anti-anchoring rules:
- The examples are ONLY to demonstrate the output format.
- NEVER copy example values into the new output.
- For the new receipt, extract fields ONLY from the text inside <NEW_DOCUMENT>.
- Ignore all text inside any <EXAMPLE_*> blocks.
"""
).strip()


SROIE_EXAMPLE_IDS = (
    "X51008142065",
    "X51008164525",
    "X51008030566",
)


def _load_prompt_example(
    doc_id: str,
    txt_path: Path,
    kvp_path: Path,
) -> Dict[str, Any]:
    """
    Load one frozen SROIE few-shot example from disk.

    The document text is stored in <doc_id>.txt.
    The KVP annotation is stored in <doc_id>_kvp.json and must contain
    valid JSON in one of these forms:

      1) {"kvp": [{"key": "...", "value": "..."}, ...]}
      2) [{"key": "...", "value": "..."}, ...]

    These examples must come from the SROIE training split.
    """
    if not txt_path.is_file():
        raise FileNotFoundError(
            f"Missing SROIE prompt-example text file for "
            f"{doc_id}: {txt_path}"
        )

    if not kvp_path.is_file():
        raise FileNotFoundError(
            f"Missing SROIE prompt-example KVP file for "
            f"{doc_id}: {kvp_path}"
        )

    doc_text = txt_path.read_text(encoding="utf-8").strip()
    kvp_text = kvp_path.read_text(encoding="utf-8").strip()

    if not doc_text:
        raise ValueError(
            f"Prompt-example document text is empty for "
            f"{doc_id}: {txt_path}"
        )

    if not kvp_text:
        raise ValueError(
            f"Prompt-example KVP file is empty for "
            f"{doc_id}: {kvp_path}"
        )

    try:
        data = json.loads(kvp_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Prompt-example KVP file is not valid JSON for "
            f"{doc_id}: {kvp_path}"
        ) from exc

    if isinstance(data, dict) and "kvp" in data:
        kvp_raw = data["kvp"]
    elif isinstance(data, list):
        kvp_raw = data
    else:
        raise ValueError(
            f"Unsupported prompt-example KVP format for "
            f"{doc_id}: {kvp_path}"
        )

    if not isinstance(kvp_raw, list):
        raise ValueError(
            f"Prompt-example KVP content must be a list for "
            f"{doc_id}: {kvp_path}"
        )

    kvp_clean: List[Dict[str, str]] = []

    for i, pair in enumerate(kvp_raw):
        if not isinstance(pair, dict):
            raise ValueError(
                f"Prompt example {doc_id}, KVP item {i} "
                "is not a JSON object."
            )

        key = pair.get("key")
        value = pair.get("value")

        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                f"Prompt example {doc_id}, KVP item {i} must contain "
                "string-valued 'key' and 'value' fields."
            )

        key = key.strip()
        value = value.strip()

        if not key or not value:
            raise ValueError(
                f"Prompt example {doc_id}, KVP item {i} "
                "has an empty key/value."
            )

        kvp_clean.append(
            {
                "key": key,
                "value": value,
            }
        )

    if not kvp_clean:
        raise ValueError(
            f"Prompt-example KVP list is empty for "
            f"{doc_id}: {kvp_path}"
        )

    return {
        "doc_id": doc_id,
        "doc_text": doc_text,
        "kvp": kvp_clean,
    }


def load_sroie_examples(
    example_files_dir: Path,
) -> List[Dict[str, Any]]:
    """Load the three frozen SROIE training demonstrations."""

    example_files_dir = Path(example_files_dir)

    examples = [
        _load_prompt_example(
            doc_id,
            example_files_dir / f"{doc_id}.txt",
            example_files_dir / f"{doc_id}_kvp.json",
        )
        for doc_id in SROIE_EXAMPLE_IDS
    ]

    loaded_example_ids = [
        example["doc_id"]
        for example in examples
    ]

    assert loaded_example_ids == list(SROIE_EXAMPLE_IDS), (
        f"Unexpected SROIE few-shot example order: "
        f"{loaded_example_ids}"
    )

    return examples


def _example_block(
    idx: int,
    doc_text: str,
    kvp_list: List[Dict[str, str]],
) -> str:
    json_obj = {"kvp": kvp_list}
    json_str = json.dumps(
        json_obj,
        ensure_ascii=False,
        indent=2,
    )

    return (
        f"<EXAMPLE_{idx}_DOCUMENT>\n"
        f"{doc_text}\n"
        f"</EXAMPLE_{idx}_DOCUMENT>\n"
        f"<EXAMPLE_{idx}_OUTPUT>\n"
        f"{json_str}\n"
        f"</EXAMPLE_{idx}_OUTPUT>\n"
    )


def _task_footer(doc_text: str) -> str:
    return (
        "Now do the task for the NEW document below.\n"
        "Extract fields ONLY from <NEW_DOCUMENT>. "
        "Ignore all <EXAMPLE_*> blocks completely.\n"
        "Return ONLY a SINGLE JSON object and stop immediately "
        "after the final '}'.\n"
        "<NEW_DOCUMENT>\n"
        f"{doc_text}\n"
        "</NEW_DOCUMENT>\n"
        "Output:\n"
    )


def build_prompt_0shot(doc_text: str) -> str:
    prompt = (
        USER_INSTRUCTIONS_BASE
        + "\n\n"
        + _task_footer(doc_text)
    )

    assert "<EXAMPLE_1_DOCUMENT>" not in prompt, (
        "0-shot prompt accidentally contains example blocks."
    )

    return prompt


def build_prompt_1shot(
    doc_text: str,
    examples: List[Dict[str, Any]],
) -> str:
    ex1 = examples[0]

    return "\n\n".join(
        [
            USER_INSTRUCTIONS_FEWSHOT,
            _example_block(
                1,
                ex1["doc_text"],
                ex1["kvp"],
            ),
            _task_footer(doc_text),
        ]
    )


def build_prompt_2shot(
    doc_text: str,
    examples: List[Dict[str, Any]],
) -> str:
    ex1 = examples[0]
    ex2 = examples[1]

    return "\n\n".join(
        [
            USER_INSTRUCTIONS_FEWSHOT,
            _example_block(
                1,
                ex1["doc_text"],
                ex1["kvp"],
            ),
            _example_block(
                2,
                ex2["doc_text"],
                ex2["kvp"],
            ),
            _task_footer(doc_text),
        ]
    )


def build_prompt_3shot(
    doc_text: str,
    examples: List[Dict[str, Any]],
) -> str:
    ex1 = examples[0]
    ex2 = examples[1]
    ex3 = examples[2]

    return "\n\n".join(
        [
            USER_INSTRUCTIONS_FEWSHOT,
            _example_block(
                1,
                ex1["doc_text"],
                ex1["kvp"],
            ),
            _example_block(
                2,
                ex2["doc_text"],
                ex2["kvp"],
            ),
            _example_block(
                3,
                ex3["doc_text"],
                ex3["kvp"],
            ),
            _task_footer(doc_text),
        ]
    )


def _ensure_examples(
    examples: Optional[List[Dict[str, Any]]],
    example_files_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    if examples is not None:
        return examples

    if example_files_dir is None:
        raise ValueError(
            "example_files_dir is required for SROIE "
            "few-shot prompts."
        )

    return load_sroie_examples(Path(example_files_dir))


def build_prompt(
    doc_text: str,
    shot_variant: str,
    *,
    examples: Optional[List[Dict[str, Any]]] = None,
    example_files_dir: Optional[Path] = None,
) -> str:
    """Build a SROIE prompt for the requested shot condition."""

    if shot_variant == "0shot":
        prompt = build_prompt_0shot(doc_text)
        expected_shots = 0
    else:
        examples = _ensure_examples(
            examples,
            example_files_dir,
        )

        if shot_variant == "1shot":
            prompt = build_prompt_1shot(
                doc_text,
                examples,
            )
            expected_shots = 1
        elif shot_variant == "2shot":
            prompt = build_prompt_2shot(
                doc_text,
                examples,
            )
            expected_shots = 2
        elif shot_variant == "3shot":
            prompt = build_prompt_3shot(
                doc_text,
                examples,
            )
            expected_shots = 3
        else:
            raise ValueError(
                f"Unknown shot_variant: {shot_variant}"
            )

    # Preserve the notebook sanity guard: exactly the intended
    # cumulative number of demonstration blocks must be present.
    for idx in range(1, 4):
        tag = f"<EXAMPLE_{idx}_DOCUMENT>"

        if idx <= expected_shots:
            assert tag in prompt, (
                f"{shot_variant} prompt is missing "
                f"expected example block {idx}."
            )
        else:
            assert tag not in prompt, (
                f"{shot_variant} prompt unexpectedly "
                f"contains example block {idx}."
            )

    return prompt
