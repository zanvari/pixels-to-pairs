"""FUNSD prompt construction used by the benchmark."""

import json
from pathlib import Path
from typing import Dict, Optional


SYSTEM_MESSAGE = (
    "You extract key–value pairs (KVP) from OCR text of business documents. "
    'Return ONLY a JSON object with the exact schema {"kvp": [{"key":..., "value":...}, ...]}. '
    "Do not output any extra text."
)


USER_INSTRUCTIONS_BASE = """
Task: Extract ALL key–value pairs from the given document OCR text.

Definition:
- key = field label as written (keep exact casing/punctuation; do NOT rename or normalize).
- value = the associated content following that key (same line or subsequent lines).

Output rules (strict):
- Output ONLY valid JSON (no markdown/code fences, no commentary).
- JSON must be exactly: {"kvp": [{"key": "...", "value": "..."}, ...]}
- Include every reasonable key–value pair you can extract.
- If nothing is extractable: {"kvp": []}
""".strip()


USER_INSTRUCTIONS_FEWSHOT = (
    USER_INSTRUCTIONS_BASE
    + """

CRITICAL anti-anchoring rules:
- Examples are ONLY to demonstrate the output format.
- NEVER copy example keys or values.
- Extract ONLY from <NEW_DOCUMENT>. Ignore all text inside any <EXAMPLE_*> blocks.
"""
).strip()


FUNSD_EXAMPLE_FILES = (
    ("91391310.txt", "91391310_kvp.json"),
    (
        "88547278_88547279.txt",
        # Historical filename; audited contents match annotation
        # 88547278_88547279 (16 pairs).
        "89368010_kvp.json",
    ),
    ("0060165115.txt", "0060165115_kvp.json"),
)


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing example file: {path}")

    return path.read_text(encoding="utf-8").strip()


def _read_kvp_json_as_prompt_obj(path: Path) -> str:
    raw = _read_text(path)
    data = json.loads(raw)

    if isinstance(data, dict) and "kvp" in data:
        obj = data
    elif isinstance(data, list):
        obj = {"kvp": data}
    else:
        raise ValueError(
            f"Unexpected example KVP format in {path}: "
            "expected list or {'kvp':[...]}"
        )

    return json.dumps(obj, ensure_ascii=False, indent=2)


def load_funsd_examples(
    example_files_dir: Path,
) -> Dict[str, Dict[str, str]]:
    """Load the three frozen FUNSD few-shot demonstrations."""

    examples: Dict[str, Dict[str, str]] = {}

    for idx, (txt_name, json_name) in enumerate(
        FUNSD_EXAMPLE_FILES,
        start=1,
    ):
        examples[f"ex{idx}"] = {
            "doc": _read_text(example_files_dir / txt_name),
            "json": _read_kvp_json_as_prompt_obj(
                example_files_dir / json_name
            ),
        }

    return examples


def _example_block(
    idx: int,
    doc_text: str,
    out_json: str,
) -> str:
    return (
        f"<EXAMPLE_{idx}_DOCUMENT>\n"
        f"{doc_text}\n"
        f"</EXAMPLE_{idx}_DOCUMENT>\n"
        f"<EXAMPLE_{idx}_OUTPUT>\n"
        f"{out_json}\n"
        f"</EXAMPLE_{idx}_OUTPUT>\n"
    )


def _task_footer(doc_text: str) -> str:
    return (
        "Now do the task for the NEW document below.\n"
        "Extract ONLY from <NEW_DOCUMENT>. "
        "Ignore all <EXAMPLE_*> blocks completely.\n"
        "Return ONLY a SINGLE JSON object and stop immediately "
        "after the final '}' character.\n"
        "<NEW_DOCUMENT>\n"
        f"{doc_text}\n"
        "</NEW_DOCUMENT>\n"
        "Output:\n"
    )


def build_prompt_0shot(doc_text: str) -> str:
    prompt = USER_INSTRUCTIONS_BASE + "\n\n" + _task_footer(doc_text)

    assert "<EXAMPLE_1_DOCUMENT>" not in prompt, (
        "0-shot prompt contains example blocks unexpectedly."
    )

    return prompt


def _ensure_examples(
    examples: Optional[Dict[str, Dict[str, str]]],
    example_files_dir: Optional[Path],
) -> Dict[str, Dict[str, str]]:
    if examples is not None:
        return examples

    if example_files_dir is None:
        raise ValueError(
            "example_files_dir is required for FUNSD few-shot prompts."
        )

    return load_funsd_examples(Path(example_files_dir))


def build_prompt_1shot(
    doc_text: str,
    examples: Dict[str, Dict[str, str]],
) -> str:
    return "\n\n".join(
        [
            USER_INSTRUCTIONS_FEWSHOT,
            _example_block(
                1,
                examples["ex1"]["doc"],
                examples["ex1"]["json"],
            ),
            _task_footer(doc_text),
        ]
    )


def build_prompt_2shot(
    doc_text: str,
    examples: Dict[str, Dict[str, str]],
) -> str:
    return "\n\n".join(
        [
            USER_INSTRUCTIONS_FEWSHOT,
            _example_block(
                1,
                examples["ex1"]["doc"],
                examples["ex1"]["json"],
            ),
            _example_block(
                2,
                examples["ex2"]["doc"],
                examples["ex2"]["json"],
            ),
            _task_footer(doc_text),
        ]
    )


def build_prompt_3shot(
    doc_text: str,
    examples: Dict[str, Dict[str, str]],
) -> str:
    return "\n\n".join(
        [
            USER_INSTRUCTIONS_FEWSHOT,
            _example_block(
                1,
                examples["ex1"]["doc"],
                examples["ex1"]["json"],
            ),
            _example_block(
                2,
                examples["ex2"]["doc"],
                examples["ex2"]["json"],
            ),
            _example_block(
                3,
                examples["ex3"]["doc"],
                examples["ex3"]["json"],
            ),
            _task_footer(doc_text),
        ]
    )


def build_prompt(
    doc_text: str,
    shot_variant: str,
    *,
    examples: Optional[Dict[str, Dict[str, str]]] = None,
    example_files_dir: Optional[Path] = None,
) -> str:
    """Build a FUNSD prompt for the requested shot condition."""

    if shot_variant == "0shot":
        return build_prompt_0shot(doc_text)

    examples = _ensure_examples(examples, example_files_dir)

    if shot_variant == "1shot":
        return build_prompt_1shot(doc_text, examples)
    if shot_variant == "2shot":
        return build_prompt_2shot(doc_text, examples)
    if shot_variant == "3shot":
        return build_prompt_3shot(doc_text, examples)

    raise ValueError(f"Unknown shot_variant: {shot_variant}")
