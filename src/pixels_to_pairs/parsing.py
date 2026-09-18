"""Conservative parsing utilities for LLM-generated KVP outputs."""

import json
from typing import Dict, List


def parse_kvp_output(raw_output: str) -> List[Dict[str, str]]:
    """
    Parse LLM output conservatively while preserving complete KVP objects.

    1. Prefer a complete valid JSON response:
       {"kvp": [{"key": "...", "value": "..."}, ...]}
    2. If the full response is malformed/truncated, recover only individually
       COMPLETE, valid JSON objects already present in the generated text
       that contain string-valued "key" and "value" fields.

    No regex KVP extraction, malformed-JSON repair, guessing, fuzzy parsing,
    GT-aware recovery, or document-text fallback is performed.
    """
    if not isinstance(raw_output, str):
        return []

    s = raw_output.strip()
    if not s:
        return []

    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        obj = None

    if isinstance(obj, dict):
        kvp_list = obj.get("kvp")
        if isinstance(kvp_list, list):
            parsed = []
            for pair in kvp_list:
                if not isinstance(pair, dict):
                    continue
                key = pair.get("key")
                value = pair.get("value")
                if not isinstance(key, str) or not isinstance(value, str):
                    continue
                key = key.strip()
                value = value.strip()
                if key == "" and value == "":
                    continue
                parsed.append({"key": key, "value": value})
            return parsed

    decoder = json.JSONDecoder()
    recovered = []

    for start, char in enumerate(s):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(s[start:])
        except json.JSONDecodeError:
            continue

        if not isinstance(candidate, dict):
            continue

        key = candidate.get("key")
        value = candidate.get("value")
        if not isinstance(key, str) or not isinstance(value, str):
            continue

        key = key.strip()
        value = value.strip()
        if key == "" and value == "":
            continue

        recovered.append({"key": key, "value": value})

    return recovered
