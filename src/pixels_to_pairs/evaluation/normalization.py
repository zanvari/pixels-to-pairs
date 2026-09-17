"""Normalization utilities used by benchmark evaluation."""

import unicodedata


def normalize_text(text):
    """Normalize text for benchmark value comparison.

    The normalization lowercases text, strips leading and trailing
    whitespace, and collapses consecutive whitespace characters.

    Args:
        text: Text value to normalize.

    Returns:
        Normalized string.
    """
    text = str(text).strip().lower()
    return " ".join(text.split())


def normalize_key(s: str) -> str:
    """
    Normalize field labels for evaluation only.

    This is intentionally formatting-tolerant but not semantic/fuzzy:
    - Unicode NFKC normalization
    - case-insensitive matching via casefold()
    - trim surrounding whitespace
    - ignore common label punctuation/decoration
    - treat slash, backslash, hyphen, underscore, and vertical bar as word separators
    - collapse repeated whitespace

    Examples that become equivalent:
      "DATE:" == "date"
      "• CASE NAME :" == "Case Name"
      "NO. OF STORES" == "no of stores"
      "P.O.S." == "POS"
      "SENDER /PHONE NUMBER:" == "sender/phone number"

    No synonym expansion, fuzzy matching, stemming, or answer repair is used.
    """
    s = unicodedata.normalize("NFKC", str(s)).casefold().strip()
    if not s:
        return ""

    # Drop purely decorative leading/trailing characters.
    while s and not s[0].isalnum():
        s = s[1:]
    while s and not s[-1].isalnum():
        s = s[:-1]

    if not s:
        return ""

    out = []
    separators = {"/", "\\", "-", "_", "|"}
    removable_punct = {
        ".", ",", ":", ";", "'", '"', "`",
        "(", ")", "[", "]", "{", "}", "<", ">",
        "!", "?", "*", "#", "~", "^"
    }

    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch.isspace() or ch in separators:
            out.append(" ")
        elif ch in removable_punct:
            # Formatting punctuation inside labels is ignored.
            continue
        else:
            # Preserve non-formatting symbols (e.g., &, +, $) as tokens
            # so genuinely different labels are not silently merged.
            out.append(f" {ch} ")

    return " ".join("".join(out).split())
