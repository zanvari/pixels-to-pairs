"""Text normalization utilities used by benchmark evaluation."""


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
