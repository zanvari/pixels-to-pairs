"""Regression tests for conservative KVP output parsing."""

from pixels_to_pairs.parsing import parse_kvp_output


def test_parse_complete_valid_json():
    raw = """
    {"kvp": [
        {"key": "Company", "value": "ABC Store"},
        {"key": "Total", "value": "$25.00"}
    ]}
    """

    assert parse_kvp_output(raw) == [
        {"key": "Company", "value": "ABC Store"},
        {"key": "Total", "value": "$25.00"},
    ]


def test_parse_json_code_fence():
    raw = """```json
{"kvp": [{"key": "Date", "value": "01/15/2025"}]}
```"""

    assert parse_kvp_output(raw) == [
        {"key": "Date", "value": "01/15/2025"},
    ]


def test_recover_complete_objects_from_truncated_output():
    raw = """
    {"kvp": [
        {"key": "Company", "value": "ABC Store"},
        {"key": "Total", "value": "$25.00"},
        {"key": "Address", "value": "123 Main
    """

    assert parse_kvp_output(raw) == [
        {"key": "Company", "value": "ABC Store"},
        {"key": "Total", "value": "$25.00"},
    ]


def test_reject_non_string_key_or_value():
    raw = """
    {"kvp": [
        {"key": "Total", "value": 25.00},
        {"key": "Date"},
        {"key": "Company", "value": "ABC Store"}
    ]}
    """

    assert parse_kvp_output(raw) == [
        {"key": "Company", "value": "ABC Store"},
    ]


def test_strip_key_and_value_whitespace():
    raw = """
    {"kvp": [
        {"key": "  Company  ", "value": "  ABC Store  "}
    ]}
    """

    assert parse_kvp_output(raw) == [
        {"key": "Company", "value": "ABC Store"},
    ]


def test_skip_pair_when_key_and_value_are_both_empty():
    raw = """
    {"kvp": [
        {"key": "", "value": ""},
        {"key": "Company", "value": "ABC Store"}
    ]}
    """

    assert parse_kvp_output(raw) == [
        {"key": "Company", "value": "ABC Store"},
    ]


def test_empty_and_non_string_inputs():
    assert parse_kvp_output("") == []
    assert parse_kvp_output("   ") == []
    assert parse_kvp_output(None) == []
    assert parse_kvp_output(123) == []


def test_valid_json_without_kvp_list_returns_no_pairs():
    assert parse_kvp_output('{"result": "nothing"}') == []


def test_parser_does_not_repair_malformed_json():
    raw = """
    {"kvp": [
        {"key": "Company", "value": "ABC Store"
    ]}
    """

    assert parse_kvp_output(raw) == []
