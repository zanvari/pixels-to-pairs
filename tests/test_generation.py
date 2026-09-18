"""Regression tests for shared model generation utilities."""

from unittest.mock import Mock, patch

import torch

from pixels_to_pairs.inference.generation import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    effective_max_input_tokens,
    maybe_apply_chat_template,
    postprocess_decoded_text,
    preferred_dtype,
    qwen_im_end_eos_id,
)


def test_benchmark_generation_defaults():
    assert DEFAULT_MAX_INPUT_TOKENS == 8192
    assert DEFAULT_MAX_NEW_TOKENS == 1024


def test_preferred_dtype_cpu():
    assert preferred_dtype("cpu") == torch.float32


def test_preferred_dtype_cuda_bfloat16():
    with patch(
        "torch.cuda.is_bf16_supported",
        return_value=True,
    ):
        assert preferred_dtype("cuda") == torch.bfloat16


def test_preferred_dtype_cuda_float16():
    with patch(
        "torch.cuda.is_bf16_supported",
        return_value=False,
    ):
        assert preferred_dtype("cuda") == torch.float16


def test_effective_max_input_tokens_uses_benchmark_limit():
    tokenizer = Mock()
    tokenizer.model_max_length = 32768

    assert effective_max_input_tokens(tokenizer) == 8192


def test_effective_max_input_tokens_respects_smaller_model_limit():
    tokenizer = Mock()
    tokenizer.model_max_length = 4096

    assert effective_max_input_tokens(tokenizer) == 4096


def test_effective_max_input_tokens_handles_missing_limit():
    tokenizer = Mock()
    tokenizer.model_max_length = None

    assert effective_max_input_tokens(tokenizer) == 8192


def test_effective_max_input_tokens_handles_sentinel_limit():
    tokenizer = Mock()
    tokenizer.model_max_length = 10**30

    assert effective_max_input_tokens(tokenizer) == 8192


def test_effective_max_input_tokens_accepts_custom_limit():
    tokenizer = Mock()
    tokenizer.model_max_length = 32768

    assert (
        effective_max_input_tokens(
            tokenizer,
            max_input_tokens=2048,
        )
        == 2048
    )


def test_chat_template_is_applied_when_available():
    tokenizer = Mock()
    tokenizer.chat_template = "template"

    tokenizer.apply_chat_template.return_value = (
        "formatted prompt"
    )

    result = maybe_apply_chat_template(
        tokenizer,
        "document text",
        "system instruction",
    )

    assert result == "formatted prompt"

    tokenizer.apply_chat_template.assert_called_once_with(
        [
            {
                "role": "system",
                "content": "system instruction",
            },
            {
                "role": "user",
                "content": "document text",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def test_chat_template_falls_back_when_missing():
    tokenizer = Mock()
    tokenizer.chat_template = None

    result = maybe_apply_chat_template(
        tokenizer,
        "document text",
        "system instruction",
    )

    assert result == "document text"


def test_chat_template_falls_back_on_error():
    tokenizer = Mock()
    tokenizer.chat_template = "template"

    tokenizer.apply_chat_template.side_effect = RuntimeError(
        "template failure"
    )

    result = maybe_apply_chat_template(
        tokenizer,
        "document text",
        "system instruction",
    )

    assert result == "document text"


def test_qwen_im_end_eos_id_returns_valid_token():
    tokenizer = Mock()
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 151645

    result = qwen_im_end_eos_id(tokenizer)

    assert result == 151645

    tokenizer.convert_tokens_to_ids.assert_called_once_with(
        "<|im_end|>"
    )


def test_qwen_im_end_eos_id_rejects_unknown_token():
    tokenizer = Mock()
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 0

    assert qwen_im_end_eos_id(tokenizer) is None


def test_qwen_im_end_eos_id_handles_lookup_failure():
    tokenizer = Mock()
    tokenizer.convert_tokens_to_ids.side_effect = RuntimeError(
        "lookup failed"
    )

    assert qwen_im_end_eos_id(tokenizer) is None


def test_postprocess_decoded_text_strips_whitespace():
    assert (
        postprocess_decoded_text(
            '  {"kvp": []}  '
        )
        == '{"kvp": []}'
    )


def test_postprocess_decoded_text_removes_qwen_marker():
    assert (
        postprocess_decoded_text(
            '{"kvp": []}<|im_end|>'
        )
        == '{"kvp": []}'
    )


def test_postprocess_decoded_text_removes_embedded_markers():
    assert (
        postprocess_decoded_text(
            '  abc<|im_end|>def<|im_end|>  '
        )
        == "abcdef"
    )
