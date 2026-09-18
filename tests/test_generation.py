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
    prompt_token_stats,
    cleanup_cuda,
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

class FakeTokenBatch:
    def __init__(self, length):
        self.shape = (1, length)


class FakePromptTokenizer:
    def __init__(
        self,
        raw_length,
        templated_length,
        model_max_length=32768,
    ):
        self.raw_length = raw_length
        self.templated_length = templated_length
        self.model_max_length = model_max_length
        self.chat_template = "template"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ):
        return "CHAT_TEMPLATE_APPLIED"

    def __call__(
        self,
        text,
        return_tensors,
        truncation,
        max_length=None,
    ):
        if text == "CHAT_TEMPLATE_APPLIED":
            length = self.templated_length
        else:
            length = self.raw_length

        if truncation and max_length is not None:
            length = min(length, max_length)

        return {
            "input_ids": FakeTokenBatch(length),
        }


def test_prompt_token_stats_uses_chat_template_for_causal_model():
    tokenizer = FakePromptTokenizer(
        raw_length=100,
        templated_length=120,
    )

    result = prompt_token_stats(
        tokenizer,
        "raw prompt",
        is_encoder_decoder=False,
        system_message="system",
    )

    assert result == (
        120,
        120,
        False,
    )


def test_prompt_token_stats_uses_raw_prompt_for_encoder_decoder():
    tokenizer = FakePromptTokenizer(
        raw_length=100,
        templated_length=120,
    )

    result = prompt_token_stats(
        tokenizer,
        "raw prompt",
        is_encoder_decoder=True,
        system_message="system",
    )

    assert result == (
        100,
        100,
        False,
    )


def test_prompt_token_stats_reports_truncation():
    tokenizer = FakePromptTokenizer(
        raw_length=100,
        templated_length=9000,
    )

    result = prompt_token_stats(
        tokenizer,
        "raw prompt",
        is_encoder_decoder=False,
        system_message="system",
    )

    assert result == (
        9000,
        8192,
        True,
    )


def test_prompt_token_stats_respects_smaller_model_limit():
    tokenizer = FakePromptTokenizer(
        raw_length=100,
        templated_length=5000,
        model_max_length=4096,
    )

    result = prompt_token_stats(
        tokenizer,
        "raw prompt",
        is_encoder_decoder=False,
        system_message="system",
    )

    assert result == (
        5000,
        4096,
        True,
    )
def test_cleanup_cuda_without_cuda():
    with (
        patch("pixels_to_pairs.inference.generation.gc.collect") as collect,
        patch(
            "pixels_to_pairs.inference.generation.torch.cuda.is_available",
            return_value=False,
        ),
        patch(
            "pixels_to_pairs.inference.generation.torch.cuda.empty_cache"
        ) as empty_cache,
        patch(
            "pixels_to_pairs.inference.generation.torch.cuda.ipc_collect"
        ) as ipc_collect,
    ):
        cleanup_cuda()

    collect.assert_called_once_with()
    empty_cache.assert_not_called()
    ipc_collect.assert_not_called()


def test_cleanup_cuda_with_cuda():
    with (
        patch("pixels_to_pairs.inference.generation.gc.collect") as collect,
        patch(
            "pixels_to_pairs.inference.generation.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "pixels_to_pairs.inference.generation.torch.cuda.empty_cache"
        ) as empty_cache,
        patch(
            "pixels_to_pairs.inference.generation.torch.cuda.ipc_collect"
        ) as ipc_collect,
    ):
        cleanup_cuda()

    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()
    ipc_collect.assert_called_once_with()

