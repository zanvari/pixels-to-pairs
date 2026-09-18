"""Shared model loading and generation utilities."""

from typing import List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


DEFAULT_MAX_INPUT_TOKENS = 8192
DEFAULT_MAX_NEW_TOKENS = 1024


def preferred_dtype(device: str) -> torch.dtype:
    """Return the benchmark dtype for the selected device."""

    if device == "cuda":
        return (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    return torch.float32


def build_model_and_tokenizer(
    model_name: str,
    family: str,
    device: str,
    hf_token: Optional[str] = None,
):
    """Load a benchmark model and its tokenizer."""

    print(
        f"Loading model: {model_name}  "
        f"(family={family})"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    kwargs = {}

    if hf_token:
        kwargs["token"] = hf_token

    config = AutoConfig.from_pretrained(
        model_name,
        **kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        **kwargs,
    )

    is_encoder_decoder = (
        bool(
            getattr(
                config,
                "is_encoder_decoder",
                False,
            )
        )
        or family == "seq2seq"
    )

    if not is_encoder_decoder:
        tokenizer.padding_side = "left"

    dtype = preferred_dtype(device)

    primary_device_map = (
        "cuda"
        if device == "cuda"
        else None
    )

    def _load(device_map):
        if is_encoder_decoder:
            return AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map=device_map,
                dtype=dtype,
                **kwargs,
            )

        return AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            **kwargs,
        )

    try:
        model = _load(primary_device_map)

    except Exception as exc:
        # Do not silently change device placement. A failed CUDA
        # load must be surfaced so the run cannot continue under
        # a different hardware configuration.
        raise RuntimeError(
            f"Failed to load {model_name} with "
            f"device_map={primary_device_map!r}. "
            "Check local disk space/model cache and GPU "
            "memory before retrying."
        ) from exc

    if (
        tokenizer.pad_token is None
        and tokenizer.eos_token is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(
        "hf_device_map:",
        getattr(model, "hf_device_map", None),
    )

    model.eval()

    return tokenizer, model, is_encoder_decoder


def effective_max_input_tokens(
    tokenizer,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> int:
    """Resolve the effective benchmark input-token limit."""

    model_max_length = getattr(
        tokenizer,
        "model_max_length",
        None,
    )

    if (
        model_max_length is None
        or model_max_length > 100000
    ):
        return max_input_tokens

    return int(
        min(
            max_input_tokens,
            model_max_length,
        )
    )


def maybe_apply_chat_template(
    tokenizer,
    user_prompt: str,
    system_message: str,
) -> str:
    """Apply the tokenizer chat template when available."""

    try:
        template = getattr(
            tokenizer,
            "chat_template",
            None,
        )

        if template:
            messages = [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]

            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    except Exception:
        pass

    return user_prompt


def qwen_im_end_eos_id(
    tokenizer,
) -> Optional[int]:
    """Return Qwen's im_end token ID when available."""

    try:
        im_end_id = tokenizer.convert_tokens_to_ids(
            "<|im_end|>"
        )

        if (
            im_end_id is not None
            and im_end_id != tokenizer.unk_token_id
        ):
            return int(im_end_id)

    except Exception:
        pass

    return None


def postprocess_decoded_text(
    text: str,
) -> str:
    """Remove the Qwen im_end marker from decoded output."""

    text = text.strip()
    text = text.replace(
        "<|im_end|>",
        "",
    ).strip()

    return text


def run_model_on_prompts(
    tokenizer,
    model,
    prompts: List[str],
    is_encoder_decoder: bool,
    device: str,
    system_message: str,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
):
    """Generate outputs plus neutral generation-length diagnostics."""

    if not is_encoder_decoder:
        prompts = [
            maybe_apply_chat_template(
                tokenizer,
                prompt,
                system_message,
            )
            for prompt in prompts
        ]

    max_in = effective_max_input_tokens(
        tokenizer,
        max_input_tokens,
    )

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_in,
    )

    if device == "cuda":
        inputs = {
            key: value.to("cuda")
            for key, value in inputs.items()
        }

    eos_id = (
        qwen_im_end_eos_id(tokenizer)
        or int(tokenizer.eos_token_id)
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(eos_id),
        "return_dict_in_generate": True,
        "output_scores": False,
    }

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            **gen_kwargs,
        )

    sequences = gen_out.sequences
    results = []

    if is_encoder_decoder:
        for i in range(sequences.shape[0]):
            ids = sequences[i]

            generated_tokens = int(
                ids.shape[-1]
            )

            text = tokenizer.decode(
                ids,
                skip_special_tokens=True,
            ).strip()

            results.append(
                {
                    "text": text,
                    "generated_tokens": generated_tokens,
                    "output_hit_max_new_tokens": bool(
                        generated_tokens
                        >= max_new_tokens
                    ),
                }
            )

        return results

    input_width = int(
        inputs["input_ids"].shape[1]
    )

    for i in range(sequences.shape[0]):
        ids = sequences[
            i,
            input_width:,
        ]

        generated_tokens = int(
            ids.shape[-1]
        )

        text = tokenizer.decode(
            ids,
            skip_special_tokens=False,
        )

        text = postprocess_decoded_text(
            text
        )

        results.append(
            {
                "text": text,
                "generated_tokens": generated_tokens,
                "output_hit_max_new_tokens": bool(
                    generated_tokens
                    >= max_new_tokens
                ),
            }
        )

    return results

def prompt_token_stats(
    tokenizer,
    prompt: str,
    is_encoder_decoder: bool,
    system_message: str,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
):
    """Measure prompt length using the representation used for generation.

    Causal/chat models are measured after applying the tokenizer's chat
    template. Encoder-decoder models are measured using the raw task prompt.

    This function provides diagnostics only and does not alter generation.
    """

    max_in = effective_max_input_tokens(
        tokenizer,
        max_input_tokens,
    )

    prompt_for_model = (
        prompt
        if is_encoder_decoder
        else maybe_apply_chat_template(
            tokenizer,
            prompt,
            system_message,
        )
    )

    tok_no_trunc = tokenizer(
        prompt_for_model,
        return_tensors="pt",
        truncation=False,
    )

    tok_trunc = tokenizer(
        prompt_for_model,
        return_tensors="pt",
        truncation=True,
        max_length=max_in,
    )

    n_no_trunc = int(
        tok_no_trunc["input_ids"].shape[-1]
    )

    n_trunc = int(
        tok_trunc["input_ids"].shape[-1]
    )

    return (
        n_no_trunc,
        n_trunc,
        n_no_trunc > n_trunc,
    )


