"""Select the FlashAttention-3 implementation used by Nano GPU runners."""

from __future__ import annotations

import functools


def configure_nano_fa3(provider: str) -> str:
    """Install the requested FA3 loader before a Nano engine is created."""
    if provider == "native":
        return provider
    if provider != "sgl_kernel":
        raise ValueError(f"unknown Nano FA3 provider: {provider}")

    from sgl_kernel import flash_attn
    import nano_vllm_uno.layers.attention as attention

    original_loader = attention._load_attention_backend

    @functools.lru_cache(maxsize=3)
    def load_backend(name: str):
        if str(name).lower() != "fa3":
            return original_loader(name)
        return (
            "fa3",
            flash_attn.flash_attn_varlen_func,
            flash_attn.flash_attn_with_kvcache,
        )

    attention._load_attention_backend = load_backend
    selected = attention._load_attention_backend("fa3")
    if (
        selected[1] is not flash_attn.flash_attn_varlen_func
        or selected[2] is not flash_attn.flash_attn_with_kvcache
    ):
        raise RuntimeError("Nano FA3 provider selection did not install SGL-Kernel")
    return provider
