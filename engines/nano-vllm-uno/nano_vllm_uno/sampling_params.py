import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_token_ids: list[int] | None = None
    mask_token_id: int | None = None
    # Deterministic_uniform is useful for reproducible schedule-invariant tests.
    noise_mode: str = "random_uniform"
    noise_salt: int | None = None

    # Forward length including the seed token. Size 1 is autoregressive;
    # larger sizes use diffusion draft/verify.
    diffusion_block_size: int = 1
    # Benchmark-only target emitted tokens per model forward.
    # Diffusion still executes verification, then overrides its accept/reject decisions.
    forced_accept_tpf: float | None = None

    def __post_init__(self):
        # Allow temperature=0 for greedy decoding
        self.temperature = float(self.temperature)
        assert self.temperature >= 0.0, "temperature must be non-negative"
        if self.top_p is not None:
            self.top_p = float(self.top_p)
            if not (0.0 < self.top_p <= 1.0):
                raise ValueError("top_p must be in (0, 1] when provided")
        if self.top_k is not None:
            self.top_k = int(self.top_k)
        if self.top_p is not None and (
            self.top_k is None or self.top_k <= 0
        ):
            raise ValueError(
                "top_p requires a positive top_k; top-p-only sampling "
                "is not supported"
            )
        if self.forced_accept_tpf is not None:
            self.forced_accept_tpf = float(self.forced_accept_tpf)
            if self.forced_accept_tpf < 1.0:
                raise ValueError("forced_accept_tpf must be >= 1.0 when set")
        self.diffusion_block_size = int(self.diffusion_block_size)
        if self.diffusion_block_size < 1:
            raise ValueError("diffusion_block_size must be >= 1")
        if self.forced_accept_tpf is not None:
            if not self.ignore_eos:
                raise ValueError("forced_accept_tpf requires ignore_eos=True")
            max_tpf = (self.diffusion_block_size + 1) / 2.0
            if self.forced_accept_tpf > max_tpf:
                logger.warning(
                    "forced_accept_tpf=%g exceeds the maximum TPF=%g for "
                    "diffusion_block_size=%d; capping to %g",
                    self.forced_accept_tpf,
                    max_tpf,
                    self.diffusion_block_size,
                    max_tpf,
                )
                self.forced_accept_tpf = max_tpf
        if (
            self.diffusion_block_size > 1
            and self.noise_mode == "mask"
            and self.mask_token_id is None
        ):
            raise ValueError(
                "Mask noise requires mask_token_id. Pass the "
                "diffusion-training mask token explicitly."
            )
        if self.stop_token_ids is not None:
            self.stop_token_ids = [int(x) for x in self.stop_token_ids]
        if self.mask_token_id is not None:
            self.mask_token_id = int(self.mask_token_id)
            if self.mask_token_id <= 1:
                raise ValueError("mask_token_id must be > 1 when provided")
        valid_noise_modes = {"deterministic_uniform", "random_uniform", "mask"}
        if self.noise_mode not in valid_noise_modes:
            raise ValueError(
                f"Unsupported noise_mode={self.noise_mode!r}. "
                f"Expected one of {sorted(valid_noise_modes)}"
            )
        if self.noise_salt is not None:
            self.noise_salt = int(self.noise_salt)
