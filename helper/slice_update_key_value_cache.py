import torch
from typing import Any


class SliceUpdateKeyValueCache:
    is_compileable = True

    def __init__(
        self,
        key_caches: list[torch.Tensor],
        value_caches: list[torch.Tensor],
        max_cache_len: int,
    ):
        self.key_caches = key_caches
        self.value_caches = value_caches
        self.max_cache_len = max_cache_len
        self.is_sliding = [False] * len(key_caches)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs

        layer_key_cache = self.key_caches[layer_idx]
        layer_value_cache = self.value_caches[layer_idx]
        key_updates = key_states.to(layer_key_cache.dtype)
        value_updates = value_states.to(layer_value_cache.dtype)

        # Sliding-window cache update: shift left and append latest token(s).
        keep_len = max(self.max_cache_len - key_updates.shape[2], 0)
        if keep_len > 0:
            next_key_cache = torch.cat(
                (
                    layer_key_cache[:, :, -keep_len:, :],
                    key_updates[:, :, -key_updates.shape[2] :, :],
                ),
                dim=2,
            )
            next_value_cache = torch.cat(
                (
                    layer_value_cache[:, :, -keep_len:, :],
                    value_updates[:, :, -value_updates.shape[2] :, :],
                ),
                dim=2,
            )
        else:
            next_key_cache = key_updates[:, :, -self.max_cache_len :, :]
            next_value_cache = value_updates[:, :, -self.max_cache_len :, :]

        layer_key_cache.copy_(next_key_cache)
        layer_value_cache.copy_(next_value_cache)

        return layer_key_cache.to(key_states.dtype), layer_value_cache.to(
            value_states.dtype
        )

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int = 0
    ) -> tuple[int, int]:
        del cache_position, layer_idx
        return self.max_cache_len, 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self.max_cache_len
