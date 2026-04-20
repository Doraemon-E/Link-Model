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
        cache_position = None
        if cache_kwargs is not None:
            cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            cache_position = torch.arange(
                key_states.shape[-2], device=key_states.device
            )

        layer_key_cache = self.key_caches[layer_idx]
        layer_value_cache = self.value_caches[layer_idx]
        one_hot = torch.nn.functional.one_hot(
            cache_position.to(torch.int64),
            num_classes=self.max_cache_len,
        ).to(layer_key_cache.dtype)
        selection = one_hot.sum(dim=0).clamp(0, 1).view(1, 1, self.max_cache_len, 1)

        key_updates = torch.matmul(
            key_states.to(layer_key_cache.dtype).permute(0, 1, 3, 2), one_hot
        ).permute(0, 1, 3, 2)
        value_updates = torch.matmul(
            value_states.to(layer_value_cache.dtype).permute(0, 1, 3, 2), one_hot
        ).permute(0, 1, 3, 2)

        layer_key_cache.mul_(1.0 - selection)
        layer_key_cache.add_(key_updates)
        layer_value_cache.mul_(1.0 - selection)
        layer_value_cache.add_(value_updates)

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
