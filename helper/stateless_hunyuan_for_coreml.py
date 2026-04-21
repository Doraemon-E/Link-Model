import torch
from transformers import AutoModelForCausalLM


class StatelessHunYuanForCoreML(torch.nn.Module):
    def __init__(self, model: AutoModelForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(torch.int64)
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.int64)
        allowed = positions.unsqueeze(0) <= positions.unsqueeze(1)
        zero_mask = torch.zeros(
            (seq_len, seq_len), dtype=torch.float16, device=input_ids.device
        )
        neg_inf_mask = torch.full(
            (seq_len, seq_len), -1.0e4, dtype=torch.float16, device=input_ids.device
        )
        attention_mask = (
            torch.where(allowed, zero_mask, neg_inf_mask).unsqueeze(0).unsqueeze(0)
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits.to(torch.float16)
