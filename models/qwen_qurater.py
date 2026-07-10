from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Optional

QUALITY_DIMENSIONS = [
    "writing_style",
    "required_expertise",
    "facts_and_trivia",
    "educational_value"
]

class QwenQuRater(nn.Module):
    """
    QwenQuRater: Backbone model + 4 independent scalar heads for QuRating.
    
    Enforces Scheme A:
    - Shared Qwen backbone.
    - 4 independent linear classification heads mapping hidden state to rating.
    - Default pooling selects the hidden state of the last non-padding token.
    """
    def __init__(
        self, 
        backbone: nn.Module, 
        hidden_size: Optional[int] = None
    ):
        super().__init__()
        self.backbone = backbone
        
        if hidden_size is None:
            hidden_size = self.backbone.config.hidden_size
            
        self.pad_token_id = self.backbone.config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.backbone.config.eos_token_id
            
        # 4 independent scalar rating heads with bias=False (matching official LlamaForSequenceClassification)
        self.rating_heads = nn.ModuleDict({
            dim: nn.Linear(hidden_size, 1, bias=False)
            for dim in QUALITY_DIMENSIONS
        })

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        dimension_id: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden = outputs.last_hidden_state
        batch_size, seq_len, hidden_size = last_hidden.shape
        
        # Pooling: Extract the hidden state of the last non-padding token.
        # Works universally for left, right, mixed, or no padding.
        non_pad_mask = torch.ne(input_ids, self.pad_token_id)
        arange = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        indices = torch.where(non_pad_mask, arange, torch.tensor(-1, device=input_ids.device))
        
        sequence_lengths = indices.max(dim=-1).values
        sequence_lengths = torch.clamp(sequence_lengths, min=0)
        
        pooled = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]

        # Scheme A: Compute outputs from heads
        if dimension_id is not None:
            # Predict only for the specified dimension index for each sample in the batch
            # dimension_id: (batch_size,) with values in [0, 1, 2, 3]
            scores = []
            for i in range(batch_size):
                dim_idx = int(dimension_id[i].item())
                dim_name = QUALITY_DIMENSIONS[dim_idx]
                score = self.rating_heads[dim_name](pooled[i])
                scores.append(score)
            return torch.stack(scores).squeeze(-1)
            
        ratings = {}
        for dim in QUALITY_DIMENSIONS:
            ratings[dim] = self.rating_heads[dim](pooled).squeeze(-1)
        return ratings
