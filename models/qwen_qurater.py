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
    QwenQuRater Model implementing QuRating pairwise quality predictors.
    
    Supports:
    - Scheme A: 4 independent classification heads (shared backbone).
    - Scheme B: Single shared linear rating head, expecting dimension tokens in the input.
    - Custom pooling: Last token (decoder default) and Mean pooling.
    """
    def __init__(
        self, 
        backbone: nn.Module, 
        hidden_size: Optional[int] = None, 
        pooling_type: str = "last_token",
        padding_side: str = "right",
        head_type: str = "A"
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling_type = pooling_type
        self.padding_side = padding_side
        self.head_type = head_type
        
        if hidden_size is None:
            hidden_size = self.backbone.config.hidden_size
            
        self.pad_token_id = self.backbone.config.pad_token_id
        if self.pad_token_id is None:
            # Fallback to EOS
            self.pad_token_id = self.backbone.config.eos_token_id
            
        # Scheme A: Four independent rating heads
        self.rating_heads = nn.ModuleDict({
            dim: nn.Linear(hidden_size, 1, bias=False)
            for dim in QUALITY_DIMENSIONS
        })
        
        # Scheme B: Single shared scalar head
        self.scalar_head = nn.Linear(hidden_size, 1, bias=False)

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
        
        # 1. Pooling hidden states
        if self.pooling_type == "last_token":
            if self.padding_side == "left":
                # For left padding, the last token is at index -1
                pooled = last_hidden[:, -1, :]
            else:
                # For right padding, locate the index of the last non-padding token
                sequence_lengths = torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), sequence_lengths]
        elif self.pooling_type == "mean":
            # Mean pooling over non-padding tokens
            hidden_mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            pooled = torch.sum(last_hidden * hidden_mask, 1) / torch.clamp(hidden_mask.sum(1), min=1e-9)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # 2. Output heads
        if self.head_type == "A":
            ratings = {}
            for dim in QUALITY_DIMENSIONS:
                ratings[dim] = self.rating_heads[dim](pooled).squeeze(-1)
            return ratings
        elif self.head_type == "B":
            # If dimension_id is provided, predict for that specific dimension.
            # Otherwise return single scalar score
            score = self.scalar_head(pooled).squeeze(-1)
            return score
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")
