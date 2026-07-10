from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Optional

DIMENSION_NAMES = [
    "writing_style",
    "required_expertise",
    "facts_and_trivia",
    "educational_value",
]

class QwenQuRater(nn.Module):
    """
    QwenQuRater: Backbone model + 1 joint rating head mapping to 4 quality dimensions.
    
    Enforces:
    - Scheme A: nn.Linear(hidden_size, 4, bias=False)
    - Default pooling selects the hidden state of the last non-padding token.
    - Dimension selection via .gather() on the dimension_id tensor.
    - Explicit validation check: raises ValueError if attention_mask is all 0.
    """
    def __init__(
        self, 
        backbone: nn.Module, 
        hidden_size: Optional[int] = None
    ):
        super().__init__()
        self.backbone = backbone
        
        config = self.backbone.config
        if hasattr(config, "text_config"):
            config = config.text_config
            
        if hidden_size is None:
            hidden_size = getattr(config, "hidden_size", getattr(config, "hidden_dim", 2560))
            
        self.pad_token_id = getattr(config, "pad_token_id", getattr(config, "eos_token_id", None))
            
        self.score = nn.Linear(hidden_size, 4, bias=False)
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
            self.score = self.score.to(dtype=backbone_dtype)
        except Exception:
            pass

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        dimension_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        Returns:
        - If dimension_id is None: scores of shape (batch_size, 4)
        - If dimension_id is provided: selected scores of shape (batch_size,)
        """
        # Ensure attention_mask is not all 0s for any item in the batch
        if attention_mask is not None and torch.eq(attention_mask, 0).all(dim=-1).any():
            raise ValueError("[CRITICAL ERROR] Attention mask is all zeros for one or more samples in the batch.")

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden = outputs.last_hidden_state
        batch_size, seq_len, hidden_size = last_hidden.shape
        
        # Last non-padding token pooling using attention_mask
        non_pad_mask = torch.eq(attention_mask, 1)
        arange = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        indices = torch.where(non_pad_mask, arange, torch.tensor(-1, device=input_ids.device))
        
        sequence_lengths = indices.max(dim=-1).values
        sequence_lengths = torch.clamp(sequence_lengths, min=0)
        
        pooled = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]

        all_scores = self.score(pooled)  # Shape: (batch_size, 4)

        if dimension_id is not None:
            selected_scores = all_scores.gather(
                dim=1,
                index=dimension_id.unsqueeze(1).long()
            ).squeeze(1)
            return selected_scores
            
        return all_scores
