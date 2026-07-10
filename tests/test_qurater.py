from __future__ import annotations
import unittest
import torch
import torch.nn as nn
import os
import json
import shutil
from typing import Dict, Any

from models.qwen_qurater import QwenQuRater, QUALITY_DIMENSIONS
from train_qurater_qwen import bradley_terry_loss

class MockConfig:
    def __init__(self, hidden_size=64, pad_token_id=0, eos_token_id=1):
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.model_type = "mock"

class MockBackboneOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state

class MockBackbone(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.config = MockConfig(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        batch_size, seq_len = input_ids.shape
        # Deterministic generation based on input_ids to make round-trip checks exact
        x = input_ids.float().unsqueeze(-1).repeat(1, 1, self.config.hidden_size)
        last_hidden = self.linear(x)
        return MockBackboneOutput(last_hidden)

def save_qurater_checkpoint(model, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save backbone
    torch.save(model.backbone.state_dict(), os.path.join(checkpoint_dir, "backbone.pt"))
    # Save rating heads
    torch.save(model.rating_heads.state_dict(), os.path.join(checkpoint_dir, "rating_heads.pt"))
    
    q_config = {
        "pooling_type": "last_token",
        "use_lora": False,
        "use_4bit": False,
        "model_path": "mock"
    }
    with open(os.path.join(checkpoint_dir, "qurater_config.json"), "w", encoding="utf-8") as f:
        json.dump(q_config, f, indent=2)

def load_qurater_checkpoint(checkpoint_dir, backbone):
    with open(os.path.join(checkpoint_dir, "qurater_config.json"), "r") as f:
        q_config = json.load(f)
        
    model = QwenQuRater(backbone=backbone)
    model.backbone.load_state_dict(torch.load(os.path.join(checkpoint_dir, "backbone.pt")))
    model.rating_heads.load_state_dict(torch.load(os.path.join(checkpoint_dir, "rating_heads.pt")))
    return model

class TestQwenQuRater(unittest.TestCase):
    def test_pooling_left_padding(self):
        """Test last non-pad token pooling when padding is on the left"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        # 0 is pad_token_id. Token at last index (4) is 99.
        input_ids = torch.tensor([[0, 0, 10, 20, 99]], dtype=torch.long)
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)
        
        outputs = backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # Manually extract output at index 4 (last token)
        expected = last_hidden[0, 4, :]
        
        ratings = model(input_ids, attention_mask)
        # Check that the pooled state matches the last non-pad token index
        # We check by evaluating the output rating for writing_style
        score_manual = model.rating_heads["writing_style"](expected)
        self.assertAlmostEqual(ratings["writing_style"][0].item(), score_manual.item(), places=5)

    def test_pooling_right_padding(self):
        """Test last non-pad token pooling when padding is on the right"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        # Token at index 2 (value 99) is the last non-pad token
        input_ids = torch.tensor([[10, 20, 99, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
        
        outputs = backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # Extract at index 2
        expected = last_hidden[0, 2, :]
        
        ratings = model(input_ids, attention_mask)
        score_manual = model.rating_heads["writing_style"](expected)
        self.assertAlmostEqual(ratings["writing_style"][0].item(), score_manual.item(), places=5)

    def test_pooling_no_padding(self):
        """Test last non-pad token pooling with no padding at all"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        input_ids = torch.tensor([[10, 20, 99]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        
        outputs = backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        
        expected = last_hidden[0, 2, :]
        
        ratings = model(input_ids, attention_mask)
        score_manual = model.rating_heads["writing_style"](expected)
        self.assertAlmostEqual(ratings["writing_style"][0].item(), score_manual.item(), places=5)

    def test_pooling_eos_token(self):
        """Test last non-pad token pooling with trailing EOS token"""
        backbone = MockBackbone(hidden_size=8)
        # Set EOS as 1
        backbone.config.eos_token_id = 1
        model = QwenQuRater(backbone=backbone)
        
        # EOS token (1) at index 3 is not pad (0)
        input_ids = torch.tensor([[10, 20, 99, 1, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
        
        outputs = backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        
        expected = last_hidden[0, 3, :]
        
        ratings = model(input_ids, attention_mask)
        score_manual = model.rating_heads["writing_style"](expected)
        self.assertAlmostEqual(ratings["writing_style"][0].item(), score_manual.item(), places=5)

    def test_pooling_max_length_truncation(self):
        """Test pooling handles sequence truncated to max_length"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        # No pad tokens, max length seq
        input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        
        outputs = backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        
        expected = last_hidden[0, 3, :]
        
        ratings = model(input_ids, attention_mask)
        score_manual = model.rating_heads["writing_style"](expected)
        self.assertAlmostEqual(ratings["writing_style"][0].item(), score_manual.item(), places=5)

    def test_pooling_batch_varying_lengths(self):
        """Test pooling on batch of varying sequence lengths"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        # Row 1 last token at 4, Row 2 last token at 2
        input_ids = torch.tensor([[10, 20, 30, 40, 99], [10, 20, 99, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.long)
        
        outputs = backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        
        ratings = model(input_ids, attention_mask)
        
        expected_0 = last_hidden[0, 4, :]
        expected_1 = last_hidden[1, 2, :]
        
        score_manual_0 = model.rating_heads["writing_style"](expected_0)
        score_manual_1 = model.rating_heads["writing_style"](expected_1)
        
        self.assertAlmostEqual(ratings["writing_style"][0].item(), score_manual_0.item(), places=5)
        self.assertAlmostEqual(ratings["writing_style"][1].item(), score_manual_1.item(), places=5)

    def test_checkpoint_round_trip(self):
        """Test checkpoint saving, loading, and score consistency verification (< 1e-5)"""
        checkpoint_dir = "./outputs/test_checkpoint_roundtrip"
        
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        # Initialize inputs
        input_ids = torch.tensor([[10, 20, 99, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        
        # Get scores before save
        scores_before = model(input_ids, attention_mask)
        
        # Save
        save_qurater_checkpoint(model, checkpoint_dir)
        
        # Release and reload
        del model
        backbone_new = MockBackbone(hidden_size=8)
        model_loaded = load_qurater_checkpoint(checkpoint_dir, backbone_new)
        
        # Get scores after load
        scores_after = model_loaded(input_ids, attention_mask)
        
        # Compare scores across all dimensions
        for dim in QUALITY_DIMENSIONS:
            diff = torch.max(torch.abs(scores_before[dim] - scores_after[dim])).item()
            self.assertLess(diff, 1e-5, f"Checkpoints round-trip difference for {dim} exceeds 1e-5 threshold: {diff}")
            
        # Clean up
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

if __name__ == "__main__":
    unittest.main()
