from __future__ import annotations
import unittest
import torch
import torch.nn as nn
from typing import Dict, Any

from models.qwen_qurater import QwenQuRater, QUALITY_DIMENSIONS
from train_qurater_qwen import bradley_terry_loss

class MockConfig:
    def __init__(self):
        self.hidden_size = 64
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.model_type = "mock"

class MockBackboneOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state

class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
        
    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        # Return random hidden states of shape (batch, seq_len, hidden_size)
        batch_size, seq_len = input_ids.shape
        last_hidden = torch.randn(batch_size, seq_len, self.config.hidden_size)
        return MockBackboneOutput(last_hidden)

class TestQwenQuRater(unittest.TestCase):
    def test_model_forward_scheme_a_last_token(self):
        backbone = MockBackbone()
        model = QwenQuRater(
            backbone=backbone,
            pooling_type="last_token",
            padding_side="right",
            head_type="A"
        )
        
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
        
        ratings = model(input_ids, attention_mask)
        self.assertIsInstance(ratings, dict)
        for dim in QUALITY_DIMENSIONS:
            self.assertIn(dim, ratings)
            self.assertEqual(ratings[dim].shape, (2,))

    def test_model_forward_scheme_a_mean_pooling(self):
        backbone = MockBackbone()
        model = QwenQuRater(
            backbone=backbone,
            pooling_type="mean",
            padding_side="right",
            head_type="A"
        )
        
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
        
        ratings = model(input_ids, attention_mask)
        self.assertIsInstance(ratings, dict)
        for dim in QUALITY_DIMENSIONS:
            self.assertEqual(ratings[dim].shape, (2,))

    def test_loss_direction_correctness(self):
        # Target is B is preferred to A (y = 1.0)
        p_b_gt_a = torch.tensor([1.0])
        
        # Test Case 1: ratings_b > ratings_a (Correct direction)
        ratings_a_1 = torch.tensor([1.5])
        ratings_b_1 = torch.tensor([3.5])
        loss_correct = bradley_terry_loss(ratings_a_1, ratings_b_1, p_b_gt_a)
        
        # Test Case 2: ratings_a > ratings_b (Wrong direction)
        ratings_a_2 = torch.tensor([3.5])
        ratings_b_2 = torch.tensor([1.5])
        loss_incorrect = bradley_terry_loss(ratings_a_2, ratings_b_2, p_b_gt_a)
        
        # Correct direction should yield smaller BCE loss
        self.assertLess(loss_correct.item(), loss_incorrect.item())

if __name__ == "__main__":
    unittest.main()
