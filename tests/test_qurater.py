from __future__ import annotations
import unittest
import torch
import torch.nn as nn
import os
import json
import shutil
from typing import Dict, Any

from models.qwen_qurater import QwenQuRater, DIMENSION_NAMES
from train_qurater_qwen import bradley_terry_loss, save_modular_checkpoint

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
        x = input_ids.float().unsqueeze(-1).repeat(1, 1, self.config.hidden_size)
        last_hidden = self.linear(x)
        return MockBackboneOutput(last_hidden)

class MockTokenizer:
    def __init__(self):
        self.padding_side = "right"
    def save_pretrained(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
            json.dump({"padding_side": "right"}, f)

class TestQwenQuRater(unittest.TestCase):
    def test_pooling_left_padding(self):
        """Test last non-pad token pooling when padding is on the left"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        input_ids = torch.tensor([[0, 0, 10, 20, 99]], dtype=torch.long)
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)
        
        outputs = backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        expected = last_hidden[0, 4, :]
        
        ratings = model(input_ids, attention_mask)
        score_manual = model.score(expected)
        
        for d in range(4):
            self.assertAlmostEqual(ratings[0, d].item(), score_manual[d].item(), places=5)

    def test_pooling_right_padding(self):
        """Test last non-pad token pooling when padding is on the right"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        input_ids = torch.tensor([[10, 20, 99, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
        
        outputs = backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        expected = last_hidden[0, 2, :]
        
        ratings = model(input_ids, attention_mask)
        score_manual = model.score(expected)
        
        for d in range(4):
            self.assertAlmostEqual(ratings[0, d].item(), score_manual[d].item(), places=5)

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
        score_manual = model.score(expected)
        
        for d in range(4):
            self.assertAlmostEqual(ratings[0, d].item(), score_manual[d].item(), places=5)

    def test_pooling_eos_token(self):
        """Test last non-pad token pooling with trailing EOS token"""
        backbone = MockBackbone(hidden_size=8)
        backbone.config.eos_token_id = 1
        model = QwenQuRater(backbone=backbone)
        
        input_ids = torch.tensor([[10, 20, 99, 1, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
        
        outputs = backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        expected = last_hidden[0, 3, :]
        
        ratings = model(input_ids, attention_mask)
        score_manual = model.score(expected)
        
        for d in range(4):
            self.assertAlmostEqual(ratings[0, d].item(), score_manual[d].item(), places=5)

    def test_pooling_max_length_truncation(self):
        """Test pooling handles sequence truncated to max_length"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        
        outputs = backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        expected = last_hidden[0, 3, :]
        
        ratings = model(input_ids, attention_mask)
        score_manual = model.score(expected)
        
        for d in range(4):
            self.assertAlmostEqual(ratings[0, d].item(), score_manual[d].item(), places=5)

    def test_pooling_batch_varying_lengths(self):
        """Test pooling on batch of varying sequence lengths"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        input_ids = torch.tensor([[10, 20, 30, 40, 99], [10, 20, 99, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.long)
        
        outputs = backbone(input_ids, attention_mask)
        last_hidden = outputs.last_hidden_state
        
        ratings = model(input_ids, attention_mask)
        
        expected_0 = last_hidden[0, 4, :]
        expected_1 = last_hidden[1, 2, :]
        
        score_manual_0 = model.score(expected_0)
        score_manual_1 = model.score(expected_1)
        
        for d in range(4):
            self.assertAlmostEqual(ratings[0, d].item(), score_manual_0[d].item(), places=5)
            self.assertAlmostEqual(ratings[1, d].item(), score_manual_1[d].item(), places=5)

    def test_attention_mask_all_ones(self):
        """Test pooling works when attention_mask is all ones"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        
        ratings = model(input_ids, attention_mask)
        self.assertEqual(ratings.shape, (1, 4))

    def test_attention_mask_all_zeros_raises_error(self):
        """Test that passing an all-zeros attention mask raises ValueError"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        input_ids = torch.tensor([[10, 20, 30, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        
        with self.assertRaises(ValueError):
            model(input_ids, attention_mask)

    def test_forward_with_dimension_id_gather(self):
        """Test that passing dimension_id gathers the correct dimension scores"""
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        
        input_ids = torch.tensor([[10, 20, 99], [10, 20, 99]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)
        dimension_ids = torch.tensor([0, 2], dtype=torch.long)
        
        ratings_all = model(input_ids, attention_mask)
        ratings_gathered = model(input_ids, attention_mask, dimension_ids)
        
        self.assertEqual(ratings_gathered.shape, (2,))
        self.assertAlmostEqual(ratings_gathered[0].item(), ratings_all[0, 0].item(), places=5)
        self.assertAlmostEqual(ratings_gathered[1].item(), ratings_all[1, 2].item(), places=5)

    def test_loss_direction_correctness(self):
        """Verify B > A target results in smaller loss when rating_b > rating_a"""
        targets = torch.tensor([1.0])
        confidences = torch.tensor([1.0])
        
        ratings_a_1 = torch.tensor([1.5])
        ratings_b_1 = torch.tensor([3.5])
        loss_correct = bradley_terry_loss(ratings_a_1, ratings_b_1, targets, confidences, 0.0)
        
        ratings_a_2 = torch.tensor([3.5])
        ratings_b_2 = torch.tensor([1.5])
        loss_incorrect = bradley_terry_loss(ratings_a_2, ratings_b_2, targets, confidences, 0.0)
        
        self.assertLess(loss_correct.item(), loss_incorrect.item())

    def test_loss_confidence_mask_empty(self):
        """Verify that when the confidence mask filters out everything, loss is 0.0 without NaN"""
        ratings_a = torch.tensor([1.5, 2.5], requires_grad=True)
        ratings_b = torch.tensor([3.5, 1.5], requires_grad=True)
        targets = torch.tensor([1.0, 0.0])
        confidences = torch.tensor([0.1, 0.2])
        
        loss = bradley_terry_loss(ratings_a, ratings_b, targets, confidences, 0.5)
        self.assertEqual(loss.item(), 0.0)
        
        loss.backward()
        self.assertIsNotNone(ratings_a.grad)

    def test_checkpoint_saving_layout(self):
        """Test checkpoint saving writes correct modular directory layout structure"""
        checkpoint_dir = "./outputs/test_checkpoint_layout"
        
        backbone = MockBackbone(hidden_size=8)
        model = QwenQuRater(backbone=backbone)
        tokenizer = MockTokenizer()
        
        class DummyArgs:
            def __init__(self):
                self.model_path = "mock"
                self.use_lora = False
                self.use_4bit = False
                self.bf16 = True
                self.seed = 42
                self.max_length = 256
                
        args = DummyArgs()
        
        save_modular_checkpoint(model, tokenizer, checkpoint_dir, args, epoch=1)
        
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "adapter")))
        self.assertTrue(
            os.path.exists(os.path.join(checkpoint_dir, "rating_head.safetensors")) or
            os.path.exists(os.path.join(checkpoint_dir, "rating_head.pt"))
        )
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "qurater_config.json")))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "tokenizer")))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "training_args.json")))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "trainer_state.pt")))
        
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

if __name__ == "__main__":
    unittest.main()
