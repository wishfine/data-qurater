import unittest

from qurater_utils import resolve_run_paths, validate_normalized_record


class TestQuRaterUtils(unittest.TestCase):
    def test_run_paths_follow_output_directory(self):
        paths = resolve_run_paths("outputs/experiment_a/checkpoints")
        self.assertEqual(paths["checkpoint_0"], "outputs/experiment_a/checkpoint-0")
        self.assertEqual(paths["metadata_dir"], "outputs/experiment_a")

    def test_normalized_record_rejects_invalid_target_and_dimension(self):
        valid = {
            "text_a": "a", "text_b": "b", "target": 0.5,
            "dimension_id": 3, "confidence": 0.0,
        }
        validate_normalized_record(valid)
        with self.assertRaises(ValueError):
            validate_normalized_record({**valid, "target": 1.1})
        with self.assertRaises(ValueError):
            validate_normalized_record({**valid, "dimension_id": 4})


if __name__ == "__main__":
    unittest.main()
