"""CPU regression tests for checkpoint and run identity; no experiment data required."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgsa_utils.model import SGSA
from sgsa_utils.checkpoints import load_checkpoint, save_checkpoint, validate_run_config


class IntegrityGuards(unittest.TestCase):
    def test_clean_split_helper_matches_benchmark_origins(self):
        sys.path.insert(0, str(ROOT / 'experiments/SDWPF/code'))
        from preprocessing import make_splits
        from data_loader import Windows
        data = SimpleNamespace(
            split={'train': (0, 14000), 'val': (14000, 15000), 'test': (16000, 20000)},
            timestamps=np.arange(20000), feature_median=np.zeros(1),
            feature_mean=np.zeros(1), feature_std=np.ones(1))
        clean = make_splits(data, 96, 48)
        for part, helper, count in zip(('train', 'val', 'test'), clean, (3465, 40, 165)):
            benchmark = Windows.for_partition(data, part, stride=4 if part == 'train' else 24)
            self.assertEqual(helper.starts, benchmark.starts)
            self.assertEqual(len(helper), count)
            self.assertGreaterEqual(helper.starts[0], 96)

    def test_rho_mismatch_is_rejected_even_when_tensor_shapes_match(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            original = SGSA(4, 2, 1, 2, 8, 0., 2, .25).eval()
            save_checkpoint(original, path)
            restored = SGSA(4, 2, 1, 2, 8, 0., 2, .25).eval()
            record = load_checkpoint(restored, path)
            self.assertTrue(record['rho_verified'])
            x = torch.randn(1, 4, 2, 1)
            with torch.no_grad():
                torch.testing.assert_close(original(x), restored(x), rtol=0, atol=0)
            wrong = SGSA(4, 2, 1, 2, 8, 0., 2, .5)
            with self.assertRaisesRegex(ValueError, 'rho'):
                load_checkpoint(wrong, path)

    def test_legacy_weights_remain_loadable_without_inventing_rho_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.pt'
            model = SGSA(4, 2, 1, 2, 8, 0., 2, .25)
            torch.save(model.state_dict(), path)
            self.assertFalse(load_checkpoint(model, path)['rho_verified'])

    def test_altered_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            model = SGSA(4, 2, 1, 2, 8, 0., 2, .25)
            save_checkpoint(model, path)
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_checkpoint(model, path)

    def test_run_configuration_cannot_be_silently_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            expected = dict(rho=.25, seed=2024, reshuffle_each_epoch=False,
                            split={'train': [0, 100]}, learning_rate=.0005)
            validate_run_config(path, expected)
            path.write_text(json.dumps(expected), encoding='utf-8')
            validate_run_config(path, expected)
            for key, value in [('rho', .5), ('seed', 2025), ('reshuffle_each_epoch', True),
                               ('split', {'train': [0, 200]}), ('learning_rate', .001)]:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    validate_run_config(path, dict(expected, **{key: value}))
            path.unlink()
            checkpoint = Path(folder) / 'best.pt'
            checkpoint.touch()
            with self.assertRaisesRegex(ValueError, 'lacks configuration'):
                validate_run_config(path, expected, [checkpoint])


if __name__ == '__main__':
    unittest.main()
