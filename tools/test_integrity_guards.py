"""CPU regression tests for checkpoint identity, run identity and window origins.

No experiment data, no GPU and no training are required:

    python tools/test_integrity_guards.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))

from SGSA import SGSA
from data_provider.common import PreparedData, window_starts
from exp.guards import FROZEN, validate_protocol
from utils.checkpoints import load_checkpoint, save_checkpoint, validate_run_config


def _fake_data(split, timestamps=20000, nodes=2, features=1):
    zeros = np.zeros((timestamps, nodes, features), dtype='float32')
    mask = np.ones((timestamps, nodes), dtype=bool)
    return PreparedData(name='synthetic', x=zeros, y=np.zeros((timestamps, nodes)), mask=mask,
                        finite_inputs=np.ones_like(zeros, dtype=bool),
                        feature_mean=np.zeros(features), feature_std=np.ones(features),
                        feature_median=np.zeros(features), target_mean=0.0, target_std=1.0,
                        feature_names=[f'f{i}' for i in range(features)], target_name='f0',
                        timestamps=np.arange(timestamps), turbine_ids=np.arange(nodes),
                        split=split)


class IntegrityGuards(unittest.TestCase):
    def test_window_origins_match_the_reported_counts(self):
        """The window helper reproduces the counts and origins stated in the paper."""
        data = _fake_data({'train': (0, 14000), 'val': (14000, 15000), 'test': (16000, 20000)})
        for dataset, expected in [('SDWPF', (3465, 40, 165))]:
            cfg = dict(FROZEN[dataset], expected_windows=dict(train=expected[0],
                                                             val=expected[1], test=expected[2]))
            counts = []
            for partition in ('train', 'val', 'test'):
                starts = window_starts(data, partition, cfg)
                counts.append(len(starts))
                self.assertGreaterEqual(int(starts[0]), cfg['seq_len'])
            self.assertEqual(tuple(counts), expected)

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
            expected = dict(rho=.25, seed=2024, shuffle_each_epoch=False,
                            split={'train': [0, 100]}, learning_rate=.0005)
            validate_run_config(path, expected)
            path.write_text(json.dumps(expected), encoding='utf-8')
            validate_run_config(path, expected)
            for key, value in [('rho', .5), ('seed', 2025), ('shuffle_each_epoch', True),
                               ('split', {'train': [0, 200]}), ('learning_rate', .001)]:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    validate_run_config(path, dict(expected, **{key: value}))
            path.unlink()
            checkpoint = Path(folder) / 'best.pt'
            checkpoint.touch()
            with self.assertRaisesRegex(ValueError, 'lacks configuration'):
                validate_run_config(path, expected, [checkpoint])

    def test_frozen_protocol_guard_rejects_a_changed_coefficient(self):
        cfg = dict(FROZEN['SDWPF'])
        self.assertTrue(validate_protocol('SDWPF', cfg))
        for key, value in [('rho', 0.5), ('epochs', 40), ('selection_key', 'val_mae')]:
            with self.subTest(key=key), self.assertRaises(SystemExit):
                validate_protocol('SDWPF', dict(cfg, **{key: value}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
