"""Protocol guards.

The manuscript reports one frozen configuration per dataset. These checks compare the
configuration actually on disk with those frozen values, so an accidental edit cannot
quietly change what is being measured. They are the only place where the reported
numbers are written down as constants; everything else reads the configuration.
"""
import torch

FROZEN = {
    'SDWPF': dict(num_nodes=134, input_dim=16, seq_len=96, pred_len=48, d_model=64,
                  graph_rank=16, rho=0.25, dropout=0.1, batch_size=32, epochs=30,
                  patience=5, learning_rate=0.0005, weight_decay=0.0001,
                  gradient_clip=1.0, selection='validation MSE',
                  selection_key='val_mse', metric_scale='training-only z-score',
                  train_stride=4, evaluation_stride=24, eval_fixed_rng=True,
                  shuffle_each_epoch=False, augmentation_seed=2024,
                  validation_noise_seed=502024, corruption_seed=910024,
                  seeds=[2024, 2025, 2026], strategies=['clean', 'noise'],
                  expected_valid_test_points=966746,
                  expected_windows=dict(train=3465, val=40, test=165),
                  benchmark=dict(warmup=20, repeats=100, batches=[32])),
    'Hill21': dict(num_nodes=21, input_dim=7, seq_len=96, pred_len=48, d_model=64,
                   graph_rank=16, rho=0.25, dropout=0.1, batch_size=32, epochs=50,
                   patience=8, learning_rate=0.001, weight_decay=0.0001,
                   gradient_clip=1.0, selection='validation MAE',
                   selection_key='val_mae', metric_scale='training-only z-score',
                   train_stride=4, evaluation_stride=24, eval_fixed_rng=False,
                   shuffle_each_epoch=True, augmentation_seed=2024,
                   validation_noise_seed=502024, corruption_seed=910024,
                   seeds=[2024, 2025, 2026], strategies=['clean', 'noise'],
                   expected_valid_test_points=26187,
                   expected_windows=dict(train=729, val=26, test=26),
                   benchmark=dict(warmup=10, repeats=30, batches=[1, 32])),
}


def validate_protocol(dataset, cfg):
    """Raise if ``cfg`` disagrees with the frozen values reported in the paper."""
    expected = FROZEN[dataset]
    changed = [key for key, value in expected.items() if cfg.get(key) != value]
    if changed:
        raise SystemExit(f'{dataset} config disagrees with the paper protocol on: '
                         + ', '.join(changed))
    return True


def check_device(device):
    """CPU is supported for frozen evaluation only; training demands CUDA."""
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():
            raise SystemExit('CUDA is unavailable; use --device cpu for frozen inference')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        return torch.device('cuda')
    return torch.device('cpu')
