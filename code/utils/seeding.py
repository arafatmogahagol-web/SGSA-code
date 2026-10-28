"""Seed handling shared by every training run."""
import random

import numpy as np
import torch


def seed_all(seed):
    """Seed python, numpy, torch CPU and every CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['py'])
    np.random.set_state(state['np'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])
