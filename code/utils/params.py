"""Parameter counting under the convention stated in the paper.

Every count reported in the manuscript is a real-valued scalar count: a complex
parameter contributes two real numbers. ``tensor_elements`` is kept alongside so
that both conventions stay auditable.
"""
import torch


def real_scalars(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def parameter_counts(model):
    params = list(model.parameters())
    return dict(tensor_elements=sum(p.numel() for p in params),
                real_scalar_parameters=sum(p.numel() * (2 if p.is_complex() else 1) for p in params),
                parameter_bytes=sum(p.numel() * p.element_size() for p in params),
                trainable_real_scalars=sum(p.numel() * (2 if p.is_complex() else 1)
                                           for p in params if p.requires_grad))


def autograd_reachable_real_scalars(model):
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in model.parameters() if p.grad is not None)


__all__ = ['real_scalars', 'parameter_counts', 'autograd_reachable_real_scalars']
