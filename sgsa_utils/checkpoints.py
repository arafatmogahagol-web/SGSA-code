"""Keep non-parameter Fourier index lists paired with checkpoint weights."""
from pathlib import Path
import hashlib
import json
import torch


def residual_coefficients(model):
    """Non-tensor SGSA coefficients are part of checkpoint identity too."""
    return {name: float(module.rho) for name, module in model.named_modules()
            if hasattr(module, 'rho') and isinstance(module.rho, (int, float))}


def validate_run_config(path, expected, artifacts=()):
    """Refuse to reuse completed/partial runs with a different protocol."""
    path = Path(path)
    if not path.exists():
        if any(Path(p).exists() for p in artifacts):
            raise ValueError('Existing run lacks configuration: ' + str(path))
        return
    saved = json.loads(path.read_text(encoding='utf-8'))
    changed = [key for key, value in expected.items()
               if key not in saved or saved[key] != value]
    if changed:
        raise ValueError('Existing run configuration differs on: ' + ', '.join(changed)
                         + '. Use a separate run directory.')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frequency_indices(model):
    result = {}
    for name, module in model.named_modules():
        attrs = {}
        for attr in ('index', 'index_q', 'index_kv'):
            value = getattr(module, attr, None)
            if isinstance(value, (list, tuple)) and value and all(isinstance(v, int) for v in value):
                attrs[attr] = list(value)
        if attrs:
            result[name] = attrs
    return result


def restore_frequency_indices(model, indices):
    current = frequency_indices(model)
    if set(current) != set(indices):
        raise ValueError('Fourier module names do not match this model')
    for name, attrs in indices.items():
        if set(attrs) != set(current[name]):
            raise ValueError('Fourier attributes do not match: ' + name)
        module = model.get_submodule(name)
        for attr, values in attrs.items():
            if (len(values) != len(current[name][attr]) or len(set(values)) != len(values)
                    or not all(isinstance(v, int) and v >= 0 for v in values)):
                raise ValueError('Invalid Fourier index list: ' + name + '.' + attr)
            setattr(module, attr, list(values))
    if frequency_indices(model) != indices:
        raise RuntimeError('Fourier index restoration failed')


def save_checkpoint(model, path):
    """Save the ordinary state_dict plus a hash-bound JSON metadata sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)
    metadata = {'schema_version': 2, 'checkpoint_sha256': sha256(path),
                'frequency_indices': frequency_indices(model),
                'residual_coefficients': residual_coefficients(model)}
    sidecar = path.with_suffix(path.suffix + '.metadata.json')
    temporary = sidecar.with_suffix(sidecar.suffix + '.tmp')
    temporary.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    temporary.replace(sidecar)
    return metadata


def load_checkpoint(model, path, recovery_dir=None):
    """Load strictly; a Fourier model cannot silently use newly sampled modes."""
    path = Path(path)
    digest = sha256(path)
    sidecar = path.with_suffix(path.suffix + '.metadata.json')
    metadata = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else None
    if metadata is None and recovery_dir is not None:
        for candidate in sorted(Path(recovery_dir).glob('*.json')):
            record = json.loads(candidate.read_text(encoding='utf-8'))
            if record.get('checkpoint_sha256') == digest:
                metadata = record
                break
    if metadata is not None and metadata.get('checkpoint_sha256') != digest:
        raise ValueError('Checkpoint hash does not match its Fourier metadata')
    if metadata is not None and 'residual_coefficients' in metadata:
        if metadata['residual_coefficients'] != residual_coefficients(model):
            raise ValueError('Checkpoint residual coefficient (rho) does not match the model')
    if frequency_indices(model):
        if metadata is None:
            raise ValueError('Missing Fourier metadata. Supply the original hash-matched '
                             'recovery record or a .pt.metadata.json sidecar.')
        restore_frequency_indices(model, metadata['frequency_indices'])
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return {'checkpoint_sha256': digest, 'frequency_indices': frequency_indices(model),
            'rho_verified': metadata is not None and 'residual_coefficients' in metadata}
