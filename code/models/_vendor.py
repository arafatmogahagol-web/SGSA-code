"""Helpers for loading the unmodified third-party implementations in ``vendor/``.

``vendor/`` sits next to ``code/`` and is never edited on disk: adapters either
import it, or load a file after an explicit, asserted source replacement that is
documented in ``docs/IMPLEMENTATION_NOTES.md``.
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

VENDOR = Path(__file__).resolve().parents[2] / 'vendor'


def module(name, path, package=False):
    """Load a module from an explicit path, cached under a private name."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, path,
        submodule_search_locations=[str(Path(path).parent)] if package else None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        # Never leave a half-executed module in the cache: a later attempt would
        # silently receive the broken object instead of a real import error.
        sys.modules.pop(name, None)
        raise
    return mod


def package(name, path):
    """Register a directory as a namespace package (needed by several vendors)."""
    pkg = ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg
    return pkg


def patched_source(name, path, replacements):
    """Load a vendor file after exact, asserted string replacements.

    Each replacement is ``(old, new)`` or ``(old, new, expected_occurrences)``.
    The assertion is what keeps the patch honest: if the upstream file changes,
    the adapter fails loudly instead of silently applying a different edit.
    """
    path = Path(path)
    source = path.read_text()
    for item in replacements:
        old, new = item[0], item[1]
        expected = item[2] if len(item) > 2 else 1
        assert source.count(old) == expected, \
            f'{path.name}: expected {expected} occurrences of {old!r}, found {source.count(old)}'
        source = source.replace(old, new)
    mod = ModuleType(name)
    exec(compile(source, str(path), 'exec'), mod.__dict__)
    return mod
