"""Segmentation architecture registry — auto-discovered from sibling modules.

Each .py file in this directory is treated as one architecture. The arch name
is derived from the filename (underscores → hyphens). Each module must expose:
  INPUT_SHAPE  — tuple, e.g. (96, 96, 96, 1)
  create()     — callable returning a fresh Keras Model
  masked_bce_loss / masked_dice_loss / masked_precision / masked_recall / masked_dice

Once an arch has been published with weights, its module is frozen — bug fixes
only if they don't alter weight shapes or forward behaviour. New ideas go in
a new file with a new name; add a legacy alias below if older metadata refers
to it under a different name.
"""
import importlib
import os

LEGACY_ALIASES = {
    None: 'unet-membrain',
    'old': 'unet-membrain',
    'current': 'unet-membrain-groupnorm',
    'lite': 'unet-easymode',
}


def _discover_module_paths():
    pkg_dir = os.path.dirname(__file__)
    out = {}
    for entry in sorted(os.listdir(pkg_dir)):
        if not entry.endswith('.py') or entry.startswith('_'):
            continue
        stem = entry[:-3]
        out[stem.replace('_', '-')] = f'easymode.segmentation.models.{stem}'
    return out


_MODULE_PATHS = _discover_module_paths()
_LOADED = {}


def list_archs():
    return sorted(_MODULE_PATHS)


def resolve_arch(name):
    if name in _MODULE_PATHS:
        return name
    if name in LEGACY_ALIASES:
        return LEGACY_ALIASES[name]
    raise ValueError(f"Unknown arch: {name!r}. Available: {list_archs()}")


def get_arch(name):
    name = resolve_arch(name)
    if name not in _LOADED:
        module = importlib.import_module(_MODULE_PATHS[name])
        _LOADED[name] = {'module': module, 'input_shape': module.INPUT_SHAPE}
    return _LOADED[name]
