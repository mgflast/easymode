"""Segmentation architecture registry — auto-discovered from sibling modules.

Each .py file in this directory is treated as one architecture. The arch name
is derived from the filename (underscores → hyphens). Each module must expose:
  INPUT_SHAPE  — tuple, e.g. (96, 96, 96, 1)
  create()     — callable returning a fresh Keras Model
  masked_bce_loss / masked_dice_loss / masked_precision / masked_recall / masked_dice
    — re-exported from easymode.segmentation.losses; shared by every arch

Once an arch has been published with weights, its module is frozen — bug fixes
only if they don't alter weight shapes or forward behaviour. New ideas go in
a new file with a new name; add a legacy alias below if older metadata refers
to it under a different name. Archs that are no longer offered for training but
have published weights that must keep loading live in legacy/ — they resolve by
name like any other arch but are excluded from list_archs(). Never delete an
arch module outright: published weights only load if the module that built them
still exists (see e0d21a9, which broke every untagged model on the hub).
"""
import importlib
import os

LEGACY_ALIASES = {
    None: 'unet-membrain',
    'old': 'unet-membrain',
    'current': 'unet-membrain-groupnorm',
    'lite': 'unet-easymode',
    'unet-membrain-groupnorm-sgd': 'unet-membrain-groupnorm',  # removed; same forward arch, weights load as-is
}


def _discover_module_paths():
    pkg_dir = os.path.dirname(__file__)
    out = {}
    legacy = set()
    for subdir in ('', 'legacy'):
        d = os.path.join(pkg_dir, subdir)
        if not os.path.isdir(d):
            continue
        pkg = 'easymode.segmentation.models' + (f'.{subdir}' if subdir else '')
        for entry in sorted(os.listdir(d)):
            if not entry.endswith('.py') or entry.startswith('_'):
                continue
            name = entry[:-3].replace('_', '-')
            out[name] = f'{pkg}.{entry[:-3]}'
            if subdir:
                legacy.add(name)
    return out, legacy


_MODULE_PATHS, _LEGACY_ARCHS = _discover_module_paths()
_LOADED = {}


def list_archs():
    return sorted(n for n in _MODULE_PATHS if n not in _LEGACY_ARCHS)


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
