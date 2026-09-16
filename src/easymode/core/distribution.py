import os, json, shutil, logging, requests
from huggingface_hub import hf_hub_download
import easymode.core.config as cfg

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

REPO_ID = "mgflast/easymode-v2"
MODEL_CACHE_DIR = cfg.settings["MODEL_DIRECTORY"]
METADATA_KEYS = ("apix", "apix_z", "arch", "normalization", "timestamp")

_online = None
_registry = None


def is_online():
    global _online
    if _online is None:
        try:
            _online = requests.get("https://huggingface.co", timeout=5).status_code == 200
        except Exception:
            _online = False
    return _online


def fetch_json(filename):
    try:
        r = requests.get(f"https://huggingface.co/{REPO_ID}/resolve/main/{filename}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def read_local_metadata(metadata_path):
    if not metadata_path or not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_ts(ts):
    ts = str(ts)
    return "20" + ts if len(ts) == 12 else ts   # 2-digit-year stamps


def _newer(remote_ts, local_ts):
    return bool(remote_ts) and (not local_ts or _normalize_ts(remote_ts) > _normalize_ts(local_ts))


# ---- registry: registry.json = {feature: {"default": tag, "models": {tag: {"weights": path, ["metadata": path], "timestamp": ..., ...}}}}

REGISTRY_CACHE = os.path.join(MODEL_CACHE_DIR, "registry.json")


def get_registry():
    global _registry
    if _registry is None:
        _registry = fetch_json("registry.json") if is_online() else None
        if _registry:
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            with open(REGISTRY_CACHE, "w") as f:
                json.dump(_registry, f, indent=2)
        else:
            _registry = read_local_metadata(REGISTRY_CACHE) or {}
    return _registry


def list_variants(feature):
    return sorted((get_registry().get(feature) or {}).get("models", {}))


def registry_entry(feature, variant=None):
    feat = get_registry().get(feature)
    if not feat:
        return None
    tag = variant or feat.get("default")
    entry = feat.get("models", {}).get(tag)
    return dict(entry, feature=feature, tag=tag) if entry else None


def root_entry(title):
    # denoisers and the tilt filter live at the repo root, outside the registry
    entry = {"feature": title, "tag": title, "weights": f"{title}.h5", "metadata": f"{title}.json"}
    meta = fetch_json(entry["metadata"]) if is_online() else None
    if meta is not None:
        entry.update({k: meta[k] for k in METADATA_KEYS if k in meta})
        return entry
    return entry if os.path.exists(_cache_path(entry["weights"])) else None


def local_entry(title):
    # a user's own model, copied into the cache root: its sidecar was not written by us, so it has no "feature"
    for ext in (".h5", ".scnm"):
        weights = os.path.join(MODEL_CACHE_DIR, title + ext)
        meta = read_local_metadata(os.path.join(MODEL_CACHE_DIR, title + ".json"))
        if os.path.exists(weights) and meta is not None and "feature" not in meta:
            return {"feature": title, "tag": "local", "weights": title + ext}
    return None


def list_local_models():
    if not os.path.isdir(MODEL_CACHE_DIR):
        return []
    stems = {os.path.splitext(f)[0] for f in os.listdir(MODEL_CACHE_DIR) if f.endswith((".h5", ".scnm"))}
    return sorted(s for s in stems if local_entry(s) is not None)


def entry_metadata(entry):
    return {k: entry[k] for k in METADATA_KEYS if k in entry}


def _cache_path(repo_path):
    return os.path.join(MODEL_CACHE_DIR, *repo_path.split("/"))


def _cache_paths(entry):
    weights_path = _cache_path(entry["weights"])
    metadata_path = _cache_path(entry["metadata"]) if entry.get("metadata") else os.path.splitext(weights_path)[0] + ".json"
    return weights_path, metadata_path   # sidecar always sits next to the weights: load_model_weights finds it by stem


def get_engine(feature, variant=None):
    entry = registry_entry(feature, variant) or local_entry(feature)
    if entry is None:
        return None
    return "2d" if entry["weights"].endswith(".scnm") else "3d"


def _download(entry, weights_path, metadata_path, silent=False):
    if not silent:
        print(f"\nDownloading '{entry['feature']} ({entry['tag']})' from {REPO_ID}...\n")
    try:
        hf_hub_download(repo_id=REPO_ID, filename=entry["weights"], cache_dir=MODEL_CACHE_DIR, local_dir=MODEL_CACHE_DIR)
    except Exception as e:
        raise RuntimeError(f"Failed to download {entry['weights']} from {REPO_ID}: {e}")
    metadata = (fetch_json(entry["metadata"]) if entry.get("metadata") else None) or entry_metadata(entry)
    metadata.update(feature=entry["feature"], tag=entry["tag"])   # the cache describes itself
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    if not silent:
        print(f"\nNetwork weights saved to cache at {weights_path}\n")
    return metadata


def get_model(model_title, force_download=False, silent=False, variant=None, _2d=None):
    entry = registry_entry(model_title, variant) or local_entry(model_title) or root_entry(model_title)
    if entry is None:
        variants = list_variants(model_title)
        if variant and variants:
            print(f"\nNo '{variant}' model for {model_title}; available: {', '.join(variants)}\n")
        else:
            print(f"\nNo model available for '{model_title}'. Run 'easymode list' to see the available features.\n")
        return None, None

    weights_path, metadata_path = _cache_paths(entry)
    local_meta = read_local_metadata(metadata_path)
    cached = os.path.exists(weights_path)

    if entry["tag"] == "local" or not is_online():
        if not cached:
            print(f"\n'{entry['feature']} ({entry['tag']})' is not in the local cache {MODEL_CACHE_DIR} and there is no internet connection to download it - aborting.\n")
            return None, None
        return weights_path, local_meta or entry_metadata(entry)

    if force_download or not cached or _newer(entry.get("timestamp"), (local_meta or {}).get("timestamp")):
        if not silent and cached and not force_download:
            print(f"\nNew version available for '{entry['feature']} ({entry['tag']})', updating...")
        local_meta = _download(entry, weights_path, metadata_path, silent=silent)
    return weights_path, local_meta or entry_metadata(entry)


def load_model_weights(weights_path):
    import tensorflow as tf
    base = os.path.basename(weights_path)
    # sidecar wins over the filename: a user model titled e.g. 'my_iso_run' would otherwise be built as the n2n denoiser
    arch_name = (read_local_metadata(os.path.splitext(weights_path)[0] + '.json') or {}).get('arch')
    if arch_name in ("n2n", "ddw", "iso") or (arch_name is None and ("n2n" in base or "ddw" in base or "iso" in base)):
        from easymode.n2n.model import create
        dummy_input = tf.zeros((1, 160, 160, 160, 1))
    elif arch_name == "tilt" or (arch_name is None and "tilt" in base):
        from easymode.tiltfilter.model import create
        dummy_input = [tf.zeros((1, 256, 256, 1)), tf.zeros((1, 256, 256, 1))]
    else:
        from easymode.segmentation.models import get_arch
        arch = get_arch(arch_name)
        create = arch['module'].create
        dummy_input = tf.zeros((1, *arch['input_shape']))

    model = create()
    _ = model(dummy_input)
    model.load_weights(weights_path)
    return model


def load_model(local_path):
    return load_model_weights(local_path)


def clear_model_cache(model_title=None):
    if not model_title:
        if os.path.exists(MODEL_CACHE_DIR):
            shutil.rmtree(MODEL_CACHE_DIR)
            print(f"Cleared model cache: {MODEL_CACHE_DIR}")
        return
    for root, _, files in os.walk(MODEL_CACHE_DIR):
        for name in files:
            sidecar = os.path.join(root, name)
            if not name.endswith(".json") or (read_local_metadata(sidecar) or {}).get("feature") != model_title:
                continue
            stem = os.path.splitext(sidecar)[0]
            for p in [sidecar] + [stem + ext for ext in (".h5", ".scnm")]:
                if os.path.exists(p):
                    os.remove(p)
                    print(f"Removed {p}")


ROOT_MODELS = ("n2n_direct", "ddw_direct", "iso_direct", "tilt")


def download_models(features=(), version=None, everything=False, silent=False):
    """Fetch models into the cache for offline use: named features (one version each), or everything."""
    if not is_online():
        print("\nAn internet connection is required to download models.\n")
        return
    if everything:
        jobs = [(f, tag) for f in sorted(get_registry()) for tag in list_variants(f)] + [(t, None) for t in ROOT_MODELS]
    else:
        jobs = [(f, version) for f in features]
    for feature, tag in jobs:
        get_model(feature, variant=tag, silent=silent)
    if everything and not silent:
        print(f"\nAll models are now in {MODEL_CACHE_DIR}.\n")


def print_notification():
    message = ((fetch_json("notification.json") or {}).get("message") or "").strip()
    if message:
        print()
        print(message)


def list_remote_models():
    if is_online():
        print_notification()
    registry = get_registry()
    if not registry:
        print(f"\nCould not read the model registry from {REPO_ID}" + ("." if is_online() else " (no internet connection)."))

    models = []
    if registry:
        print("\neasymode can currently segment the following features:\n")
        print(f"     {''.ljust(30)} versions")
        for feature in sorted(registry):
            tags = list_variants(feature)
            default = (registry[feature] or {}).get("default")
            label = ", ".join(t + "*" if t == default else t for t in tags)
            print(f"   > {feature.ljust(30)} {label}")
            models.append({"title": feature, "variants": tags, "default": default})
        print("\n   *default model. use --version to select a specific model variant.")
    print()
    return models
