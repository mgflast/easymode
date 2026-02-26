import os, json, requests
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
import easymode.core.config as cfg

HF_REPO_ID = "mgflast/easymode"
MODEL_CACHE_DIR = cfg.settings["MODEL_DIRECTORY"]


def get_model_info(model_title, _2d=False):
    if _2d:
        weights_filename = f"{model_title}.scnm"
        metadata_filename = f"{model_title}_2d.json"
    else:
        weights_filename = f"{model_title}.h5"
        metadata_filename = f"{model_title}.json"
    return {
        "repo_id": HF_REPO_ID,
        "model_title": model_title,
        "weights_filename": weights_filename,
        "metadata_filename": metadata_filename,
        "weights_path": os.path.join(MODEL_CACHE_DIR, weights_filename),
        "metadata_path": os.path.join(MODEL_CACHE_DIR, metadata_filename),
    }


def is_online():
    try:
        r = requests.get("https://huggingface.co", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def read_local_metadata(metadata_path):
    if not os.path.exists(metadata_path): return None
    try:
        with open(metadata_path, "r") as f: return json.load(f)
    except Exception:
        return None


def get_remote_metadata(model_title, _2d=False):
    prev = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    filename = f"{model_title}_2d.json" if _2d else f"{model_title}.json"
    metadata = None
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename, cache_dir=MODEL_CACHE_DIR)
        with open(path, "r") as f: metadata = json.load(f)
    except Exception as e:
        print(f"Warning: Could not get remote metadata for {model_title} ({filename}): {e}")
    finally:
        if prev is None: os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else: os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev
    return metadata


def is_remote_newer(local_meta, remote_meta):
    if not local_meta or not remote_meta:
        return False

    local_ts = local_meta.get("timestamp")
    remote_ts = remote_meta.get("timestamp")

    if not local_ts or not remote_ts:
        return False

    return remote_ts > local_ts


def download_model_files(info, remote_metadata=None, silent=False):
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    if not silent: print(f"\nDownloading {info['model_title']} from {info['repo_id']}...\n")
    try:
        hf_hub_download(repo_id=info["repo_id"], filename=info["weights_filename"], cache_dir=MODEL_CACHE_DIR, local_dir=MODEL_CACHE_DIR)
        prev = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        hf_hub_download(repo_id=info["repo_id"], filename=info["metadata_filename"], cache_dir=MODEL_CACHE_DIR, local_dir=MODEL_CACHE_DIR)
        if prev is None: os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else: os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev
    except Exception as e:
        raise RuntimeError(f"Failed to download {info['model_title']} from {info['repo_id']}: {e}")
    metadata = remote_metadata or read_local_metadata(info["metadata_path"])
    if not silent: print(f"\nNetwork weights saved to cache at {info['weights_path']}\n")
    return info["weights_path"], metadata


def get_model(model_title, force_download=False, silent=False, _2d=False):
    info = get_model_info(model_title, _2d=_2d)
    online = is_online()
    weights_local = os.path.exists(info["weights_path"])
    local_meta = read_local_metadata(info["metadata_path"])

    if not online:
        if not weights_local:
            print(f"\nThe required network weights for {model_title} are not available in the local cache {MODEL_CACHE_DIR} and there is no internet connection available to download them - aborting...\n")
            return None, None
        if not silent:
            print("\nLocal model found. There may be updates available, but we cannot check without an internet connection.\n")
        return info["weights_path"], local_meta

    remote_meta = get_remote_metadata(model_title, _2d=_2d)
    if remote_meta is None:
        if weights_local:
            if not silent: print(f"Remote metadata for {model_title} not found; using existing local weights.")
            return info["weights_path"], local_meta
        print(f"\nModel '{model_title}' not found. For an up-to-date list of available models, run 'easymode list'\n")
        return None, None

    needs_update = is_remote_newer(local_meta, remote_meta)
    if force_download or not weights_local or needs_update:
        if not silent:
            if not weights_local: print(f"\nThe required network weights for {model_title} are not available in the local cache.")
            elif needs_update: print(f"\nNew version available for {model_title}, updating...")
            else: print(f"\nForce downloading {model_title}...")
        return download_model_files(info, remote_metadata=remote_meta, silent=silent)

    return info["weights_path"], local_meta or remote_meta


def load_model_weights(weights_path):
    import tensorflow as tf
    if "n2n" in os.path.basename(weights_path):
        from easymode.n2n.model import create
        dummy_input = tf.zeros((1, 160, 160, 160, 1))
    elif "ddw" in os.path.basename(weights_path):
        from easymode.ddw.model import create
        dummy_input = tf.zeros((1, 160, 160, 160, 1))
    elif "tilt" in os.path.basename(weights_path):
        from easymode.tiltfilter.model import create
        dummy_input = [tf.zeros((1, 256, 256, 1)), tf.zeros((1, 256, 256, 1))]
    else:
        # Check file size to determine architecture version
        # Large models (>400 MB) use old architecture, smaller models use new architecture
        file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

        if file_size_mb > 400:
            print('importing segmentation model with default architecture')
            from easymode.segmentation.model import create
        else:
            print('importing segmentation model with lightweight architecture')
            from easymode.segmentation.model_lite import create

        dummy_input = tf.zeros((1, 160, 160, 160, 1))

    model = create()
    _ = model(dummy_input)
    model.load_weights(weights_path)
    return model


def load_model(local_path):
    return load_model_weights(local_path)


def clear_model_cache(model_title=None):
    if model_title:
        for _2d in (False, True):
            info = get_model_info(model_title, _2d=_2d)
            for p in (info["weights_path"], info["metadata_path"]):
                if os.path.exists(p):
                    os.remove(p)
                    print(f"Removed {p}")
    else:
        import shutil
        if os.path.exists(MODEL_CACHE_DIR):
            shutil.rmtree(MODEL_CACHE_DIR)
            print(f"Cleared model cache: {MODEL_CACHE_DIR}")


def list_remote_models():
    """List features and whether they have 3d, 2d, or both models."""
    if not is_online():
        print("Cannot list remote models: No internet connection")
        return []

    try:
        api = HfApi()
        repo_files = api.list_repo_files(HF_REPO_ID)

        # collect bases from h5 (3d)
        h5_bases = {
            os.path.splitext(os.path.basename(f))[0]
            for f in repo_files
            if f.endswith(".h5") and "ddw" not in f and "n2n" not in f and not "tilt" in f
        }

        # collect bases from scnm (2d)
        scnm_bases = {
            os.path.splitext(os.path.basename(f))[0]
            for f in repo_files
            if f.endswith(".scnm")
        }

        # union of all bases
        all_bases = sorted(h5_bases | scnm_bases)

        print("\neasymode can currently segment the following features:\n")
        models = []

        for base in all_bases:
            has_3d = base in h5_bases
            has_2d = base in scnm_bases

            if has_3d and has_2d:
                dim = "3D/2D"
            elif has_3d:
                dim = "3D only"
            else:
                dim = "2D only (--2d)"

            print(f"   > {base.ljust(30)} {dim}")

            models.append({
                "title": base,
                "dim": dim,
                "has_3d": has_3d,
                "has_2d": has_2d,
            })

        print()
        return models

    except Exception as e:
        print(f"Error listing remote models: {e}")
        return []