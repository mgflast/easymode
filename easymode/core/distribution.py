import os
import requests
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from easymode.core.model import create
import tensorflow as tf
import easymode.core.config as cfg

# Configuration
HF_REPO_ID = "mgflast/easymode"  # Single repo for all models
MODEL_CACHE_DIR = cfg.settings["MODEL_DIRECTORY"]
VERSION_FILE = "model_info.json"


def get_model_info(model_title):
    """Get model repository and filename info."""
    filename = f"{model_title}_3d.h5"

    return {
        'repo_id': HF_REPO_ID,
        'filename': filename,
        'local_path': os.path.join(MODEL_CACHE_DIR, filename),
        'version_path': os.path.join(MODEL_CACHE_DIR, f"{model_title}_3d_info.json")
    }


def is_online():
    """Check if internet connection is available."""
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_remote_version(repo_id):
    """Get latest version info from Hugging Face."""
    try:
        api = HfApi()
        repo_info = api.repo_info(repo_id)

        # Use last modified time as version identifier
        last_modified = repo_info.last_modified
        return {
            'version': last_modified.isoformat() if last_modified else "unknown",
            'commit_hash': repo_info.sha[:8] if repo_info.sha else "unknown"
        }
    except Exception as e:
        print(f"Warning: Could not get remote version info: {e}")
        return None


def get_local_version(version_path):
    """Get local version info."""
    if not os.path.exists(version_path):
        return None

    try:
        with open(version_path, 'r') as f:
            return json.load(f)
    except:
        return None


def save_version_info(version_path, version_info):
    """Save version info locally."""
    os.makedirs(os.path.dirname(version_path), exist_ok=True)
    with open(version_path, 'w') as f:
        json.dump(version_info, f, indent=2)


def download_model(repo_id, filename, local_path, version_path):
    """Download model from Hugging Face."""
    print(f"Downloading {repo_id}/{filename}...")

    try:
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=MODEL_CACHE_DIR,
            local_dir=os.path.dirname(local_path)
        )

        # Get and save version info
        remote_version = get_remote_version(repo_id)
        if remote_version:
            save_version_info(version_path, remote_version)

        print(f"\nDownloaded successfully to {local_path}")
        return local_path

    except Exception as e:
        raise RuntimeError(f"Failed to download {repo_id}: {e}")


def load_model_weights(weights_path):
    model = create()
    dummy_input = tf.zeros((1, 160, 160, 160, 1))
    _ = model(dummy_input)
    model.load_weights(weights_path)
    return model

def cache_model(model_title, force_download=False, silent=False):
    info = get_model_info(model_title)
    online = is_online()

    # Check if local file exists
    local_exists = os.path.exists(info['local_path'])

    if force_download or not local_exists:
        # Need to download
        if not online:
            if local_exists:
                print("Local model found. There may be updates available, but we cannot check without an internet connection.")
            else:
                print(f"The required network weights are not available in the local cache {MODEL_CACHE_DIR} and there is no internet connection available to download them - aborting...")
                exit()
        else:
            print(f"The required network weights for {model_title} are not available in the local cache. Downloading now...")
            download_model(info['repo_id'], info['filename'],
                           info['local_path'], info['version_path'])

    elif local_exists and online:
        # Check if we need to update
        local_version = get_local_version(info['version_path'])
        remote_version = get_remote_version(info['repo_id'])

        if remote_version and local_version:
            if remote_version['version'] != local_version['version']:
                if not silent:
                    print(f"New version available for {model_title}, updating...")
                download_model(info['repo_id'], info['filename'], info['local_path'], info['version_path'])
        elif remote_version and not local_version:
            if not silent:
                print("No local version info, checking for updates...")
            save_version_info(info['version_path'], remote_version)

    return info['local_path']

def load_model(local_path):
    return load_model_weights(local_path)

def clear_model_cache(model_title=None):
    """Clear local model cache."""
    if model_title:
        # Clear specific model
        info = get_model_info(model_title)
        files_to_remove = [info['local_path'], info['version_path']]

        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed {file_path}")
    else:
        # Clear all models
        import shutil
        if os.path.exists(MODEL_CACHE_DIR):
            shutil.rmtree(MODEL_CACHE_DIR)
            print(f"Cleared model cache: {MODEL_CACHE_DIR}")

def list_remote_models():
    """List all available models in the Hugging Face repository."""
    if not is_online():
        print("Cannot list remote models: No internet connection")
        return []

    try:
        api = HfApi()
        repo_files = api.list_repo_files(HF_REPO_ID)

        # Filter for .h5 model files
        model_files = [f for f in repo_files if f.endswith('.h5')]

        if not model_files:
            print("No model files found in repository")
            return []

        print()
        print(f"Easymode can currently segment the following features:")
        print()
        models = []

        for model_file in sorted(model_files):
            model_name = model_file.replace('.h5', '')
            if '_' in model_name:
                title, dim = model_name.rsplit('_', 1)
                models.append({'title': title, 'filename': model_file})

                # Check if we have it locally
                local_path = os.path.join(MODEL_CACHE_DIR, model_file)
                local_status = "weights local" if os.path.exists(local_path) else "weights available for download"

                print(f"   > {title} - [{local_status}]")
            else:
                pass

        return models

    except Exception as e:
        print(f"Error listing remote models: {e}")
        return []
