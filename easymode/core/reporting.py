import requests
import uuid
import json
import os
from datetime import datetime, timezone
from tqdm import tqdm

WORKER_URL = "https://easymode-reporting.easymode.workers.dev"


def get_presigned_url(filename, size, content_type):
    resp = requests.get(WORKER_URL, params={"filename": filename, "size": str(size), "contentType": content_type})

    # Handle rate limiting (429 Too Many Requests)
    if resp.status_code == 429:
        try:
            data = resp.json()
            message = data.get("message", "Rate limit exceeded. Please try again later.")
            print(f"Upload failed: {message}")
        except Exception:
            print("Upload rate limit exceeded. Please try again after 5 minutes.")
        exit(1)

    # If the quota is exhausted, the worker returns status 507 (Insufficient
    # Storage). Exit early with an informative message.
    if resp.status_code == 507:
        print(
            "The easymode reporting bucket is currently at capacity. Sorry! Please try again later."
        )
        exit(1)

    # Handle file too large errors
    if resp.status_code == 413:
        print(f"File is too large. Maximum allowed size is currently 2.0 GB. Please contact me if you would like to submit larger files: mlast@mrc-lmb.cam.ac.uk")
        exit(1)

    # Handle invalid file type errors
    if resp.status_code == 415:
        print(f"Invalid file type. Only .mrc files are accepted.")
        exit(1)

    # Raise an exception for other HTTP errors (4xx, 5xx) so callers can
    # investigate. Success responses return JSON with presigned URL info.
    resp.raise_for_status()
    return resp.json()


def put(url, data, content_type, show_progress=False):
    headers = {"Content-Type": content_type}

    if show_progress and len(data) > 1024 * 1024:
        from io import BytesIO
        bio = BytesIO(data)
        headers["Content-Length"] = str(len(data))
        with tqdm.wrapattr(bio, "read", total=len(data), unit='B', unit_scale=True, unit_divisor=1024) as wrapped:
            resp = requests.put(url, data=wrapped, headers=headers)
    else:
        resp = requests.put(url, data=data, headers=headers)

    resp.raise_for_status()


def report(volume_path, model, contact, comment):
    vol = os.path.abspath(volume_path)
    if not os.path.isfile(vol):
        print(f"File '{vol}' could not be found.")
        exit()

    ext = os.path.splitext(vol)[1].lower()
    if ext != ".mrc":
        print(f"The volume must be an .mrc file (got: '{ext}').")
        exit()

    base = f"{datetime.now(timezone.utc).strftime('%y%m%d_%H%M')}_{uuid.uuid4().hex[:8]}"

    print(f'\033[96mReading .mrc...\033[0m')
    with open(vol, 'rb') as f:
        vol_bytes = f.read()

    size_mb = len(vol_bytes) / (1024 * 1024)
    print(f'\033[96mUploading volume ({size_mb:.1f} MB)...\033[0m')
    presigned = get_presigned_url(f"{base}.mrc", len(vol_bytes), "application/octet-stream")
    put(presigned["url"], vol_bytes, "application/octet-stream", show_progress=True)

    print(f'\033[96mUploading metadata...\033[0m')
    metadata = {
        "date": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "contact": contact,
        "comment": comment,
    }

    metadata_bytes = json.dumps(metadata).encode("utf-8")
    presigned_json = get_presigned_url(f"{base}.json", len(metadata_bytes), "application/json")
    put(presigned_json["url"], metadata_bytes, "application/json")

    print()
    print(f'\033[92m✓ Report submitted successfully!\033[0m')
    print(f'\033[96mThank you for helping to improve easymode!\033[0m')
    print()
