"""
ReActor V3 model bootstrap helpers.

Downloads required models on first actual swap attempt and stores them in
shared WebUI model directories.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, List

from modules.util import load_file_from_url


_BOOTSTRAP_LOCK = threading.Lock()
_LAST_ATTEMPT_TS = 0.0
_RETRY_COOLDOWN_SECONDS = 60.0


def _get_required_models() -> List[Dict[str, object]]:
    """Return required model metadata with optional env var URL overrides."""
    gpen_512_override = os.environ.get("REACTOR_V3_GPEN512_URL")
    gpen_512_urls = [
        "https://huggingface.co/yangxy/GPEN/resolve/main/GPEN-BFR-512.onnx",
        "https://hf-mirror.com/yangxy/GPEN/resolve/main/GPEN-BFR-512.onnx",
    ]
    if gpen_512_override:
        gpen_512_urls = [gpen_512_override] + gpen_512_urls

    inswapper_override = os.environ.get("REACTOR_V3_INSWAPPER_URL")
    inswapper_urls = [
        # Common community mirrors on Hugging Face.
        "https://huggingface.co/facefusion/facefusion-assets/resolve/main/models/inswapper_128.onnx",
        "https://huggingface.co/JavaFXpert/inswapper/resolve/main/inswapper_128.onnx",
        "https://hf-mirror.com/facefusion/facefusion-assets/resolve/main/models/inswapper_128.onnx",
        "https://hf-mirror.com/JavaFXpert/inswapper/resolve/main/inswapper_128.onnx",
    ]
    if inswapper_override:
        inswapper_urls = [inswapper_override] + inswapper_urls

    return [
        {
            "file_name": "GPEN-BFR-512.onnx",
            "relative_dir": "facerestore_models",
            "urls": gpen_512_urls,
        },
        {
            "file_name": "inswapper_128.onnx",
            "relative_dir": "insightface",
            "urls": inswapper_urls,
        },
    ]


def _download_one(file_name: str, target_dir: str, urls: List[str]) -> bool:
    """Try candidate URLs until one succeeds."""
    last_error = None

    for url in urls:
        try:
            print(f"[ReActor V3] Downloading {file_name} from: {url}")
            load_file_from_url(
                url,
                model_dir=target_dir,
                file_name=file_name,
                progress=True,
            )
            print(f"[ReActor V3] Download complete: {file_name}")
            return True
        except Exception as exc:
            last_error = exc
            print(f"[ReActor V3] Download failed from this URL: {exc}")

    print(f"[ReActor V3] WARNING: Could not download {file_name}: {last_error}")
    return False


def ensure_reactor_v3_models(models_root: str) -> Dict[str, List[str]]:
    """
    Ensure required ReActor V3 models exist under shared WebUI models path.

    This function is thread-safe and rate-limited to avoid repeated network
    attempts when users generate many images in one batch.
    """
    global _LAST_ATTEMPT_TS

    required = _get_required_models()
    existing: List[str] = []
    downloaded: List[str] = []
    failed: List[str] = []

    for model in required:
        target_dir = os.path.join(models_root, model["relative_dir"])
        target_file = os.path.join(target_dir, model["file_name"])
        if os.path.exists(target_file):
            existing.append(model["file_name"])

    missing_names = [
        model["file_name"]
        for model in required
        if model["file_name"] not in existing
    ]
    if not missing_names:
        return {"existing": existing, "downloaded": downloaded, "failed": failed}

    with _BOOTSTRAP_LOCK:
        # Re-check under lock to avoid duplicate downloads in concurrent calls.
        existing = []
        for model in required:
            target_dir = os.path.join(models_root, model["relative_dir"])
            target_file = os.path.join(target_dir, model["file_name"])
            if os.path.exists(target_file):
                existing.append(model["file_name"])

        missing = [
            model
            for model in required
            if model["file_name"] not in existing
        ]
        if not missing:
            return {"existing": existing, "downloaded": downloaded, "failed": failed}

        now = time.time()
        if now - _LAST_ATTEMPT_TS < _RETRY_COOLDOWN_SECONDS:
            failed = [model["file_name"] for model in missing]
            print(
                "[ReActor V3] Skipping repeated model download attempt "
                f"(cooldown {_RETRY_COOLDOWN_SECONDS:.0f}s)."
            )
            return {"existing": existing, "downloaded": downloaded, "failed": failed}

        _LAST_ATTEMPT_TS = now
        print("[ReActor V3] Checking required first-run models...")

        for model in missing:
            file_name = model["file_name"]
            target_dir = os.path.join(models_root, model["relative_dir"])
            os.makedirs(target_dir, exist_ok=True)
            target_file = os.path.join(target_dir, file_name)

            if os.path.exists(target_file):
                existing.append(file_name)
                continue

            ok = _download_one(file_name, target_dir, list(model["urls"]))
            if ok and os.path.exists(target_file):
                downloaded.append(file_name)
            else:
                failed.append(file_name)

    return {"existing": existing, "downloaded": downloaded, "failed": failed}
