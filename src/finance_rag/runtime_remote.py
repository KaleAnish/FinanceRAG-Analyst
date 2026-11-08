"""Publish a runtime bundle to R2 and fetch it at server start-up.

Layout in the bucket::

    <prefix>/build_manifest.json      (file list + SHA-256 for every runtime file)
    <prefix>/<runtime file>           (chunks_df.parquet, chunk_embeddings.npy, ...)

Every downloaded file is verified against the manifest before it is used, and
already-present files with a matching hash are not downloaded again.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger("finance_rag.runtime_remote")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _storage():
    from .storage import create_storage
    storage = create_storage()
    if not hasattr(storage, "client"):
        raise ValueError("Remote runtime requires STORAGE_BACKEND=r2")
    return storage


def publish_runtime(build_dir: Path, prefix: str, storage=None) -> dict:
    build_dir = Path(build_dir)
    manifest_path = build_dir / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    storage = storage or _storage()
    prefix = prefix.strip("/")

    for name, info in manifest["files"].items():
        path = build_dir / "runtime" / name
        if _sha256(path) != info["sha256"]:
            raise ValueError(f"{name} does not match build_manifest.json; rebuild before publishing")
        log.info("uploading %s (%.1f MB)", name, info["bytes"] / 1e6)
        storage.client.upload_file(str(path), storage.bucket, f"{prefix}/{name}")
    # Manifest last: a reader never sees a manifest whose files are missing.
    storage.client.upload_file(str(manifest_path), storage.bucket, f"{prefix}/build_manifest.json")
    return manifest


def ensure_runtime(prefix: str, runtime_dir: Path, storage=None) -> dict:
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    storage = storage or _storage()
    prefix = prefix.strip("/")

    manifest_path = runtime_dir / "build_manifest.json"
    storage.client.download_file(storage.bucket, f"{prefix}/build_manifest.json", str(manifest_path))
    manifest = json.loads(manifest_path.read_text())

    for name, info in manifest["files"].items():
        target = runtime_dir / name
        if target.is_file() and target.stat().st_size == info["bytes"] and _sha256(target) == info["sha256"]:
            continue
        log.info("downloading %s (%.1f MB)", name, info["bytes"] / 1e6)
        tmp = target.with_suffix(target.suffix + ".part")
        storage.client.download_file(storage.bucket, f"{prefix}/{name}", str(tmp))
        if _sha256(tmp) != info["sha256"]:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {name}")
        tmp.replace(target)
    log.info("runtime %s ready in %s", manifest.get("runtime_version"), runtime_dir)
    return manifest
