import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from finance_rag import api as api_module
from finance_rag.rate_limit import SlidingWindowLimiter
from finance_rag.runtime_remote import ensure_runtime, publish_runtime


class FakeS3:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key):
        self.objects[key] = open(path, "rb").read()

    def download_file(self, bucket, key, path):
        if key not in self.objects:
            raise FileNotFoundError(key)
        open(path, "wb").write(self.objects[key])


class FakeStorage:
    bucket = "b"

    def __init__(self):
        self.client = FakeS3()


def _build(tmp_path):
    runtime = tmp_path / "build" / "runtime"
    runtime.mkdir(parents=True)
    files = {}
    for name, payload in {"a.npy": b"x" * 100, "b.parquet": b"y" * 50}.items():
        (runtime / name).write_bytes(payload)
        files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    (tmp_path / "build" / "build_manifest.json").write_text(
        json.dumps({"runtime_version": "rt-test", "files": files}))
    return tmp_path / "build"


def test_publish_then_fetch_round_trip(tmp_path):
    storage = FakeStorage()
    publish_runtime(_build(tmp_path), "runtime/rt-test", storage=storage)
    assert "runtime/rt-test/build_manifest.json" in storage.client.objects

    target = tmp_path / "server_runtime"
    manifest = ensure_runtime("runtime/rt-test", target, storage=storage)
    assert manifest["runtime_version"] == "rt-test"
    assert (target / "a.npy").read_bytes() == b"x" * 100

    # second start-up: files already verified, nothing re-downloaded
    storage.client.objects["runtime/rt-test/a.npy"] = b"tampered"
    ensure_runtime("runtime/rt-test", target, storage=storage)
    assert (target / "a.npy").read_bytes() == b"x" * 100


def test_fetch_rejects_corrupted_download(tmp_path):
    storage = FakeStorage()
    publish_runtime(_build(tmp_path), "runtime/rt-test", storage=storage)
    storage.client.objects["runtime/rt-test/b.parquet"] = b"corrupt"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ensure_runtime("runtime/rt-test", tmp_path / "srv", storage=storage)
    assert not (tmp_path / "srv" / "b.parquet").exists()


def test_publish_refuses_stale_files(tmp_path):
    build = _build(tmp_path)
    (build / "runtime" / "a.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        publish_runtime(build, "runtime/x", storage=FakeStorage())


def test_sliding_window_limiter():
    now = [0.0]
    limiter = SlidingWindowLimiter(per_minute=2, per_day=3, clock=lambda: now[0])
    assert limiter.check("ip") == 0
    assert limiter.check("ip") == 0
    assert limiter.check("ip") > 0          # 3rd within a minute blocked
    assert limiter.check("other") == 0      # other clients unaffected
    now[0] = 61
    assert limiter.check("ip") == 0          # minute window passed
    now[0] = 130
    assert limiter.check("ip") > 0           # daily cap of 3 reached


class _Retriever:
    def search(self, query, **kwargs):
        import pandas as pd
        return pd.DataFrame([{"chunk_id": "c1", "ticker": "AAPL", "rank": 1, "text": "t"}])


class _Response:
    def to_dict(self):
        return {"answer": "ok [S1]"}


class _RateLimitedService:
    retriever = _Retriever()

    def search(self, *a, **k):
        class GenerationRateLimitError(RuntimeError):
            pass
        raise GenerationRateLimitError("minute")


class _OkService:
    retriever = _Retriever()

    def search(self, *a, **k):
        return _Response()


def test_provider_rate_limit_maps_to_429():
    app = api_module.create_app(service_factory=lambda project_dir: _RateLimitedService())
    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"query": "q"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"


def test_ask_rate_limit_per_client(monkeypatch):
    monkeypatch.setenv("ASK_LIMIT_PER_MINUTE", "2")
    app = api_module.create_app(service_factory=lambda project_dir: _OkService())
    with TestClient(app) as client:
        codes = [client.post("/v1/ask", json={"query": "q"}).status_code for _ in range(3)]
        other = client.post("/v1/ask", json={"query": "q"}, headers={"X-Forwarded-For": "9.9.9.9"})
    assert codes == [200, 200, 429]
    assert other.status_code == 200


def test_cors_allows_configured_origin(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://financerag.vercel.app")
    app = api_module.create_app(service_factory=lambda project_dir: _OkService())
    with TestClient(app) as client:
        ok = client.options("/v1/ask", headers={
            "Origin": "https://financerag.vercel.app", "Access-Control-Request-Method": "POST"})
        bad = client.options("/v1/ask", headers={
            "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "https://financerag.vercel.app"
    assert "access-control-allow-origin" not in bad.headers
