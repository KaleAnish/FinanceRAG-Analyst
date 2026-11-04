import pytest

from finance_rag.storage import LocalStorage, StorageConfig, create_storage


def test_local_storage_round_trip(tmp_path):
    storage = LocalStorage(tmp_path)
    key = "raw/sec/0000000001/test-filing.txt"
    payload = b"FinanceRAG local storage test"

    storage.put_bytes(key, payload, content_type="text/plain")

    assert storage.exists(key)
    assert storage.get_bytes(key) == payload
    assert storage.list_keys("raw/sec") == [key]

    storage.delete(key)

    assert not storage.exists(key)
    assert storage.list_keys("raw/sec") == []


def test_local_storage_rejects_path_traversal(tmp_path):
    storage = LocalStorage(tmp_path)

    with pytest.raises(ValueError):
        storage.put_bytes("../outside.txt", b"unsafe")


def test_storage_config_defaults_to_local(monkeypatch, tmp_path):
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    monkeypatch.setenv("STORAGE_LOCAL_ROOT", str(tmp_path))

    config = StorageConfig.from_env(env_file="/nonexistent/.env")

    assert config.backend == "local"
    assert config.local_root == str(tmp_path)


def test_factory_creates_local_backend(tmp_path):
    config = StorageConfig(
        backend="local",
        local_root=str(tmp_path),
    )

    storage = create_storage(config)

    assert isinstance(storage, LocalStorage)
