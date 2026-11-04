from pathlib import Path
from typing import Protocol


class StorageBackend(Protocol):
    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        ...

    def get_bytes(self, key: str) -> bytes:
        ...

    def exists(self, key: str) -> bool:
        ...

    def delete(self, key: str) -> None:
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        ...
