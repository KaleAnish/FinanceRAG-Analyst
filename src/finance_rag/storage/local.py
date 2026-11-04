from pathlib import Path, PurePosixPath

from .base import StorageBackend


def _validate_key(key: str) -> str:
    normalized = key.replace("\\", "/")
    path = PurePosixPath(normalized)

    if not normalized or path.is_absolute():
        raise ValueError("Storage key must be a non-empty relative path")

    if any(part in ("", ".", "..") for part in normalized.split("/")):
        raise ValueError("Storage key contains an invalid path component")

    return path.as_posix()


class LocalStorage:
    def __init__(self, root: str | Path = "data/storage"):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe_key = _validate_key(key)
        destination = (self.root / safe_key).resolve()

        if not destination.is_relative_to(self.root):
            raise ValueError("Storage key escapes the configured storage root")

        return destination

    def put_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")

        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_file():
            path.unlink()

    def list_keys(self, prefix: str = "") -> list[str]:
        if prefix:
            prefix = _validate_key(prefix).rstrip("/")
            search_root = self._path(prefix)
            if search_root.is_file():
                return [prefix]
            if not search_root.exists():
                search_root = self.root / prefix
        else:
            search_root = self.root

        if not search_root.exists():
            return []

        if search_root.is_file():
            return [search_root.relative_to(self.root).as_posix()]

        return sorted(
            path.relative_to(self.root).as_posix()
            for path in search_root.rglob("*")
            if path.is_file()
        )
