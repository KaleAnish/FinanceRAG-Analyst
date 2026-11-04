import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class StorageConfig:
    backend: str
    local_root: str = "data/storage"
    r2_bucket: str | None = None
    r2_endpoint_url: str | None = None
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_region: str = "auto"

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "StorageConfig":
        load_dotenv(env_file)

        backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()

        if backend not in {"local", "r2"}:
            raise ValueError("STORAGE_BACKEND must be 'local' or 'r2'")

        config = cls(
            backend=backend,
            local_root=os.getenv("STORAGE_LOCAL_ROOT", "data/storage"),
            r2_bucket=os.getenv("R2_BUCKET"),
            r2_endpoint_url=os.getenv("R2_ENDPOINT_URL"),
            r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            r2_region=os.getenv("R2_REGION", "auto"),
        )

        if backend == "r2":
            required = {
                "R2_BUCKET": config.r2_bucket,
                "R2_ENDPOINT_URL": config.r2_endpoint_url,
                "R2_ACCESS_KEY_ID": config.r2_access_key_id,
                "R2_SECRET_ACCESS_KEY": config.r2_secret_access_key,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    f"Missing R2 environment variables: {', '.join(missing)}"
                )

        return config
