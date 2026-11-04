from .config import StorageConfig
from .local import LocalStorage
from .r2 import R2Storage


def create_storage(config: StorageConfig | None = None):
    config = config or StorageConfig.from_env()

    if config.backend == "local":
        return LocalStorage(root=config.local_root)

    if config.backend == "r2":
        return R2Storage(
            bucket=config.r2_bucket,
            endpoint_url=config.r2_endpoint_url,
            access_key_id=config.r2_access_key_id,
            secret_access_key=config.r2_secret_access_key,
            region=config.r2_region,
        )

    raise ValueError(f"Unsupported storage backend: {config.backend}")
