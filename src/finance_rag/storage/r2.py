from botocore.exceptions import ClientError

from .base import StorageBackend


class R2Storage:
    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        region: str = "auto",
        client=None,
    ):
        if not all([bucket, endpoint_url, access_key_id, secret_access_key]):
            raise ValueError("R2 bucket, endpoint and credentials are required")

        self.bucket = bucket

        if client is None:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                region_name=region,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            )

        self.client = client

    def put_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")

        kwargs = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": data,
        }

        if content_type:
            kwargs["ContentType"] = content_type

        self.client.put_object(**kwargs)

    def get_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"]

        try:
            return body.read()
        finally:
            body.close()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = exc.response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            )
            if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                return False
            raise

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def list_keys(self, prefix: str = "") -> list[str]:
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
        ):
            keys.extend(
                item["Key"]
                for item in page.get("Contents", [])
            )

        return sorted(keys)
