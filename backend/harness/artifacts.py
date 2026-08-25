"""Object-store-first artifact contracts.

The control plane keeps small, queryable references. Workers write replay
payloads directly to an artifact store so high-volume trajectories never pass
through the API process or SQLite metadata store.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from .models import ArtifactReference


class ArtifactStore(Protocol):
    id: str

    def put_bytes(self, *, kind: str, key: str, payload: bytes, media_type: str) -> ArtifactReference: ...

    def get_bytes(self, reference: ArtifactReference) -> bytes: ...

    def delete(self, reference: ArtifactReference) -> None: ...


class FileArtifactStore:
    """Local development implementation with the same immutable reference contract."""

    id = "file"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, *, kind: str, key: str, payload: bytes, media_type: str) -> ArtifactReference:
        digest = hashlib.sha256(payload).hexdigest()
        target = self.root / kind / digest[:2] / f"{key}-{digest}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(payload)
        return ArtifactReference(kind=kind, uri=target.resolve().as_uri(), content_sha256=digest, size_bytes=len(payload), media_type=media_type)

    def get_bytes(self, reference: ArtifactReference) -> bytes:
        if not reference.uri.startswith("file://"):
            raise ValueError(f"FileArtifactStore cannot read {reference.uri}")
        return Path(reference.uri.removeprefix("file://")).read_bytes()

    def delete(self, reference: ArtifactReference) -> None:
        if not reference.uri.startswith("file://"):
            raise ValueError(f"FileArtifactStore cannot delete {reference.uri}")
        target = Path(reference.uri.removeprefix("file://")).resolve()
        root = self.root.resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Artifact is outside the configured store: {target}")
        target.unlink(missing_ok=True)


class S3ArtifactStore:
    """S3-compatible extension point; inject a client to avoid an SDK dependency."""

    id = "s3"

    def __init__(self, bucket: str, prefix: str = "harness", client: object | None = None) -> None:
        self.bucket, self.prefix, self.client = bucket, prefix.strip("/"), client

    def put_bytes(self, *, kind: str, key: str, payload: bytes, media_type: str) -> ArtifactReference:
        if self.client is None:
            raise RuntimeError("S3ArtifactStore needs an injected S3-compatible client.")
        digest = hashlib.sha256(payload).hexdigest()
        object_key = f"{self.prefix}/{kind}/{digest[:2]}/{key}-{digest}.json"
        self.client.put_object(Bucket=self.bucket, Key=object_key, Body=payload, ContentType=media_type)  # type: ignore[attr-defined]
        return ArtifactReference(kind=kind, uri=f"s3://{self.bucket}/{object_key}", content_sha256=digest, size_bytes=len(payload), media_type=media_type)

    def get_bytes(self, reference: ArtifactReference) -> bytes:
        if self.client is None:
            raise RuntimeError("S3ArtifactStore needs an injected S3-compatible client.")
        bucket, key = reference.uri.removeprefix("s3://").split("/", 1)
        return self.client.get_object(Bucket=bucket, Key=key)["Body"].read()  # type: ignore[attr-defined]

    def delete(self, reference: ArtifactReference) -> None:
        if self.client is None:
            raise RuntimeError("S3ArtifactStore needs an injected S3-compatible client.")
        bucket, key = reference.uri.removeprefix("s3://").split("/", 1)
        self.client.delete_object(Bucket=bucket, Key=key)  # type: ignore[attr-defined]
