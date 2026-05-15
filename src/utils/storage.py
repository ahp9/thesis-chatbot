from pathlib import Path
from typing import Any

import aiofiles
from chainlit.data.storage_clients.base import BaseStorageClient


class LocalStorageClient(BaseStorageClient):
    """Local filesystem storage provider for Chainlit uploaded elements."""

    def __init__(self, root: str | Path = "./uploaded_files"):
        """Initialise with the directory that will hold uploaded files."""
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, object_key: str) -> Path:
        """Resolve an object key to a safe local file path, creating parent dirs."""
        safe_key = object_key.lstrip("/").replace("..", "_")
        path = self.root / safe_key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def close(self):
        """No-op: nothing to close for local storage."""
        pass

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> dict[str, Any]:
        """Write data to disk and return its local URI and object key."""
        path = self._path_for_key(object_key)
        if path.exists() and not overwrite:
            return {"url": path.resolve().as_uri(), "object_key": object_key}

        write_data = data.encode("utf-8") if isinstance(data, str) else data

        async with aiofiles.open(path, "wb") as f:
            await f.write(write_data)

        return {
            "url": path.resolve().as_uri(),
            "object_key": object_key,
            "content_disposition": content_disposition,
        }

    async def delete_file(self, object_key: str):
        """Delete the file for the given object key if it exists."""
        path = self._path_for_key(object_key)
        if path.exists():
            path.unlink()

    async def get_read_url(self, object_key: str):
        """Return the local URI for the file, or raise FileNotFoundError."""
        path = self._path_for_key(object_key)
        if not path.exists():
            raise FileNotFoundError(f"Stored file not found for object_key={object_key}")
        return path.resolve().as_uri()
