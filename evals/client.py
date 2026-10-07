"""A minimal JSON client for the aios operator API (stdlib only).

Reads ``AIOS_URL`` and ``AIOS_API_KEY`` from the environment. The key is an operator
key for the account the eval runs in; it is never printed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol


class Api(Protocol):
    def get(self, path: str, **params: Any) -> Any: ...
    def post(self, path: str, body: Any) -> Any: ...
    def put(self, path: str, body: Any) -> Any: ...


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status


class Client:
    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        self.url = (url or os.environ["AIOS_URL"]).rstrip("/")
        self._key = api_key or os.environ["AIOS_API_KEY"]

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.url + path,
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise ApiError(exc.code, exc.read().decode(errors="replace")) from exc
        return json.loads(raw) if raw else None

    def get(self, path: str, **params: Any) -> Any:
        query = {k: v for k, v in params.items() if v is not None}
        suffix = "?" + urllib.parse.urlencode(query) if query else ""
        return self._call("GET", path + suffix)

    def post(self, path: str, body: Any) -> Any:
        return self._call("POST", path, body)

    def put(self, path: str, body: Any) -> Any:
        return self._call("PUT", path, body)
