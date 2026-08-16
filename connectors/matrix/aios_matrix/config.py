from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class MatrixConfig(BaseSettings):
    """Infrastructure configuration shared with the Synapse registration."""

    model_config = SettingsConfigDict(env_prefix="MATRIX_", populate_by_name=True)

    hs_url: str
    server_name: str
    as_token: str
    hs_token: str
    sender_localpart: str = "_aios"
    user_namespace_regex: str
    listen_addr: str = "0.0.0.0:29328"
    database_url: str

    @property
    def listen(self) -> tuple[str, int]:
        host, separator, port = self.listen_addr.rpartition(":")
        if not separator or not host:
            raise ValueError("MATRIX_LISTEN_ADDR must have the form host:port")
        return host, int(port)
