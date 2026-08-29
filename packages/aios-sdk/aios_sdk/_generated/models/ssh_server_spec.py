from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.ssh_permission_policy import SshPermissionPolicy


T = TypeVar("T", bound="SshServerSpec")


@_attrs_define
class SshServerSpec:
    """One entry in an agent's ``ssh_servers`` list.

    Declares a remote host the agent can run shell commands on via the ``ssh``
    built-in tool. ``credential`` names the ``ssh_key`` vault credential (its
    ``secret_name``) resolved at call time from the session's bound vaults — the
    private key is loaded only in worker memory and never enters the sandbox or
    the model context. ``host_keys`` is a REQUIRED non-empty pin-set of public
    host-key lines the server must present (no trust-on-first-use, no store, no
    insecure mode): a server key outside the set aborts the connection.

    There is NO command grammar: the grant is per-server whole-shell (OpenSSH
    hands the remote login shell a single command string, which it re-parses, so
    an aios-side allowlist could never equal its own effect). Restrict the blast
    radius server-side — ``authorized_keys`` forced commands / ``restrict`` /
    dedicated low-privilege users — and/or gate every command with
    ``permission_policy: {type: always_ask}``.

        Attributes:
            name (str):
            host (str):
            username (str):
            host_keys (list[str]):
            credential (str):
            port (int | Unset):  Default: 22.
            description (None | str | Unset):
            permission_policy (None | SshPermissionPolicy | Unset):
            suppress (bool | None | Unset):
            enabled (bool | Unset):  Default: True.
    """

    name: str
    host: str
    username: str
    host_keys: list[str]
    credential: str
    port: int | Unset = 22
    description: None | str | Unset = UNSET
    permission_policy: None | SshPermissionPolicy | Unset = UNSET
    suppress: bool | None | Unset = UNSET
    enabled: bool | Unset = True

    def to_dict(self) -> dict[str, Any]:
        from ..models.ssh_permission_policy import SshPermissionPolicy

        name = self.name

        host = self.host

        username = self.username

        host_keys = self.host_keys

        credential = self.credential

        port = self.port

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        permission_policy: dict[str, Any] | None | Unset
        if isinstance(self.permission_policy, Unset):
            permission_policy = UNSET
        elif isinstance(self.permission_policy, SshPermissionPolicy):
            permission_policy = self.permission_policy.to_dict()
        else:
            permission_policy = self.permission_policy

        suppress: bool | None | Unset
        if isinstance(self.suppress, Unset):
            suppress = UNSET
        else:
            suppress = self.suppress

        enabled = self.enabled

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
                "host": host,
                "username": username,
                "host_keys": host_keys,
                "credential": credential,
            }
        )
        if port is not UNSET:
            field_dict["port"] = port
        if description is not UNSET:
            field_dict["description"] = description
        if permission_policy is not UNSET:
            field_dict["permission_policy"] = permission_policy
        if suppress is not UNSET:
            field_dict["suppress"] = suppress
        if enabled is not UNSET:
            field_dict["enabled"] = enabled

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.ssh_permission_policy import SshPermissionPolicy

        d = dict(src_dict)
        name = d.pop("name")

        host = d.pop("host")

        username = d.pop("username")

        host_keys = cast(list[str], d.pop("host_keys"))

        credential = d.pop("credential")

        port = d.pop("port", UNSET)

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        def _parse_permission_policy(
            data: object,
        ) -> None | SshPermissionPolicy | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                permission_policy_type_0 = SshPermissionPolicy.from_dict(data)

                return permission_policy_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | SshPermissionPolicy | Unset, data)

        permission_policy = _parse_permission_policy(d.pop("permission_policy", UNSET))

        def _parse_suppress(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        suppress = _parse_suppress(d.pop("suppress", UNSET))

        enabled = d.pop("enabled", UNSET)

        ssh_server_spec = cls(
            name=name,
            host=host,
            username=username,
            host_keys=host_keys,
            credential=credential,
            port=port,
            description=description,
            permission_policy=permission_policy,
            suppress=suppress,
            enabled=enabled,
        )

        return ssh_server_spec
