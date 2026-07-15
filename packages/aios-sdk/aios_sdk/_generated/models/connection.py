from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.actor import Actor
    from ..models.allow_all import AllowAll
    from ..models.allow_list import AllowList
    from ..models.allow_senders import AllowSenders
    from ..models.connection_metadata import ConnectionMetadata
    from ..models.deny_all import DenyAll


T = TypeVar("T", bound="Connection")


@_attrs_define
class Connection:
    """Read view of a connection.

    Mode is implicit in the populated field:

    * ``session_id`` set → single_session
    * ``session_template_id`` set → per_chat
    * neither → detached

    ``session_id`` / ``session_template_id`` / ``attached_at`` are
    projected from the connection's active binding row at read time.
    ``attached_at`` is "when did the active binding land," not "when
    was this connection first attached," so detach+re-attach moves
    it forward — operator dashboards keying off the timestamp see
    that motion.

    Secrets are *write-only* on the operator surface — the model carries
    ``secrets_set: bool`` rather than the values themselves.  The only
    decryption path is the runtime-scoped
    ``GET /v1/connectors/runtime/secrets``, which returns the dict
    for a connection of the caller's connector type.

        Attributes:
            id (str):
            connector (str):
            external_account_id (str):
            metadata (ConnectionMetadata):
            created_at (datetime.datetime):
            updated_at (datetime.datetime):
            session_id (None | str | Unset):
            session_template_id (None | str | Unset):
            secrets_set (bool | Unset):  Default: False.
            created_by (Actor | None | Unset):
            attached_at (datetime.datetime | None | Unset):
            archived_at (datetime.datetime | None | Unset):
            inbound_policy (AllowAll | AllowList | AllowSenders | DenyAll | None | Unset):
            inbound_policy_effective (AllowAll | AllowList | AllowSenders | DenyAll | Unset): Server-derived, read-only echo
                of the *effective* inbound-admission policy: the stored ``inbound_policy`` union member, or the server default
                ``DenyAll`` (fail-closed) when the column is NULL. Lets an operator see both the fail-open (``allow_all``) and
                fail-closed (``deny_all`` / ``allow_list``) posture on create / get / list without a second round-trip. **Never
                an input** — it is rejected on the ``ConnectionCreate`` write model; set the policy via ``PUT
                /v1/connections/{id}/inbound-policy``.
    """

    id: str
    connector: str
    external_account_id: str
    metadata: ConnectionMetadata
    created_at: datetime.datetime
    updated_at: datetime.datetime
    session_id: None | str | Unset = UNSET
    session_template_id: None | str | Unset = UNSET
    secrets_set: bool | Unset = False
    created_by: Actor | None | Unset = UNSET
    attached_at: datetime.datetime | None | Unset = UNSET
    archived_at: datetime.datetime | None | Unset = UNSET
    inbound_policy: AllowAll | AllowList | AllowSenders | DenyAll | None | Unset = UNSET
    inbound_policy_effective: AllowAll | AllowList | AllowSenders | DenyAll | Unset = (
        UNSET
    )
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.actor import Actor
        from ..models.allow_all import AllowAll
        from ..models.allow_list import AllowList
        from ..models.allow_senders import AllowSenders
        from ..models.deny_all import DenyAll

        id = self.id

        connector = self.connector

        external_account_id = self.external_account_id

        metadata = self.metadata.to_dict()

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        session_id: None | str | Unset
        if isinstance(self.session_id, Unset):
            session_id = UNSET
        else:
            session_id = self.session_id

        session_template_id: None | str | Unset
        if isinstance(self.session_template_id, Unset):
            session_template_id = UNSET
        else:
            session_template_id = self.session_template_id

        secrets_set = self.secrets_set

        created_by: dict[str, Any] | None | Unset
        if isinstance(self.created_by, Unset):
            created_by = UNSET
        elif isinstance(self.created_by, Actor):
            created_by = self.created_by.to_dict()
        else:
            created_by = self.created_by

        attached_at: None | str | Unset
        if isinstance(self.attached_at, Unset):
            attached_at = UNSET
        elif isinstance(self.attached_at, datetime.datetime):
            attached_at = self.attached_at.isoformat()
        else:
            attached_at = self.attached_at

        archived_at: None | str | Unset
        if isinstance(self.archived_at, Unset):
            archived_at = UNSET
        elif isinstance(self.archived_at, datetime.datetime):
            archived_at = self.archived_at.isoformat()
        else:
            archived_at = self.archived_at

        inbound_policy: dict[str, Any] | None | Unset
        if isinstance(self.inbound_policy, Unset):
            inbound_policy = UNSET
        elif isinstance(self.inbound_policy, AllowAll):
            inbound_policy = self.inbound_policy.to_dict()
        elif isinstance(self.inbound_policy, AllowList):
            inbound_policy = self.inbound_policy.to_dict()
        elif isinstance(self.inbound_policy, AllowSenders):
            inbound_policy = self.inbound_policy.to_dict()
        elif isinstance(self.inbound_policy, DenyAll):
            inbound_policy = self.inbound_policy.to_dict()
        else:
            inbound_policy = self.inbound_policy

        inbound_policy_effective: dict[str, Any] | Unset
        if isinstance(self.inbound_policy_effective, Unset):
            inbound_policy_effective = UNSET
        elif isinstance(self.inbound_policy_effective, AllowAll):
            inbound_policy_effective = self.inbound_policy_effective.to_dict()
        elif isinstance(self.inbound_policy_effective, AllowList):
            inbound_policy_effective = self.inbound_policy_effective.to_dict()
        elif isinstance(self.inbound_policy_effective, AllowSenders):
            inbound_policy_effective = self.inbound_policy_effective.to_dict()
        else:
            inbound_policy_effective = self.inbound_policy_effective.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "connector": connector,
                "external_account_id": external_account_id,
                "metadata": metadata,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if session_id is not UNSET:
            field_dict["session_id"] = session_id
        if session_template_id is not UNSET:
            field_dict["session_template_id"] = session_template_id
        if secrets_set is not UNSET:
            field_dict["secrets_set"] = secrets_set
        if created_by is not UNSET:
            field_dict["created_by"] = created_by
        if attached_at is not UNSET:
            field_dict["attached_at"] = attached_at
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at
        if inbound_policy is not UNSET:
            field_dict["inbound_policy"] = inbound_policy
        if inbound_policy_effective is not UNSET:
            field_dict["inbound_policy_effective"] = inbound_policy_effective

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.actor import Actor
        from ..models.allow_all import AllowAll
        from ..models.allow_list import AllowList
        from ..models.allow_senders import AllowSenders
        from ..models.connection_metadata import ConnectionMetadata
        from ..models.deny_all import DenyAll

        d = dict(src_dict)
        id = d.pop("id")

        connector = d.pop("connector")

        external_account_id = d.pop("external_account_id")

        metadata = ConnectionMetadata.from_dict(d.pop("metadata"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        def _parse_session_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        session_id = _parse_session_id(d.pop("session_id", UNSET))

        def _parse_session_template_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        session_template_id = _parse_session_template_id(
            d.pop("session_template_id", UNSET)
        )

        secrets_set = d.pop("secrets_set", UNSET)

        def _parse_created_by(data: object) -> Actor | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                created_by_type_0 = Actor.from_dict(data)

                return created_by_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(Actor | None | Unset, data)

        created_by = _parse_created_by(d.pop("created_by", UNSET))

        def _parse_attached_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                attached_at_type_0 = isoparse(data)

                return attached_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        attached_at = _parse_attached_at(d.pop("attached_at", UNSET))

        def _parse_archived_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                archived_at_type_0 = isoparse(data)

                return archived_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        archived_at = _parse_archived_at(d.pop("archived_at", UNSET))

        def _parse_inbound_policy(
            data: object,
        ) -> AllowAll | AllowList | AllowSenders | DenyAll | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                inbound_policy_type_0_type_0 = AllowAll.from_dict(data)

                return inbound_policy_type_0_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                inbound_policy_type_0_type_1 = AllowList.from_dict(data)

                return inbound_policy_type_0_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                inbound_policy_type_0_type_2 = AllowSenders.from_dict(data)

                return inbound_policy_type_0_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                inbound_policy_type_0_type_3 = DenyAll.from_dict(data)

                return inbound_policy_type_0_type_3
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(
                AllowAll | AllowList | AllowSenders | DenyAll | None | Unset, data
            )

        inbound_policy = _parse_inbound_policy(d.pop("inbound_policy", UNSET))

        def _parse_inbound_policy_effective(
            data: object,
        ) -> AllowAll | AllowList | AllowSenders | DenyAll | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                inbound_policy_effective_type_0 = AllowAll.from_dict(data)

                return inbound_policy_effective_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                inbound_policy_effective_type_1 = AllowList.from_dict(data)

                return inbound_policy_effective_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                inbound_policy_effective_type_2 = AllowSenders.from_dict(data)

                return inbound_policy_effective_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            inbound_policy_effective_type_3 = DenyAll.from_dict(data)

            return inbound_policy_effective_type_3

        inbound_policy_effective = _parse_inbound_policy_effective(
            d.pop("inbound_policy_effective", UNSET)
        )

        connection = cls(
            id=id,
            connector=connector,
            external_account_id=external_account_id,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
            session_id=session_id,
            session_template_id=session_template_id,
            secrets_set=secrets_set,
            created_by=created_by,
            attached_at=attached_at,
            archived_at=archived_at,
            inbound_policy=inbound_policy,
            inbound_policy_effective=inbound_policy_effective,
        )

        connection.additional_properties = d
        return connection

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
