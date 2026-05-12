from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="FileUploadResponse")


@_attrs_define
class FileUploadResponse:
    """Wire shape returned from ``POST /v1/sessions/<id>/files``.

    ``file_id`` rather than ``id`` matches the #324 contract so callers can
    reference the file in future inbound attachment payloads without rebinding.

        Attributes:
            file_id (str):
            in_sandbox_path (str):
            filename (str):
            size (int):
            content_type (str):
            sha256 (str):
    """

    file_id: str
    in_sandbox_path: str
    filename: str
    size: int
    content_type: str
    sha256: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        file_id = self.file_id

        in_sandbox_path = self.in_sandbox_path

        filename = self.filename

        size = self.size

        content_type = self.content_type

        sha256 = self.sha256

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "file_id": file_id,
                "in_sandbox_path": in_sandbox_path,
                "filename": filename,
                "size": size,
                "content_type": content_type,
                "sha256": sha256,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        file_id = d.pop("file_id")

        in_sandbox_path = d.pop("in_sandbox_path")

        filename = d.pop("filename")

        size = d.pop("size")

        content_type = d.pop("content_type")

        sha256 = d.pop("sha256")

        file_upload_response = cls(
            file_id=file_id,
            in_sandbox_path=in_sandbox_path,
            filename=filename,
            size=size,
            content_type=content_type,
            sha256=sha256,
        )

        file_upload_response.additional_properties = d
        return file_upload_response

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
