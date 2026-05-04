"""Regression for #223 §8: signal-cli's JSON-RPC daemon mode omits the
``file`` field from attachment envelopes.  The connector falls back to
the ``<config_dir>/attachments/<id>`` storage convention so inbound
photos still reach the model as ``image_url`` parts.
"""

from __future__ import annotations

from pathlib import Path

from aios_signal.config import Settings
from aios_signal.connector import SignalConnector
from aios_signal.parse import Attachment, InboundMessage


def _make_connector(config_dir: Path) -> SignalConnector:
    cfg = Settings(
        phones=["+15550001"],
        config_dir=config_dir,
        cli_bin="/usr/bin/signal-cli",
    )
    return SignalConnector(cfg)


def _make_msg(attachments: tuple[Attachment, ...]) -> InboundMessage:
    return InboundMessage(
        chat_type="dm",
        raw_chat_id="11111111-2222-3333-4444-555555555555",
        sender_uuid="11111111-2222-3333-4444-555555555555",
        sender_name="Alice",
        chat_name=None,
        timestamp_ms=1700000000000,
        text="",
        attachments=attachments,
        reply=None,
        reaction=None,
    )


def test_fallback_to_config_dir_when_file_field_omitted(tmp_path: Path) -> None:
    """The daemon's JSON-RPC envelope omits ``file`` but ``id``
    points at ``<config_dir>/attachments/<id>``.  Plant the file
    there and assert the SDK Attachment surfaces with that path."""
    config_dir = tmp_path / "signal-cfg"
    (config_dir / "attachments").mkdir(parents=True)
    expected_path = config_dir / "attachments" / "xyz-789"
    expected_path.write_bytes(b"fake-png-bytes")

    connector = _make_connector(config_dir)
    msg = _make_msg(
        (
            Attachment(
                content_type="image/png",
                filename="photo.png",
                host_path=None,
                id="xyz-789",
            ),
        )
    )

    sdk_atts = connector._build_sdk_attachments(msg)

    assert len(sdk_atts) == 1
    assert sdk_atts[0].host_path == str(expected_path)
    assert sdk_atts[0].content_type == "image/png"


def test_fallback_skips_when_file_missing_on_disk(tmp_path: Path) -> None:
    """If the file isn't where we'd expect (signal-cli didn't auto-download
    or the daemon's storage layout differs), as_params rejects and the
    attachment is logged + skipped."""
    config_dir = tmp_path / "signal-cfg"
    (config_dir / "attachments").mkdir(parents=True)

    connector = _make_connector(config_dir)
    msg = _make_msg(
        (
            Attachment(
                content_type="image/png",
                filename="photo.png",
                host_path=None,
                id="never-downloaded",
            ),
        )
    )

    sdk_atts = connector._build_sdk_attachments(msg)
    assert sdk_atts == []


def test_no_id_no_host_path_skips_with_warning(tmp_path: Path) -> None:
    """Records that have neither host_path nor id (shouldn't happen, but
    a malformed daemon could produce it) get logged + skipped, not
    crashed on."""
    connector = _make_connector(tmp_path / "cfg")
    msg = _make_msg(
        (
            Attachment(
                content_type="image/png",
                filename="photo.png",
                host_path=None,
                id=None,
            ),
        )
    )

    assert connector._build_sdk_attachments(msg) == []


def test_explicit_host_path_wins_over_id_fallback(tmp_path: Path) -> None:
    """When the envelope DOES include ``file`` (legacy CLI shape), the
    explicit host_path is used and the id-based fallback isn't consulted."""
    config_dir = tmp_path / "signal-cfg"
    config_dir.mkdir()
    explicit = tmp_path / "explicit-path.png"
    explicit.write_bytes(b"x")

    connector = _make_connector(config_dir)
    msg = _make_msg(
        (
            Attachment(
                content_type="image/png",
                filename="photo.png",
                host_path=str(explicit),
                id="ignored",
            ),
        )
    )

    sdk_atts = connector._build_sdk_attachments(msg)
    assert len(sdk_atts) == 1
    assert sdk_atts[0].host_path == str(explicit)
