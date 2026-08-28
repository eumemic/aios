"""Tests for the export tar filter (#2280).

The filter sits on the critical path of session durability: everything it
forwards becomes the session's filesystem, and everything it drops is gone.
So the tests are written around two properties rather than examples —
*nothing outside the dropped prefixes is ever altered*, and *the result is
always a tar Python can read back* — plus the specific shapes (long names,
pax headers, hardlinks, base-256 sizes) where a naive 512-byte scanner
silently corrupts a stream.
"""

from __future__ import annotations

import io
import tarfile

import pytest

from aios.sandbox._tar_filter import TarPrefixFilter, _is_ephemeral


def _build_tar(entries: list[tuple[str, bytes]], **kwargs: object) -> bytes:
    """A tar containing ``entries`` as regular files."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", **kwargs) as tf:  # type: ignore[call-overload]
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def _run(raw: bytes, chunk_size: int = 1024 * 1024) -> bytes:
    """Feed ``raw`` through the filter in ``chunk_size`` pieces."""
    filt = TarPrefixFilter()
    out = bytearray()
    for start in range(0, len(raw), chunk_size):
        out += filt.feed(raw[start : start + chunk_size])
    out += filt.flush()
    return bytes(out)


def _names(raw: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tf:
        return tf.getnames()


class TestPrefixMatching:
    @pytest.mark.parametrize(
        "name",
        ["tmp/x", "tmp/deep/nested/file", "var/tmp/y", "run/lock", "./tmp/x", "/tmp/x"],
    )
    def test_inside_a_dropped_tree(self, name: str) -> None:
        assert _is_ephemeral(name, ("tmp/", "var/tmp/", "run/")) is True

    @pytest.mark.parametrize(
        "name",
        [
            "tmp",  # the mount point itself must survive
            "tmp/",
            "./tmp",
            "var",
            "var/tmp",  # ditto
            "workspace/tmp/x",  # only ROOT-anchored prefixes match
            "tmpfoo/x",  # not a path component boundary
            "root/.cache/ms-playwright/x",  # deliberately durable
            "app/.venv/lib/x",
        ],
    )
    def test_kept(self, name: str) -> None:
        assert _is_ephemeral(name, ("tmp/", "var/tmp/", "run/")) is False


class TestFiltering:
    def test_drops_only_the_ephemeral_entries(self) -> None:
        raw = _build_tar(
            [
                ("workspace/keep.txt", b"keep me"),
                ("tmp/junk.bin", b"x" * 4096),
                ("var/tmp/more.bin", b"y" * 2048),
                ("run/sock.lock", b"z"),
                ("root/.cache/ms-playwright/webkit", b"expensive"),
            ]
        )
        out = _run(raw)
        assert _names(out) == ["workspace/keep.txt", "root/.cache/ms-playwright/webkit"]

    def test_kept_payloads_are_byte_identical(self) -> None:
        payload = bytes(range(256)) * 40
        raw = _build_tar([("workspace/data.bin", payload), ("tmp/junk", b"q" * 9000)])
        with tarfile.open(fileobj=io.BytesIO(_run(raw)), mode="r") as tf:
            member = tf.extractfile("workspace/data.bin")
            assert member is not None
            assert member.read() == payload

    def test_stream_with_no_matches_is_unchanged(self) -> None:
        raw = _build_tar([("workspace/a", b"a" * 5000), ("etc/hosts", b"127.0.0.1")])
        assert _run(raw) == raw

    @pytest.mark.parametrize("chunk_size", [1, 7, 512, 513, 1024, 65536])
    def test_chunk_boundaries_do_not_matter(self, chunk_size: int) -> None:
        """The relay reads 1 MiB at a time with no regard for tar's 512-byte
        records, so a header or payload can be split anywhere."""
        raw = _build_tar(
            [("workspace/keep", b"k" * 3000), ("tmp/drop", b"d" * 3000), ("etc/keep2", b"e")]
        )
        out = _run(raw, chunk_size=chunk_size)
        assert _names(out) == ["workspace/keep", "etc/keep2"]

    def test_reports_what_it_dropped(self) -> None:
        filt = TarPrefixFilter()
        raw = _build_tar([("tmp/a", b"x" * 10_000), ("tmp/b", b"y" * 10_000)])
        filt.feed(raw)
        filt.flush()
        assert filt.dropped_entries == 2
        assert filt.dropped_bytes >= 20_000

    def test_empty_stream(self) -> None:
        assert _run(b"") == b""


class TestAwkwardHeaders:
    def test_gnu_long_names(self) -> None:
        """>100-char names are emitted as a separate ``L`` record preceding the
        entry, so the drop decision has to be deferred to the real header."""
        long_kept = "workspace/" + "a" * 150 + "/keep.txt"
        long_dropped = "tmp/" + "b" * 150 + "/junk.bin"
        raw = _build_tar(
            [(long_kept, b"keep"), (long_dropped, b"z" * 4096)],
            format=tarfile.GNU_FORMAT,
        )
        out = _run(raw)
        assert _names(out) == [long_kept]

    def test_pax_headers(self) -> None:
        long_kept = "workspace/" + "a" * 150 + "/keep.txt"
        long_dropped = "tmp/" + "b" * 150 + "/junk.bin"
        raw = _build_tar(
            [(long_kept, b"keep"), (long_dropped, b"z" * 4096)],
            format=tarfile.PAX_FORMAT,
        )
        out = _run(raw)
        assert _names(out) == [long_kept]

    def test_hardlink_into_a_dropped_tree_is_dropped(self) -> None:
        """Its target won't be in the output, so forwarding the link would
        produce an archive ``docker import`` refuses."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo("tmp/original")
            info.size = 10
            tf.addfile(info, io.BytesIO(b"0123456789"))
            link = tarfile.TarInfo("workspace/alias")
            link.type = tarfile.LNKTYPE
            link.linkname = "tmp/original"
            tf.addfile(link)
        out = _run(buf.getvalue())
        assert _names(out) == []

    def test_hardlink_to_a_kept_file_survives(self) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo("workspace/original")
            info.size = 10
            tf.addfile(info, io.BytesIO(b"0123456789"))
            link = tarfile.TarInfo("workspace/alias")
            link.type = tarfile.LNKTYPE
            link.linkname = "workspace/original"
            tf.addfile(link)
        out = _run(buf.getvalue())
        assert _names(out) == ["workspace/original", "workspace/alias"]

    def test_directories_and_symlinks_pass_through(self) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            d = tarfile.TarInfo("workspace")
            d.type = tarfile.DIRTYPE
            tf.addfile(d)
            mount_point = tarfile.TarInfo("tmp")
            mount_point.type = tarfile.DIRTYPE
            tf.addfile(mount_point)
            s = tarfile.TarInfo("workspace/link")
            s.type = tarfile.SYMTYPE
            s.linkname = "../etc/hosts"
            tf.addfile(s)
        out = _run(buf.getvalue())
        # ``tmp`` itself survives: the bind mount needs a mount point.
        assert _names(out) == ["workspace", "tmp", "workspace/link"]


class TestFailureModes:
    def test_garbage_is_forwarded_verbatim(self) -> None:
        """A stream we cannot parse must come out unchanged. Fat beats corrupt."""
        garbage = b"this is definitely not a tar archive" * 100
        assert _run(garbage) == garbage

    def test_truncated_archive_is_not_rewritten(self) -> None:
        raw = _build_tar([("workspace/a", b"a" * 4096)])
        truncated = raw[: len(raw) - 700]
        assert _run(truncated) == truncated

    def test_trailing_data_after_end_marker_survives(self) -> None:
        raw = _build_tar([("workspace/a", b"a")])
        assert _run(raw + b"trailing") == raw + b"trailing"


class TestLargeEntries:
    def test_base256_size_header(self) -> None:
        """Entries >8 GiB encode their size in GNU base-256 rather than octal;
        misreading one desynchronises the parser for the rest of the stream."""
        from aios.sandbox._tar_filter import _parse_size

        header = bytearray(512)
        size = 10 * 1024**3
        raw = size.to_bytes(11, "big")
        header[124] = 0x80
        header[125:136] = raw
        assert _parse_size(bytes(header)) == size

    def test_payload_spanning_many_chunks(self) -> None:
        big = b"Q" * (3 * 1024 * 1024)
        raw = _build_tar([("tmp/big", big), ("workspace/small", b"s")])
        out = _run(raw, chunk_size=1024 * 1024)
        assert _names(out) == ["workspace/small"]
        assert len(out) < len(raw) - 2 * 1024 * 1024
