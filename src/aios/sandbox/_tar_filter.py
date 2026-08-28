"""Streaming tar filter that drops ephemeral paths from a container export.

``docker export`` emits the container's whole filesystem as a tar stream.
For a durable session sandbox that stream *is* the snapshot, so anything in
it is preserved forever — including the accumulated scratch under ``/tmp``
that nobody ever wanted kept. One production session reached 18 GiB of which
16.4 GiB was ``/tmp`` (eumemic/aios#2280).

The structural fix is the ``/tmp`` bind mount
(:func:`aios.sandbox.volumes.session_tmp_dir`), which Docker omits from both
snapshot verbs. This filter exists for the images that already exist: a
session snapshotted before the mount landed carries its old ``/tmp`` *inside*
the image, and a mount does not remove what is already baked in. Re-exporting
such an image would copy that payload into its successor forever. Filtering
the export breaks the chain — each affected session sheds its historical
scratch on its next flatten, and the mount keeps it lean thereafter.

Design notes
------------
* **Pure and incremental.** :class:`TarPrefixFilter` is fed arbitrary byte
  chunks (the relay reads 1 MiB at a time, unrelated to tar's 512-byte record
  boundaries) and returns the bytes to forward. It holds at most one partial
  record plus one pending metadata record — never the stream.

* **Entries are dropped, not emptied.** A dropped entry's header and payload
  are both suppressed, so ``docker import`` never learns the path existed.

* **The mount points survive.** ``tmp/`` itself is kept while everything
  beneath it is dropped: the directory must exist in the image for the bind
  to have something to mount onto.

* **Extended headers travel with their target.** GNU ``L``/``K`` and pax
  ``x``/``X`` records describe the *next* entry, so the drop decision cannot
  be made until that entry's own header arrives. They are buffered and then
  emitted or dropped together with it.

* **Hardlinks into a dropped tree are dropped**, not rewritten. Tar encodes
  the second and later names of an inode as a link to the first; if the first
  is gone the link dangles and ``docker import`` fails. Every prefix we drop
  is ephemeral, so a hardlink into one is ephemeral too.

* **Unparseable input is forwarded verbatim.** A stream we cannot understand
  is passed through unchanged rather than truncated: the failure mode of this
  filter must be "the snapshot is as fat as it used to be", never "the
  snapshot is corrupt". Corrupting a session's filesystem to save disk would
  be a far worse bug than the one being fixed.
"""

from __future__ import annotations

from aios.logging import get_logger

log = get_logger("aios.sandbox.tar_filter")

_BLOCK = 512

# Paths dropped from the export, relative to the container root (tar members
# are emitted without a leading slash). Each is scratch the sandbox itself
# treats as disposable across boots. ``root/.cache`` is deliberately NOT here:
# it holds Playwright browsers and package caches that are expensive to
# re-fetch, so it stays durable state.
EPHEMERAL_PREFIXES: tuple[str, ...] = ("tmp/", "var/tmp/", "run/")

# Record types describing the FOLLOWING entry rather than a file of their own:
# GNU long name / long link name, pax extended / global headers.
_META_TYPES = frozenset({b"L", b"K", b"x", b"X"})


def _normalize(name: str) -> str:
    """Strip the leading ``./`` or ``/`` docker may emit, so a member name can
    be compared against :data:`EPHEMERAL_PREFIXES`."""
    if name.startswith("./"):
        name = name[2:]
    return name.lstrip("/")


def _is_ephemeral(name: str, prefixes: tuple[str, ...]) -> bool:
    """True if ``name`` is *inside* one of ``prefixes``.

    The prefix directories themselves are kept — see the mount-point note in
    the module docstring.
    """
    norm = _normalize(name).rstrip("/")
    for prefix in prefixes:
        stem = prefix.rstrip("/")
        if norm == stem:
            return False  # the mount point itself
        if norm.startswith(f"{stem}/"):
            return True
    return False


def _header_name(header: bytes) -> str:
    """The member name from a ustar header, honouring the ``prefix`` field."""
    name = header[0:100].split(b"\0")[0].decode("utf-8", errors="replace")
    prefix = header[345:500].split(b"\0")[0].decode("utf-8", errors="replace")
    return f"{prefix}/{name}" if prefix else name


def _parse_size(header: bytes) -> int:
    """Payload size from a tar header: octal, or GNU base-256 for >8 GiB."""
    raw = header[124:136]
    if raw[0] & 0x80:  # GNU base-256 extension
        value = 0
        for byte in raw[1:]:
            value = (value << 8) | byte
        return value
    text = raw.split(b"\0")[0].strip()
    return int(text, 8) if text else 0


class TarPrefixFilter:
    """Incremental tar-stream filter dropping :data:`EPHEMERAL_PREFIXES`.

    Feed chunks to :meth:`feed` and write whatever it returns; call
    :meth:`flush` at EOF for any buffered remainder. ``dropped_bytes`` and
    ``dropped_entries`` report what was removed.
    """

    def __init__(self, prefixes: tuple[str, ...] = EPHEMERAL_PREFIXES) -> None:
        self._prefixes = prefixes
        self._buf = bytearray()
        # Bytes of the current entry's payload still to be forwarded/skipped.
        self._payload_remaining = 0
        self._skipping_payload = False
        # A buffered GNU/pax metadata record awaiting its target entry, plus
        # the long name it carries (which overrides the 100-byte name field).
        self._pending_meta = bytearray()
        self._pending_name: str | None = None
        self._passthrough = False
        self.dropped_bytes = 0
        self.dropped_entries = 0

    def feed(self, chunk: bytes) -> bytes:
        """Consume ``chunk``; return the bytes to forward downstream."""
        if self._passthrough:
            return chunk
        self._buf.extend(chunk)
        out = bytearray()
        try:
            self._drain(out)
        except Exception:
            # Unparseable: stop filtering and forward everything verbatim from
            # here on. A fat snapshot beats a corrupt one.
            log.exception("tar_filter.parse_failed_passing_through")
            self._passthrough = True
            out += self._pending_meta
            self._pending_meta.clear()
            out += self._buf
            self._buf.clear()
        return bytes(out)

    def _drain(self, out: bytearray) -> None:
        """Consume as many whole records as the buffer currently holds."""
        while True:
            if self._payload_remaining:
                take = min(self._payload_remaining, len(self._buf))
                if not take:
                    return
                if self._skipping_payload:
                    self.dropped_bytes += take
                else:
                    out += self._buf[:take]
                del self._buf[:take]
                self._payload_remaining -= take
                continue

            if len(self._buf) < _BLOCK:
                return
            header = bytes(self._buf[:_BLOCK])

            if header == b"\0" * _BLOCK:
                # End-of-archive marker: emit it and everything after verbatim.
                out += self._pending_meta
                self._pending_meta.clear()
                out += self._buf
                self._buf.clear()
                self._passthrough = True
                return

            size = _parse_size(header)
            padded = (size + _BLOCK - 1) // _BLOCK * _BLOCK
            typeflag = header[156:157]

            if typeflag in _META_TYPES:
                # Metadata for the NEXT entry: buffer the whole record and
                # decide when that entry's own header arrives.
                need = _BLOCK + padded
                if len(self._buf) < need:
                    return  # wait for the rest of the record
                if typeflag == b"L":
                    raw_name = bytes(self._buf[_BLOCK : _BLOCK + size])
                    self._pending_name = raw_name.rstrip(b"\0").decode("utf-8", errors="replace")
                self._pending_meta += self._buf[:need]
                del self._buf[:need]
                continue

            name = self._pending_name if self._pending_name is not None else _header_name(header)
            drop = _is_ephemeral(name, self._prefixes) or self._links_into_dropped(header, typeflag)
            if drop:
                self.dropped_entries += 1
                self.dropped_bytes += len(self._pending_meta) + _BLOCK
                self._skipping_payload = True
            else:
                out += self._pending_meta
                out += header
                self._skipping_payload = False
            self._pending_meta.clear()
            self._pending_name = None
            del self._buf[:_BLOCK]
            self._payload_remaining = padded

    def _links_into_dropped(self, header: bytes, typeflag: bytes) -> bool:
        """True for a hardlink whose target lives in a dropped tree — the
        target will not be in the output, so the link would dangle."""
        if typeflag != b"1":
            return False
        target = header[157:257].split(b"\0")[0].decode("utf-8", errors="replace")
        return _is_ephemeral(target, self._prefixes)

    def flush(self) -> bytes:
        """Return any buffered bytes at EOF.

        A truncated trailing record is forwarded as-is: the filter never
        invents padding, so a stream that arrived malformed stays exactly as
        malformed rather than becoming differently malformed.
        """
        out = bytes(self._pending_meta) + bytes(self._buf)
        self._pending_meta.clear()
        self._buf.clear()
        return out
