"""``read`` paging must count lines the way ``cat -n``/``sed`` do (``\\n`` only)."""

import subprocess
from pathlib import Path

import pytest

from aios.tools.read import _fit_text_result


def _numbered(p: Path, start: int = 1) -> str:
    return subprocess.run(
        ["bash", "-c", f"cat -n -- {p} | sed -n '{start},$p'"], capture_output=True, check=True
    ).stdout.decode()


def test_next_offset_follows_last_shown_line_with_cr(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_bytes("".join(f"line{i:05d} 10%\r100% {'x' * 200}\n" for i in range(1, 501)).encode())
    out = subprocess.run(
        ["bash", "-c", f"cat -n -- {p} | sed -n '1,2000p'"], capture_output=True
    ).stdout.decode()
    r = _fit_text_result(str(p), out, offset=1, budget=16000)
    last = [line for line in r["content"].split("\n") if "\t" in line][-1].split("\t")[0].strip()
    assert r["next_offset"] == int(last) + 1


@pytest.mark.parametrize(
    "sep", ["\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
)
def test_paging_covers_every_line_once(tmp_path: Path, sep: str) -> None:
    n = 600
    p = tmp_path / "f.txt"
    p.write_bytes("".join(f"l{i} a{sep}b{sep}c {'y' * 120}\n" for i in range(1, n + 1)).encode())
    seen: list[int] = []
    offset = 1
    for _ in range(n + 1):
        r = _fit_text_result(str(p), _numbered(p, offset), offset=offset, budget=8000)
        seen += [
            int(line.split("\t")[0])
            for line in r["content"].split("\n")
            if "\t" in line and line.split("\t")[0].strip().isdigit()
        ]
        if not r.get("truncated"):
            break
        offset = r["next_offset"]
    assert seen == list(range(1, n + 1))
