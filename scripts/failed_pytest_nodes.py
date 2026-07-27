"""Print failed pytest node IDs from a JUnit XML report.

This is used by CI instead of ``pytest --lf``: xdist workers maintain separate
last-failed caches, so the controller's cache can be empty and accidentally
turn a retry into a full-suite replay.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def _node_id(case: ET.Element) -> str:
    classname = case.attrib["classname"]
    name = case.attrib["name"]
    parts = classname.split(".")
    module_index = next(index for index, part in enumerate(parts) if part.startswith("test_"))
    path = "/".join(parts[: module_index + 1]) + ".py"
    selectors = [*parts[module_index + 1 :], name]
    return "::".join([path, *selectors])


def failed_nodes(report: Path) -> list[str]:
    root = ET.parse(report).getroot()
    nodes: list[str] = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            nodes.append(_node_id(case))
    return list(dict.fromkeys(nodes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(*failed_nodes(args.report), sep="\n")


if __name__ == "__main__":
    main()
