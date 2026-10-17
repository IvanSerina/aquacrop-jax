"""Regression comparison for AquaCrop standalone outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Difference:
    filename: str
    line: int
    expected: str
    actual: str


def normalized_lines(path: Path) -> list[str]:
    lines = path.read_bytes().decode("utf-8", errors="surrogateescape").splitlines(keepends=True)
    if lines and "Output created on (date)" in lines[0]:
        ending = "\r\n" if lines[0].endswith("\r\n") else ("\n" if lines[0].endswith("\n") else "")
        lines[0] = lines[0].split("Output created on (date)", 1)[0] + ending
    return lines


def compare_directories(actual: str | Path, expected: str | Path) -> list[Difference]:
    actual_dir, expected_dir = Path(actual), Path(expected)
    if not actual_dir.is_dir():
        raise FileNotFoundError(f"Generated-output directory does not exist: {actual_dir}")
    if not expected_dir.is_dir():
        raise FileNotFoundError(f"Reference-output directory does not exist: {expected_dir}")
    if not any(actual_dir.glob("*.OUT")):
        raise FileNotFoundError(
            f"No generated .OUT files found in {actual_dir}; run an AquaCrop "
            "simulator before using compare"
        )
    if not any(expected_dir.glob("*.OUT")):
        raise FileNotFoundError(f"No reference .OUT files found in {expected_dir}")
    differences: list[Difference] = []
    for extra in sorted(p.name for p in actual_dir.glob("*.OUT")
                        if not (expected_dir / p.name).exists()):
        differences.append(Difference(extra, 0, "absent", "unexpected file"))
    for expected_file in sorted(expected_dir.glob("*.OUT")):
        actual_file = actual_dir / expected_file.name
        if not actual_file.exists():
            differences.append(Difference(expected_file.name, 0, "file exists", "missing"))
            continue
        want, got = normalized_lines(expected_file), normalized_lines(actual_file)
        for n in range(max(len(want), len(got))):
            left = want[n] if n < len(want) else "<EOF>"
            right = got[n] if n < len(got) else "<EOF>"
            if left != right:
                differences.append(Difference(expected_file.name, n + 1, left, right))
                break
    return differences
