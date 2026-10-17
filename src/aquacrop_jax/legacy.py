"""Readers for AquaCrop standalone-program text formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_NONE = {"(None)", "KeepSWC"}


@dataclass(frozen=True)
class FileReference:
    role: str
    name: str
    directory: str

    @property
    def is_external(self) -> bool:
        return self.name not in _NONE

    def resolve(self, project_file: Path) -> Path | None:
        if not self.is_external:
            return None
        candidate = (project_file.parent / self.directory / self.name).resolve()
        if candidate.exists():
            return candidate
        # Standalone projects normally live in LIST/ while paths in the PRM
        # are relative to the testcase/run directory containing LIST/.
        return (project_file.parent.parent / self.directory / self.name).resolve()


@dataclass(frozen=True)
class SimulationRun:
    cultivation_year: int
    simulation_first_day: int
    simulation_last_day: int
    crop_first_day: int
    crop_last_day: int
    files: tuple[FileReference, ...]


@dataclass(frozen=True)
class Project:
    description: str
    version: float
    runs: tuple[SimulationRun, ...]


_ROLE = re.compile(r"^(?:--\s*)?(?:\d+(?:\.\d+)?)\.?\s*(.+?)\s+file\s*$", re.I)


def _value(line: str) -> str:
    return line.split(":", 1)[0].strip()


def read_project(path: str | Path) -> Project:
    """Parse a PRM file without changing AquaCrop day-number semantics."""
    project_file = Path(path).resolve()
    lines = project_file.read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise ValueError(f"Not an AquaCrop project file: {project_file}")
    description = lines[0].strip()
    version = float(_value(lines[1]))
    runs: list[SimulationRun] = []
    i = 2
    while i < len(lines):
        if "Year number of cultivation" not in lines[i]:
            i += 1
            continue
        cultivation_year = int(_value(lines[i])); i += 1
        days = [int(_value(lines[i + k])) for k in range(4)]
        i += 4
        refs: list[FileReference] = []
        while i < len(lines) and "Year number of cultivation" not in lines[i]:
            match = _ROLE.match(lines[i].strip())
            if match and i + 2 < len(lines):
                refs.append(FileReference(match.group(1), lines[i + 1].strip(), lines[i + 2].strip().strip("'\"")))
                i += 3
            else:
                i += 1
        runs.append(SimulationRun(cultivation_year, *days, tuple(refs)))
    if not runs:
        raise ValueError(f"No simulation runs found in {project_file}")
    return Project(description, version, tuple(runs))


@dataclass(frozen=True)
class ClimateSeries:
    data_type: int
    first_day: int
    first_month: int
    first_year: int
    values: tuple[tuple[float, ...], ...]


def read_climate_series(path: str | Path, columns: int) -> ClimateSeries:
    """Read ETo/PLU/Tnx numeric records after their fixed legacy header."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 9:
        raise ValueError(f"Climate file is too short: {path}")
    metadata = [int(_value(lines[n])) for n in range(1, 5)]
    values: list[tuple[float, ...]] = []
    for line in lines[8:]:
        fields = line.split()
        if not fields:
            continue
        try:
            row = tuple(float(v) for v in fields)
        except ValueError:
            continue
        if len(row) != columns:
            raise ValueError(f"Expected {columns} values in {path}: {line!r}")
        values.append(row)
    return ClimateSeries(*metadata, tuple(values))


def read_project_list(path: str | Path) -> tuple[Path, ...]:
    """Resolve non-empty project names from LIST/ListProjects.txt."""
    list_file = Path(path).resolve()
    projects = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            projects.append((list_file.parent / name).resolve())
    return tuple(projects)


def read_annotated_values(path: str | Path) -> dict[str, float]:
    """Read numeric `value : description` records used by CRO/PPn files."""
    result: dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        raw, description = line.split(":", 1)
        try:
            result[description.strip()] = float(raw.strip())
        except ValueError:
            continue
    return result
