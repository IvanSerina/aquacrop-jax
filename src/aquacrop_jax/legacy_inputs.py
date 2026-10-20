"""Legacy daily-forcing adapters for the coupled diagnostic runner."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .climate import date_to_day_number, day_number_to_date


def _number(line):
    return float(line.split(':', 1)[0].strip())


@dataclass(frozen=True)
class ProgramTemperatureParameters:
    gdd_method: int = 3
    senescence_exponent: float = 1.0


def program_temperature_parameters(project_file):
    """Read temperature controls by CLI PP1/PPn record position.

    InitializeSettings supplies defaults if the project parameter file is
    absent. LoadProgramParametersProjectPlugIn clamps GDD method to 1..3;
    its lower clamp differs from the generic GUI loader. Comments are not
    semantic input. Other program parameters are not represented here yet.
    """
    project = Path(project_file)
    extension = '.PP1' if project.suffix == '.PRO' else '.PPn'
    path = project.parent.parent / 'PARAM' / (project.stem + extension)
    if not path.exists():
        return ProgramTemperatureParameters()
    records = [line for line in path.read_text().splitlines() if line.strip()]
    if len(records) < 25:
        raise ValueError('Program parameter file ends before its 25 CLI records')
    try:
        exponent = float(records[9].split()[0].replace('D', 'e').replace('d', 'e'))
        method = int(records[20].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError('Invalid temperature control in program parameter file') from exc
    if not np.isfinite(exponent):
        raise ValueError('Program senescence exponent must be finite')
    if not -128 <= method <= 127:
        raise ValueError('Program GDD method exceeds native int8 input range')
    return ProgramTemperatureParameters(min(max(method, 1), 3), exponent)


def _records(lines, columns):
    rows = []
    for line in lines:
        fields = line.split(':', 1)[0].split()
        try:
            values = tuple(float(v) for v in fields)
        except ValueError:
            continue
        if values:
            if len(values) != columns or not np.all(np.isfinite(values)):
                raise ValueError(f'Expected {columns} finite values: {line!r}')
            rows.append(values)
    return rows


def field_fertility_stress(path):
    """Read the integer fertility stress from ``LoadManagement``'s MAN header."""
    if path is None:
        return 0
    lines = Path(path).read_text().splitlines()
    if len(lines) < 5:
        raise ValueError('MAN header ends before the fertility-stress record')
    try:
        return int(lines[4].split(':', 1)[0].strip())
    except ValueError as exc:
        raise ValueError('MAN fertility stress must be an integer') from exc


@dataclass(frozen=True)
class ManualCuttings:
    day_numbers: tuple[int, ...]
    canopy_after_cut_pct: int
    absolute_reference: bool = False


def manual_cuttings(path, crop_first_day):
    """Resolve v7 MAN harvest offsets as ``OpenHarvestInfo`` does."""
    if path is None:
        return ManualCuttings((), 30)
    lines = Path(path).read_text().splitlines()
    if len(lines) < 21:
        raise ValueError('MAN header ends before the cuttings records')
    if _number(lines[1]) < 7.0:
        raise NotImplementedError('Cuttings before MAN v7 are not ported')
    if int(_number(lines[12])) == 0:
        return ManualCuttings((), int(_number(lines[13])))
    if int(_number(lines[17])) != 0:
        raise NotImplementedError('Generated cutting schedules are not yet ported')
    first = int(_number(lines[20]))
    absolute_reference = first != -9
    first = crop_first_day if first == -9 else first
    days = []
    for line in lines[24:]:
        if line.strip():
            offset = int(_number(line))
            days.append(first + offset - 1)
    return ManualCuttings(tuple(days), int(_number(lines[13])), absolute_reference)


def manual_irrigation_forcing(path, days, crop_first_day):
    """Load v7 manual IRR, including IrriManual's legacy EC lookahead."""
    days = np.asarray(days, dtype=int)
    depth, ec = np.zeros(days.shape), np.zeros(days.shape)
    if path is None:
        return depth, ec
    lines = Path(path).read_text().splitlines()
    version, method, wetted, mode = map(_number, lines[1:5])
    if version < 7.0:
        raise NotImplementedError('IRR before v7 is not ported')
    if mode == 0:
        return depth, ec
    if mode != 1:
        raise NotImplementedError('Legacy runner currently supports manual IRR only')
    if method != 1 or wetted != 100:
        raise NotImplementedError('Partial wetting and non-sprinkler irrigation need evaporation wiring')
    first = int(_number(lines[5]))
    first = crop_first_day if first == -9 else first
    rows = _records(lines[8:], 3)
    event_days = [first + int(np.rint(r[0])) - 1 for r in rows]
    if any(b <= a for a, b in zip(event_days, event_days[1:])):
        raise ValueError('Manual irrigation days must be strictly increasing')
    for index, (day, row) in enumerate(zip(event_days, rows)):
        match = days == day
        depth[match] = max(float(np.rint(row[1])), 0.0)
        # run.f90 IrriManual reads the next record before the daily salt budget.
        ec[match] = rows[min(index + 1, len(rows) - 1)][2]
    if np.any(ec < 0.0):
        raise ValueError('Irrigation EC must be nonnegative')
    return depth, ec


def groundwater_forcing(path, days):
    days = np.asarray(days, dtype=int)
    if path is None:
        return np.full(days.shape, -1.0), np.zeros(days.shape)
    lines = Path(path).read_text().splitlines()
    mode = int(_number(lines[2]))
    if mode == 0:
        return np.full(days.shape, -1.0), np.zeros(days.shape)
    offset, first = 3, 1
    if mode != 1:
        day, month, year = map(lambda l: int(_number(l)), lines[3:6])
        if year == 1901:
            raise NotImplementedError('Yearless cyclic groundwater observations are not ported')
        first = date_to_day_number(date(year, month, day))
        offset = 6
    rows = _records(lines[offset + 3:], 3)
    if not rows:
        raise ValueError('GWT contains no observations')
    records = np.asarray(rows)
    observation_days = first + np.rint(records[:, 0]) - 1
    if np.any(np.diff(observation_days) <= 0.0):
        raise ValueError('GWT observations must have increasing days')
    if np.any(records[:, 1:] < 0.0):
        raise ValueError('GWT depth and EC must be nonnegative')
    if mode == 1:
        depth = np.full(days.shape, records[0, 1])
        ec = np.full(days.shape, records[0, 2])
    else:
        depth = np.interp(days, observation_days, records[:, 1])
        ec = np.interp(days, observation_days, records[:, 2])
    # LoadGroundWater rounds depth to integer centimetres before the budget.
    return np.rint(100.0 * depth) / 100.0, ec


@dataclass(frozen=True)
class InitialConditions:
    theta: np.ndarray
    ece: np.ndarray
    canopy_pct: float
    biomass: float
    root_depth: float


def initial_conditions(path, thickness, field_capacity_pct, saturation_pct):
    thickness = np.asarray(thickness, dtype=float)
    if path is None:
        return InitialConditions(np.full(thickness.shape, field_capacity_pct / 100.0),
                                  np.zeros_like(thickness), -9., 0., -9.)
    lines = Path(path).read_text().splitlines()
    if _number(lines[1]) < 4.1:
        raise NotImplementedError('SW0 before v4.1 is not ported')
    cover, biomass, root, surface, surface_ec, points, count = map(_number, lines[2:9])
    if surface != 0.0:
        raise NotImplementedError('Initial surface ponding is not ported')
    if points != 0:
        raise NotImplementedError('SW0 point profiles need the legacy piecewise integration')
    rows = _records(lines[12:], 3)
    if len(rows) != int(count) or not rows:
        raise ValueError('SW0 layer count does not match the data')
    values = np.asarray(rows)
    if np.any(values[:, 0] <= 0.0) or np.any(values[:, 1:] < 0.0):
        raise ValueError('SW0 layer thickness must be positive, water and ECe nonnegative')
    ends = np.cumsum(thickness)
    starts = ends - thickness
    layer_ends = np.cumsum(values[:, 0])
    layer_ends[-1] = max(layer_ends[-1], ends[-1])
    layer_starts = np.concatenate(([0.], layer_ends[:-1]))
    overlap = np.maximum(np.minimum(ends[:, None], layer_ends) -
                         np.maximum(starts[:, None], layer_starts), 0.)
    theta = np.minimum(overlap @ values[:, 1] / thickness, saturation_pct) / 100.0
    ece = overlap @ values[:, 2] / thickness
    return InitialConditions(theta, ece, cover, biomass, root)


def co2_for_period(path, first_day, last_day):
    first_year, last_year = [day_number_to_date(d).year for d in (first_day, last_day)]
    if first_year == 1901 or last_year == 1901 or path is None:
        return 369.41
    rows = _records(Path(path).read_text().splitlines()[3:], 2)
    if not rows:
        raise ValueError('CO2 file contains no records')
    values = np.asarray(rows)
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError('CO2 years must be strictly increasing')
    return float(np.mean(np.interp([first_year, last_year], values[:, 0], values[:, 1])))
