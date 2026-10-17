"""Typed daily OUT boundary and numerical diagnostics for AquaCrop 7.3.

The current schema is the full seven-output-group, 12-compartment layout.
Formats are from run.f90 WriteDailyResults. Parsing/formatting parity is
independent of simulator parity; a writer requires every numerical field.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re

import numpy as np

from .climate import date_to_day_number

DAILY_COLUMNS = tuple('''day month year dap stage water_mm rain irrigation surface infiltration runoff drainage capillary_rise groundwater_depth potential_evaporation evaporation evaporation_ratio potential_transpiration transpiration transpiration_ratio potential_et actual_et et_ratio gdd root_depth expansion_stress stomatal_stress senescence_stress salinity_stress weed_stress canopy_pct weed_canopy_pct transpiration_stress crop_coefficient crop_potential_transpiration crop_transpiration weed_transpiration crop_transpiration_ratio water_productivity biomass harvest_index yield_dry yield_fresh relative_biomass water_productivity_et biomass_in biomass_out profile_water root_max_water profile_root_depth root_water root_sat root_fc root_expansion root_stomatal root_senescence root_wp salt_in salt_out salt_up salt_total salt_root salt_root_depth root_ece root_ecsw salt_stress groundwater_salt_depth groundwater_ec'''.split()) + tuple(f'theta{i:02d}_pct' for i in range(1, 13)) + tuple(f'ece{i:02d}' for i in range(1, 13)) + ('climate_rain', 'eto', 'tmin', 'tavg', 'tmax', 'co2')

# (width, precision); None precision denotes integer output.
DAILY_FORMATS = tuple(
    [(6, None)] * 5 +
    [(10, 1), (8, 1), (9, 1)] + [(7, 1)] * 3 + [(9, 1)] * 2 + [(8, 2)] +
    [(9, 1)] * 2 + [(7, None)] + [(9, 1)] * 2 + [(6, None), (9, 1), (8, 1), (8, None)] +
    [(9, 1), (8, 2)] + [(7, None)] * 5 + [(8, 1)] * 2 + [(7, None), (9, 2)] +
    [(9, 1)] * 3 + [(6, None), (8, 1), (10, 3), (8, 1), (9, 3), (9, 3), (8, None), (12, 2)] + [(9, 3)] * 2 +
    [(10, 1), (9, 1), (8, 2), (8, 1)] + [(10, 1)] * 6 +
    [(9, 3)] + [(10, 3)] * 4 + [(8, 2), (9, 2), (8, 2), (7, None), (8, 2), (8, 2)] +
    [(11, 1)] * 24 + [(9, 1)] + [(10, 1)] * 4 + [(10, 2)])
assert len(DAILY_COLUMNS) == len(DAILY_FORMATS) == 98


@dataclass(frozen=True)
class DailyRecord:
    run: int
    values: tuple[float, ...]

    def mapping(self):
        return dict(zip(DAILY_COLUMNS, self.values, strict=True))

    @property
    def key(self):
        day, month, year = (int(x) for x in self.values[:3])
        return self.run, date_to_day_number(date(year, month, day))


@dataclass(frozen=True)
class DailyOutput:
    records: tuple[DailyRecord, ...]
    # Literal header/footer lines and record indices preserve native metadata.
    layout: tuple[str | int, ...]


def read_daily_output(path):
    records, layout = [], []
    run = None
    expected_width = sum(width for width, _ in DAILY_FORMATS)
    seen = set()
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        match = re.fullmatch(r'\s*Run:\s*(\d+)\s*', line)
        if match:
            run = int(match.group(1))
        if re.match(r'^\s*\d+\s+\d+\s+\d+\s+', line):
            if run is None:
                raise ValueError(f'{path}:{lineno}: missing PRM run header')
            if len(line) != expected_width:
                raise ValueError(f'{path}:{lineno}: unsupported daily OUT layout/width ({len(line)} != {expected_width})')
            values, offset = [], 0
            for name, (width, precision) in zip(DAILY_COLUMNS, DAILY_FORMATS):
                field = line[offset:offset + width]
                try:
                    value = float(field)
                except ValueError as exc:
                    raise ValueError(f'{path}:{lineno}: invalid field {name}: {field!r}') from exc
                if not np.isfinite(value) or (precision is None and value != int(value)):
                    raise ValueError(f'{path}:{lineno}: nonfinite or invalid integer {name}')
                values.append(value)
                offset += width
            record = DailyRecord(run, tuple(values))
            if record.key in seen:
                raise ValueError(f'{path}:{lineno}: duplicate run/day {record.key}')
            seen.add(record.key)
            layout.append(len(records))
            records.append(record)
        else:
            layout.append(line)
    if not records:
        raise ValueError(f'{path}: no full-layout daily records')
    return DailyOutput(tuple(records), tuple(layout))


def format_daily_record(values):
    if set(values) != set(DAILY_COLUMNS):
        missing = sorted(set(DAILY_COLUMNS) - set(values))
        extra = sorted(set(values) - set(DAILY_COLUMNS))
        raise ValueError(f'Daily OUT requires a complete record; missing={missing}, extra={extra}')
    fields = []
    for name, (width, precision) in zip(DAILY_COLUMNS, DAILY_FORMATS):
        value = float(values[name])
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite daily OUT field: {name}')
        if precision is None:
            if value != int(value):
                raise ValueError(f'Noninteger daily OUT field: {name}')
            formatted = f'{int(value):{width}d}'
        else:
            formatted = f'{value:{width}.{precision}f}'
        if len(formatted) > width:
            formatted = '*' * width  # native Fortran field overflow
        fields.append(formatted)
    return ''.join(fields)


def write_daily_output(path, output):
    lines = [format_daily_record(output.records[item].mapping()) if isinstance(item, int)
             else item for item in output.layout]
    Path(path).write_text('\n'.join(lines) + '\n')


# CSV units differ from OUT for compartment theta, canopy and salt mass.
TRACE_COLUMNS = {
    'rain': ('rain', 1.), 'eto': ('eto', 1.), 'tmin': ('tmin', 1.),
    'tmax': ('tmax', 1.), 'co2': ('co2', 1.), 'gdd': ('gdd', 1.),
    'water_mm': ('water_mm', 1.), 'runoff': ('runoff', 1.),
    'drainage': ('drainage', 1.), 'infiltration': ('infiltration', 1.),
    'evaporation': ('evaporation', 1.), 'transpiration': ('transpiration', 1.),
    'irrigation': ('irrigation', 1.), 'root_depth': ('root_depth', 1.),
    'canopy': ('canopy_pct', 100.), 'biomass': ('biomass', 1.),
    'yield_mass': ('yield_dry', 1.), 'rootzone_ece': ('root_ece', 1.),
    'salt_g_m2': ('salt_total', .01),
    **{f'theta{i:02d}': (f'theta{i:02d}_pct', 100.) for i in range(1, 13)},
    **{f'ece{i:02d}': (f'ece{i:02d}', 1.) for i in range(1, 13)},
}


def compare_trace_to_daily(trace, reference):
    """Report coverage and per-column errors; this never relaxes exact compare."""
    import csv
    with Path(trace).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    actual = {}
    for row in rows:
        key = int(row['run']), int(row['daynr'])
        if key in actual:
            raise ValueError(f'Duplicate trace run/day: {key}')
        actual[key] = row
    expected = {r.key: r.mapping() for r in read_daily_output(reference).records}
    common = sorted(actual.keys() & expected.keys())
    if not common:
        raise ValueError('No matching run/day records between trace and OUT')
    stats = {}
    for trace_name, (out_name, scale) in TRACE_COLUMNS.items():
        if any(trace_name not in actual[key] for key in common):
            continue
        errors = np.asarray([float(actual[k][trace_name]) * scale - expected[k][out_name] for k in common])
        if not np.all(np.isfinite(errors)):
            raise ValueError(f'Nonfinite trace field: {trace_name}')
        stats[trace_name] = dict(max_abs=float(np.max(np.abs(errors))),
            rmse=float(np.sqrt(np.mean(errors ** 2))), mean_error=float(np.mean(errors)))
    return dict(matched_days=len(common), missing_days=sorted(expected.keys() - actual.keys()),
                extra_days=sorted(actual.keys() - expected.keys()), columns=stats)
