"""Legacy climate alignment and the first recurrent JAX pipeline."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .kernels import degrees_day
from .legacy import FileReference, Project, read_annotated_values, read_climate_series


EPOCH = date(1901, 1, 1)


def date_to_day_number(value: date) -> int:
    return (value - EPOCH).days + 1


def day_number_to_date(day_number: int) -> date:
    return EPOCH + timedelta(days=day_number - 1)


def _reference(run, text: str) -> FileReference:
    for ref in run.files:
        if text.lower() in ref.role.lower():
            return ref
    raise ValueError(f"Project has no {text} file reference")


def _slice(series, first_day_number: int, last_day_number: int, label: str):
    if series.data_type != 1:
        raise NotImplementedError(f"{label}: only daily climate records are ported")
    start = date_to_day_number(date(series.first_year, series.first_month, series.first_day))
    left, right = first_day_number - start, last_day_number - start + 1
    if left < 0 or right > len(series.values):
        raise ValueError(f"{label} does not cover DayNr {first_day_number}..{last_day_number}")
    return series.values[left:right]


def run_climate_stage(project: Project, project_file: Path, output: Path) -> int:
    """Create a calculated daily climate/GDD trace for all project runs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run", "daynr", "day", "month", "year", "rain", "eto",
                         "tmin", "tavg", "tmax", "gdd", "sum_gdd"))
        for run_index, run in enumerate(project.runs, 1):
            temp_path = _reference(run, "Temperature").resolve(project_file)
            eto_path = _reference(run, "Reference ET").resolve(project_file)
            rain_path = _reference(run, "Rain").resolve(project_file)
            crop_path = _reference(run, "Crop").resolve(project_file)
            assert temp_path and eto_path and rain_path and crop_path
            temps = _slice(read_climate_series(temp_path, 2), run.simulation_first_day,
                           run.simulation_last_day, "temperature")
            etos = _slice(read_climate_series(eto_path, 1), run.simulation_first_day,
                          run.simulation_last_day, "ETo")
            rains = _slice(read_climate_series(rain_path, 1), run.simulation_first_day,
                           run.simulation_last_day, "rain")
            crop = read_annotated_values(crop_path)
            tbase = next(v for k, v in crop.items() if k.startswith("Base temperature"))
            tupper = next(v for k, v in crop.items() if k.startswith("Upper temperature"))
            tensor = jnp.asarray(temps, dtype=jnp.float64)
            gdd = degrees_day(tbase, tupper, tensor[:, 0], tensor[:, 1], 3)
            # AquaCrop crop GDD is reset outside crop periods. This first trace
            # exposes both daily GDD and the crop-period cumulative recurrence.
            in_crop = jnp.asarray([
                run.crop_first_day <= n <= run.crop_last_day
                for n in range(run.simulation_first_day, run.simulation_last_day + 1)
            ])
            crop_gdd = jnp.where(in_crop, gdd, jnp.zeros_like(gdd))
            cumulative = np.asarray(jnp.cumsum(crop_gdd))
            for offset, ((tmin, tmax), (eto,), (rain,), daily_gdd, total) in enumerate(
                    zip(temps, etos, rains, np.asarray(gdd), cumulative)):
                daynr = run.simulation_first_day + offset
                current = day_number_to_date(daynr)
                writer.writerow((run_index, daynr, current.day, current.month, current.year,
                                 rain, eto, tmin, (tmin + tmax) / 2.0, tmax,
                                 float(daily_gdd), float(total)))
                row_count += 1
    return row_count
