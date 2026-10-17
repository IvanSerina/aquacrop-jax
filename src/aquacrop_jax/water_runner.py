"""End-to-end water-only runner for the progressive Ottawa port."""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .climate import _reference, _slice, day_number_to_date
from .hydrology import run_water_trajectory
from .legacy import Project, read_annotated_values, read_climate_series
from .soil import (discretize_profile, drainage_tau, read_soil_profile,
                   resize_for_root_depth)


def _parameter(values: dict[str, float], prefix: str) -> float:
    try:
        return next(value for name, value in values.items() if name.startswith(prefix))
    except StopIteration as exc:
        raise ValueError(f"Missing crop parameter: {prefix}") from exc


def run_water_stage(project: Project, project_file: Path, output: Path) -> int:
    """Run the current water-only recurrence over all runs in one PRM."""
    output.parent.mkdir(parents=True, exist_ok=True)
    current_theta = None
    rows = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run", "daynr", "day", "month", "year", "rain",
                         "infiltration", "runoff", "drainage", "water_mm",
                         "balance_error", *(f"theta{i:02d}" for i in range(1, 13))))
        for run_number, run in enumerate(project.runs, 1):
            soil_path = _reference(run, "Soil profile").resolve(project_file)
            crop_path = _reference(run, "Crop").resolve(project_file)
            rain_path = _reference(run, "Rain").resolve(project_file)
            assert soil_path and crop_path and rain_path
            soil = read_soil_profile(soil_path)
            if len(soil.layers) != 1:
                raise NotImplementedError("water milestone currently supports one soil layer")
            crop = read_annotated_values(crop_path)
            root_max = _parameter(crop, "Maximum effective rooting depth")
            compartments = resize_for_root_depth(
                soil, discretize_profile(soil), root_max)
            layer = soil.layers[0]
            rain_rows = _slice(read_climate_series(rain_path, 1),
                               run.simulation_first_day, run.simulation_last_day, "rain")
            rains = jnp.asarray([row[0] for row in rain_rows], jnp.float64)
            if current_theta is None:
                current_theta = jnp.full((len(compartments.thickness),),
                                         layer.field_capacity / 100.0, jnp.float64)
            trajectory = run_water_trajectory(
                current_theta, rains, compartments.thickness,
                [layer.gravel_volume] * len(compartments.thickness),
                [layer.field_capacity / 100.0] * len(compartments.thickness),
                layer.saturation, layer.field_capacity, drainage_tau(layer.ksat),
                layer.ksat, soil.curve_number)
            capacity = jnp.asarray(compartments.thickness, jnp.float64) * 1000.0 * (
                1.0 - layer.gravel_volume / 100.0)
            storage = np.asarray(jnp.sum(trajectory.theta * capacity[None, :], axis=1))
            arrays = [np.asarray(x) for x in trajectory]
            for offset in range(len(rain_rows)):
                daynr = run.simulation_first_day + offset
                current = day_number_to_date(daynr)
                writer.writerow((run_number, daynr, current.day, current.month, current.year,
                                 rain_rows[offset][0], arrays[3][offset], arrays[1][offset],
                                 arrays[2][offset], storage[offset], arrays[4][offset],
                                 *arrays[0][offset]))
                rows += 1
            current_theta = trajectory.theta[-1]
    return rows
