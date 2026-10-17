"""Ottawa runner for the coupled water and soil-evaporation recurrence."""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .climate import _reference, _slice, day_number_to_date
from .legacy import Project, read_annotated_values, read_climate_series
from .soil import discretize_profile, drainage_tau, read_soil_profile, resize_for_root_depth
from .water_evap import run_water_evap_trajectory


def _crop_value(values, prefix):
    return next(value for name, value in values.items() if name.startswith(prefix))


def run_water_evap_stage(project: Project, project_file: Path, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    theta = None; surface_store = 0.0; evap_depth = 0.15; stage2_start = -1.0; rows = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run", "daynr", "day", "month", "year", "rain", "eto",
                         "infiltration", "runoff", "drainage", "evaporation",
                         "water_mm", "evap_surface_water", "evap_depth",
                         "stage2_start", "balance_error",
                         *(f"theta{i:02d}" for i in range(1, 13))))
        for run_number, run in enumerate(project.runs, 1):
            soil_path = _reference(run, "Soil profile").resolve(project_file)
            crop_path = _reference(run, "Crop").resolve(project_file)
            rain_path = _reference(run, "Rain").resolve(project_file)
            eto_path = _reference(run, "Reference ET").resolve(project_file)
            assert soil_path and crop_path and rain_path and eto_path
            soil = read_soil_profile(soil_path); layer = soil.layers[0]
            if len(soil.layers) != 1:
                raise NotImplementedError("water-evap stage currently supports one soil layer")
            crop = read_annotated_values(crop_path)
            compartments = resize_for_root_depth(
                soil, discretize_profile(soil),
                _crop_value(crop, "Maximum effective rooting depth"))
            rain_rows = _slice(read_climate_series(rain_path, 1),
                               run.simulation_first_day, run.simulation_last_day, "rain")
            eto_rows = _slice(read_climate_series(eto_path, 1),
                              run.simulation_first_day, run.simulation_last_day, "ETo")
            if theta is None:
                theta = jnp.full((12,), layer.field_capacity / 100.0,
                                 dtype=jnp.float64)
            trajectory = run_water_evap_trajectory(
                theta, [x[0] for x in rain_rows], [x[0] for x in eto_rows],
                compartments.thickness, [layer.gravel_volume] * 12,
                [layer.field_capacity / 100.0] * 12, layer.saturation,
                layer.field_capacity, layer.wilting_point, drainage_tau(layer.ksat),
                layer.ksat, soil.curve_number, soil.readily_evaporable_water,
                initial_surface_water_mm=surface_store,
                initial_evap_depth_m=evap_depth,
                initial_stage2_start_pct=stage2_start)
            arrays = [np.asarray(value) for value in trajectory]
            capacity = jnp.asarray(compartments.thickness, jnp.float64) * 1000.0 * (
                1.0 - layer.gravel_volume / 100.0)
            storage = np.asarray(jnp.sum(trajectory.theta * capacity[None, :], axis=1))
            for offset in range(len(rain_rows)):
                daynr = run.simulation_first_day + offset; current = day_number_to_date(daynr)
                writer.writerow((run_number, daynr, current.day, current.month, current.year,
                    rain_rows[offset][0], eto_rows[offset][0], arrays[3][offset],
                    arrays[1][offset], arrays[2][offset], arrays[4][offset], storage[offset],
                    arrays[5][offset], arrays[6][offset], arrays[7][offset], arrays[8][offset],
                    *arrays[0][offset]))
                rows += 1
            theta = trajectory.theta[-1]
            surface_store = float(trajectory.evap_surface_water[-1])
            evap_depth = float(trajectory.evap_depth[-1])
            stage2_start = float(trajectory.stage2_start[-1])
    return rows
