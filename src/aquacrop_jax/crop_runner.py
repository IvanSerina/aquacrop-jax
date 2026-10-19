"""Integrated progressive runner: climate, crop demand, water/salt and production.

The CSV is a diagnostic trajectory. Actual canopy feedback, fertility calibration,
cutting/regrowth and perennial assimilate transfers still prevent legacy OUT parity.
Those omissions are recorded alongside each trace; no oracle values enter simulation.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from jax import lax, vmap
import jax.numpy as jnp
import numpy as np

from .canopy import calculate_potential_et, potential_canopy_gdd
from .climate import (_reference, _slice, calendar_days_for_gdd,
                      date_to_day_number, day_number_to_date)
from .kernels import degrees_day
from .legacy import Project, read_annotated_values, read_climate_series
from .legacy_inputs import (co2_for_period, groundwater_forcing,
                            initial_conditions, manual_irrigation_forcing)
from .production import run_production_trajectory
from .roots import actual_rooting_depth_gdd
from .rounding import round_fortran
from .salinity import SaltState
from .soil import (discretize_profile, drainage_tau, read_soil_profile,
                   resize_for_root_depth)
from .water_crop import run_water_crop_trajectory
from .water_runner import _parameter


LIMITATIONS = [
    'Potential canopy only: actual canopy water/salinity feedback is not yet coupled.',
    'MAN fertility calibration, cuttings/regrowth and perennial assimilate transfers are not yet coupled.',
    'Delayed anaerobic stress and germination delays are not yet coupled.',
    'Calibrated ECsw polynomial must be supplied explicitly to enable stomatal salt feedback.',
    'Only homogeneous soils and daily climate are supported; CSV is not legacy day.OUT.',
]


@dataclass(frozen=True)
class CropRunResult:
    run: int
    columns: dict[str, np.ndarray]


def simulate_project(project: Project, project_file: Path, salt_stomatal_coefficients=None):
    """Calculate every simulation day, resetting/preserving explicit state per SW0."""
    theta = salt = previous_geometry = None
    surface, evaporation_depth, stage2_start = 0.0, .15, -1.0
    previous_end = None
    for number, run in enumerate(project.runs, 1):
        def path(role):
            return _reference(run, role).resolve(project_file)

        soil = read_soil_profile(path('Soil profile'))
        if len(soil.layers) != 1:
            raise NotImplementedError('Coupled runner requires a homogeneous soil profile')
        layer = soil.layers[0]
        crop = read_annotated_values(path('Crop'))
        crop_type = int(path('Crop').read_text().splitlines()[3].split(':', 1)[0])
        value = lambda prefix: _parameter(crop, prefix)
        zmin, zmax = value('Minimum effective rooting depth'), value('Maximum effective rooting depth')
        if crop_type == 4:
            # LoadSimulationRun calls AdjustYearPerennials before root expansion.
            # Older crop files default RootMinYear1 to their ordinary RootMin.
            first_year_min = next((float(v) for key, v in crop.items()
                if key.startswith('Minimum effective rooting depth (m) in first year')), zmin)
            zmin = first_year_min if run.cultivation_year == 1 else zmax
        profile = resize_for_root_depth(soil, discretize_profile(soil), zmax)
        dz = np.asarray(profile.thickness)
        days = np.arange(run.simulation_first_day, run.simulation_last_day + 1)
        if days.size == 0:
            raise ValueError('Simulation period must contain at least one day')
        in_crop = (days >= run.crop_first_day) & (days <= run.crop_last_day)
        dap = days - run.crop_first_day + 1

        def climate(role, columns):
            return np.asarray(_slice(read_climate_series(path(role), columns),
                run.simulation_first_day, run.simulation_last_day, role))

        rain, eto = climate('Rain', 1)[:, 0], climate('Reference ET', 1)[:, 0]
        temperature_record = read_climate_series(path('Temperature'), 2)
        temperature = np.asarray(_slice(temperature_record, run.simulation_first_day,
                                        run.simulation_last_day, 'Temperature'))
        gdd = degrees_day(value('Base temperature'), value('Upper temperature'),
                          temperature[:, 0], temperature[:, 1], 3)
        cumulative = jnp.cumsum(jnp.where(in_crop, gdd, 0.0))
        initial = _reference(run, 'Initial conditions')
        keep = initial.name == 'KeepSWC'
        if keep:
            if theta is None:
                raise ValueError('KeepSWC requires a previous simulation run')
            if previous_end + 1 != run.simulation_first_day:
                raise NotImplementedError('KeepSWC with unmodelled days between runs is unsupported')
            if previous_geometry != tuple(dz):
                raise NotImplementedError('KeepSWC compartment remapping is not yet ported')
            initial_biomass, initial_ece = 0.0, None
        else:
            conditions = initial_conditions(initial.resolve(project_file), dz,
                                             layer.field_capacity, layer.saturation)
            if conditions.canopy_pct != -9.0 or conditions.root_depth != -9.0:
                raise NotImplementedError('Explicit initial canopy/root development needs crop-state recurrence')
            theta, initial_ece, initial_biomass = conditions.theta, conditions.ece, conditions.biomass
            salt = None
            surface, evaporation_depth, stage2_start = 0.0, .15, -1.0

        cycle = int(run.crop_last_day - run.crop_first_day + 1)
        mode = int(value('Determination of crop cycle'))
        if mode != 0:
            raise NotImplementedError('Calendar root development is not yet ported')
        root = actual_rooting_depth_gdd(dap, cycle, value('GDDays: from sowing to emergence'),
            value('GDDays: from sowing to maximum rooting depth'), cumulative,
            zmin, zmax, value('Shape factor describing root zone expansion'))
        initial_cover = value('Soil surface covered by an individual seedling') * value('Number of plants per hectare') / 1e8
        maximum_cover = value('Maximum canopy cover')
        # AdjustCalendarCrop converts GDD milestones with the actual daily
        # temperature record, starting on Crop.Day1.  Calendar values in the
        # CRO file are defaults and are not the loaded GDD-mode crop calendar.
        record_first_day = date_to_day_number(date(
            temperature_record.first_year, temperature_record.first_month,
            temperature_record.first_day))
        offset = run.crop_first_day - record_first_day
        if offset < 0 or offset >= len(temperature_record.values):
            raise ValueError('Temperature record does not cover crop sowing day')
        future_temperature = np.asarray(temperature_record.values[offset:])
        future_gdd = degrees_day(value('Base temperature'), value('Upper temperature'),
                                 future_temperature[:, 0], future_temperature[:, 1], 3)
        emergence_day = int(calendar_days_for_gdd(
            future_gdd, int(value('GDDays: from sowing to emergence'))))
        full_canopy_gdd = int(value('GDDays: from sowing to emergence')) + int(
            round_fortran(np.log((.25 * maximum_cover * maximum_cover / initial_cover) /
                                 (maximum_cover - .98 * maximum_cover)) /
                          value('CGC for GGDays')))
        full_canopy_gdd = min(full_canopy_gdd, int(value('GDDays: from sowing to maturity')))
        full_canopy_day = int(calendar_days_for_gdd(future_gdd, full_canopy_gdd))
        if emergence_day == -9 or full_canopy_day == -9:
            raise NotImplementedError('Temperature record ends before a GDD crop milestone')
        canopy = potential_canopy_gdd(cumulative, value('GDDays: from sowing to emergence'),
            value('GDDays: from sowing to start senescence'), value('GDDays: from sowing to maturity'),
            initial_cover, maximum_cover, value('CGC for GGDays'), value('CDC for GGDays'))
        canopy = jnp.where(in_crop, canopy, 0.0)
        # BUDGET_module recomputes ET's calendar DAP from the rounded GDD
        # total.  This may differ from daynr - Crop.Day1 + 1 on cold days.
        et_dap = vmap(calendar_days_for_gdd, in_axes=(None, 0))(
            future_gdd, round_fortran(cumulative))
        co2 = co2_for_period(path('Atmospheric CO2'), run.crop_first_day, run.crop_last_day)
        potential = calculate_potential_et(et_dap, emergence_day,
            full_canopy_day, value('Calendar Days: from sowing to start senescence'), cycle,
            canopy, eto, value('Crop coefficient when canopy is complete'),
            value('Cumulative decrease'), maximum_cover, lax.associative_scan(jnp.maximum, canopy),
            value('Effect of canopy cover in reducing soil evaporation'), co2, gdd,
            value('Minimum growing degrees required for full crop transpiration'))
        irrigation, irrigation_ec = manual_irrigation_forcing(path('Irrigation management'), days, run.crop_first_day)
        groundwater, groundwater_ec = groundwater_forcing(path('Groundwater table'), days)
        water = run_water_crop_trajectory(theta, rain, eto, root, potential.transpiration_mm,
            dz, [layer.gravel_volume] * len(dz), [layer.field_capacity / 100.] * len(dz),
            layer.saturation, layer.field_capacity, layer.wilting_point,
            drainage_tau(layer.ksat), layer.ksat, soil.curve_number,
            soil.readily_evaporable_water, value('Soil water depletion fraction for stomatal'),
            value('Maximum root water extraction (m3water/m3soil.day) in top'),
            value('Maximum root water extraction (m3water/m3soil.day) in bottom'),
            stomatal_shape=value('Shape factor for water stress coefficient for stomatal'),
            initial_surface_water_mm=surface, initial_evap_depth_m=evaporation_depth,
            initial_stage2_start_pct=stage2_start, daily_irrigation=irrigation,
            groundwater_depth_m=groundwater, capillary_a=[layer.capillary_a] * len(dz),
            capillary_b=[layer.capillary_b] * len(dz), minimum_root_depth_m=zmin,
            initial_salt=salt, initial_ece=initial_ece, irrigation_ec=irrigation_ec,
            groundwater_ec=groundwater_ec, salinity_minimum_ece=value('Electrical Conductivity of soil saturation extract at which crop starts'),
            salinity_maximum_ece=value('Electrical Conductivity of soil saturation extract at which crop can no longer'),
            salinity_response_pct=value('Calibrated response'),
            salt_stomatal_coefficients=salt_stomatal_coefficients,
            potential_evaporation=potential.evaporation_mm,
            salinity_growing_mask=in_crop)
        reproductive = crop_type in (2, 3)
        # DetermineBiomassAndYield passes dayi - Crop.Day1 to HarvestIndexDay.
        # The diagnostic/output DAP column remains one-based.
        production = run_production_trajectory(water.transpiration, potential.transpiration_mm,
            eto, jnp.full(days.shape, co2), dap - 1, value('Water Productivity normalized for ETo and CO2 (WP*)'),
            value('Sink strength'), value('Calendar Days: from sowing to flowering'),
            value('Reference Harvest Index'), value('Reference Harvest Index') / value('Building up of Harvest Index'),
            value('Water Productivity normalized for ETo and CO2 during yield formation'),
            reproductive_crop=reproductive, grain_crop=(crop_type == 2), initial_biomass=initial_biomass)
        dates = [day_number_to_date(int(day)) for day in days]
        storage = np.sum(np.asarray(water.theta) * (1000. * dz * (1. - layer.gravel_volume / 100.)), axis=1)
        columns = dict(daynr=days, day=np.array([d.day for d in dates]),
            month=np.array([d.month for d in dates]), year=np.array([d.year for d in dates]),
            dap=dap, et_dap=et_dap, rain=rain, eto=eto,
            tmin=temperature[:, 0], tmax=temperature[:, 1],
            gdd=gdd, sum_gdd=cumulative, co2=np.full(days.shape, co2), root_depth=root,
            canopy=canopy, potential_evaporation=potential.evaporation_mm,
            potential_transpiration=potential.transpiration_mm, water_mm=storage)
        for field in ('runoff', 'drainage', 'infiltration', 'evaporation', 'transpiration',
                      'irrigation', 'capillary_rise', 'groundwater_inflow', 'residual',
                      'salt_infiltrated', 'salt_drained', 'salt_groundwater', 'salt_residual',
                      'rootzone_ece', 'salinity_stress', 'mean_salinity_stress_pct',
                      'salinity_growing_days', 'salinity_stomatal_reduction_pct'):
            columns[field] = getattr(water, field)
        columns['salt_g_m2'] = jnp.sum(water.salt_dissolved + water.salt_deposited, axis=(1, 2))
        for field in production._fields:
            columns[field] = getattr(production, field)
        for index in range(len(dz)):
            columns[f'theta{index + 1:02d}'] = water.theta[:, index]
            columns[f'ece{index + 1:02d}'] = water.ece[:, index]
        yield CropRunResult(number, {k: np.asarray(v) for k, v in columns.items()})
        theta = water.theta[-1]
        salt = SaltState(water.salt_dissolved[-1], water.salt_deposited[-1])
        surface, evaporation_depth, stage2_start = [float(v[-1]) for v in
            (water.evap_surface_water, water.evap_depth, water.stage2_start)]
        previous_end, previous_geometry = int(days[-1]), tuple(dz)


def run_crop_stage(project: Project, project_file: Path, output: Path,
                   salt_stomatal_coefficients=None):
    results = list(simulate_project(project, project_file, salt_stomatal_coefficients))
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        names = list(results[0].columns)
        writer.writerow(['run', *names])
        for result in results:
            if list(result.columns) != names:
                raise ValueError('Compartment count must remain constant across the trace')
            for values in zip(*(result.columns[k] for k in names)):
                writer.writerow([result.run, *values])
                rows += 1
    metadata = dict(stage='water-crop', daily_rows=rows, legacy_out_parity=False,
        limitations=LIMITATIONS, salt_stomatal_coefficients=salt_stomatal_coefficients,
        maximum_water_residual_mm=max(float(np.max(np.abs(r.columns['residual']))) for r in results),
        maximum_salt_residual_g_m2=max(float(np.max(np.abs(r.columns['salt_residual']))) for r in results))
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return rows
