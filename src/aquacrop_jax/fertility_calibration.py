"""Fertility reference sample generation and polynomial calibration."""
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .climate import (_reference, _slice, calendar_days_for_gdd,
                      date_to_day_number, day_number_to_date)
from .fertility import (FertilityBiomassCoefficients, days_to_reach_cover,
    fit_fertility_biomass_response, soil_fertility_effect, time_to_max_canopy_sf)
from .kernels import degrees_day
from .legacy import read_annotated_values, read_climate_series
from .legacy_inputs import (co2_for_period, field_fertility_stress,
                            program_temperature_parameters)
from .reference_biomass import normalized_biomass_trajectory
from .reference_calendar import ReferenceCropCalendar, gdd_reference_calendar
from .reference_climate import (crop_reference_temperature, daily_reference_temperature,
    reference_calendar_days, reference_monthly_temperature)
from .reference_et import reference_crop_transpiration
from .rounding import round_fortran

F64 = jnp.float64


class ReferenceFertilityParameters(NamedTuple):
    calendar_days: tuple
    thermal_days: tuple
    initial_cover: object
    maximum_cover: object
    calendar_growth: object
    thermal_growth: object
    calendar_decline: object
    thermal_decline: object
    kc_top: object
    kc_decline_pct: object
    late_evaporation_pct: object
    co2_ppm: object
    minimum_gdd: object
    water_productivity: object
    fertility_shapes: tuple
    flowering_day: object = 0
    flowering_length: object = 0
    determinacy_linked: object = False
    reproductive_crop: object = False
    gdd_mode: object = True
    initial_cover_day: object = 0
    initial_cover_gdd: object = 0
    wp_yield_pct: object = 100.
    yield_formation_days: object = 0
    switch_day: object = 0
    senescence_exponent: object = 1.


class ReferenceFertilityCalibration(NamedTuple):
    coefficients: FertilityBiomassCoefficients
    biomass_percentages: jax.Array
    normalized_biomass: jax.Array
    sum_kc_top: jax.Array
    full_canopy_sf_day: jax.Array
    full_canopy_sf_gdd: jax.Array
    reduction_growth_pct: jax.Array
    reduction_cover_pct: jax.Array
    calendar_valid: jax.Array


def reference_fertility_calibration(daily_gdd, calendar_gdd, parameters):
    """StressBiomassRelationshipForTnxReference with explicit climate arrays.

    daily_gdd is the file-based climate over the static calibration season;
    calendar_gdd is the 365-day native-SP array used for calendar conversion.
    Reproductive yield/switch periods must be prepared by the caller, just as
    the native routine prepares them before calling Bnormalized.
    """
    p = parameters
    if len(p.fertility_shapes) != 4:
        raise ValueError("Fertility calibration requires four response shapes")
    common = (daily_gdd, p.calendar_days, p.thermal_days, p.initial_cover,
        p.maximum_cover, p.calendar_growth, p.thermal_growth,
        p.calendar_decline, p.thermal_decline, p.kc_top, p.kc_decline_pct,
        p.late_evaporation_pct, p.co2_ppm, p.minimum_gdd)
    initial = dict(gdd_mode=p.gdd_mode, initial_cover_day=p.initial_cover_day,
                   initial_cover_gdd=p.initial_cover_gdd,
                   senescence_exponent=p.senescence_exponent)
    sum_kc_top = reference_crop_transpiration(*common, **initial).sum_kc

    def sample(stress):
        effect = soil_fertility_effect(stress, *p.fertility_shapes)
        timing = time_to_max_canopy_sf(p.initial_cover, p.calendar_growth,
            p.maximum_cover, *p.calendar_days[:3], p.flowering_day,
            p.flowering_length, p.determinacy_linked,
            effect.reduction_growth_pct, effect.reduction_cover_pct, stress)
        changed = (effect.reduction_growth_pct != 0.) | (effect.reduction_cover_pct != 0.)
        # Preserve the native call even though a field called GDDL12SF
        # receives calendar days obtained by treating L12SF as the GDD target.
        converted = reference_calendar_days(calendar_gdd, timing.full_canopy_day)
        use_conversion = jnp.asarray(p.gdd_mode) & changed
        full_sf_gdd = jnp.where(use_conversion, converted.days, p.thermal_days[1])
        denominator = jnp.asarray(p.thermal_days[2], F64) - full_sf_gdd
        ratio = jnp.where(use_conversion & (full_sf_gdd < p.thermal_days[2]),
            (jnp.asarray(p.calendar_days[2], F64) - timing.full_canopy_day) /
            jnp.where(denominator != 0., denominator, 1.), 1.)
        production = normalized_biomass_trajectory(*common, **initial,
            sum_kc_top=sum_kc_top, full_canopy_sf_day=timing.full_canopy_day,
            full_canopy_sf_gdd=full_sf_gdd, days_per_gdd=ratio,
            reduction_growth_pct=timing.reduction_growth_pct,
            reduction_cover_pct=timing.reduction_cover_pct,
            canopy_decline=effect.canopy_decline,
            water_productivity=p.water_productivity, fertility_stress_pct=stress,
            reduction_wp_pct=effect.reduction_water_productivity_pct,
            reproductive_crop=p.reproductive_crop, flowering_day=p.flowering_day,
            wp_yield_pct=p.wp_yield_pct, yield_formation_days=p.yield_formation_days,
            switch_day=p.switch_day)
        return (production.biomass, timing.full_canopy_day, full_sf_gdd,
                timing.reduction_growth_pct, timing.reduction_cover_pct,
                (~use_conversion) | converted.valid)

    biomass, full_days, full_gdd, red_growth, red_cover, valid = jax.vmap(sample)(
        jnp.arange(8, dtype=jnp.int32) * 10)
    reference = biomass[0]
    percentages = jnp.where(reference > .00001,
        100. * biomass / jnp.where(reference > .00001, reference, 1.), 100.)
    percentages = percentages.at[0].set(100.)
    coefficients = fit_fertility_biomass_response(percentages)
    return ReferenceFertilityCalibration(coefficients, percentages, biomass,
        sum_kc_top, full_days, full_gdd, red_growth, red_cover, jnp.all(valid))


@dataclass(frozen=True)
class PreparedFertilityReference:
    daily_gdd: jax.Array
    calendar_gdd: jax.Array
    parameters: ReferenceFertilityParameters
    reference_calendar: ReferenceCropCalendar
    loaded_calendar_days: tuple[int, int, int, int]
    management_stress_pct: int
    reference_year: int
    gdd_method: int


def prepare_seeded_forage_reference(project_file: Path, run):
    """Load the calibrated, sown first-year forage GDD path from real inputs.

    Perennial regrowth preparation and calendar-mode crops are separate
    adapters; the numerical kernel itself supports their explicit parameters.
    No native executable, generated file or oracle fixture enters this path.
    """
    project_file = Path(project_file)
    def path(role):
        return _reference(run, role).resolve(project_file)
    crop_path = path("Crop")
    crop = read_annotated_values(crop_path)
    lines = crop_path.read_text().splitlines()
    if (int(lines[3].split(":", 1)[0]) != 4 or run.cultivation_year != 1 or
            int(lines[4].split(":", 1)[0]) != 1 or
            not next((bool(v) for key, v in crop.items()
                if key.startswith("Crop is sown in 1st year (for perennials)")), True)):
        raise NotImplementedError("Reference adapter requires first-year sown forage")
    def value(prefix):
        return next(v for key, v in crop.items() if key.startswith(prefix))
    if int(value("Determination of crop cycle")) != 0:
        raise NotImplementedError("Reference adapter requires GDD crop mode")
    shapes = tuple(value(prefix) for prefix in (
        "Shape factor for the response of canopy expansion",
        "Shape factor for the response of maximum canopy cover",
        "Shape factor for the response of crop Water Productivity",
        "Shape factor for the response of decline of canopy cover"))
    if all(shape > 24.9 for shape in shapes):
        raise ValueError("Crop soil fertility response is not calibrated")
    program = program_temperature_parameters(project_file)
    senescence_exponent, method = program.senescence_exponent, program.gdd_method
    if path("Temperature") is None:
        raise NotImplementedError("Reference adapter requires a daily temperature file")
    series = read_climate_series(path("Temperature"), 2)
    actual = np.asarray(_slice(series, run.crop_first_day, run.crop_last_day, "Temperature"))
    base, upper = value("Base temperature"), value("Upper temperature")
    actual_gdd = degrees_day(base, upper, actual[:, 0], actual[:, 1], method)
    # GrowingDegreeDays accumulates chronologically before roundc. A tree
    # reduction can select the other integer at a half-GDD boundary.
    end_total, _ = jax.lax.scan(lambda total, day: (total + day, None),
                               jnp.asarray(0., F64), actual_gdd)
    end_gdd = int(round_fortran(end_total))
    difference = (value("GDDays: from sowing to maturity") -
                  value("GDDays: from sowing to start senescence"))
    sen_gdd = min(end_gdd - int(difference), end_gdd)
    initial = value("Soil surface covered by an individual seedling") * value("Number of plants per hectare") / 1e8
    maximum = value("Maximum canopy cover")
    growth = value("CGC for GGDays")
    emergence_gdd = int(value("GDDays: from sowing to emergence"))
    full_gdd = min(int(days_to_reach_cover(.98 * maximum, initial, maximum,
                                         growth, emergence_gdd)), end_gdd)
    season = run.crop_last_day - run.crop_first_day + 1
    loaded_calendar = (int(calendar_days_for_gdd(actual_gdd, emergence_gdd)),
        int(calendar_days_for_gdd(actual_gdd, full_gdd)),
        season if sen_gdd >= end_gdd else int(calendar_days_for_gdd(actual_gdd, sen_gdd)), season)
    reference = reference_monthly_temperature(series)
    annual = daily_reference_temperature(reference.temperature)
    start = day_number_to_date(run.crop_first_day)
    day_of_year = sum((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[:start.month - 1]) + start.day
    temperature = crop_reference_temperature(annual, day_of_year)
    native_sp = temperature.astype(jnp.float32).astype(F64)
    calendar_gdd = degrees_day(base, upper, native_sp[:, 0], native_sp[:, 1], method)
    thermal = (emergence_gdd, full_gdd, sen_gdd, end_gdd)
    # LoadCrop resets both flowering clocks for forage; only code 1 links
    # determinancy (all other codes select false in the native select case).
    determinacy = value("Crop determinancy") == 1
    calendar = gdd_reference_calendar(calendar_gdd, loaded_calendar + (0, 0),
        thermal + (0, 0), growth, value("CDC for GGDays"), maximum,
        value("Reference Harvest Index"), value("GDDays: building-up of Harvest Index"),
        forage_crop=True)
    if not bool(calendar.valid):
        raise ValueError("Reference climate cannot produce a finite crop calendar")
    reference_days = int(calendar.maturity_day)
    crop_temperature = crop_reference_temperature(annual, day_of_year, days=reference_days)
    daily_gdd = degrees_day(base, upper, crop_temperature[:, 0], crop_temperature[:, 1], method)
    year = reference.reference_year
    co2 = (369.41 if year == 2000 else co2_for_period(path("Atmospheric CO2"),
        date_to_day_number(date(year, 1, 1)), date_to_day_number(date(year, 12, 31))))
    p = ReferenceFertilityParameters(
        (calendar.emergence_day, calendar.full_canopy_day, calendar.senescence_day, calendar.maturity_day),
        thermal, initial, maximum, calendar.calendar_growth, growth,
        calendar.calendar_decline, value("CDC for GGDays"),
        value("Crop coefficient when canopy is complete"), value("Cumulative decrease"),
        value("Effect of canopy cover in reducing soil evaporation"), co2,
        value("Minimum growing degrees required for full crop transpiration"),
        value("Water Productivity normalized for ETo and CO2 (WP*)"), shapes,
        wp_yield_pct=value("Water Productivity normalized for ETo and CO2 during yield formation"),
        flowering_day=calendar.flowering_day, flowering_length=calendar.flowering_length,
        determinacy_linked=determinacy,
        senescence_exponent=senescence_exponent)
    return PreparedFertilityReference(daily_gdd, calendar_gdd, p, calendar,
        loaded_calendar, field_fertility_stress(path("Field management")), year, method)
