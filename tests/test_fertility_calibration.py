"""Full numerical reference fertility calibration from legacy input files."""
from dataclasses import replace
from ctypes.util import find_library
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.fertility import fertility_biomass_fraction
from aquacrop_jax.fertility_calibration import (prepare_seeded_forage_reference,
                                               reference_fertility_calibration)
from aquacrop_jax.legacy import read_project

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "aquacrop-fortran/testcase/LIST/Ottawa.PRM"
FIXTURE = json.loads((ROOT / "tests/fixtures/fertility_setup_fortran73.json").read_text())


@pytest.fixture(scope="module")
def prepared():
    return prepare_seeded_forage_reference(PROJECT, read_project(PROJECT).runs[0])


@pytest.fixture(scope="module")
def native_fit_source_probe(tmp_path_factory):
    """Compile the verbatim fit block from the immutable native routine.

    The complete native routine generates its own biomass samples. This probe
    exposes only its unchanged fit block so we can feed identical JAX/native
    samples, instead of conflating input rounding with fit arithmetic error.
    """
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    if not shutil.which("gfortran") or not find_library("gfortran") or not objects:
        pytest.skip("Native fit extraction requires gfortran and reference objects")
    source = (reference / "preparefertilitysalinity.f90").read_text()
    routine = source.split("subroutine StressBiomassRelationshipForTnxReference(", 1)[1].split(
        "end subroutine StressBiomassRelationshipForTnxReference", 1)[0]
    types = routine[routine.index("    type StressIndexes"):]
    types = types[:types.index("end type StressIndexes") + len("end type StressIndexes")]
    fit = routine[routine.index("    ! 5. Stress - Biomass relationship"):]
    program = """program extracted_native_fit
use ac_kinds, only: dp, int8, int32
use ac_utils, only: roundc
use ac_global, only: undef_int
implicit none
""" + types + """
type(StressIndexes) :: StressMatrix(8)
integer(int32) :: count, row
integer(int8) :: Si
real(dp) :: x(8), b0, b1, b2, BM10, BM20, BM30, BM40, BM50, BM60, BM70
real(dp) :: Yavg, X1avg, X2avg, y, x1, x2, x1y, x2y, x1Sq, x2Sq, x1x2
real(dp) :: SUMx1y, SUMx2y, SUMx1Sq, SUMx2Sq, SUMx1x2
read(*,*) count
do row=1,count
  read(*,*) x
  do Si=1,8
    StressMatrix(Si)%StressProc=10*(Si-1)
    StressMatrix(Si)%BioMProc=x(Si)
    StressMatrix(Si)%BioMSquare=x(Si)*x(Si)
  end do
""" + fit + """
  write(*,'(3(es24.16,1x))') b0,b1,b2
end do
end program extracted_native_fit
"""
    directory = tmp_path_factory.mktemp("native-fit-source")
    path, probe = directory / "fit.f90", directory / "probe"
    path.write_text(program)
    compiled = subprocess.run(["gfortran", "-O0", "-I", str(reference), "-o", str(probe),
        str(path), *map(str, objects)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    return probe


def test_reference_preparation_uses_project_inputs_without_native_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No native process or fixture may enter reference preparation")
    monkeypatch.setattr(subprocess, "run", forbidden)
    actual = prepare_seeded_forage_reference(PROJECT, read_project(PROJECT).runs[0])
    p = actual.parameters
    assert actual.management_stress_pct == 50
    assert actual.reference_year == FIXTURE["reference_year"]
    assert actual.gdd_method == 3
    assert p.co2_ppm == FIXTURE["reference_co2"]
    assert p.senescence_exponent == FIXTURE["senescence_exponent"] == 1.
    assert p.thermal_days == (5, 461, 1803, 1803)
    assert actual.loaded_calendar_days == (1, 39, 164, 164)
    np.testing.assert_array_equal(p.calendar_days, [1, 40, 164, 164])
    assert actual.daily_gdd.shape == (164,)
    assert actual.calendar_gdd.shape == (365,)
    assert actual.daily_gdd.dtype == jnp.float64


def test_ottawa_reference_calibration_matches_native_samples_fit_and_fractions(prepared):
    levels = np.asarray(FIXTURE["reference_levels"])
    expected = np.asarray(FIXTURE["biomass_percentages"])
    # Composed native public routines reproduce the original monolithic
    # reference routine, making the intermediate trace an independent oracle.
    np.testing.assert_array_equal(100. * levels[:, -1] / levels[0, -1], expected)
    stresses = jnp.asarray([case["stress"] for case in FIXTURE["cases"]])
    calibrated = jnp.asarray([case["calibrated"] for case in FIXTURE["cases"]])
    fractions = np.asarray([case["fraction"] for case in FIXTURE["cases"]])

    def evaluate(daily, calendar, p):
        result = reference_fertility_calibration(daily, calendar, p)
        fraction = jax.vmap(lambda stress, enabled: fertility_biomass_fraction(
            stress, *result.coefficients, calibrated=enabled))(stresses, calibrated)
        return result, fraction

    for operation in (evaluate, jax.jit(evaluate)):
        result, fraction = operation(prepared.daily_gdd, prepared.calendar_gdd, prepared.parameters)
        np.testing.assert_array_equal(result.full_canopy_sf_day, levels[:, 1])
        np.testing.assert_array_equal(result.full_canopy_sf_gdd, levels[:, 2])
        np.testing.assert_array_equal(result.reduction_growth_pct, levels[:, 3])
        np.testing.assert_array_equal(result.reduction_cover_pct, levels[:, 4])
        np.testing.assert_allclose(result.normalized_biomass, levels[:, -1], rtol=0., atol=1e-12)
        np.testing.assert_allclose(result.sum_kc_top, FIXTURE["reference_sum_kc_top"], rtol=0., atol=1e-12)
        # At most two ULP observed after the 164-day reduction and relative
        # normalization; allow four ULP here without changing inverse parity.
        np.testing.assert_array_max_ulp(result.biomass_percentages, expected, maxulp=4)
        assert float(result.biomass_percentages[0]) == 100.
        np.testing.assert_allclose(fraction, fractions, rtol=0., atol=1e-14)
        assert bool(result.calendar_valid)
        for leaf in jax.tree_util.tree_leaves(result):
            assert np.all(np.isfinite(leaf))


def test_generated_sample_fit_matches_native_with_identical_inputs(prepared, native_fit_source_probe):
    results = [operation(prepared.daily_gdd, prepared.calendar_gdd, prepared.parameters)
               for operation in (reference_fertility_calibration, jax.jit(reference_fertility_calibration))]
    samples = [FIXTURE["biomass_percentages"],
               *[np.asarray(result.biomass_percentages).tolist() for result in results]]
    inputs = str(len(samples)) + "\n" + "\n".join(
        " ".join(format(value, ".17g") for value in row) for row in samples) + "\n"
    captured = subprocess.run([str(native_fit_source_probe)], input=inputs,
        capture_output=True, text=True, check=True)
    native = np.asarray([list(map(float, line.split())) for line in captured.stdout.splitlines()])
    # The extracted source agrees with the original monolithic native object
    # on its own samples before it is used as a like-input arithmetic oracle.
    np.testing.assert_array_equal(native[0], FIXTURE["fitted_coefficients"])
    for result, expected in zip(results, native[1:], strict=True):
        np.testing.assert_array_max_ulp(np.asarray(result.coefficients), expected, maxulp=64)


def test_project_senescence_parameter_materially_changes_reference_fit(prepared):
    operation = jax.jit(reference_fertility_calibration)
    wrong = operation(prepared.daily_gdd, prepared.calendar_gdd,
        prepared.parameters._replace(senescence_exponent=2.5))
    fraction = fertility_biomass_fraction(50, *wrong.coefficients)
    assert abs(float(fraction) - FIXTURE["setup"][4]) > .01


def test_unprepared_perennial_regrowth_is_explicitly_unsupported():
    run = replace(read_project(PROJECT).runs[0], cultivation_year=2)
    with pytest.raises(NotImplementedError, match="first-year sown forage"):
        prepare_seeded_forage_reference(PROJECT, run)


def test_project_program_overrides_match_native_changed_inputs(tmp_path):
    if (not shutil.which("gfortran") or not find_library("gfortran") or
            not (ROOT.parent / "aquacrop-fortran/src/run.o").exists()):
        pytest.skip("Program override oracle requires gfortran and reference objects")
    testcase = tmp_path / "case"
    shutil.copytree(PROJECT.parent.parent, testcase,
                    ignore=shutil.ignore_patterns("OUTP", "OUTP_REF"))
    program = testcase / "PARAM/Ottawa.PPn"
    lines = program.read_text().splitlines()
    for i, line in enumerate(lines):
        if "Exponent of senescence factor" in line:
            lines[i] = "      2.50 : " + line.split(":", 1)[1]
        if "Default method for the calculation of growing degree days" in line:
            lines[i] = "      2 : " + line.split(":", 1)[1]
    program.write_text("\n".join(lines) + "\n")
    project = testcase / "LIST/Ottawa.PRM"
    prepared = prepare_seeded_forage_reference(project, read_project(project).runs[0])
    assert prepared.gdd_method == 2
    assert prepared.parameters.senescence_exponent == 2.5
    output = tmp_path / "native.json"
    result = subprocess.run([sys.executable,
        str(ROOT / "scripts/generate_fertility_setup_oracle.py"),
        "--testcase", str(testcase), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    native = json.loads(output.read_text())
    thermal = dict(zip(native["thermal_fields"], native["thermal_parameters"], strict=True))
    np.testing.assert_array_equal(prepared.parameters.thermal_days,
        [thermal[key] for key in ("emergence_gdd", "full_canopy_gdd", "senescence_gdd", "maturity_gdd")])
    calibrated = jax.jit(reference_fertility_calibration)(
        prepared.daily_gdd, prepared.calendar_gdd, prepared.parameters)
    np.testing.assert_array_max_ulp(calibrated.biomass_percentages,
                                   native["biomass_percentages"], maxulp=4)
    fraction = fertility_biomass_fraction(prepared.management_stress_pct, *calibrated.coefficients)
    np.testing.assert_allclose(fraction, native["setup"][4], rtol=0., atol=1e-14)
