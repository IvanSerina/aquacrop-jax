"""Unrounded native HI parity, with the crop-kind wiring tested through the runner."""
from dataclasses import replace
from ctypes.util import find_library
import json
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from aquacrop_jax.crop_runner import simulate_project
from aquacrop_jax.legacy import Project, read_project
from aquacrop_jax.production import reference_harvest_index, run_production_trajectory

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/production_hi_fortran73.json").read_text())
CASES = FIXTURE["cases"]


def case_id(case):
    return f"kind{case['crop_file_kind']}-HI{case['maximum_hi_pct']}-rate{case['daily_hi_rate_pct']}"


HI_VECTOR = jax.vmap(reference_harvest_index, in_axes=(0, None, None, None, None))
HI_COMPILED = jax.jit(HI_VECTOR)


@jax.jit
def hi_scan(days, flowering, maximum, rate, grain):
    def body(carry, day):
        return carry, reference_harvest_index(day, flowering, maximum, rate, grain)
    return lax.scan(body, (), days)[1]


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_harvest_index_matches_native_boundaries_eager_jit_and_scan(case):
    days = jnp.asarray([r["dap"] for r in case["records"]], jnp.float64)
    expected = np.asarray([r["hi"] for r in case["records"]])
    arguments = (days, case["flowering_day"], case["maximum_hi_pct"],
                 case["daily_hi_rate_pct"], case["crop_file_kind"] == 2)
    # Native diagnostic measured errors <= 7.2e-15 in the unaffected cases.
    # 1e-12 absolute retains ample margin for float64 exp and fused operations.
    for operation in (HI_VECTOR, HI_COMPILED, hi_scan):
        result = operation(*arguments)
        assert result.dtype == jnp.float64
        np.testing.assert_allclose(result, expected, rtol=0., atol=1e-12)
    if case["crop_file_kind"] in (2, 3):
        count = len(days)
        trajectory = run_production_trajectory(
            jnp.full(count, 2.), jnp.full(count, 3.), jnp.full(count, 5.),
            jnp.full(count, 369.41), days, 15., 100., case["flowering_day"],
            case["maximum_hi_pct"], case["daily_hi_rate_pct"],
            reproductive_crop=True, grain_crop=(case["crop_file_kind"] == 2))
        np.testing.assert_allclose(trajectory.harvest_index_pct, expected, rtol=0., atol=1e-12)
        biomass = .06 * np.arange(1, count + 1)
        np.testing.assert_allclose(trajectory.biomass, biomass, rtol=0., atol=1e-12)
        np.testing.assert_allclose(trajectory.yield_mass, biomass * expected / 100., rtol=0., atol=1e-12)


@pytest.fixture(scope="module")
def production_probe(tmp_path_factory):
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    if not shutil.which("gcc") or not objects or not find_library("gfortran"):
        pytest.skip("Optional native probe requires gcc and prebuilt reference objects")
    probe = tmp_path_factory.mktemp("production-probe") / "probe"
    result = subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/production_probe.c"), *map(str, objects), "-l:libgfortran.so.5", "-lm"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return probe


@pytest.mark.parametrize("kind", [2, 3])
def test_slow_hi_fixture_matches_live_native_objects(production_probe, kind):
    case = next(c for c in CASES if c["crop_file_kind"] == kind and c["daily_hi_rate_pct"] == .25)
    records = case["records"]
    text = f"{kind} 50 .25 10 {len(records)}\n" + " ".join(str(r["dap"]) for r in records) + "\n"
    result = subprocess.run([str(production_probe)], input=text, text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    coefficient, switch, rate = map(float, lines[0].split())
    np.testing.assert_allclose([coefficient, switch, rate],
        [case["coefficient"], case["switch_day"], case["linear_rate"]], rtol=0., atol=1e-12)
    actual = np.asarray([list(map(float, line.split())) for line in lines[1:]])
    expected = np.asarray([[r["dap"], r["hi"], r["lag_pct"], r["final"]] for r in records])
    np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-12)


@pytest.mark.parametrize("kind", [2, 3])
def test_runner_selects_native_hi_branch_for_grain_and_tuber(tmp_path, kind):
    path = ROOT.parent / "aquacrop-fortran/testcase/LIST/Ottawa.PRM"
    original = read_project(path).runs[0]
    original_crop = next(ref for ref in original.files if "Crop" in ref.role)
    lines = original_crop.resolve(path).read_text().splitlines()
    lines[3] = f"{kind} : Crop kind for HI branch fixture"
    updates = {"Calendar Days: from sowing to flowering": 10,
               "Building up of Harvest Index": 50, "Reference Harvest Index": 50}
    for index, line in enumerate(lines):
        if ":" in line:
            description = line.split(":", 1)[1].strip()
            for prefix, value in updates.items():
                if description.startswith(prefix):
                    lines[index] = f"{value} : {description}"
    crop = tmp_path / "HI.CRO"
    crop.write_text("\n".join(lines) + "\n")
    files = tuple(replace(ref, name=crop.name, directory=str(tmp_path))
                  if ref == original_crop else ref for ref in original.files)
    run = replace(original, simulation_last_day=original.simulation_first_day + 64, files=files)
    result, = simulate_project(Project("HI branch wiring", 7.3, (run,)), path)
    case = next(c for c in CASES if c["crop_file_kind"] == kind and
                c["maximum_hi_pct"] == 50 and c["daily_hi_rate_pct"] == 1.)
    expected = {r["dap"]: r["hi"] for r in case["records"]}
    columns = result.columns
    # Fortran's production clock is dayi - Crop.Day1, while OUT DAP starts at 1.
    native_days = columns["daynr"] - run.crop_first_day
    np.testing.assert_array_equal(columns["dap"], native_days + 1)
    matched = [i for i, day in enumerate(native_days) if int(day) in expected]
    native = np.asarray([expected[int(native_days[i])] for i in matched])
    assert any(native_days[i] > case["flowering_day"] + case["switch_day"] for i in matched)
    np.testing.assert_allclose(columns["harvest_index_pct"][matched], native, rtol=0., atol=1e-12)
    np.testing.assert_allclose(columns["yield_mass"][matched],
                               columns["biomass"][matched] * native / 100., rtol=0., atol=1e-12)
    assert np.max(np.abs(columns["residual"])) < 1e-9
    assert np.all(columns["salt_residual"] == 0.)
