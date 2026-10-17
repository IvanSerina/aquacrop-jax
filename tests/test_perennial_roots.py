"""Crop-year root initialization from native AdjustYearPerennials semantics."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aquacrop_jax.crop_runner import simulate_project
from aquacrop_jax.legacy import Project, read_project
from aquacrop_jax.outputs import read_daily_output

ROOT = Path(__file__).resolve().parents[1]
TESTCASE = ROOT.parent / "aquacrop-fortran/testcase"


@pytest.mark.parametrize("year", [1, 2, 3])
def test_initial_root_depths_match_native_ottawa_years(year):
    path = TESTCASE / "LIST/Ottawa.PRM"
    original = read_project(path).runs[year - 1]
    # RootMin=RootMax makes regrowth depth independent of antecedent soil water.
    # Reset water for this isolated three-day initialization check, retaining
    # the original crop, cultivation year and forcing dates.
    files = tuple(replace(ref, name="(None)", directory="(None)")
                  if "Initial conditions" in ref.role else ref for ref in original.files)
    run = replace(original, simulation_first_day=original.crop_first_day,
                  simulation_last_day=original.crop_first_day + 2, files=files)
    result, = simulate_project(Project("native root initialization", 7.3, (run,)), path)
    native = {r.key: r.mapping()["root_depth"] for r in
              read_daily_output(TESTCASE / "OUTP_REF/OttawaPRMday.OUT").records}
    expected = [native[year, int(day)] for day in result.columns["daynr"]]
    # Native OUT prints metres to 0.01; use its half-unit quantization bound.
    np.testing.assert_allclose(result.columns["root_depth"], expected, rtol=0., atol=.005000000001)
    if year > 1:
        np.testing.assert_array_equal(result.columns["root_depth"], [3., 3., 3.])


@pytest.mark.parametrize("kind,year,expected", [(4, 1, .3), (4, 2, 3.), (3, 2, .8)])
def test_for_first_year_uses_root_min_year1_and_leaves_other_crop_kinds_alone(tmp_path, kind, year, expected):
    path = TESTCASE / "LIST/Ottawa.PRM"
    original = read_project(path).runs[0]
    crop_ref = next(ref for ref in original.files if "Crop" in ref.role)
    lines = crop_ref.resolve(path).read_text().splitlines()
    lines[3] = f"{kind} : crop kind"
    for index, line in enumerate(lines):
        if ":" in line:
            _, description = line.split(":", 1)
            if description.strip() == "Minimum effective rooting depth (m)":
                lines[index] = ".8 : Minimum effective rooting depth (m)"
    crop = tmp_path / "roots.CRO"
    crop.write_text("\n".join(lines) + "\n")
    files = tuple(replace(ref, name=crop.name, directory=str(tmp_path))
                  if ref == crop_ref else ref for ref in original.files)
    run = replace(original, cultivation_year=year,
                  simulation_last_day=original.simulation_first_day, files=files)
    result, = simulate_project(Project("root parameter routing", 7.3, (run,)), path)
    assert result.columns["root_depth"][0] == expected
    assert abs(result.columns["residual"][0]) < 1e-9
