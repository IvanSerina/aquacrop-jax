import csv
from pathlib import Path

from aquacrop_jax.legacy import read_project
from aquacrop_jax.water_runner import run_water_stage

FIXTURE = Path(__file__).parents[2] / "aquacrop-fortran" / "testcase"


def test_ottawa_water_runner_writes_all_892_days(tmp_path):
    prm = (FIXTURE / "LIST" / "Ottawa.PRM").resolve()
    target = tmp_path / "water.csv"
    assert run_water_stage(read_project(prm), prm, target) == 892
    with target.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 892
    assert rows[0]["daynr"] == "41414"
    assert rows[-1]["daynr"] == "42305"
    assert len([name for name in rows[0] if name.startswith("theta")]) == 12
    assert max(abs(float(row["balance_error"])) for row in rows) < 1e-9


def test_ottawa_water_runner_preserves_state_across_run_boundary(tmp_path):
    prm = (FIXTURE / "LIST" / "Ottawa.PRM").resolve()
    target = tmp_path / "water.csv"
    run_water_stage(read_project(prm), prm, target)
    with target.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    last_run1 = next(row for row in rows if row["daynr"] == "41577")
    first_run2 = next(row for row in rows if row["daynr"] == "41578")
    # No reset to field capacity: the second run starts from the recurrent
    # state, then applies that day's fluxes.
    assert any(abs(float(first_run2[f"theta{i:02d}"]) - 0.29) > 1e-8
               for i in range(1, 13))
    assert last_run1["run"] == "1" and first_run2["run"] == "2"
