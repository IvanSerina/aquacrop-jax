import csv
from pathlib import Path

from aquacrop_jax.legacy import read_project
from aquacrop_jax.water_evap_runner import run_water_evap_stage

FIXTURE = Path(__file__).parents[2] / "aquacrop-fortran" / "testcase"


def test_ottawa_water_evap_runner_writes_balanced_892_day_trace(tmp_path):
    prm = (FIXTURE / "LIST" / "Ottawa.PRM").resolve(); target = tmp_path / "trace.csv"
    assert run_water_evap_stage(read_project(prm), prm, target) == 892
    with target.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 892
    assert max(abs(float(row["balance_error"])) for row in rows) < 1e-8
    assert sum(float(row["evaporation"]) for row in rows) > 0.0
    assert all(0.15 <= float(row["evap_depth"]) <= 0.30 for row in rows)
