from pathlib import Path

from aquacrop_jax.legacy import read_climate_series, read_project

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "aquacrop-fortran" / "testcase"


def test_ottawa_project_contract():
    project = read_project(FIXTURE / "LIST" / "Ottawa.PRM")
    assert project.version == 7.3
    assert len(project.runs) == 3
    assert project.runs[0].simulation_first_day == 41414
    assert sum(r.simulation_last_day - r.simulation_first_day + 1 for r in project.runs) == 892
    assert project.runs[1].files[11].name == "KeepSWC"
    assert all(ref.resolve(FIXTURE / "LIST" / "Ottawa.PRM") is None or ref.resolve(FIXTURE / "LIST" / "Ottawa.PRM").exists()
               for run in project.runs for ref in run.files)


def test_daily_temperature_reader():
    series = read_climate_series(FIXTURE / "DATA" / "Ottawa.Tnx", 2)
    assert (series.data_type, series.first_year) == (1, 2014)
    assert series.values[0] == (-31.3, -21.4)
