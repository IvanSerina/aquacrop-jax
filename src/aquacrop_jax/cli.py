from __future__ import annotations

import argparse
from pathlib import Path

from .compat import compare_directories
from .legacy import read_project, read_project_list


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aquacrop-jax")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_p = sub.add_parser("inspect", help="validate and summarize a legacy PRM")
    inspect_p.add_argument("project")
    compare_p = sub.add_parser("compare", help="compare generated and reference OUT files")
    compare_p.add_argument("actual")
    compare_p.add_argument("expected")
    trace_p = sub.add_parser("compare-trace", help="measure a diagnostic CSV against native daily OUT")
    trace_p.add_argument("trace")
    trace_p.add_argument("reference")
    trace_p.add_argument("--output", help="save numerical JSON report")
    run_p = sub.add_parser("run", help="run an implemented progressive simulation stage")
    run_p.add_argument("project_list")
    run_p.add_argument("--stage", choices=("climate", "water", "water-evap", "water-crop"), required=True)
    run_p.add_argument("--output", default=None)
    run_p.add_argument("--salt-stomatal-coefficients", nargs=3, type=float, metavar=("B0", "B1", "B2"),
                       help="calibrated ECsw response polynomial for water-crop")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        project_path = Path(args.project).resolve()
        project = read_project(project_path)
        print(f"{project.description} | AquaCrop {project.version:g} | {len(project.runs)} runs")
        for index, run in enumerate(project.runs, 1):
            print(f"run {index}: {run.simulation_first_day}..{run.simulation_last_day} ({len(run.files)} file references)")
            for ref in run.files:
                resolved = ref.resolve(project_path)
                status = "special" if resolved is None else ("ok" if resolved.exists() else "missing")
                print(f"  {ref.role}: {ref.name} [{status}]")
        return 0
    if args.command == "compare-trace":
        import json
        from .outputs import compare_trace_to_daily
        report = compare_trace_to_daily(args.trace, args.reference)
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered)
        print(rendered, end="")
        return int(bool(report["missing_days"] or report["extra_days"] or any(
            value["max_abs"] > 0.0 for value in report["columns"].values())))
    if args.command == "run":
        from .climate import run_climate_stage
        projects = read_project_list(args.project_list)
        if not projects:
            parser.error("project list is empty")
        total = 0
        output = Path(args.output or f"build/{args.stage}.csv")
        if len(projects) != 1:
            parser.error("diagnostic stages currently support one project per list")
        project_file = projects[0]
        project = read_project(project_file)
        if args.stage == "climate":
            total += run_climate_stage(project, project_file, output)
        elif args.stage == "water":
            from .water_runner import run_water_stage
            total += run_water_stage(project, project_file, output)
        elif args.stage == "water-evap":
            from .water_evap_runner import run_water_evap_stage
            total += run_water_evap_stage(project, project_file, output)
        else:
            from .crop_runner import run_crop_stage
            total += run_crop_stage(project, project_file, output, args.salt_stomatal_coefficients)
        print(f"wrote {total} calculated daily rows to {output}")
        return 0
    try:
        differences = compare_directories(args.actual, args.expected)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    for diff in differences:
        print(f"{diff.filename}:{diff.line}: expected {diff.expected!r}; got {diff.actual!r}")
    print("compatible" if not differences else f"{len(differences)} incompatible files")
    return int(bool(differences))


if __name__ == "__main__":
    raise SystemExit(main())
