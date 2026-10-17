from pathlib import Path
import csv

import pytest

from aquacrop_jax.outputs import (compare_trace_to_daily, format_daily_record,
                                  read_daily_output, write_daily_output)

REFERENCE = Path(__file__).parents[2] / 'aquacrop-fortran/testcase/OUTP_REF/OttawaPRMday.OUT'


def test_full_daily_out_roundtrip_is_byte_identical(tmp_path):
    table = read_daily_output(REFERENCE)
    assert len(table.records) == 892
    assert [sum(r.run == run for r in table.records) for run in [1, 2, 3]] == [164, 358, 370]
    output = tmp_path / 'day.OUT'
    write_daily_output(output, table)
    assert output.read_bytes() == REFERENCE.read_bytes()


def test_writer_requires_all_fields_and_rejects_nonfinite_values():
    with pytest.raises(ValueError, match='complete record'):
        format_daily_record({'rain': 1.})
    values = read_daily_output(REFERENCE).records[0].mapping()
    values['rain'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        format_daily_record(values)


def test_numerical_comparison_checks_dates_units_and_coverage(tmp_path):
    record = read_daily_output(REFERENCE).records[0]
    reference = record.mapping()
    trace = tmp_path / 'trace.csv'
    with trace.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['run', 'daynr', 'rain', 'theta01'])
        writer.writeheader()
        writer.writerow(dict(run=record.run, daynr=record.key[1], rain=reference['rain'],
                             theta01=reference['theta01_pct'] / 100.))
    result = compare_trace_to_daily(trace, REFERENCE)
    assert result['matched_days'] == 1
    assert len(result['missing_days']) == 891
    assert result['columns']['rain']['max_abs'] == 0.
    assert result['columns']['theta01']['max_abs'] == pytest.approx(0., abs=1e-14)
