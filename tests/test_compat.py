from pathlib import Path

from aquacrop_jax.compat import compare_directories


def test_reference_directory_is_self_compatible():
    ref = Path(__file__).parents[2] / "aquacrop-fortran" / "testcase" / "OUTP_REF"
    assert compare_directories(ref, ref) == []


def test_empty_generated_directory_has_actionable_error(tmp_path):
    ref = Path(__file__).parents[2] / "aquacrop-fortran" / "testcase" / "OUTP_REF"
    try:
        compare_directories(tmp_path, ref)
    except FileNotFoundError as exc:
        assert "run an AquaCrop simulator" in str(exc)
    else:
        raise AssertionError("empty generated directory must be rejected")


def test_comparison_rejects_extra_files_and_trailing_whitespace(tmp_path):
    actual, expected = tmp_path / 'actual', tmp_path / 'expected'
    actual.mkdir(); expected.mkdir()
    (actual / 'data.OUT').write_text('value \n')
    (expected / 'data.OUT').write_text('value\n')
    (actual / 'extra.OUT').write_text('unexpected\n')
    differences = compare_directories(actual, expected)
    assert {d.filename for d in differences} == {'extra.OUT', 'data.OUT'}


def test_comparison_ignores_only_creation_timestamp(tmp_path):
    actual, expected = tmp_path / 'actual', tmp_path / 'expected'
    actual.mkdir(); expected.mkdir()
    (actual / 'data.OUT').write_text('AquaCrop 7.3 - Output created on (date) : today\nvalue\n')
    (expected / 'data.OUT').write_text('AquaCrop 7.3 - Output created on (date) : yesterday\nvalue\n')
    assert compare_directories(actual, expected) == []
    (actual / 'data.OUT').write_text('AquaCrop 7.2 - Output created on (date) : today\nvalue\n')
    assert len(compare_directories(actual, expected)) == 1


def test_comparison_detects_missing_final_newline(tmp_path):
    actual, expected = tmp_path / 'actual', tmp_path / 'expected'
    actual.mkdir(); expected.mkdir()
    (actual / 'data.OUT').write_bytes(b'value')
    (expected / 'data.OUT').write_bytes(b'value\n')
    assert len(compare_directories(actual, expected)) == 1
