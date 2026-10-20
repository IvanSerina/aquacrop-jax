"""Exercise the verbatim native HarvestNow block on controlled cutting state.

This observes the event application, not scheduling or full seasonal parity.
No original source, object or project input is modified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'aquacrop-fortran/src'
INPUTS = ('occurred', 'count', 'last_cut_day', 'interval_days', 'interval_gdd',
    'previous_cover', 'actual_cover', 'initial_adjusted', 'maximum_adjusted',
    'withered_cover', 'potential_withered_cover', 'previous_biomass',
    'previous_yield', 'cumulative_biomass', 'cumulative_yield', 'day_in_season', 'cut_cover_pct')
OUTPUTS = ('count', 'last_cut_day', 'interval_days', 'interval_gdd',
    'previous_cover', 'actual_cover', 'initial_adjusted', 'maximum_adjusted',
    'withered_cover', 'potential_withered_cover', 'previous_biomass',
    'previous_yield', 'cumulative_biomass', 'cumulative_yield', 'biomass_harvested', 'yield_harvested')


def cases():
    base = dict(zip(INPUTS, (1, 2, 40, 30, 333.25, .8, .82, .05, .9, .95,
        .97, 2., .5, 5., 2., 100, 25), strict=True))
    rows = []
    def add(name, **updates):
        rows.append(dict(name=name, input=base | updates))
    add('ordinary-cut')
    add('no-event', occurred=0)
    add('cover-below-threshold', previous_cover=math.nextafter(.25, 0.))
    add('cover-equals-threshold', previous_cover=.25)
    add('cover-above-threshold', previous_cover=math.nextafter(.25, math.inf))
    add('zero-cover', previous_cover=0.)
    add('cut-to-zero', cut_cover_pct=0)
    add('cut-at-100-percent', cut_cover_pct=100)
    add('first-event', count=0, last_cut_day=-9, interval_days=1,
        interval_gdd=10., previous_biomass=0., previous_yield=0.)
    add('negative-recorded-increments', previous_biomass=10., previous_yield=4.)
    add('percent57-below', cut_cover_pct=57, previous_cover=math.nextafter(.57, 0.))
    add('percent57-equal', cut_cover_pct=57, previous_cover=.57)
    add('percent57-above', cut_cover_pct=57, previous_cover=math.nextafter(.57, math.inf))
    return rows


def generate(output):
    source = (REFERENCE / 'run.f90').read_text()
    advance = source.split('subroutine AdvanceOneTimeStep(WPi, HarvestNow)', 1)[1].split(
        'end subroutine AdvanceOneTimeStep', 1)[0]
    block = advance[advance.index('        if (HarvestNow .eqv. .true.) then'):].split(
        '\n    end if\n\n    ! 14. Write results', 1)[0]
    if not block.rstrip().endswith('end if'):
        raise RuntimeError('Unexpected native cutting block boundary')
    program = '''program extracted_native_cutting
use ac_kinds, only: dp, int32
use ac_global
use ac_run, only: SetNrCut,GetNrCut,SetDayLastCut,GetDayLastCut, &
    SetSumInterval,GetSumInterval,SetSumGDDcuts,GetSumGDDcuts, &
    SetBprevSum,GetBprevSum,SetYprevSum,GetYprevSum, &
    SetCCxWitheredTpotNoS,GetCCxWitheredTpotNoS,RecordHarvest
implicit none
integer(int32) :: n,row,flag,ncut,last,interval,DayInSeason,cutpct
real(dp) :: gdd,ccprev,ccactual,ccoadj,ccxadj,withered,potwithered
real(dp) :: bprev,yprev,biomass,yield,record_b,record_y
logical :: HarvestNow
call SetPart1Mult(.false.)
read(*,*) n
do row=1,n
read(*,*) flag,ncut,last,interval,gdd,ccprev,ccactual,ccoadj,ccxadj, &
    withered,potwithered,bprev,yprev,biomass,yield,DayInSeason,cutpct
HarvestNow = flag /= 0
call SetNrCut(ncut)
call SetDayLastCut(last)
call SetSumInterval(interval)
call SetSumGDDcuts(gdd)
call SetCCiPrev(ccprev)
call SetCCiActual(ccactual)
call SetCrop_CCoAdjusted(ccoadj)
call SetCrop_CCxAdjusted(ccxadj)
call SetCrop_CCxWithered(withered)
call SetCCxWitheredTpotNoS(potwithered)
call SetBprevSum(bprev)
call SetYprevSum(yprev)
call SetSumWaBal_Biomass(biomass)
call SetSumWaBal_YieldPart(yield)
call SetManagement_Cuttings_CCcut(cutpct)
record_b=0._dp
record_y=0._dp
if (HarvestNow) then
    ! Same raw mass differences recorded by native RecordHarvest, before reset.
    record_b=GetSumWaBal_Biomass()-GetBprevSum()
    record_y=GetSumWaBal_YieldPart()-GetYprevSum()
end if
''' + block + '''
write(*,'(3(i0,1x),13(es24.16,1x))') GetNrCut(),GetDayLastCut(),GetSumInterval(), &
    GetSumGDDcuts(),GetCCiPrev(),GetCCiActual(),GetCrop_CCoAdjusted(), &
    GetCrop_CCxAdjusted(),GetCrop_CCxWithered(),GetCCxWitheredTpotNoS(), &
    GetBprevSum(),GetYprevSum(),GetSumWaBal_Biomass(),GetSumWaBal_YieldPart(),record_b,record_y
end do
end program extracted_native_cutting
'''
    objects = sorted(p for p in REFERENCE.glob('*.o') if p.name != 'aquacrop.o')
    if not objects:
        raise RuntimeError('Prebuilt immutable native objects required')
    rows = cases()
    with tempfile.TemporaryDirectory(prefix='native-cutting-') as work:
        path, executable = Path(work) / 'probe.f90', Path(work) / 'probe'
        path.write_text(program)
        subprocess.run(['gfortran', '-O0', '-I', str(REFERENCE), '-o', str(executable),
            str(path), *map(str, objects)], check=True, capture_output=True, text=True)
        inputs = str(len(rows)) + '\n' + '\n'.join(' '.join(
            format(case['input'][key], '.17g') for key in INPUTS) for case in rows) + '\n'
        native = subprocess.run([str(executable)], input=inputs,
            capture_output=True, text=True, check=True)
        expected = [list(map(float, line.split())) for line in native.stdout.splitlines()]
    if len(expected) != len(rows) or any(len(row) != len(OUTPUTS) for row in expected):
        raise RuntimeError('Native cutting probe returned unexpected output shape')
    for case, values in zip(rows, expected, strict=True):
        case['expected'] = values
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        source='Verbatim run.f90::AdvanceOneTimeStep HarvestNow application block',
        scope='Event application only; caller supplies scheduling decision and interval counters',
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        block_sha256=hashlib.sha256(block.encode()).hexdigest(),
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        output_columns=OUTPUTS, cases=rows), indent=2, allow_nan=False) + '\n')
    print(f'{len(rows)} native cutting cases: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
        default=ROOT / 'tests/fixtures/cutting_event_fortran73.json')
    generate(parser.parse_args().output)
