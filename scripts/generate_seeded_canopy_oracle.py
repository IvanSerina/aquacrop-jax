"""Native seeded clock -> germination -> DetermineCCiGDD, shared protection."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'aquacrop-fortran/src'


def generate(output):
    # Main fields retain the original germination probe format. Additional
    # input is previous CC and simulation start, testing the sowing-day reset.
    rows = []
    for day in (99, 100, 101, 102):
        for gdd in (0., 40.):
            for germ, protected, theta, previous in ((0, 0, .10, 0.), (0, 0, .30, 0.),
                (1, 1, .30, .05), (1, 0, .10, .07)):
                rows.append(([day, 100, 120, germ, protected, 0, gdd, gdd,
                    theta, 30., 10., 25, 1, 5.], [previous, 99]))
    rows += [([102, 100, 120, 1, 1, 0, 2200., 2200., .3, 30., 10., 25, 1, 5.], [.07, 99]),
             ([102, 100, 120, 1, 1, 0, 0., 0., .3, 30., 10., 25, 1, 0.], [.07, 99]),
             ([101, 100, 120, 1, 0, 0, 40., 40., .1, 30., 10., 25, 1, 5.], [.07, 101])]
    rows = [(main, extra + [0]) for main, extra in rows]
    rows.append(([102, 100, 120, 1, 1, 0, 40., 40., .3, 30., 10., 25, 1, 5.], [.07, 99, 1]))
    run_source = (REFERENCE / 'run.f90').read_text()
    seeded = run_source.split('    else\n        ! sown or transplanted\n', 1)[1]
    clock = seeded.split('        ! CC initial', 1)[0]
    initialize = re.search(r'        if \(\(GetDayNri\(\) == \(GetCrop_Day1\(\).*?        end if', seeded, re.S).group()
    template = (ROOT / 'scripts/germination_clock_probe.f90').read_text()
    source = template.replace('only: CheckGermination', 'only: CheckGermination, DetermineCCiGDD')
    source = source.replace('GetGDDayi, SetGDDVariablesNextDay',
        'GetGDDayi, SetGDDVariablesNextDay, GetCCoTotal, SetCCoTotal')
    source = source.replace('    implicit none', '''    implicit none
    real(dp), parameter :: CCoTotal = .05_dp
    real(dp) :: previous_cover, leaf, sen_stress, time_sen
    integer(int32) :: sim_first, no_more_input
    logical :: no_more
''')
    source = source.replace('        call SetDayNri(day)', '''        read(*,*) previous_cover,sim_first,no_more_input
        call SetSimulation_FromDayNr(sim_first)
        call SetCCoTotal(CCoTotal)
        call SetCrop_DaysToCCini(0_int32)
        call SetCrop_subkind(subkind_Forage)
        call SetCrop_DeterminancyLinked(.false.)
        call SetCrop_DaysToGermination(1_int32)
        call SetCrop_DaysToFullCanopy(60_int32)
        call SetCrop_DaysToFullCanopySF(60_int32)
        call SetCrop_DaysToSenescence(180_int32)
        call SetCrop_DaysToHarvest(210_int32)
        call SetCrop_GDDaysToGermination(5_int32)
        call SetCrop_GDDaysToFullCanopy(600_int32)
        call SetCrop_GDDaysToFullCanopySF(600_int32)
        call SetCrop_GDDaysToSenescence(1920_int32)
        call SetCrop_GDDaysToHarvest(2200_int32)
        call SetCrop_CCo(CCoTotal)
        call SetCrop_CCx(.95_dp)
        call SetCrop_CCoAdjusted(CCoTotal)
        call SetCrop_CCxAdjusted(CCoTotal)
        call SetCrop_CCxWithered(CCoTotal)
        call SetCrop_CGC(.12_dp)
        call SetCrop_GDDCGC(.012_dp)
        call SetCrop_CDC(.06_dp)
        call SetCrop_GDDCDC(.006_dp)
        call SetCrop_pMethod(0_int8)
        call SetCrop_pLeafDefUL(.15_dp)
        call SetCrop_pLeafDefLL(.55_dp)
        call SetCrop_KsShapeFactorLeaf(3._dp)
        call SetCrop_pSenescence(.65_dp)
        call SetCrop_KsShapeFactorSenescence(3._dp)
        call SetCrop_SumEToDelaySenescence(40_int32)
        call SetSimulation_EffectStress_RedCGC(0_int8)
        call SetSimulation_EffectStress_RedCCX(0_int8)
        call SetSimulation_EffectStress_CDecline(0._dp)
        call SetSimulation_SumEToStress(0._dp)
        call SetSimulation_EvapLimitON(.false.)
        call SetCCiTopEarlySen(0._dp)
        call SetCCiPrev(previous_cover)
        call SetCCiActual(previous_cover)
        call SetETo(5._dp)
        leaf = -9._dp
        sen_stress = -9._dp
        time_sen = 0._dp
        no_more = no_more_input /= 0
        call SetDayNri(day)''')
    source = source.replace('        ! NATIVE_SEEDED_CLOCK', clock + initialize)
    # Add DetermineCCiGDD after the check, before printing the clock: its
    # protection flag must be the same persistent variable in both records.
    source = source.replace("        write(*,'(11(es24.16,1x))')", '''        if (.not. no_more) then
        call DetermineCCiGDD(.95_dp,CCoTotal,leaf,0._dp,.false.,.false., &
            SumGDDadjCC,VirtualTimeCC,sen_stress,time_sen,no_more, &
            .06_dp,1._dp,GetGDDayi(),.006_dp,0_int32)
        end if
        write(*,'(11(es24.16,1x))')''')
    source = source.replace('    end do', '''        write(*,'(10(es24.16,1x))') GetCCiActual(),GetCrop_CCoAdjusted(), &
            GetCrop_CCxAdjusted(),time_sen,GetCCiTopEarlySen(),GetCrop_CCxWithered(), &
            GetSimulation_SumEToStress(),merge(1._dp,0._dp,GetSimulation_EvapLimitON()), &
            merge(1._dp,0._dp,no_more),leaf
    end do''')
    objects = sorted(p for p in REFERENCE.glob('*.o') if p.name != 'aquacrop.o')
    if not objects:
        raise RuntimeError('Requires original native objects')
    with tempfile.TemporaryDirectory(prefix='native-seeded-canopy-') as tmp:
        work = Path(tmp)
        program, probe = work / 'probe.f90', work / 'probe'
        program.write_text(source)
        built = subprocess.run(['gfortran', '-O0', '-I', str(REFERENCE), '-o', str(probe),
            str(program), *map(str, objects)], capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stderr)
        text = str(len(rows)) + '\n' + '\n'.join(' '.join(format(v, '.17g') for v in main) +
            '\n' + ' '.join(map(str, extra)) for main, extra in rows) + '\n'
        native = subprocess.run([str(probe)], input=text, capture_output=True, text=True)
        if native.returncode:
            raise RuntimeError(native.stderr)
        lines = [list(map(float, line.split())) for line in native.stdout.splitlines()]
    assert len(lines) == 2 * len(rows)
    cases = [dict(input=main, previous_cover=extra[0], simulation_first_day=extra[1], no_more=bool(extra[2]),
        expected_clock=lines[2 * i][:7], root_water=lines[2 * i][7:10],
        daily_gdd=lines[2 * i][10], expected_canopy=lines[2 * i + 1])
        for i, (main, extra) in enumerate(rows)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(source='Verbatim seeded AdvanceOneTimeStep assignments, native CheckGermination and DetermineCCiGDD, no fertility stress',
        cases=cases, source_sha256={name: hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
            for name in ('run.f90', 'simul.f90', 'global.f90')},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects}),
        indent=2, allow_nan=False) + '\n')
    print(f'{len(cases)} original seeded clock/germination/canopy cases: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'tests/fixtures/seeded_canopy_fortran73.json')
    generate(parser.parse_args().output)
