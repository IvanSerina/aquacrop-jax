! Execute the original Ottawa project, observing harvests after each day.
! Run only inside a disposable testcase copy; native output files are produced.
program cutting_season_probe
    use ac_kinds, only: dp, int8, int32
    use ac_global, only: InitializeGlobalStrings, typeproject_typeprm, undef_int, &
        GetSimulation_ToDayNr, GetSimulation_NrRuns, GetCrop_Day1, GetCCiActual, GetCCiPrev, &
        GetCrop_CCoAdjusted, GetCrop_CCxAdjusted, GetCrop_CCxWithered, &
        GetCrop_GDDaysToSenescence, GetSimulation_SumGDD, &
        GetSumWaBal_Biomass, GetSumWaBal_YieldPart, GetManagement_Cuttings_CCcut
    use ac_startunit, only: InitializeTheProgram, InitializeProject
    use ac_run, only: InitializeSimulation, InitializeRunPart1, InitializeClimate, &
        InitializeRunPart2, AdvanceOneTimeStep, ReadClimateNextDay, SetGDDVariablesNextDay, &
        SetRepeatToDay, GetRepeatToDay, FinalizeRun1, FinalizeRun2, FinalizeSimulation, &
        GetNrCut, GetDayLastCut, GetSumInterval, GetSumGDDcuts, GetBprevSum, GetYprevSum, &
        GetCCxWitheredTpotNoS, GetDayNri, GetGDDayi, SetNextSimFromDayNr
    implicit none
    integer(int32) :: day, iteration, interval_before
    integer(int8) :: run_number
    real(dp) :: wp, previous_biomass, previous_yield, gdd_interval_before
    logical :: harvest
    call InitializeGlobalStrings()
    call InitializeTheProgram()
    call InitializeProject(1_int32, 'Ottawa.PRM', typeproject_typeprm)
    call SetNextSimFromDayNr(undef_int)
    call InitializeSimulation('Ottawa.PRM', typeproject_typeprm)
    do run_number = 1,GetSimulation_NrRuns()
    call InitializeRunPart1(run_number, typeproject_typeprm)
    call InitializeClimate()
    call InitializeRunPart2(run_number, typeproject_typeprm)
    call SetRepeatToDay(GetSimulation_ToDayNr())
    wp = 0._dp
    harvest = .false.
    do iteration = 1,10000
        day = GetDayNri()
        interval_before = GetSumInterval()
        gdd_interval_before = GetSumGDDcuts()
        previous_biomass = GetBprevSum()
        previous_yield = GetYprevSum()
        call AdvanceOneTimeStep(wp, harvest)
        if (harvest) then
            write(*,'(a,1x,8(i0,1x),14(es24.16,1x))') 'cut', run_number, day, GetCrop_Day1(), GetNrCut(), &
                GetDayLastCut(), interval_before+1, GetManagement_Cuttings_CCcut(), &
                GetCrop_GDDaysToSenescence(), gdd_interval_before+GetGDDayi(), &
                GetSimulation_SumGDD(), GetCCiActual(), GetCCiPrev(), &
                GetCrop_CCoAdjusted(), GetCrop_CCxAdjusted(), GetCrop_CCxWithered(), &
                GetCCxWitheredTpotNoS(), GetSumWaBal_Biomass(), GetSumWaBal_YieldPart(), &
                GetBprevSum(), GetYprevSum(), &
                GetSumWaBal_Biomass()-previous_biomass, GetSumWaBal_YieldPart()-previous_yield
        end if
        call ReadClimateNextDay()
        call SetGDDVariablesNextDay()
        if ((GetDayNri()-1) == GetRepeatToDay()) exit
    end do
    if (iteration > 10000) stop 2
    call FinalizeRun1(run_number, 'Ottawa.PRM', typeproject_typeprm)
    call FinalizeRun2(run_number, typeproject_typeprm)
    end do
    call FinalizeSimulation()
end program cutting_season_probe
