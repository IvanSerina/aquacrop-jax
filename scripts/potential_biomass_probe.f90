! Original first-year forage initialization and unlimited-biomass operator.
! Run in a disposable testcase copy; source/objects are immutable.
program potential_biomass_probe
    use ac_kinds, only: dp, int8, int32
    use ac_global
    use ac_startunit, only: InitializeTheProgram, InitializeProject
    use ac_run, only: InitializeRunPart1
    use ac_simul, only: DeterminePotentialBiomass
    implicit none
    integer(int32) :: count, row, virtual, delayed, dap
    real(dp) :: case_heat, case_daily, case_eto, case_co2, case_withered, case_biomass
    real(dp) :: input_withered, input_biomass
    character(len=32) :: mode
    call InitializeGlobalStrings()
    call InitializeTheProgram()
    call InitializeProject(1_int32, 'Ottawa.PRM', typeproject_typeprm)
    call InitializeRunPart1(1_int8, typeproject_typeprm)
    call get_command_argument(1, mode)
    write(*,'(8(i0,1x),12(es24.16,1x))') &
        GetCrop_DaysToGermination(),GetCrop_DaysToFullCanopy(),GetCrop_DaysToSenescence(),GetCrop_DaysToHarvest(), &
        GetCrop_GDDaysToGermination(),GetCrop_GDDaysToFullCanopy(), &
        GetCrop_GDDaysToSenescence(),GetCrop_GDDaysToHarvest(), &
        GetCrop_CCo(),GetCrop_CCx(),GetCrop_GDDCGC(),GetCrop_GDDCDC(), &
        GetCrop_KcTop(),GetCrop_KcDeclineCumul(),real(GetCrop_CCEffectEvapLate(),dp), &
        GetCrop_GDtranspLow(),GetCrop_WP(),real(GetCrop_AdaptedToCO2(),dp), &
        GetSimulParam_KcWetBare(),GetSimulParam_ExpFsen()
    read(*,*) count
    do row=1,count
        read(*,*) virtual,case_heat,case_daily,case_eto,case_co2,delayed,input_withered,input_biomass
        if (row == 1 .or. trim(mode) /= '--scan') then
            case_withered = input_withered
            case_biomass = input_biomass
        end if
        call SetSimulation_DelayedDays(delayed)
        call SetETo(case_eto)
        call DeterminePotentialBiomass(virtual,case_heat,case_co2,case_daily,case_withered,case_biomass)
        write(*,'(2(es24.16,1x))') case_withered,case_biomass
    end do
end program potential_biomass_probe
