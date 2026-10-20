! Execute original daily fertility forcing on the actually loaded Ottawa crop.
! Initialization writes files; run only in a disposable testcase copy.
program daily_fertility_probe
    use ac_kinds, only: dp, int8, int32
    use ac_global
    use ac_startunit, only: InitializeTheProgram, InitializeProject
    use ac_run, only: InitializeRunPart1
    use ac_simul, only: EffectSoilFertilitySalinityStress
    use ac_tempprocessing, only: TminRecord => Tmin
    implicit none
    integer(int32) :: case_count, case_index, case_stress, case_management, case_virtual, case_germinated
    type(rep_shapes) :: crop_shapes
    call InitializeGlobalStrings()
    call InitializeTheProgram()
    call InitializeProject(1_int32, 'Ottawa.PRM', typeproject_typeprm)
    call InitializeRunPart1(1_int8, typeproject_typeprm)
    crop_shapes = GetCrop_StressResponse()
    write(*,'(a,1x,3(es24.16,1x),8(i0,1x))') 'crop', &
        GetCrop_CCo(),GetCrop_CCx(),GetCrop_CGC(),GetCrop_DaysToGermination(), &
        GetCrop_DaysToFullCanopy(),GetCrop_DaysToSenescence(),GetCrop_DaysToHarvest(), &
        GetCrop_DaysToFlowering(),GetCrop_LengthFlowering(), &
        merge(1,0,GetCrop_DeterminancyLinked()),GetCrop_DayN()-GetCrop_Day1()
    write(*,'(a,1x,4(i0,1x),4(es24.16,1x))') 'thermal', &
        GetCrop_GDDaysToGermination(),GetCrop_GDDaysToFullCanopy(), &
        GetCrop_GDDaysToSenescence(),GetCrop_GDDaysToHarvest(), &
        crop_shapes%ShapeCGC,crop_shapes%ShapeCCX,crop_shapes%ShapeWP,crop_shapes%ShapeCDecline
    write(*,'(a,1x,5(i0,1x),2(es24.16,1x))') 'record',size(TminRecord), &
        GetTemperatureRecord_FromDayNr(),GetTemperatureRecord_ToDayNr(),GetCrop_Day1(), &
        int(GetSimulParam_GDDMethod()),GetCrop_Tbase(),GetCrop_Tupper()
    call SetSimulation_SalinityConsidered(.false.)
    read(*,*) case_count
    do case_index = 1,case_count
        read(*,*) case_stress,case_management,case_virtual,case_germinated
        call SetManagement_FertilityStress(case_management)
        call SetSimulation_Germinate(case_germinated /= 0)
        call SetCrop_DaysToFullCanopySF(GetCrop_DaysToFullCanopy())
        call SetCrop_GDDaysToFullCanopySF(GetCrop_GDDaysToFullCanopy())
        call EffectSoilFertilitySalinityStress(case_stress,0._dp,0._dp,0._dp,0_int32,0._dp,case_virtual)
        write(*,'(7(i0,1x),es24.16)') case_stress,GetCrop_DaysToFullCanopySF(), &
            GetCrop_GDDaysToFullCanopySF(),int(GetSimulation_EffectStress_RedCGC()), &
            int(GetSimulation_EffectStress_RedCCX()),int(GetSimulation_EffectStress_RedWP()), &
            int(GetSimulation_EffectStress_RedKsSto()),GetSimulation_EffectStress_CDecline()
    end do
end program daily_fertility_probe
