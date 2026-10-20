! Run only in a disposable copy of the native testcase: initialization writes
! temporary climate files. Reference source and objects are never modified.
program fertility_setup_probe
    use ac_kinds, only: dp, int8, int32
    use ac_global, only: InitializeGlobalStrings, typeproject_typeprm, &
        GetManagement_FertilityStress, SetManagement_FertilityStress, &
        GetSimulation_EffectStress_RedCGC, GetSimulation_EffectStress_RedCCX, &
        GetSimulation_EffectStress_RedWP, GetSimulation_EffectStress_CDecline, &
        GetCrop_DaysToFullCanopySF, GetCrop_GDDaysToFullCanopySF
    use ac_global, only: SetCrop_StressResponse_Calibrated, GetCrop_CCo, &
        GetCrop_CGC, GetCrop_CCx, GetCrop_DaysToGermination, &
        GetCrop_DaysToFullCanopy, GetCrop_DaysToSenescence, &
        GetCrop_DaysToFlowering, GetCrop_LengthFlowering, GetCrop_DeterminancyLinked
    use ac_global, only: GetCrop_DaysToCCini, GetCrop_GDDaysToCCini, GetCrop_DaysToGermination, &
        GetCrop_DaysToFullCanopy, GetCrop_DaysToSenescence, GetCrop_DaysToHarvest, &
        GetCrop_DaysToFlowering, GetCrop_LengthFlowering, GetCrop_GDDaysToGermination, &
        GetCrop_GDDaysToFullCanopy, GetCrop_GDDaysToSenescence, GetCrop_GDDaysToHarvest, &
        GetCrop_WPy, GetCrop_HI, GetCrop_CCo, &
        GetCrop_CCx, GetCrop_CGC, GetCrop_GDDCGC, &
        GetCrop_CDC, GetCrop_GDDCDC, GetCrop_KcTop, &
        GetCrop_KcDeclineCumul, GetCrop_CCEffectEvapLate, GetCrop_Tbase, &
        GetCrop_Tupper, GetSimulParam_Tmin, GetSimulParam_Tmax, &
        GetCrop_GDtranspLow, GetCrop_WP, GetCrop_dHIdt, &
        GetCrop_Day1, GetCrop_DeterminancyLinked, GetCrop_StressResponse, &
        GetCrop_subkind, GetCrop_ModeCycle, GetCrop_GDDaysToFlowering, &
        GetCrop_GDDLengthFlowering, GetCrop_GDDaysToHIo, GetCrop_Planting, &
        GetCrop_DaysToHIo
    use ac_preparefertilitysalinity, only: ReferenceStressBiomassRelationship
    use ac_startunit, only: InitializeTheProgram, InitializeProject
    use ac_run, only: InitializeRunPart1, RelationshipsForFertilityAndSaltStress, &
        GetCoeffb0, GetCoeffb1, GetCoeffb2, GetFracBiomassPotSF
    implicit none
    real(dp) :: b0, b1, b2, samples(7)
    integer(int32) :: count, index, stress, calibrated

    call InitializeGlobalStrings()
    call InitializeTheProgram()
    call InitializeProject(1_int32, 'Ottawa.PRM', typeproject_typeprm)
    call InitializeRunPart1(1_int8, typeproject_typeprm)
    write(*, '(a,1x,i0,1x,4(es24.16,1x),5(i0,1x),es24.16)') &
        'setup', GetManagement_FertilityStress(), GetCoeffb0(), GetCoeffb1(), &
        GetCoeffb2(), GetFracBiomassPotSF(), &
        int(GetSimulation_EffectStress_RedCGC()), &
        int(GetSimulation_EffectStress_RedCCX()), &
        int(GetSimulation_EffectStress_RedWP()), &
        GetCrop_DaysToFullCanopySF(), GetCrop_GDDaysToFullCanopySF(), &
        GetSimulation_EffectStress_CDecline()
    write(*, '(a,1x,3(es24.16,1x),6(i0,1x))') 'timing', GetCrop_CCo(), &
        GetCrop_CGC(), GetCrop_CCx(), GetCrop_DaysToGermination(), &
        GetCrop_DaysToFullCanopy(), GetCrop_DaysToSenescence(), &
        GetCrop_DaysToFlowering(), GetCrop_LengthFlowering(), &
        merge(1, 0, GetCrop_DeterminancyLinked())
    b0 = 0._dp
    b1 = 0._dp
    b2 = 0._dp
    samples = 0._dp
        call ReferenceStressBiomassRelationship(GetCrop_DaysToCCini(), &
                                  GetCrop_GDDaysToCCini(), &
                                  GetCrop_DaysToGermination(), &
                                  GetCrop_DaysToFullCanopy(), &
                                  GetCrop_DaysToSenescence(), &
                                  GetCrop_DaysToHarvest(), &
                                  GetCrop_DaysToFlowering(), &
                                  GetCrop_LengthFlowering(), &
                                  GetCrop_GDDaysToGermination(), &
                                  GetCrop_GDDaysToFullCanopy(), &
                                  GetCrop_GDDaysToSenescence(), &
                                  GetCrop_GDDaysToHarvest(), &
                                  GetCrop_WPy(), GetCrop_HI(), &
                                  GetCrop_CCo(), GetCrop_CCx(), &
                                  GetCrop_CGC(), GetCrop_GDDCGC(), &
                                  GetCrop_CDC(), GetCrop_GDDCDC(), &
                                  GetCrop_KcTop(), GetCrop_KcDeclineCumul(), &
                                  real(GetCrop_CCEffectEvapLate(), kind= dp), &
                                  GetCrop_Tbase(), &
                                  GetCrop_Tupper(), GetSimulParam_Tmin(), &
                                  GetSimulParam_Tmax(), GetCrop_GDtranspLow(), &
                                  GetCrop_WP(), GetCrop_dHIdt(), &
                                  GetCrop_Day1(), GetCrop_DeterminancyLinked(), &
                                  GetCrop_StressResponse(),GetCrop_subkind(), &
                                  GetCrop_ModeCycle(), b0, b1, &
                                  b2, samples(1), samples(2), samples(3), samples(4), samples(5), samples(6), samples(7), &
                                  GetCrop_GDDaysToFlowering(), GetCrop_GDDLengthFlowering(), &
                                  GetCrop_GDDaysToHIo(), GetCrop_Planting(), GetCrop_DaysToHIo())
    write(*, '(a,1x,11(es24.16,1x))') 'samples', b0, b1, b2, 100._dp, samples
    read(*, *) count
    do index = 1, count
        read(*, *) stress, calibrated
        call SetManagement_FertilityStress(stress)
        call SetCrop_StressResponse_Calibrated(calibrated /= 0)
        call RelationshipsForFertilityAndSaltStress()
        write(*, '(i0,1x,4(es24.16,1x))') stress, GetCoeffb0(), &
            GetCoeffb1(), GetCoeffb2(), GetFracBiomassPotSF()
    end do
end program fertility_setup_probe
