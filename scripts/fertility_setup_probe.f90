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
    use ac_startunit, only: InitializeTheProgram, InitializeProject
    use ac_run, only: InitializeRunPart1, RelationshipsForFertilityAndSaltStress, &
        GetCoeffb0, GetCoeffb1, GetCoeffb2, GetFracBiomassPotSF
    implicit none
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
