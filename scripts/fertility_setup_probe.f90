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
    use ac_global, only: GetTnxReferenceYear, DetermineDate, DetermineDayNr, GetSimulParam_ExpFsen, &
        SeasonalSumOfKcPot, CropStressParametersSoilFertility, TimeToMaxCanopySF, &
        SumCalendarDaysReferenceTnx, rep_EffectStress
    use ac_tempprocessing, only: Bnormalized
    use ac_preparefertilitysalinity, only: ReferenceStressBiomassRelationship, &
        CO2ForTnxReferenceYear, AdjustCalendarDaysReferenceTnx
    use ac_startunit, only: InitializeTheProgram, InitializeProject
    use ac_run, only: InitializeRunPart1, RelationshipsForFertilityAndSaltStress, &
        GetCoeffb0, GetCoeffb1, GetCoeffb2, GetFracBiomassPotSF, &
        GetCO2i, GetSumKcTop, GetSumKcTopStress
    implicit none
    real(dp) :: b0, b1, b2, samples(7), ref_cgc, ref_cdc, ref_hi_rate
    real(dp) :: reference_kc, reference_co2, ratio, normalized
    type(rep_EffectStress) :: effect
    integer(int32) :: count, index, stress, calibrated
    integer(int32) :: ref_days(7), crop_day, crop_month, crop_year, reference_start
    integer(int32) :: level, full_sf, gfull_sf, stress_class

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
    write(*, '(a,1x,i0,1x,es24.16)') 'reference', GetTnxReferenceYear(), &
        CO2ForTnxReferenceYear(GetTnxReferenceYear())
    write(*, '(a,1x,5(i0,1x),16(es24.16,1x))') 'thermal', &
        GetCrop_GDDaysToGermination(), GetCrop_GDDaysToFullCanopy(), &
        GetCrop_GDDaysToSenescence(), GetCrop_GDDaysToHarvest(), GetCrop_GDDaysToHIo(), &
        GetCrop_GDDCGC(), GetCrop_GDDCDC(), GetCrop_CGC(), GetCrop_CDC(), &
        GetCrop_WP(), GetCrop_KcTop(), GetCrop_KcDeclineCumul(), &
        real(GetCrop_CCEffectEvapLate(), dp), GetCrop_Tbase(), GetCrop_Tupper(), &
        GetCrop_GDtranspLow(), GetSimulParam_Tmin(), GetSimulParam_Tmax(), &
        GetCO2i(), GetSumKcTop(), GetSumKcTopStress()
    call DetermineDate(GetCrop_Day1(), crop_day, crop_month, crop_year)
    call DetermineDayNr(crop_day, crop_month, 1901_int32, reference_start)
    ref_days = [GetCrop_DaysToGermination(), GetCrop_DaysToFullCanopy(), &
        GetCrop_DaysToFlowering(), GetCrop_LengthFlowering(), GetCrop_DaysToSenescence(), &
        GetCrop_DaysToHarvest(), GetCrop_DaysToHIo()]
    ref_cgc = GetCrop_CGC()
    ref_cdc = GetCrop_CDC()
    ref_hi_rate = GetCrop_dHIdt()
    call AdjustCalendarDaysReferenceTnx(reference_start, GetCrop_subkind(), &
        GetCrop_Tbase(), GetCrop_Tupper(), GetSimulParam_Tmin(), GetSimulParam_Tmax(), &
        GetCrop_GDDaysToGermination(), GetCrop_GDDaysToFullCanopy(), &
        GetCrop_GDDaysToFlowering(), GetCrop_GDDLengthFlowering(), &
        GetCrop_GDDaysToSenescence(), GetCrop_GDDaysToHarvest(), GetCrop_GDDaysToHIo(), &
        GetCrop_GDDCGC(), GetCrop_GDDCDC(), GetCrop_CCo(), GetCrop_CCx(), GetCrop_HI(), &
        GetCrop_DaysToCCini(), GetCrop_GDDaysToCCini(), GetCrop_Planting(), &
        ref_days(1), ref_days(2), ref_days(3), ref_days(4), ref_days(5), ref_days(6), &
        ref_days(7), ref_cgc, ref_cdc, ref_hi_rate)
    write(*, '(a,1x,7(i0,1x),3(es24.16,1x))') 'refcalendar', &
        ref_days, ref_cgc, ref_cdc, ref_hi_rate
    reference_co2 = CO2ForTnxReferenceYear(GetTnxReferenceYear())
    reference_kc = SeasonalSumOfKcPot(GetCrop_DaysToCCini(), GetCrop_GDDaysToCCini(), &
        ref_days(1), ref_days(2), ref_days(5), ref_days(6), ref_days(6), &
        GetCrop_GDDaysToGermination(), GetCrop_GDDaysToFullCanopy(), &
        GetCrop_GDDaysToSenescence(), GetCrop_GDDaysToHarvest(), GetCrop_CCo(), &
        GetCrop_CCx(), ref_cgc, GetCrop_GDDCGC(), ref_cdc, GetCrop_GDDCDC(), &
        GetCrop_KcTop(), GetCrop_KcDeclineCumul(), real(GetCrop_CCEffectEvapLate(), dp), &
        GetCrop_Tbase(), GetCrop_Tupper(), GetSimulParam_Tmin(), GetSimulParam_Tmax(), &
        GetCrop_GDtranspLow(), reference_co2, GetCrop_ModeCycle(), .true.)
    write(*, '(a,1x,2(es24.16,1x))') 'refkc', reference_kc, GetSimulParam_ExpFsen()
    do level = 0, 70, 10
        call CropStressParametersSoilFertility(GetCrop_StressResponse(), level, effect)
        full_sf = ref_days(2)
        gfull_sf = GetCrop_GDDaysToFullCanopy()
        ratio = 1._dp
        stress_class = level
        if ((effect%RedCGC /= 0) .or. (effect%RedCCX /= 0)) then
            call TimeToMaxCanopySF(GetCrop_CCo(), ref_cgc, GetCrop_CCx(), ref_days(1), &
                ref_days(2), ref_days(5), ref_days(3), ref_days(4), GetCrop_DeterminancyLinked(), &
                full_sf, effect%RedCGC, effect%RedCCX, stress_class)
            gfull_sf = SumCalendarDaysReferenceTnx(full_sf, reference_start, reference_start, &
                GetCrop_Tbase(), GetCrop_Tupper(), GetSimulParam_Tmin(), GetSimulParam_Tmax())
            if (gfull_sf < GetCrop_GDDaysToSenescence()) &
                ratio = real(ref_days(5) - full_sf, dp) / (GetCrop_GDDaysToSenescence() - gfull_sf)
        end if
        normalized = Bnormalized(GetCrop_DaysToCCini(), GetCrop_GDDaysToCCini(), &
            ref_days(1), ref_days(2), full_sf, ref_days(5), ref_days(6), ref_days(6), ref_days(3), &
            GetCrop_GDDaysToGermination(), GetCrop_GDDaysToFullCanopy(), gfull_sf, &
            GetCrop_GDDaysToSenescence(), GetCrop_GDDaysToHarvest(), GetCrop_WPy(), &
            0_int32, 0_int32, GetCrop_CCo(), GetCrop_CCx(), ref_cgc, GetCrop_GDDCGC(), &
            ref_cdc, GetCrop_GDDCDC(), GetCrop_KcTop(), GetCrop_KcDeclineCumul(), &
            real(GetCrop_CCEffectEvapLate(), dp), GetCrop_WP(), reference_co2, GetCrop_Tbase(), &
            GetCrop_Tupper(), GetSimulParam_Tmin(), GetSimulParam_Tmax(), GetCrop_GDtranspLow(), &
            ratio, reference_kc, level, effect%RedCGC, effect%RedCCX, effect%RedWP, effect%RedKsSto, &
            0_int8, 0_int32, effect%CDecline, -.01_dp, GetCrop_ModeCycle(), .true., .true.)
        write(*, '(a,1x,6(i0,1x),3(es24.16,1x))') 'level', level, full_sf, gfull_sf, &
            int(effect%RedCGC), int(effect%RedCCX), int(effect%RedWP), effect%CDecline, ratio, normalized
    end do
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
