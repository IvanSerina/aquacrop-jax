! Calls the immutable AquaCrop reference routine with controlled cumulative
! biomass and fertility calibration inputs. Input: count, then one row per
! call: management calibrated Btotal Bunlimited fraction b0 b1 b2 previous
! SumKcTop ETo Tact grain determinate day delayed crop_day1 flowering length.
! Output: stress previous Btotal Bunlimited SumKcTopStress.
program fertility_stress_probe
    use ac_kinds, only: dp, int8, int32
    use ac_global, only: CO2Ref, subkind_Forage, plant_regrowth, &
        SetCrop_subkind, SetCrop_Planting, SetCrop_HI, SetCrop_WP, &
        SetCrop_Day1, SetCrop_DaysToFlowering, &
        SetCrop_StressResponse_Calibrated, SetManagement_FertilityStress, &
        SetSimulation_DelayedDays, SetSimulation_RCadj, &
        SetSimulation_EffectStress_RedWP, subkind_grain, plant_seed, &
        SetCrop_DeterminancyLinked, SetCrop_LengthFlowering, &
        SetCrop_dHIdt, SetCrop_aCoeff, SetCrop_bCoeff, &
        SetCrop_DaysToSenescence, SetCrop_CCo, SetCrop_CCxAdjusted, &
        SetCrop_CCxWithered, SetSimulation_HIfinal, SetCCiActual, &
        SetSimulParam_PercCCxHIfinal, SetNrCompartments, &
        SetCompartment_Thickness, SetCompartment_theta, SetCompartment_Layer, &
        SetSoilLayer_SAT, SetSoilLayer_FC, SetSoilLayer_WP, &
        SetSoilLayer_GravelVol, SetRootingDepth, SetCrop_pLeafAct, &
        SetCrop_pActStom, SetCrop_pSenAct, SetSimulParam_ThicknessTopSWC, &
        SetSimulation_SWCtopSoilConsidered
    use ac_simul, only: DetermineBiomassAndYield
    implicit none

    integer(int32) :: count, index, management, calibrated, previous
    integer(int32) :: stress, day, interval, virtual_time
    integer(int32) :: grain, determinate, delayed, crop_day1, flowering, length
    real(dp) :: biomass, potential_biomass, unlimited, total, yield, wp
    real(dp) :: hi_before, score1, score2, hi_at1, hi_at2, hi_at
    real(dp) :: alfa, alfa_max, sum_kctop_stress, sum_kci
    real(dp) :: weed_rc, ccw, trw, to_mobilize, mobilized, bin, bout, testval
    real(dp) :: fraction_pot_sf, fraction_assim, sum_kctop, cgc_ref, gdd_cgc_ref
    real(dp) :: b0, b1, b2, salt_b0, salt_b1, salt_b2, salt_stress
    real(dp) :: sum_gdd, cc, eto, tact
    logical :: store_assimilates, mobilize_assimilates, calibrated_logical
    logical :: probe_wp
    character(len=32) :: mode
    integer(int8) :: reduction_wp
    real(dp) :: initial_sum_kci, initial_top_stress

    call get_command_argument(1, mode)
    probe_wp = trim(mode) == '--fertility-wp'
    read(*, *) count
    call SetCrop_subkind(subkind_Forage)
    call SetCrop_Planting(plant_regrowth)
    call SetCrop_HI(50_int32)
    call SetCrop_WP(15._dp)
    call SetCrop_Day1(100_int32)
    call SetCrop_DaysToFlowering(100_int32)
    call SetSimulation_DelayedDays(0_int32)
    call SetSimulation_RCadj(0_int8)
    call SetSimulation_EffectStress_RedWP(0_int8)
    ! A wet, one-compartment root zone makes the post-flowering HI path
    ! well-defined. The probe observes fertility, not yield or water fluxes.
    call SetNrCompartments(1_int32)
    call SetCompartment_Thickness(1_int32, .3_dp)
    call SetCompartment_theta(1_int32, .3_dp)
    call SetCompartment_Layer(1_int32, 1_int32)
    call SetSoilLayer_SAT(1_int32, 46._dp)
    call SetSoilLayer_FC(1_int32, 30._dp)
    call SetSoilLayer_WP(1_int32, 13._dp)
    call SetSoilLayer_GravelVol(1_int32, 0._dp)
    call SetRootingDepth(.3_dp)
    call SetCrop_pLeafAct(.15_dp)
    call SetCrop_pActStom(.65_dp)
    call SetCrop_pSenAct(.7_dp)
    call SetSimulParam_ThicknessTopSWC(10_int8)
    call SetCrop_dHIdt(100._dp)
    call SetCrop_aCoeff(-9._dp)
    call SetCrop_bCoeff(-9._dp)
    call SetCrop_DaysToSenescence(180_int32)
    call SetCrop_CCo(.05_dp)
    call SetCrop_CCxAdjusted(.95_dp)
    call SetCrop_CCxWithered(0._dp)
    call SetCCiActual(0._dp)
    call SetSimulParam_PercCCxHIfinal(5_int8)

    do index = 1, count
        read(*, *) management, calibrated, total, unlimited, fraction_pot_sf, &
            b0, b1, b2, previous, sum_kctop, eto, tact, &
            grain, determinate, day, delayed, crop_day1, flowering, length
        call SetCrop_subkind(subkind_Forage)
        call SetCrop_Planting(plant_regrowth)
        if (grain /= 0) then
            call SetCrop_subkind(subkind_grain)
            call SetCrop_Planting(plant_seed)
        end if
        call SetCrop_DeterminancyLinked(determinate /= 0)
        call SetCrop_Day1(crop_day1)
        call SetCrop_DaysToFlowering(flowering)
        call SetCrop_LengthFlowering(length)
        call SetSimulation_DelayedDays(delayed)
        call SetSimulation_HIfinal(50_int32)
        call SetSimulation_SWCtopSoilConsidered(.false.)
        call SetManagement_FertilityStress(management)
        calibrated_logical = calibrated /= 0
        call SetCrop_StressResponse_Calibrated(calibrated_logical)

        interval = 0_int32
        virtual_time = 0_int32
        biomass = 0._dp
        potential_biomass = fraction_pot_sf * unlimited
        yield = 0._dp
        wp = 0._dp
        hi_before = 1._dp
        score1 = 0._dp
        score2 = 0._dp
        hi_at1 = 0._dp
        hi_at2 = 0._dp
        hi_at = 1._dp
        alfa = 0._dp
        alfa_max = 0._dp
        sum_kctop_stress = 0._dp
        sum_kci = 0._dp
        if (probe_wp) then
            read(*, *) reduction_wp, initial_sum_kci, initial_top_stress
            call SetSimulation_EffectStress_RedWP(reduction_wp)
            sum_kci = initial_sum_kci
            sum_kctop_stress = initial_top_stress
        end if
        weed_rc = 0._dp
        ccw = 0._dp
        trw = 0._dp
        stress = previous
        store_assimilates = .false.
        mobilize_assimilates = .false.
        to_mobilize = 0._dp
        mobilized = 0._dp
        bin = 0._dp
        bout = 0._dp
        testval = 0._dp
        fraction_assim = 0._dp
        cgc_ref = 0._dp
        gdd_cgc_ref = 0._dp
        salt_b0 = 0._dp
        salt_b1 = 0._dp
        salt_b2 = 0._dp
        salt_stress = 0._dp
        sum_gdd = 0._dp
        cc = 0._dp

        call DetermineBiomassAndYield(day, eto, 15._dp, 25._dp, CO2Ref, &
            10._dp, tact, sum_kctop, cgc_ref, gdd_cgc_ref, b0, b1, b2, &
            fraction_pot_sf, salt_b0, salt_b1, salt_b2, salt_stress, &
            sum_gdd, cc, fraction_assim, virtual_time, interval, biomass, &
            potential_biomass, unlimited, total, yield, wp, hi_before, &
            score1, score2, hi_at1, hi_at2, hi_at, alfa, alfa_max, &
            sum_kctop_stress, sum_kci, weed_rc, ccw, trw, stress, previous, &
            store_assimilates, mobilize_assimilates, to_mobilize, mobilized, &
            bin, bout, testval)

        if (probe_wp) then
            write(*, '(3(es24.16,1x))') wp, sum_kci, biomass
        else
            write(*, '(i0,1x,i0,1x,3(es24.16,1x))') stress, previous, total, &
                unlimited, sum_kctop_stress
        end if
    end do
end program fertility_stress_probe
