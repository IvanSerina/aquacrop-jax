! Native Bnormalized calibration, zero weeds, with each daily prefix observed.
program normalized_biomass_probe
    use ac_kinds, only: dp, int8, int32, intEnum
    use ac_global, only: SeasonalSumOfKcPot, SetTemperatureFile, &
        SetSimulParam_GDDMethod, SetSimulParam_KcWetBare, SetSimulParam_ExpFsen, &
        SetSimulation_EvapLimitON, SetCrop_subkind, subkind_grain, subkind_forage
    use ac_tempprocessing, only: Bnormalized
    implicit none
    integer(int32) :: count, index, n, day, initial_day, initial_gdd
    integer(int32) :: l0, l12, l123, maturity, g0, g12, g123, g_maturity
    integer(int32) :: stress, full_sf, gfull_sf, yield_wp, yield_days, switch, flower
    integer(int32) :: reproductive, fertility
    integer(int8) :: red_growth, red_cover, red_wp, red_stomatal
    integer(intEnum) :: mode
    real(dp) :: cco, ccx, cgc, gddcgc, cdc, gddcdc, kc, ageing, late
    real(dp) :: base, upper, tmin, tmax, minimum_gdd, co2, total, sum_kc
    real(dp) :: decline, ratio, wp, kc_override, senescence_exponent
    call SetTemperatureFile('(None)')
    call SetSimulParam_GDDMethod(3_int8)
    call SetSimulParam_KcWetBare(1.1_dp)
    call SetSimulParam_ExpFsen(2.5_dp)
    call SetSimulation_EvapLimitON(.true.)
    read(*, *) count
    do index = 1, count
        read(*, *) mode, initial_day, initial_gdd, n, l0, l12, l123, maturity, &
            g0, g12, g123, g_maturity, cco, ccx, cgc, gddcgc, cdc, gddcdc, &
            kc, ageing, late, base, upper, tmin, tmax, minimum_gdd, co2, &
            stress, red_growth, red_cover, red_wp, decline, full_sf, gfull_sf, &
            ratio, wp, yield_wp, yield_days, switch, flower, reproductive, &
            fertility, red_stomatal, kc_override, senescence_exponent
        call SetSimulParam_ExpFsen(senescence_exponent)
        call SetCrop_subkind(subkind_forage)
        if (reproductive /= 0) call SetCrop_subkind(subkind_grain)
        sum_kc = SeasonalSumOfKcPot(initial_day, initial_gdd, l0, l12, l123, &
            maturity, n, g0, g12, g123, g_maturity, cco, ccx, cgc, gddcgc, &
            cdc, gddcdc, kc, ageing, late, base, upper, tmin, tmax, &
            minimum_gdd, co2, mode, .true.)
        if (kc_override >= 0._dp) sum_kc = kc_override
        write(*, '(es24.16)') sum_kc
        do day = 1, n
            total = Bnormalized(initial_day, initial_gdd, l0, l12, full_sf, &
                l123, maturity, day, flower, g0, g12, gfull_sf, g123, &
                g_maturity, yield_wp, yield_days, switch, cco, ccx, &
                cgc, gddcgc, cdc, gddcdc, kc, ageing, late, wp, co2, &
                base, upper, tmin, tmax, minimum_gdd, ratio, sum_kc, &
                stress, red_growth, red_cover, red_wp, red_stomatal, &
                0_int8, 0_int32, decline, -.01_dp, mode, fertility /= 0, .true.)
            write(*, '(es24.16)') total
        end do
    end do
end program normalized_biomass_probe
