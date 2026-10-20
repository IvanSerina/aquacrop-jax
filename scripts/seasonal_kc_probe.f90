! Actual native SeasonalSumOfKcPot for each prefix of a constant-climate run.
program seasonal_kc_probe
    use ac_kinds, only: dp, int8, int32, intEnum
    use ac_global, only: SeasonalSumOfKcPot, SetTemperatureFile, &
        SetSimulParam_GDDMethod, SetSimulParam_KcWetBare, SetSimulParam_ExpFsen, &
        SetSimulation_EvapLimitON
    implicit none
    integer(int32) :: count, index, n, day, initial_day, initial_gdd
    integer(int32) :: l0, l12, l123, maturity, g0, g12, g123, g_maturity
    integer(intEnum) :: mode
    real(dp) :: cco, ccx, cgc, gddcgc, cdc, gddcdc, kc, ageing, late
    real(dp) :: base, upper, tmin, tmax, minimum_gdd, co2, total
    call SetTemperatureFile('(None)')
    call SetSimulParam_GDDMethod(3_int8)
    call SetSimulParam_KcWetBare(1.1_dp)
    call SetSimulParam_ExpFsen(2.5_dp)
    call SetSimulation_EvapLimitON(.true.)
    read(*, *) count
    do index = 1, count
        read(*, *) mode, initial_day, initial_gdd, n, l0, l12, l123, maturity, &
            g0, g12, g123, g_maturity, cco, ccx, cgc, gddcgc, cdc, gddcdc, &
            kc, ageing, late, base, upper, tmin, tmax, minimum_gdd, co2
        do day = 1, n
            total = SeasonalSumOfKcPot(initial_day, initial_gdd, l0, l12, l123, &
                maturity, day, g0, g12, g123, g_maturity, cco, ccx, cgc, gddcgc, &
                cdc, gddcdc, kc, ageing, late, base, upper, tmin, tmax, &
                minimum_gdd, co2, mode, .true.)
            write(*, '(es24.16)') total
        end do
    end do
end program seasonal_kc_probe
