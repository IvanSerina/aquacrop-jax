! Direct calls to immutable global.f90::CCiNoWaterStressSF.
program fertility_decline_probe
    use ac_kinds, only: dp, int8, int32, intEnum
    use ac_global, only: CCiNoWaterStressSF, SetSimulation_DelayedDays
    implicit none
    integer(int32) :: count, i, day, l0, full, sen, maturity, g0, gfull, gsen, gmature
    integer(int8) :: red_growth, red_cover
    integer(intEnum) :: mode
    real(dp) :: initial, maximum, cgc, gddcgc, cdc, gddcdc, gdd, ratio, decline, cover
    call SetSimulation_DelayedDays(0_int32)
    read(*, *) count
    do i = 1, count
        read(*, *) mode, day, l0, full, sen, maturity, g0, gfull, gsen, gmature, &
            initial, maximum, cgc, gddcgc, cdc, gddcdc, gdd, ratio, &
            red_growth, red_cover, decline
        cover = CCiNoWaterStressSF(day, l0, full, sen, maturity, g0, gfull, &
            gsen, gmature, initial, maximum, cgc, gddcgc, cdc, gddcdc, &
            gdd, ratio, red_growth, red_cover, decline, mode)
        write(*, '(es24.16)') cover
    end do
end program fertility_decline_probe
