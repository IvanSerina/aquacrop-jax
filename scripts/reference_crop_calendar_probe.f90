! Native GDD reference calendar for controlled crop types and regrowth states.
program reference_crop_calendar_probe
    use ac_kinds, only: dp, sp, int8, int32, intEnum
    use ac_global, only: SetTnxReferenceFile, SetTminCropReferenceRun, &
        SetTmaxCropReferenceRun, SetSimulParam_GDDMethod, plant_seed, &
        subkind_vegetative, subkind_grain, subkind_tuber, subkind_forage
    use ac_preparefertilitysalinity, only: AdjustCalendarDaysReferenceTnx
    implicit none
    integer(int32) :: count, index, initial_day, initial_gdd, hi_gdd, hi, legacy_kind
    integer(int32) :: d(7), g(6)
    integer(intEnum) :: kind
    real(dp) :: temperature, cgc, gddcgc, cdc, gddcdc, maximum, initial, rate
    real(sp) :: daily(365)
    call SetTnxReferenceFile('(External)')
    call SetSimulParam_GDDMethod(3_int8)
    read(*, *) count
    do index = 1, count
        read(*, *) legacy_kind, initial_day, initial_gdd, temperature, d(1), d(2), &
            d(5), d(6), d(3), d(4), g, hi_gdd, hi, initial, maximum, &
            cgc, gddcgc, cdc, gddcdc, d(7), rate
        ! Match LoadCrop: CRO codes 1..4 are not the internal enum 0..3.
        select case (legacy_kind)
        case (1)
            kind = subkind_vegetative
        case (2)
            kind = subkind_grain
        case (3)
            kind = subkind_tuber
        case (4)
            kind = subkind_forage
        case default
            error stop 'Invalid legacy crop kind'
        end select
        daily = real(temperature, sp)
        call SetTminCropReferenceRun(daily)
        call SetTmaxCropReferenceRun(daily)
        call AdjustCalendarDaysReferenceTnx(1_int32, kind, 5._dp, 30._dp, &
            temperature, temperature, g(1), g(2), g(5), g(6), g(3), g(4), &
            hi_gdd, gddcgc, gddcdc, initial, maximum, hi, initial_day, initial_gdd, &
            plant_seed, d(1), d(2), d(3), d(4), d(5), d(6), d(7), cgc, cdc, rate)
        write(*, '(7(i0,1x),3(es24.16,1x))') d, cgc, cdc, rate
    end do
end program reference_crop_calendar_probe
