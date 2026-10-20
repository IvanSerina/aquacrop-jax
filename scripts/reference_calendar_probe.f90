! Original array branch of SumCalendarDaysReferenceTnx, with SP temperatures.
program reference_calendar_probe
    use ac_kinds, only: dp, sp, int8, int32
    use ac_global, only: SumCalendarDaysReferenceTnx, SetTnxReferenceFile, &
        SetTminCropReferenceRun, SetTmaxCropReferenceRun, SetSimulParam_GDDMethod
    implicit none
    integer(int32) :: count, row, i, target, offset, pattern, days
    real(dp) :: first, second, last
    real(sp) :: temperatures(365)
    call SetTnxReferenceFile('(External)')
    call SetSimulParam_GDDMethod(3_int8)
    read(*, *) count
    do row = 1, count
        read(*, *) target, offset, pattern, first, second, last
        temperatures = real(first, sp)
        if (pattern == 1) then
            do i = 2, 364, 2
                temperatures(i) = real(second, sp)
            end do
        else if (pattern == 2) then
            temperatures(1) = real(second, sp)
        end if
        temperatures(365) = real(last, sp)
        call SetTminCropReferenceRun(temperatures)
        call SetTmaxCropReferenceRun(temperatures)
        days = SumCalendarDaysReferenceTnx(target, 1_int32, 1_int32 + offset, &
            5._dp, 30._dp, first, first)
        write(*, '(i0)') days
    end do
end program reference_calendar_probe
