! Observe the actual CLI loader, without initializing a season or writing files.
program program_temperature_probe
    use ac_kinds, only: dp, int8
    use ac_global, only: SetSimulParam_GDDMethod, SetSimulParam_ExpFsen, &
        GetSimulParam_GDDMethod, GetSimulParam_ExpFsen, DegreesDay
    use ac_startunit, only: LoadProgramParametersProjectPlugIn
    implicit none
    character(len=4096) :: filename
    character(len=16) :: mode
    real(dp) :: base, upper, tmin, tmax
    integer :: ios
    logical :: available
    call get_command_argument(1, filename)
    call SetSimulParam_GDDMethod(3_int8)
    call SetSimulParam_ExpFsen(1._dp)
    available = .false.
    call LoadProgramParametersProjectPlugIn(trim(filename), available)
    write(*, '(i0,1x,es24.16)') GetSimulParam_GDDMethod(), GetSimulParam_ExpFsen()
    call get_command_argument(2, mode)
    if (trim(mode) == '--gdd') then
        do
            read(*, *, iostat=ios) base, upper, tmin, tmax
            if (ios /= 0) exit
            write(*, '(es24.16)') DegreesDay(base, upper, tmin, tmax, GetSimulParam_GDDMethod())
        end do
    end if
end program program_temperature_probe
