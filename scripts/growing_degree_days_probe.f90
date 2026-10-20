! Observe GrowingDegreeDays with the original daily temperature-file loaders.
! Controlled Tbase=5,Tupper=30,GDDmethod3; no seasonal runner is initialized.
program growing_degree_days_probe
    use ac_kinds, only: dp, int8, int32
    use ac_global
    use ac_tempprocessing, only: GrowingDegreeDays, SumCalendarDays, TminRecord => Tmin, &
        ReadTemperatureFilefull, TemperatureFilefull_exists
    implicit none
    integer(int32) :: count, row, period, offset, value, undefined
    type(rep_clim) :: record
    character(len=256) :: description
    character(len=32) :: mode
    character(len=32) :: operation
    call InitializeGlobalStrings()
    call SetTemperatureFile('Ottawa.Tnx')
    call SetTemperatureFilefull('DATA/Ottawa.Tnx')
    TemperatureFilefull_exists = FileExists(GetTemperatureFilefull())
    call ReadTemperatureFilefull()
    description = ''
    call LoadClim(GetTemperatureFilefull(), description, record)
    call SetTemperatureRecord(record)
    call SetSimulParam_GDDMethod(3_int8)
    call SetSimulParam_Tmin(12._dp)
    call SetSimulParam_Tmax(28._dp)
    call SetCrop_Tbase(5._dp)
    call SetCrop_Tupper(30._dp)
    call get_command_argument(1, mode)
    call get_command_argument(2, operation)
    if (trim(mode) == '--base-zero') call SetCrop_Tbase(0._dp)
    undefined = 0
    if (FullUndefinedRecord(GetTemperatureRecord_FromY(), GetTemperatureRecord_FromD(), &
        GetTemperatureRecord_FromM(), GetTemperatureRecord_ToD(), GetTemperatureRecord_ToM())) undefined = 1
    write(*,'(6(i0,1x),4(es24.16,1x))') size(TminRecord), GetTemperatureRecord_FromDayNr(), &
        GetTemperatureRecord_ToDayNr(), GetTemperatureRecord_FromDayNr()+140, &
        int(GetSimulParam_GDDMethod()), undefined, &
        GetCrop_Tbase(), GetCrop_Tupper(), GetSimulParam_Tmin(), GetSimulParam_Tmax()
    read(*,*) count
    do row = 1,count
        read(*,*) period,offset
        if (trim(operation) == '--calendar-days') then
            value = SumCalendarDays(period, GetTemperatureRecord_FromDayNr()+offset, &
                GetCrop_Tbase(), GetCrop_Tupper(), GetSimulParam_Tmin(), GetSimulParam_Tmax())
        else
            value = GrowingDegreeDays(period, GetTemperatureRecord_FromDayNr()+offset, &
                GetCrop_Tbase(), GetCrop_Tupper(), GetSimulParam_Tmin(), GetSimulParam_Tmax())
        end if
        write(*,'(3(i0,1x))') period,offset,value
    end do
end program growing_degree_days_probe
