! Original GDDCDCToCDC, actual daily climate; no modified reference sources.
program calendar_decline_probe
    use ac_kinds, only: dp, int8, int32
    use ac_global
    use ac_tempprocessing, only: GDDCDCToCDC, ReadTemperatureFilefull, TemperatureFilefull_exists
    implicit none
    integer(int32) :: count, row, sen, maturity, day, offset
    real(dp) :: maximum, thermal, result
    type(rep_clim) :: record
    character(len=256) :: description
    call InitializeGlobalStrings()
    call SetTemperatureFile('Ottawa.Tnx')
    call SetTemperatureFilefull('DATA/Ottawa.Tnx')
    TemperatureFilefull_exists = FileExists(GetTemperatureFilefull())
    call ReadTemperatureFilefull()
    description = ''
    call LoadClim(GetTemperatureFilefull(), description, record)
    call SetTemperatureRecord(record)
    call SetSimulParam_GDDMethod(3_int8)
    read(*,*) count
    do row=1,count
        read(*,*) maximum,thermal,sen,maturity,day,offset
        result = -999._dp
        call GDDCDCToCDC(GetTemperatureRecord_FromDayNr()+offset, day, sen, maturity, &
            maximum,thermal,5._dp,30._dp,12._dp,28._dp,result,.false.)
        write(*,'(es24.16)') result
    end do
end program calendar_decline_probe
