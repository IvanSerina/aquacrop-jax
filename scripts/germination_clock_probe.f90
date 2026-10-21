! Template: seeded-clock assignments inserted verbatim from AdvanceOneTimeStep.
! Direct native routines update daily GDD and post-input-water germination.
program germination_clock_probe
    use ac_kinds, only: dp, int8, int32
    use ac_global
    use ac_run, only: SetDayNri, GetDayNri, GetGDDayi, SetGDDVariablesNextDay
    use ac_simul, only: CheckGermination
    implicit none
    integer(int32) :: count, row, day, first, last, germ, protected, delay, seeded
    integer(int32) :: VirtualTimeCC
    real(dp) :: gdd, total, theta_input, fc_input, wp_input, heat_input
    real(dp) :: SumGDDadjCC, actual_water, fc_water, wp_water
    integer(int32) :: percentage
    logical :: top_selected
    call SetNrCompartments(1_int32)
    call SetCompartment_Thickness(1_int32,.1_dp)
    call SetCompartment_Layer(1_int32,1_int32)
    call SetSoilLayer_SAT(1_int32,46._dp)
    call SetSoilLayer_GravelVol(1_int32,0._dp)
    call SetCrop_RootMin(.1_dp)
    call SetCrop_pLeafAct(.2_dp)
    call SetCrop_pActStom(.5_dp)
    call SetCrop_pSenAct(.7_dp)
    call SetSimulParam_ThicknessTopSWC(10_int8)
    call SetCrop_ModeCycle(modeCycle_GDDays)
    call SetCrop_Tbase(5._dp)
    call SetCrop_Tupper(30._dp)
    call SetSimulParam_GDDMethod(3_int8)
    read(*,*) count
    do row=1,count
        read(*,*) day,first,last,germ,protected,delay,gdd,total, &
            theta_input,fc_input,wp_input,percentage,seeded,heat_input
        call SetDayNri(day)
        call SetCrop_Day1(first)
        call SetSimulation_ToDayNr(last)
        call SetSimulation_Germinate(germ /= 0)
        call SetSimulation_ProtectedSeedling(protected /= 0)
        call SetSimulation_DelayedDays(delay)
        call SetSimulation_SumGDD(gdd)
        call SetSimulation_SumGDDfromDay1(total)
        call SetCompartment_Theta(1_int32,theta_input)
        call SetSoilLayer_FC(1_int32,fc_input)
        call SetSoilLayer_WP(1_int32,wp_input)
        call SetSimulParam_TAWGermination(int(percentage,int8))
        if (seeded /= 0) then
            call SetCrop_Planting(plant_seed)
        else
            call SetCrop_Planting(plant_transplant)
        end if
        call SetTmin(5._dp+heat_input)
        call SetTmax(5._dp+heat_input)
        call SetGDDVariablesNextDay()
        ! NATIVE_SEEDED_CLOCK
        top_selected = .false.
        call DetermineRootZoneWC(GetCrop_RootMin(),top_selected)
        actual_water = GetRootZoneWC_Actual()
        fc_water = GetRootZoneWC_FC()
        wp_water = GetRootZoneWC_WP()
        if ((.not. GetSimulation_Germinate()) .and. (day >=GetCrop_Day1())) then
            call CheckGermination()
        end if
        write(*,'(11(es24.16,1x))') real(VirtualTimeCC,dp),SumGDDadjCC, &
            merge(1._dp,0._dp,GetSimulation_Germinate()), &
            merge(1._dp,0._dp,GetSimulation_ProtectedSeedling()), &
            real(GetSimulation_DelayedDays(),dp),GetSimulation_SumGDD(), &
            GetSimulation_SumGDDfromDay1(),actual_water,fc_water,wp_water,GetGDDayi()
    end do
end program germination_clock_probe
