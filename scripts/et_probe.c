/* Native CalculateETpot, immutable Fortran objects. Each input row has
 * DAP,L0,L12,L123,LHarvest,DayLastCut,DelayedDays,EvapLimitON then
 * CCi,ETo,Kc,KcDecline,CCx,CCxWithered,CCeffect,CO2,GDD,TempLow,KcBare,ExpFsen.
 */
#include <stdint.h>
#include <stdio.h>
#define G(name) __ac_global_MOD_##name
extern void G(setsimulation_delayeddays)(const int32_t *);
extern void G(setsimulation_evaplimiton)(const int32_t *);
extern void G(setsimulparam_kcwetbare)(const double *);
extern void G(setsimulparam_expfsen)(const double *);
extern void G(calculateetpot)(const int32_t *, const int32_t *, const int32_t *,
    const int32_t *, const int32_t *, const int32_t *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const double *, double *, double *);

int main(void) {
    int count;
    if (scanf("%d", &count) != 1 || count < 1 || count > 10000) return 2;
    for (int row = 0; row < count; ++row) {
        int32_t days[8];
        double values[12], transpiration = 0., evaporation = 0.;
        for (int i = 0; i < 8; ++i) {
            int value;
            if (scanf("%d", &value) != 1) return 2;
            days[i] = value;
        }
        for (int i = 0; i < 12; ++i)
            if (scanf("%lf", &values[i]) != 1) return 2;
        G(setsimulation_delayeddays)(&days[6]);
        G(setsimulation_evaplimiton)(&days[7]);
        G(setsimulparam_kcwetbare)(&values[10]);
        G(setsimulparam_expfsen)(&values[11]);
        G(calculateetpot)(&days[0], &days[1], &days[2], &days[3], &days[4], &days[5],
            &values[0], &values[1], &values[2], &values[3], &values[4], &values[5],
            &values[6], &values[7], &values[8], &values[9], &transpiration, &evaporation);
        printf("%.17g %.17g\n", transpiration, evaporation);
    }
    return 0;
}
