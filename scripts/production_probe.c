/* Link the immutable reference objects; do not reimplement the HI equations.
 * stdin: CRO kind (1..4), HImax, dHIdt, flowering day, count; then count DAPs.
 * Healthy canopy, no delayed germination, seed planting, no HIfinal restriction.
 */
#include <stdint.h>
#include <stdio.h>

#define G(name) __ac_global_MOD_##name
extern void G(setcrop_subkind)(const int8_t *);
extern void G(setcrop_cco)(const double *);
extern void G(setsimulation_delayeddays)(const int32_t *);
extern double G(harvestindexgrowthcoefficient)(const double *, const double *);
extern void G(getdayswitchtolinear)(const int32_t *, const double *, const double *, int32_t *, double *);
extern double G(harvestindexday)(const int32_t *, const int32_t *, const int32_t *,
    const double *, const double *, const double *, const double *, const int8_t *,
    const int8_t *, int8_t *, int32_t *);

int main(void) {
    int kind, maximum, flowering, count;
    double rate;
    if (scanf("%d%d%lf%d%d", &kind, &maximum, &rate, &flowering, &count) != 5 ||
        kind < 1 || kind > 4 || maximum < 2 || maximum > 100 ||
        rate <= 0 || count < 1 || count > 10000) return 2;
    int8_t subkind = (int8_t)(kind - 1), planting = 0, threshold = 5, lag = 0;
    int32_t delayed = 0, hi_max = maximum, final = maximum, flower = flowering, switch_day = 0;
    double initial_cover = .05, cover = .9, maximum_cover = .95, withered_cover = .95;
    double max_real = maximum, coefficient = 0., linear_rate = 0.;
    G(setcrop_subkind)(&subkind);
    G(setcrop_cco)(&initial_cover);
    G(setsimulation_delayeddays)(&delayed);
    if (rate <= 99.) {
        coefficient = G(harvestindexgrowthcoefficient)(&max_real, &rate);
        G(getdayswitchtolinear)(&hi_max, &rate, &coefficient, &switch_day, &linear_rate);
    }
    printf("%.17g %d %.17g\n", coefficient, switch_day, linear_rate);
    for (int i = 0; i < count; ++i) {
        int day;
        if (scanf("%d", &day) != 1) return 2;
        int32_t dap = day;
        double hi = G(harvestindexday)(&dap, &flower, &hi_max, &rate, &cover,
            &maximum_cover, &withered_cover, &threshold, &planting, &lag, &final);
        printf("%d %.17g %d %d\n", day, hi, (int)lag, final);
    }
    return 0;
}
