/* Native CanopyCoverNoStressSF. No reference source is modified.
 * stdin: mode(0 GDD, 1 calendar), delayed days, count, emergence, senescence,
 * maturity, initial cover, maximum cover, growth, decline, RedCGC, RedCCx;
 * followed by count times (integer DAP in calendar mode).
 */
#include <stdint.h>
#include <stdio.h>

#define G(name) __ac_global_MOD_##name
extern void G(setsimulation_delayeddays)(const int32_t *);
extern double G(canopycovernostresssf)(const int32_t *, const int32_t *,
    const int32_t *, const int32_t *, const int32_t *, const int32_t *,
    const int32_t *, const double *, const double *, const double *,
    const double *, const double *, const double *, const double *,
    const int8_t *, const int8_t *, const int8_t *);

int main(void) {
    int mode, delayed, count, emergence, senescence, maturity, red_growth, red_cover;
    double initial, maximum, growth, decline;
    if (scanf("%d%d%d%d%d%d%lf%lf%lf%lf%d%d", &mode, &delayed, &count,
        &emergence, &senescence, &maturity, &initial, &maximum, &growth,
        &decline, &red_growth, &red_cover) != 12 || mode < 0 || mode > 1 ||
        count < 1 || count > 10000 || red_growth < 0 || red_growth > 100 ||
        red_cover < 0 || red_cover > 100) return 2;
    int32_t delay = delayed, lo = emergence, ls = senescence, lm = maturity;
    int8_t kind = mode, rg = red_growth, rc = red_cover;
    G(setsimulation_delayeddays)(&delay);
    for (int i = 0; i < count; ++i) {
        double time;
        if (scanf("%lf", &time) != 1) return 2;
        int32_t dap = (int32_t)time;
        double cover = G(canopycovernostresssf)(&dap, &lo, &ls, &lm,
            &lo, &ls, &lm, &initial, &maximum, &growth, &decline,
            &growth, &decline, &time, &kind, &rg, &rc);
        printf("%.17g %.17g\n", time, cover);
    }
    return 0;
}
