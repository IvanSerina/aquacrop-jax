/* Native TimeToMaxCanopySF from the immutable AquaCrop 7.3 object files. */
#include <stdint.h>
#include <stdio.h>

extern void __ac_global_MOD_timetomaxcanopysf(
    const double *, const double *, const double *,
    const int32_t *, const int32_t *, const int32_t *, const int32_t *,
    const int32_t *, const int32_t *, int32_t *, int8_t *, int8_t *, int32_t *);

int main(void) {
    int count;
    if (scanf("%d", &count) != 1 || count < 1 || count > 1000) return 2;
    for (int index = 0; index < count; ++index) {
        double initial, growth, maximum;
        int32_t emergence, full, senescence, flowering, length, determinant;
        int32_t stress_class, adjusted;
        int red_growth, red_cover;
        int8_t reduction_growth, reduction_cover;
        if (scanf("%lf%lf%lf%d%d%d%d%d%d%d%d%d", &initial, &growth, &maximum,
                  &emergence, &full, &senescence, &flowering, &length,
                  &determinant, &red_growth, &red_cover, &stress_class) != 12)
            return 2;
        reduction_growth = (int8_t)red_growth;
        reduction_cover = (int8_t)red_cover;
        adjusted = -999;
        __ac_global_MOD_timetomaxcanopysf(
            &initial, &growth, &maximum, &emergence, &full, &senescence,
            &flowering, &length, &determinant, &adjusted,
            &reduction_growth, &reduction_cover, &stress_class);
        printf("%d %d %d %d\n", adjusted, (int)reduction_growth,
               (int)reduction_cover, stress_class);
    }
    return 0;
}
