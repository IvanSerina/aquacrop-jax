/* Native CropStressParametersSoilFertility; objects/source remain untouched. */
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

typedef struct {
    int8_t stress;
    double shape_cgc;
    double shape_ccx;
    double shape_wp;
    double shape_decline;
    int32_t calibrated;
} Shapes;

typedef struct {
    int8_t red_cgc;
    int8_t red_ccx;
    int8_t red_wp;
    double decline;
    int8_t red_ks_sto;
} EffectStress;

#define G(name) __ac_global_MOD_##name
extern void G(cropstressparameterssoilfertility)(const Shapes *,
    const int32_t *, EffectStress *);

int main(void) {
    int count;
    if (sizeof(Shapes) != 48 || offsetof(Shapes, shape_cgc) != 8 ||
        offsetof(Shapes, calibrated) != 40 || sizeof(EffectStress) != 24 ||
        offsetof(EffectStress, decline) != 8 || offsetof(EffectStress, red_ks_sto) != 16)
        return 3;
    if (scanf("%d", &count) != 1 || count < 1 || count > 10000) return 2;
    for (int i = 0; i < count; ++i) {
        int stress;
        Shapes shapes = {0};
        EffectStress effect = {0};
        if (scanf("%d%lf%lf%lf%lf", &stress, &shapes.shape_cgc,
            &shapes.shape_ccx, &shapes.shape_wp, &shapes.shape_decline) != 5) return 2;
        shapes.stress = (int8_t)stress;
        int32_t level = stress;
        G(cropstressparameterssoilfertility)(&shapes, &level, &effect);
        printf("%d %d %d %.17g %d\n", (int)effect.red_cgc,
            (int)effect.red_ccx, (int)effect.red_wp,
            effect.decline, (int)effect.red_ks_sto);
    }
    return 0;
}
