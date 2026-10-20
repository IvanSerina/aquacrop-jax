/* Call the original DetermineCCiGDD through the gfortran module ABI.
 * The reference objects and source are never changed. See probe_input() in
 * generate_senescence_oracle.py for the input columns. LOGICAL is int32,
 * crop enums and percent reductions are int8 in the reference build.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define G(name) __ac_global_MOD_##name
#define DS(name) extern void G(set##name)(const double *)
#define IS(name) extern void G(set##name)(const int32_t *)
#define BS(name) extern void G(set##name)(const int8_t *)
#define DG(name) extern double G(get##name)(void)
#define IG(name) extern int32_t G(get##name)(void)
#define D(name, value) G(set##name)(&(double){value})
#define I(name, value) G(set##name)(&(int32_t){value})
#define B(name, value) G(set##name)(&(int8_t){value})

BS(crop_modecycle); BS(crop_subkind); BS(crop_pmethod);
BS(simulparam_beta); BS(simulation_effectstress_redcgc);
BS(simulation_effectstress_redccx);
IS(crop_determinancylinked); IS(crop_daystoccini);
IS(crop_daystogermination); IS(crop_daystofullcanopysf);
IS(crop_daystosenescence); IS(crop_daystoharvest);
IS(crop_gddaystogermination); IS(crop_gddaystofullcanopysf);
IS(crop_gddaystosenescence); IS(crop_gddaystoharvest);
IS(crop_gddaystoflowering); IS(crop_gddlengthflowering);
IS(crop_sumetodelaysenescence); IS(simulation_delayeddays);
IS(simulation_protectedseedling); IS(simulation_swctopsoilconsidered);
IS(simulation_evaplimiton);
DS(crop_pleafdeful); DS(crop_pleafdefll); DS(crop_psenescence);
DS(crop_ksshapefactorleaf); DS(crop_ksshapefactorsenescence);
DS(crop_cco); DS(crop_ccoadjusted); DS(crop_ccx);
DS(crop_ccxadjusted); DS(crop_ccxwithered); DS(crop_cgc); DS(crop_gddcgc);
DS(crop_cdc); DS(crop_gddcdc); DS(simulparam_padjfao);
DS(simulation_effectstress_cdecline); DS(simulation_sumetostress);
DS(rootzonewc_actual); DS(rootzonewc_fc); DS(rootzonewc_wp);
DS(rootzonewc_ztopact); DS(rootzonewc_ztopfc); DS(rootzonewc_ztopwp);
DS(cciprev); DS(cciactual); DS(ccitopearlysen); DS(eto);
DG(cciactual); DG(ccitopearlysen); DG(crop_ccoadjusted);
DG(crop_ccxadjusted); DG(crop_ccxwithered); DG(simulation_sumetostress);
DG(crop_psenact); IG(simulation_protectedseedling); IG(simulation_evaplimiton);

extern void __ac_simul_MOD_determineccigdd(
    const double *, const double *, double *, const double *,
    const int32_t *, const int32_t *, const double *, const int32_t *,
    double *, double *, int32_t *, const double *, const double *,
    const double *, const double *, const int32_t *);

int main(int argc, char **argv) {
    int count, rg, rc, fao, beta, delay, protected, evap, dead;
    double cco, ccx, growth, decline, psen, shape, factor;
    double previous, initial_adjusted, maximum_adjusted, time, top, withered, sumeto;
    double fertility_decline = 0.;
    int virtual_start = 10;
    int final_growth = 1920, full_sf = 600, senescence_gdd = 1920, maturity_gdd = 2200;
    if (scanf("%d%lf%lf%lf%lf%d%d%lf%lf%d%lf%d%d", &count, &cco, &ccx,
        &growth, &decline, &rg, &rc, &psen, &shape, &fao, &factor, &beta, &delay) != 13
        || count < 1 || count > 10000 || rg < 0 || rg > 100 || rc < 0 || rc > 100
        || fao < 0 || fao > 1 || beta < 0 || beta > 100 || delay < 0) return 2;
    if (scanf("%lf%lf%lf%d%lf%lf%lf%lf%d%d", &previous, &initial_adjusted,
        &maximum_adjusted, &protected, &time, &top, &withered, &sumeto, &evap, &dead) != 10)
        return 2;
    if (argc > 1 && strcmp(argv[1], "--fertility-decline") == 0) {
        if (scanf("%lf%d", &fertility_decline, &virtual_start) != 2) return 2;
    }
    if (argc > 1 && strcmp(argv[1], "--nonexpanding") == 0) {
        if (scanf("%lf%d%d%d%d%d", &fertility_decline, &virtual_start, &final_growth,
            &full_sf, &senescence_gdd, &maturity_gdd) != 6) return 2;
    }

    B(crop_modecycle, 0); B(crop_subkind, 0); I(crop_determinancylinked, 0);
    if (final_growth < senescence_gdd) {
        B(crop_subkind, 1); I(crop_determinancylinked, 1);
        I(crop_gddaystoflowering, final_growth); I(crop_gddlengthflowering, 0);
    }
    B(crop_pmethod, fao); D(crop_pleafdeful, .15); D(crop_pleafdefll, .55);
    D(crop_psenescence, psen); D(crop_ksshapefactorleaf, 3.);
    D(crop_ksshapefactorsenescence, shape);
    D(crop_cco, cco); D(crop_ccoadjusted, initial_adjusted);
    D(crop_ccx, ccx); D(crop_ccxadjusted, maximum_adjusted);
    D(crop_ccxwithered, withered); D(crop_gddcgc, growth); D(crop_gddcdc, decline);
    D(crop_cgc, .12); D(crop_cdc, .06);
    I(crop_daystoccini, 0); I(crop_daystogermination, 1);
    I(crop_daystofullcanopysf, 60); I(crop_daystosenescence, 180);
    I(crop_daystoharvest, 210); I(crop_gddaystogermination, 5);
    I(crop_gddaystofullcanopysf, full_sf); I(crop_gddaystosenescence, senescence_gdd);
    I(crop_gddaystoharvest, maturity_gdd); I(crop_sumetodelaysenescence, delay);
    D(simulparam_padjfao, factor); B(simulparam_beta, beta);
    B(simulation_effectstress_redcgc, rg); B(simulation_effectstress_redccx, rc);
    D(simulation_effectstress_cdecline, fertility_decline); I(simulation_delayeddays, 0);
    I(simulation_protectedseedling, protected); I(simulation_evaplimiton, evap);
    D(simulation_sumetostress, sumeto); D(cciprev, previous);
    D(cciactual, previous); D(ccitopearlysen, top);
    double leaf = -9., senescence = -9.;
    int32_t no_more = dead;
    for (int i = 0; i < count; ++i) {
        double total, daily, eto, actual, fc, wp, top_actual, top_fc, top_wp;
        int topsoil;
        if (scanf("%lf%lf%lf%lf%lf%lf%lf%lf%lf%d", &total, &daily, &eto,
            &actual, &fc, &wp, &top_actual, &top_fc, &top_wp, &topsoil) != 10) return 2;
        D(eto, eto); D(rootzonewc_actual, actual); D(rootzonewc_fc, fc);
        D(rootzonewc_wp, wp); D(rootzonewc_ztopact, top_actual);
        D(rootzonewc_ztopfc, top_fc); D(rootzonewc_ztopwp, top_wp);
        I(simulation_swctopsoilconsidered, topsoil);
        __ac_simul_MOD_determineccigdd(&ccx, &cco, &leaf, &(double){0.},
            &(int32_t){0}, &(int32_t){0}, &total, &(int32_t){i + virtual_start},
            &senescence, &time, &no_more, &(double){.06}, &(double){1.},
            &daily, &decline, &(int32_t){0});
        printf("%.17g %.17g %.17g %d %.17g %.17g %.17g %.17g %d %d %.17g %.17g %.17g\n",
            G(getcciactual)(), G(getcrop_ccoadjusted)(), G(getcrop_ccxadjusted)(),
            G(getsimulation_protectedseedling)(), time, G(getccitopearlysen)(),
            G(getcrop_ccxwithered)(), G(getsimulation_sumetostress)(),
            G(getsimulation_evaplimiton)(), no_more, leaf, senescence, G(getcrop_psenact)());
        D(cciprev, G(getcciactual)());
    }
    return 0;
}
