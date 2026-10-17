/* Existing Fortran objects only. stdin: n SAT FC tau Ksat gravel;
 * n rows of thickness (m), theta (fraction), adjusted FC (percent).
 * stdout: n theta/flux rows, then bottom drainage and water residual (mm).
 */
#include <stdint.h>
#include <stdio.h>
#define G(name) __ac_global_MOD_##name
#define LAYER(name) extern void G(name)(const int32_t *, const double *)
LAYER(setsoillayer_sat); LAYER(setsoillayer_fc); LAYER(setsoillayer_tau);
LAYER(setsoillayer_infrate); LAYER(setsoillayer_gravelvol);
LAYER(setcompartment_thickness); LAYER(setcompartment_theta); LAYER(setcompartment_fcadj);
extern void G(setnrcompartments)(const int32_t *);
extern void G(setcompartment_layer)(const int32_t *, const int32_t *);
extern double G(getcompartment_theta)(const int32_t *);
extern double G(getcompartment_fluxout)(const int32_t *);
extern double G(getdrain)(void);
extern void __ac_simul_MOD_calculate_drainage(void);

int main(void) {
    int n;
    double sat, fc, tau, ksat, gravel, dz[12], begin = 0., end = 0.;
    if (scanf("%d%lf%lf%lf%lf%lf", &n, &sat, &fc, &tau, &ksat, &gravel) != 6 ||
        n < 1 || n > 12 || gravel < 0 || gravel >= 100) return 2;
    int32_t count = n, layer = 1;
    G(setnrcompartments)(&count);
    G(setsoillayer_sat)(&layer, &sat); G(setsoillayer_fc)(&layer, &fc);
    G(setsoillayer_tau)(&layer, &tau); G(setsoillayer_infrate)(&layer, &ksat);
    G(setsoillayer_gravelvol)(&layer, &gravel);
    for (int32_t i = 1; i <= n; ++i) {
        double theta, adjusted_fc;
        if (scanf("%lf%lf%lf", &dz[i-1], &theta, &adjusted_fc) != 3 || dz[i-1] <= 0) return 2;
        G(setcompartment_layer)(&i, &layer);
        G(setcompartment_thickness)(&i, &dz[i-1]);
        G(setcompartment_theta)(&i, &theta);
        G(setcompartment_fcadj)(&i, &adjusted_fc);
        begin += theta * 1000. * dz[i-1] * (1. - gravel / 100.);
    }
    __ac_simul_MOD_calculate_drainage();
    for (int32_t i = 1; i <= n; ++i) {
        double theta = G(getcompartment_theta)(&i);
        end += theta * 1000. * dz[i-1] * (1. - gravel / 100.);
        printf("%.17g %.17g\n", theta, G(getcompartment_fluxout)(&i));
    }
    double bottom = G(getdrain)();
    printf("%.17g %.17g\n", bottom, begin - end - bottom);
    return 0;
}
