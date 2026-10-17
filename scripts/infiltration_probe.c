/* Native daily rainfall entry without bunds: extra runoff then infiltration.
 * stdin: n SAT FC tau Ksat gravel water; then n rows: dz theta FCadj fluxout.
 */
#include <stdint.h>
#include <stdio.h>
#define G(name) __ac_global_MOD_##name
#define LAYER(name) extern void G(name)(const int32_t *, const double *)
#define VALUE(name) extern void G(name)(const double *)
LAYER(setsoillayer_sat); LAYER(setsoillayer_fc); LAYER(setsoillayer_tau);
LAYER(setsoillayer_infrate); LAYER(setsoillayer_gravelvol);
LAYER(setcompartment_thickness); LAYER(setcompartment_theta);
LAYER(setcompartment_fcadj); LAYER(setcompartment_fluxout);
VALUE(setrain); VALUE(setrunoff); VALUE(setdrain); VALUE(setsurfacestorage);
VALUE(setirrigation); VALUE(setmanagement_bundheight);
extern void G(setnrcompartments)(const int32_t *);
extern void G(setcompartment_layer)(const int32_t *, const int32_t *);
extern void G(setrainrecord_datatype)(const int8_t *);
extern double G(getcompartment_theta)(const int32_t *);
extern double G(getcompartment_fluxout)(const int32_t *);
extern double G(getdrain)(void);
extern double G(getrunoff)(void);
extern void __ac_simul_MOD_calculate_extra_runoff(double *, double *, double *, double *);
extern void __ac_simul_MOD_calculate_infiltration(double *, double *, double *, double *);

int main(void) {
    int n;
    double sat, fc, tau, ksat, gravel, water, dz[12], begin = 0., end = 0., initial_drain = 0.;
    if (scanf("%d%lf%lf%lf%lf%lf%lf", &n, &sat, &fc, &tau, &ksat, &gravel, &water) != 7 ||
        n < 1 || n > 12 || gravel < 0 || gravel >= 100 || water < 0) return 2;
    int32_t count = n, layer = 1;
    int8_t daily = 0;
    double zero = 0., rain = 0., irrigation = 0., storage = 0., subdrain = 0.;
    G(setnrcompartments)(&count);
    G(setsoillayer_sat)(&layer, &sat); G(setsoillayer_fc)(&layer, &fc);
    G(setsoillayer_tau)(&layer, &tau); G(setsoillayer_infrate)(&layer, &ksat);
    G(setsoillayer_gravelvol)(&layer, &gravel);
    G(setrainrecord_datatype)(&daily); G(setmanagement_bundheight)(&zero);
    G(setrain)(&water); G(setirrigation)(&zero); G(setrunoff)(&zero); G(setsurfacestorage)(&zero);
    for (int32_t i = 1; i <= n; ++i) {
        double theta, adjusted_fc, flux;
        if (scanf("%lf%lf%lf%lf", &dz[i-1], &theta, &adjusted_fc, &flux) != 4 || dz[i-1] <= 0) return 2;
        G(setcompartment_layer)(&i, &layer);
        G(setcompartment_thickness)(&i, &dz[i-1]); G(setcompartment_theta)(&i, &theta);
        G(setcompartment_fcadj)(&i, &adjusted_fc); G(setcompartment_fluxout)(&i, &flux);
        initial_drain = flux;
        begin += theta * 1000. * dz[i-1] * (1. - gravel / 100.);
    }
    G(setdrain)(&initial_drain);
    __ac_simul_MOD_calculate_extra_runoff(&rain, &irrigation, &storage, &subdrain);
    __ac_simul_MOD_calculate_infiltration(&rain, &irrigation, &storage, &subdrain);
    for (int32_t i = 1; i <= n; ++i) {
        double theta = G(getcompartment_theta)(&i);
        end += theta * 1000. * dz[i-1] * (1. - gravel / 100.);
        printf("%.17g %.17g\n", theta, G(getcompartment_fluxout)(&i));
    }
    double extra = G(getdrain)() - initial_drain, rejected = G(getrunoff)();
    printf("%.17g %.17g %.17g %.17g\n", extra, rejected, rain + irrigation + storage,
        begin + water - end - extra - rejected);
    return 0;
}
