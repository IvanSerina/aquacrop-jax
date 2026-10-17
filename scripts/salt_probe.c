/* Read-only link against the reference's existing Fortran object files.
 * A scalar C driver exposes calculate_saltcontent at full double precision.
 * stdin: n ksat sat fc gravel solub rain irrigation ec; then each
 * compartment: thickness theta fluxout; then 11 dissolved/deposited pairs.
 */
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#define G(name) __ac_global_MOD_##name
#define REAL_LAYER(name) extern void G(name)(const int*, const double*)
#define REAL_VALUE(name) extern void G(name)(const double*)
REAL_LAYER(setsoillayer_sat); REAL_LAYER(setsoillayer_fc);
REAL_LAYER(setsoillayer_gravelvol); REAL_LAYER(setsoillayer_ul);
REAL_LAYER(setsoillayer_dx); REAL_LAYER(setcompartment_thickness);
REAL_LAYER(setcompartment_theta); REAL_LAYER(setcompartment_fluxout);
REAL_VALUE(setsimulation_irriecw); REAL_VALUE(setdrain);
REAL_VALUE(setecstorage);
extern void G(setnrcompartments)(const int*);
extern void G(setcrop_day1)(const int*);
extern void G(setcrop_dayn)(const int*);
extern void G(setcompartment_layer)(const int*, const int*);
extern void G(setsoillayer_sc)(const int*, const int8_t*);
extern void G(setsoillayer_scp1)(const int*, const int8_t*);
extern void G(setsimulparam_saltsolub)(const int8_t*);
extern void G(calculate_saltmobility)(const int*, const int8_t*, const int8_t*, double*);
extern void G(setsoillayer_saltmobility)(const int*, const double*);
extern void G(setcompartment_salt)(const int*, const int*, const double*);
extern void G(setcompartment_depo)(const int*, const int*, const double*);
extern double G(getcompartment_salt)(const int*, const int*);
extern double G(getcompartment_depo)(const int*, const int*);
extern double G(getecdrain)(void);
extern void __ac_simul_MOD_calculate_saltcontent(const double*, const double*, const double*, const double*, const int*);

extern void __ac_simul_MOD_concentratesalts(void);
extern void G(setsimulation_salinityconsidered)(const int*);
extern double G(kssalinity)(const int*,const int8_t*,const int8_t*,const double*,const double*);
extern double G(adjustedksstotoecsw)(const int8_t*,const int8_t*,const int*,const double*,const double*,const double*,const double*,const double*,const double*,const double*,const double*);

struct effect_stress { int8_t cgc, ccx, wp; double decline; int8_t sto; };
extern void __ac_tempprocessing_MOD_cropstressparameterssoilsalinity(
    const int8_t*,const int8_t*,const double*,const double*,const double*,const double*,
    const int*,const int*,const int*,const int*,const int*,const int*,const int*,const int*,const int*,const int*,struct effect_stress*);

int main(int argc, char **argv) {
    if(argc>1 && strcmp(argv[1],"canopy")==0) {
        int red,distortion,full,flower,length,harvest,gfull,gflower,glength,gharvest,determinate,gdd;
        double initial,maximum,cgc,gddcgc;
        if(scanf("%d%d%lf%lf%lf%lf%d%d%d%d%d%d%d%d%d%d",&red,&distortion,&initial,&maximum,&cgc,&gddcgc,&full,&flower,&length,&harvest,&gfull,&gflower,&glength,&gharvest,&determinate,&gdd)!=16) return 2;
        int8_t reduction=red, distortion8=distortion;
        int mode=gdd ? 0 : 1;
        struct effect_stress result={0};
        __ac_tempprocessing_MOD_cropstressparameterssoilsalinity(&reduction,&distortion8,&initial,&maximum,&cgc,&gddcgc,&determinate,&full,&flower,&length,&harvest,&gfull,&gflower,&glength,&gharvest,&mode,&result);
        printf("%d %d %d %.17g %d\n",result.cgc,result.ccx,result.wp,result.decline,result.sto);
        return 0;
    }
    if(argc>1 && strcmp(argv[1],"stress")==0) {
        double ece,ecsw,ecfc,wrel,shape,b0,b1,b2,ks;
        int lo,hi,response,considered;
        if(scanf("%lf%lf%lf%lf%d%d%d%lf%lf%lf%lf%lf%d", &ece,&ecsw,&ecfc,&wrel,&lo,&hi,&response,&shape,&b0,&b1,&b2,&ks,&considered)!=13) return 2;
        int8_t lower=lo, upper=hi;
        G(setsimulation_salinityconsidered)(&considered);
        printf("%.17g %.17g\n",G(kssalinity)(&considered,&lower,&upper,&ece,&shape),
            G(adjustedksstotoecsw)(&lower,&upper,&response,&ece,&ecsw,&ecfc,&wrel,&b0,&b1,&b2,&ks));
        return 0;
    }
    int n, layer=1, day=1, last=1000;
    double ksat,sat,fc,gravel,solub,rain,irrigation,ec;
    if (scanf("%d%lf%lf%lf%lf%lf%lf%lf%lf", &n,&ksat,&sat,&fc,&gravel,&solub,&rain,&irrigation,&ec)!=9 || n<1 || n>12) return 2;
    int8_t count = ksat<=112 ? 11 : (int8_t)nearbyint(1.6+1000/ksat);
    if(count<2) count=2;
    int8_t sc=count-1, diffusion=20, macro=(int8_t)nearbyint(fc), sol=(int8_t)solub;
    double upper=sat/100*sc/(sc+2.), width=upper/sc, mobility[11]={0}, zero=0;
    G(setnrcompartments)(&n); G(setcrop_day1)(&day); G(setcrop_dayn)(&last);
    G(setsoillayer_sat)(&layer,&sat); G(setsoillayer_fc)(&layer,&fc);
    G(setsoillayer_gravelvol)(&layer,&gravel); G(setsoillayer_ul)(&layer,&upper);
    G(setsoillayer_dx)(&layer,&width); G(setsoillayer_sc)(&layer,&sc);
    G(setsoillayer_scp1)(&layer,&count); G(setsimulparam_saltsolub)(&sol);
    G(calculate_saltmobility)(&layer,&diffusion,&macro,mobility);
    G(setsoillayer_saltmobility)(&layer,mobility);
    G(setsimulation_irriecw)(&ec); G(setecstorage)(&zero);
    for(int i=1;i<=n;i++) {
        double dz,theta,flux;
        if(scanf("%lf%lf%lf",&dz,&theta,&flux)!=3) return 2;
        G(setcompartment_layer)(&i,&layer); G(setcompartment_thickness)(&i,&dz);
        G(setcompartment_theta)(&i,&theta); G(setcompartment_fluxout)(&i,&flux);
        if(i==n) G(setdrain)(&flux);
        for(int c=1;c<=11;c++) {
            double salt,depo;
            if(scanf("%lf%lf",&salt,&depo)!=2) return 2;
            G(setcompartment_salt)(&i,&c,&salt); G(setcompartment_depo)(&i,&c,&depo);
        }
    }
    if(argc>1 && strcmp(argv[1],"concentrate")==0)
        __ac_simul_MOD_concentratesalts();
    else
        __ac_simul_MOD_calculate_saltcontent(&rain,&irrigation,&zero,&zero,&day);
    for(int i=1;i<=n;i++) for(int c=1;c<=11;c++)
        printf("%.17g %.17g\n",G(getcompartment_salt)(&i,&c),G(getcompartment_depo)(&i,&c));
    printf("%.17g\n",G(getecdrain)());
    return 0;
}
