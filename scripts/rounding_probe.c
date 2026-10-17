/* Native ac_utils::roundc for int8 and int32. stdin: count then doubles. */
#include <stdint.h>
#include <stdio.h>
extern int32_t __ac_utils_MOD_roundc_int32(const double *, const int32_t *);
extern int8_t __ac_utils_MOD_roundc_int8(const double *, const int8_t *);
int main(void) {
    int count;
    if (scanf("%d", &count) != 1 || count < 1 || count > 10000) return 2;
    const int32_t mold32 = 1;
    const int8_t mold8 = 1;
    for (int i = 0; i < count; ++i) {
        double value;
        if (scanf("%lf", &value) != 1) return 2;
        printf("%d %d\n", (int)__ac_utils_MOD_roundc_int32(&value, &mold32),
            (int)__ac_utils_MOD_roundc_int8(&value, &mold8));
    }
    return 0;
}
