/*
 * stubs.c - Stub functions for bare-metal builds
 *
 * These stubs provide missing symbols from newlib/libc_nano
 * that are not needed for our bare-metal environment.
 */

#include <stddef.h>
#include <reent.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Wide string printf stub - missing from libc_nano
 * This function is called by _svfwprintf_r for wide string output.
 * In bare-metal environment, we don't use wide strings, so return error.
 */
int __ssputws_r(struct _reent *ptr, void *uio) {
    (void)ptr;
    (void)uio;
    return -1;  /* Return error - wide strings not supported */
}

#ifdef __cplusplus
}
#endif
