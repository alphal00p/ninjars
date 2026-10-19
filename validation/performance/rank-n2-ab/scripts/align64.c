// LD_PRELOAD shim (diagnostic only): every heap block of >= 64 bytes is
// 64-byte aligned. Build: gcc -O2 -shared -fPIC -o libalign64.so align64.c
// Rust's System allocator calls malloc/calloc/realloc for alignments <= 16,
// so this fixes the cache-line offset of every engine and replay buffer.
#include <stddef.h>
#include <string.h>
#include <stdint.h>
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);
extern void __libc_free(void *);
void *malloc(size_t n) { return n >= 64 ? __libc_memalign(64, n) : __libc_malloc(n); }
void free(void *p) { __libc_free(p); }
void *calloc(size_t a, size_t b) {
  size_t n = a * b;
  if (n < 64) return __libc_calloc(a, b);
  void *p = __libc_memalign(64, n);
  if (p) memset(p, 0, n);
  return p;
}
void *realloc(void *p, size_t n) {
  if (!p) return malloc(n);
  if (n < 64) return __libc_realloc(p, n);
  void *q = __libc_realloc(p, n);
  if (!q || ((uintptr_t)q & 63) == 0) return q;
  void *r = __libc_memalign(64, n);
  if (!r) return r;
  memcpy(r, q, n);
  __libc_free(q);
  return r;
}
