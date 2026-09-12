#ifndef QWEN_SSD_H
#define QWEN_SSD_H
#include <stdint.h>
typedef struct qs_store qs_store;
typedef struct { int fd; uint64_t offset, expert_bytes; } qs_source;
typedef struct {
    uint64_t ram_bytes, transient_peak_bytes, read_bytes, expert_reads;
    double io_seconds, wait_seconds;
} qs_stats;
/* Anonymous, sparse address space preserves the existing quantized CPU layout.
 * Only pinned experts and the bounded current batch have physical RAM pages.
 * Disk access is explicit pread, never a fault through a file-backed mapping. */
qs_store *qs_create(const qs_source *sources,int layers,const int *ranking,
                    int ram_experts,uint64_t buffer_bytes,int io_threads,char *error);
void qs_destroy(qs_store *s);
const void *qs_tensor(qs_store *s,int layer,int matrix);
int qs_is_hot(qs_store *s,int layer,int expert);
/* Begin a token prefix that fits the cold buffer; return its token count.
 * Wait before CPU access, release after it. No simultaneous inference callers. */
int qs_begin(qs_store *s,int layer,const int *ids,int tokens);
int qs_wait(qs_store *s);
int qs_release(qs_store *s);
const char *qs_error(qs_store *s);
void qs_get_stats(qs_store *s,qs_stats *out);
#endif
