#ifndef QWEN_CPU_H
#define QWEN_CPU_H
#include <stddef.h>
typedef struct qc_workspace qc_workspace;
const char *qc_backend_name(void);
int qc_backend_available(const char *name);
qc_workspace *qc_create(int capacity,int threads);
void qc_destroy(qc_workspace *w);
/* Reconfigure only between complete MoE calls; buffers and arithmetic stay unchanged. */
int qc_set_threads(qc_workspace *w,int threads);
int qc_threads(const qc_workspace *w);
size_t qc_row_bytes(int type,int cols);
float qc_weight_at(const void *row,int type,int index);
int qc_matmul(float *out,const void *weights,int type,int rows,int cols,const float *x,int nt,int threads);
int qc_moe(qc_workspace *w,float *out,const float *x,const int *ids,const float *weights,int nt,
           const void *gate,int gate_type,const void *up,int up_type,const void *down,int down_type);
#endif
