/* Baseline x86-64 only: unsupported instructions are never entered. */
#include "cpu.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#define DECLARE(p) \
extern qc_workspace *p##create(int,int); \
extern void p##destroy(qc_workspace*); \
extern int p##set_threads(qc_workspace*,int); \
extern int p##threads(const qc_workspace*); \
extern size_t p##row_bytes(int,int); \
extern float p##weight_at(const void*,int,int); \
extern int p##matmul(float*,const void*,int,int,int,const float*,int,int); \
extern int p##moe(qc_workspace*,float*,const float*,const int*,const float*,int,const void*,int,const void*,int,const void*,int)
DECLARE(qcs_); DECLARE(qc2_); DECLARE(qc5_);
typedef struct {
    const char *name;
    qc_workspace *(*create)(int,int);void (*destroy)(qc_workspace*);
    int (*set_threads)(qc_workspace*,int);int (*threads)(const qc_workspace*);
    size_t (*row_bytes)(int,int);float (*weight_at)(const void*,int,int);
    int (*matmul)(float*,const void*,int,int,int,const float*,int,int);
    int (*moe)(qc_workspace*,float*,const float*,const int*,const float*,int,const void*,int,const void*,int,const void*,int);
} backend;
#define ENTRY(n,p) {n,p##create,p##destroy,p##set_threads,p##threads,p##row_bytes,p##weight_at,p##matmul,p##moe}
static const backend variants[]={ENTRY("scalar",qcs_),ENTRY("avx2",qc2_),ENTRY("avx512",qc5_)};
static const backend *selected;
static pthread_once_t once=PTHREAD_ONCE_INIT;
int qc_backend_available(const char *name){
    if(!name)return 0;
    if(!strcmp(name,"scalar"))return 1;
    __builtin_cpu_init();
    int base=__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma")&&__builtin_cpu_supports("f16c");
    if(!strcmp(name,"avx2"))return base;
    if(!strcmp(name,"avx512"))return base&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl");
    return 0;
}
static void choose(void){
    const char *forced=getenv("QWEN_CPU_BACKEND");
    for(int i=2;i>=0;i--)if((!forced||!strcmp(forced,"auto")||!strcmp(forced,variants[i].name))&&qc_backend_available(variants[i].name)){selected=&variants[i];return;}
}
static const backend *get(void){pthread_once(&once,choose);return selected;}
const char *qc_backend_name(void){const backend*b=get();return b?b->name:"unsupported";}
qc_workspace *qc_create(int c,int t){const backend*b=get();return b?b->create(c,t):NULL;}
void qc_destroy(qc_workspace*w){const backend*b=get();if(b)b->destroy(w);}
int qc_set_threads(qc_workspace*w,int t){const backend*b=get();return b?b->set_threads(w,t):-1;}
int qc_threads(const qc_workspace*w){const backend*b=get();return b?b->threads(w):0;}
size_t qc_row_bytes(int t,int c){return qcs_row_bytes(t,c);}
float qc_weight_at(const void*w,int t,int i){const backend*b=get();return b?b->weight_at(w,t,i):NAN;}
int qc_matmul(float*o,const void*w,int t,int r,int c,const float*x,int n,int th){const backend*b=get();return b?b->matmul(o,w,t,r,c,x,n,th):-1;}
int qc_moe(qc_workspace*w,float*o,const float*x,const int*i,const float*p,int n,const void*g,int gt,const void*u,int ut,const void*d,int dt){const backend*b=get();return b?b->moe(w,o,x,i,p,n,g,gt,u,ut,d,dt):-1;}
