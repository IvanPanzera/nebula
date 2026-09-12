#define _GNU_SOURCE
#include "ssd.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { EXPERTS=512, MATRICES=3, MAX_LAYERS=48, MAX_THREADS=4 };
typedef struct {qs_source src;unsigned char *dst;int error;} read_job;
struct qs_store {
    int layers,threads,started,stop,next,jobs,pending,active_layer,cold_count;
    uint64_t budget;
    qs_source source[MAX_LAYERS*MATRICES];
    unsigned char *data[MAX_LAYERS*MATRICES],hot[MAX_LAYERS][EXPERTS];
    int cold[EXPERTS];
    read_job job[EXPERTS*MATRICES];
    pthread_t worker[MAX_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work,done;
    double begin,finished;
    qs_stats stats;
    char error[256];
};
static double seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void *reader(void *arg){
    qs_store *s=arg;pthread_mutex_lock(&s->mutex);
    for(;;){
        while(!s->stop&&s->next==s->jobs)pthread_cond_wait(&s->work,&s->mutex);
        if(s->stop)break;
        read_job *j=&s->job[s->next++];pthread_mutex_unlock(&s->mutex);
        uint64_t n=0;
        while(n<j->src.expert_bytes){
            size_t want=j->src.expert_bytes-n;if(want>64ULL*1024*1024)want=64ULL*1024*1024;
            ssize_t got=pread(j->src.fd,j->dst+n,want,(off_t)(j->src.offset+n));
            if(got<0&&errno==EINTR)continue;
            if(got<=0){j->error=got<0?errno:EIO;break;}n+=(uint64_t)got;
        }
        /* The kernel may still cache source pages. This advice does not affect
         * correctness and avoids deliberately accumulating a second model copy. */
        (void)posix_fadvise(j->src.fd,(off_t)j->src.offset,(off_t)j->src.expert_bytes,POSIX_FADV_DONTNEED);
        pthread_mutex_lock(&s->mutex);
        if(!--s->pending){s->finished=seconds();pthread_cond_signal(&s->done);}
    }
    pthread_mutex_unlock(&s->mutex);return NULL;
}
static void add_expert(qs_store *s,int layer,int expert,int *n){
    for(int k=0;k<3;k++){
        int i=layer*3+k;read_job *j=&s->job[(*n)++];
        j->src=s->source[i];j->src.offset+=expert*j->src.expert_bytes;
        j->dst=s->data[i]+expert*j->src.expert_bytes;j->error=0;
    }
}
static void submit(qs_store *s,int jobs){
    pthread_mutex_lock(&s->mutex);s->next=0;s->jobs=s->pending=jobs;
    s->finished=s->begin=seconds();pthread_cond_broadcast(&s->work);pthread_mutex_unlock(&s->mutex);
}
int qs_wait(qs_store *s){
    if(!s)return -1;
    double start=seconds();pthread_mutex_lock(&s->mutex);
    while(s->pending)pthread_cond_wait(&s->done,&s->mutex);
    s->stats.wait_seconds+=seconds()-start;
    s->stats.io_seconds+=s->finished-s->begin;
    for(int i=0;i<s->jobs;i++){
        read_job *j=&s->job[i];
        if(j->error){snprintf(s->error,sizeof(s->error),"SSD read at offset %llu: %s",(unsigned long long)j->src.offset,strerror(j->error));pthread_mutex_unlock(&s->mutex);return -1;}
        s->stats.read_bytes+=j->src.expert_bytes;
    }
    s->stats.expert_reads+=s->jobs/3;s->jobs=s->next=0;
    pthread_mutex_unlock(&s->mutex);
    return 0;
}
qs_store *qs_create(const qs_source *src,int layers,const int *ranking,int hot,uint64_t budget,int threads,char *error){
    qs_store *s=NULL;
    if(!src||!ranking||layers<1||layers>MAX_LAYERS||hot<0||hot>512||threads<1||threads>MAX_THREADS){snprintf(error,256,"invalid SSD store configuration");return NULL;}
    s=calloc(1,sizeof(*s));if(!s){snprintf(error,256,"allocate SSD metadata");return NULL;}
    s->layers=layers;s->threads=threads;s->budget=budget;s->active_layer=-1;
    int rc=pthread_mutex_init(&s->mutex,NULL);if(rc){free(s);snprintf(error,256,"SSD mutex: %s",strerror(rc));return NULL;}
    rc=pthread_cond_init(&s->work,NULL);if(rc){pthread_mutex_destroy(&s->mutex);free(s);snprintf(error,256,"SSD condition: %s",strerror(rc));return NULL;}
    rc=pthread_cond_init(&s->done,NULL);if(rc){pthread_cond_destroy(&s->work);pthread_mutex_destroy(&s->mutex);free(s);snprintf(error,256,"SSD condition: %s",strerror(rc));return NULL;}
    long page=sysconf(_SC_PAGESIZE);
    for(int l=0;l<layers;l++){
        uint64_t bytes=0;unsigned char seen[512]={0};
        for(int j=0;j<512;j++){
            int e=ranking[l*512+j];if(e<0||e>=512||seen[e]){snprintf(s->error,256,"invalid full ranking at layer %d",l);goto bad;}
            seen[e]=1;if(j<hot)s->hot[l][e]=1;
        }
        for(int k=0;k<3;k++){
            int i=l*3+k;qs_source v=src[i];struct stat st;
            /* Whole-expert page alignment makes releasing cold pages independent
             * of adjacent hot experts; supported Qwen quantizations satisfy it. */
            if(page<1||!v.expert_bytes||v.expert_bytes%(uint64_t)page||v.expert_bytes>(uint64_t)SIZE_MAX/512||
               v.offset>(uint64_t)INT64_MAX||v.expert_bytes>((uint64_t)INT64_MAX-v.offset)/512||
               fstat(v.fd,&st)||v.offset+v.expert_bytes*512>(uint64_t)st.st_size){snprintf(s->error,256,"invalid/page-unaligned expert source layer %d matrix %d",l,k);goto bad;}
            s->source[i]=v;bytes+=v.expert_bytes;
            void *p=mmap(NULL,v.expert_bytes*512,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
            if(p==MAP_FAILED){snprintf(s->error,256,"allocate sparse expert buffer: %s",strerror(errno));goto bad;}
            s->data[i]=p;
            if(madvise(p,v.expert_bytes*512,MADV_NOHUGEPAGE)){snprintf(s->error,256,"disable huge pages for bounded expert cache: %s",strerror(errno));goto bad;}
        }
        if(hot<512&&budget<bytes*10){snprintf(s->error,256,"SSD buffer must fit ten cold experts (%llu bytes)",(unsigned long long)(bytes*10));goto bad;}
        s->stats.ram_bytes+=bytes*hot;
    }
    for(int i=0;i<threads;i++){
        rc=pthread_create(&s->worker[i],NULL,reader,s);
        if(rc){snprintf(s->error,256,"SSD read worker: %s",strerror(rc));goto bad;}s->started++;
    }
    for(int l=0;l<layers;l++){
        int jobs=0;for(int e=0;e<512;e++)if(s->hot[l][e])add_expert(s,l,e,&jobs);
        submit(s,jobs);if(qs_wait(s))goto bad;
    }
    s->stats.read_bytes=s->stats.expert_reads=0;s->stats.io_seconds=s->stats.wait_seconds=0;
    return s;
bad:snprintf(error,256,"%s",s->error);qs_destroy(s);return NULL;
}
int qs_begin(qs_store *s,int l,const int *ids,int nt){
    if(!s||!ids||l<0||l>=s->layers||nt<1)return -1;
    if(s->active_layer>=0){snprintf(s->error,256,"release previous SSD batch before reuse");return -1;}
    unsigned char seen[512]={0};int rows=0,ncold=0;
    uint64_t bytes=s->source[l*3].expert_bytes+s->source[l*3+1].expert_bytes+s->source[l*3+2].expert_bytes;
    for(int t=0;t<nt;t++){
        int added[10],n=0;unsigned char token[512]={0};
        for(int k=0;k<10;k++){
            int e=ids[t*10+k];if(e<=-2&&e>=-513)continue;
            if(e<0||e>=512||token[e]){snprintf(s->error,256,"invalid SSD route ID");return -1;}
            token[e]=1;if(!s->hot[l][e]&&!seen[e])added[n++]=e;
        }
        if((uint64_t)(ncold+n)*bytes>s->budget)break;
        for(int k=0;k<n;k++){seen[added[k]]=1;s->cold[ncold++]=added[k];}rows++;
    }
    if(!rows){snprintf(s->error,256,"SSD buffer cannot fit this token");return -1;}
    s->active_layer=l;s->cold_count=ncold;
    uint64_t transient=bytes*ncold;if(transient>s->stats.transient_peak_bytes)s->stats.transient_peak_bytes=transient;
    int jobs=0;for(int e=0;e<512;e++)if(seen[e])add_expert(s,l,e,&jobs);
    submit(s,jobs);return rows;
}
int qs_release(qs_store *s){
    if(!s)return -1;
    if(s->active_layer<0)return 0;
    pthread_mutex_lock(&s->mutex);int busy=s->pending;pthread_mutex_unlock(&s->mutex);
    if(busy){snprintf(s->error,256,"SSD release before reads complete");return -1;}
    for(int j=0;j<s->cold_count;j++)for(int k=0;k<3;k++){
        int i=s->active_layer*3+k;uint64_t bytes=s->source[i].expert_bytes;
        if(madvise(s->data[i]+s->cold[j]*bytes,bytes,MADV_DONTNEED)){snprintf(s->error,256,"release cold expert pages: %s",strerror(errno));return -1;}
    }
    s->active_layer=-1;s->cold_count=0;return 0;
}
const void *qs_tensor(qs_store *s,int l,int k){return s&&l>=0&&l<s->layers&&k>=0&&k<3?s->data[l*3+k]:NULL;}
int qs_is_hot(qs_store *s,int l,int e){return s&&l>=0&&l<s->layers&&e>=0&&e<512&&s->hot[l][e];}
const char *qs_error(qs_store *s){return s?s->error:"missing SSD store";}
void qs_get_stats(qs_store *s,qs_stats *out){if(out){memset(out,0,sizeof(*out));if(s)*out=s->stats;}}
void qs_destroy(qs_store *s){
    if(!s)return;
    pthread_mutex_lock(&s->mutex);s->stop=1;pthread_cond_broadcast(&s->work);pthread_mutex_unlock(&s->mutex);
    for(int i=0;i<s->started;i++)pthread_join(s->worker[i],NULL);
    for(int i=0;i<s->layers*3;i++)if(s->data[i])munmap(s->data[i],s->source[i].expert_bytes*512);
    pthread_cond_destroy(&s->work);pthread_cond_destroy(&s->done);pthread_mutex_destroy(&s->mutex);free(s);
}
