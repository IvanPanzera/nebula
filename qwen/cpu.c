/* CPU fallback for the existing GGML expert layouts. Activations stay F32;
 * quantized weights are expanded a 32-value group at a time in registers. */
#include "cpu_variant.h"
#include "cpu.h"
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH=2560, FF=640, EXPERTS=512, TOP=10, TILE=4 };
struct qc_workspace {
    int capacity,threads;
    float *input,*gate,*up,*mid,*output,*slots,*small_mid;
    int *map;
};
size_t qc_row_bytes(int type,int cols){
    if(cols<1)return 0;
    if(type==0)return (size_t)cols*4;
    if(type==1||type==30)return (size_t)cols*2;
    if(cols%32)return 0;
    if(type==2||type==20)return (size_t)cols/32*18;
    if(type==6)return (size_t)cols/32*22;
    if(type==12)return cols%256?0:(size_t)cols/256*144;
    if(type==13)return cols%256?0:(size_t)cols/256*176;
    if(type==14)return cols%256?0:(size_t)cols/256*210;
    if(type==7)return (size_t)cols/32*24;
    if(type==8)return (size_t)cols/32*34;
    return 0;
}
static float half(const unsigned char *p){
    uint16_t h;memcpy(&h,p,2);
#if defined(QC_AVX512)||defined(QC_AVX2)
    return _cvtsh_ss(h);
#else
    unsigned sign=(unsigned)(h&0x8000)<<16,exp=(h>>10)&31,m=h&1023;uint32_t bits;
    if(!exp){if(!m)bits=sign;else{int e=-14;while(!(m&1024)){m<<=1;e--;}bits=sign|((unsigned)(e+127)<<23)|((m&1023)<<13);}}
    else if(exp==31)bits=sign|0x7f800000|(m<<13)|(m?0x400000:0);
    else bits=sign|((exp+112)<<23)|(m<<13);
    float value;memcpy(&value,&bits,4);return value;
#endif
}
float qc_weight_at(const void *ptr,int type,int i){
    const unsigned char *row=ptr;
    if(type==0)return ((const float*)ptr)[i];
    if(type==1)return half(row+i*2);
    if(type==30){uint16_t h;uint32_t u;float f;memcpy(&h,row+i*2,2);u=(uint32_t)h<<16;memcpy(&f,&u,4);return f;}
    if(type==8){const unsigned char *p=row+(i/32)*34;return half(p)*((const signed char*)(p+2))[i%32];}
    if(type==2||type==20||type==6||type==7){
        int bytes=type==7?24:type==6?22:18,j=i%32,off=type==7?8:type==6?6:2;
        const unsigned char *p=row+(i/32)*bytes;int q=(p[off+j%16]>>((j/16)*4))&15;
        if(type==6||type==7){uint32_t high;memcpy(&high,p+(type==7?4:2),4);q|=((high>>j)&1)<<4;}
        if(type==20){const signed char lut[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};return half(p)*lut[q];}
        return type==7?fmaf(half(p),q,half(p+2)):half(p)*(q-(type==2?8:type==6?16:0));
    }
    int k=i%256,g=k/32,l=k%32;
    const unsigned char *p=row+(i/256)*(type==12?144:type==13?176:210);
    if(type==14){int h=k/128,s=(k%128)/32;
        int lo=(p[h*64+(s%2)*32+l]>>((s/2)*4))&15,hi=(p[128+h*32+l]>>(s*2))&3;
        return half(p+208)*((const signed char*)(p+192))[h*8+s*2+l/16]*((lo|(hi<<4))-32);}
    const unsigned char *sc=p+4;
    int s=g<4?(sc[g]&63):((sc[g+4]&15)|((sc[g-4]>>6)<<4));
    int m=g<4?(sc[g+4]&63):((sc[g+4]>>4)|((sc[g]>>6)<<4));
    int q=(p[(type==12?16:48)+(g/2)*32+l]>>((g%2)*4))&15;
    if(type==13)q|=((p[16+l]>>g)&1)<<4;
    return fmaf(half(p)*s,q,-half(p+2)*m);
}
#if defined(QC_AVX512)
static __m512 quant16(__m128i bytes,int upper){
    if(upper)bytes=_mm_srli_epi16(bytes,4);
    bytes=_mm_and_si128(bytes,_mm_set1_epi8(15));
    return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes));
}
static inline __attribute__((always_inline)) void unpack32(const unsigned char *row,int type,int k,__m512 *a,__m512 *b){
    if(type==0){*a=_mm512_loadu_ps((const float*)row+k);*b=_mm512_loadu_ps((const float*)row+k+16);
    }else if(type==1){*a=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(row+k*2)));*b=_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(row+(k+16)*2)));
    }else if(type==30){
        *a=_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(row+k*2))),16));
        *b=_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(row+(k+16)*2))),16));
    }else if(type==2||type==20){
        const unsigned char *p=row+(k/32)*18;__m128i bytes=_mm_loadu_si128((const __m128i*)(p+2));
        __m512 q0=quant16(bytes,0),q1=quant16(bytes,1);
        if(type==20){const __m512i lut=_mm512_setr_epi32(-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113);
            q0=_mm512_cvtepi32_ps(_mm512_permutexvar_epi32(_mm512_cvttps_epi32(q0),lut));
            q1=_mm512_cvtepi32_ps(_mm512_permutexvar_epi32(_mm512_cvttps_epi32(q1),lut));
        }else{q0=_mm512_sub_ps(q0,_mm512_set1_ps(8));q1=_mm512_sub_ps(q1,_mm512_set1_ps(8));}
        *a=_mm512_mul_ps(_mm512_set1_ps(half(p)),q0);*b=_mm512_mul_ps(_mm512_set1_ps(half(p)),q1);
    }else if(type==6||type==13||type==14){
        float values[32];for(int j=0;j<32;j++)values[j]=qc_weight_at(row,type,k+j);
        *a=_mm512_loadu_ps(values);*b=_mm512_loadu_ps(values+16);
    }else if(type==8){
        const unsigned char *p=row+(size_t)(k/32)*34;
        __m512 scale=_mm512_set1_ps(half(p));
        *a=_mm512_mul_ps(scale,_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(p+2)))));
        *b=_mm512_mul_ps(scale,_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(p+18)))));
    }else if(type==7){
        const unsigned char *p=row+(size_t)(k/32)*24;uint32_t high;memcpy(&high,p+4,4);
        __m128i bytes=_mm_loadu_si128((const __m128i*)(p+8));
        const __m512i indices=_mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        __m512i h0=_mm512_srlv_epi32(_mm512_set1_epi32((int)high),indices);
        __m512i h1=_mm512_srlv_epi32(_mm512_set1_epi32((int)(high>>16)),indices);
        __m512 q0=_mm512_add_ps(quant16(bytes,0),_mm512_cvtepi32_ps(_mm512_slli_epi32(_mm512_and_si512(h0,_mm512_set1_epi32(1)),4)));
        __m512 q1=_mm512_add_ps(quant16(bytes,1),_mm512_cvtepi32_ps(_mm512_slli_epi32(_mm512_and_si512(h1,_mm512_set1_epi32(1)),4)));
        __m512 scale=_mm512_set1_ps(half(p)),offset=_mm512_set1_ps(half(p+2));
        *a=_mm512_fmadd_ps(scale,q0,offset);*b=_mm512_fmadd_ps(scale,q1,offset);
    }else{
        const unsigned char *p=row+(size_t)(k/256)*144,*sc=p+4;int g=(k%256)/32;
        int s=g<4?(sc[g]&63):((sc[g+4]&15)|((sc[g-4]>>6)<<4));
        int m=g<4?(sc[g+4]&63):((sc[g+4]>>4)|((sc[g]>>6)<<4));
        const unsigned char *q=p+16+(g/2)*32;
        __m512 scale=_mm512_set1_ps(half(p)*s),offset=_mm512_set1_ps(-half(p+2)*m);
        *a=_mm512_fmadd_ps(scale,quant16(_mm_loadu_si128((const __m128i*)q),g&1),offset);
        *b=_mm512_fmadd_ps(scale,quant16(_mm_loadu_si128((const __m128i*)(q+16)),g&1),offset);
    }
}
#else
#include "cpu_portable.h"
#endif
int qc_matmul(float *out,const void *weights,int type,int rows,int cols,const float *x,int nt,int threads){
    size_t stride=qc_row_bytes(type,cols);
    if(!out||!weights||!x||!stride||cols%32||rows<1||nt<1||nt>2048||threads<1||threads>28)return -1;
    /* Reproduce CUDA's 32 lane accumulators and reduction tree. This avoids
     * a different dot-product order when the same expert changes device. */
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for(int row=0;row<rows;row++)for(int first=0;first<nt;first+=TILE){
        int n=nt-first<TILE?nt-first:TILE;
#if defined(QC_AVX512)
        __m512 lo[TILE],hi[TILE];
        for(int t=0;t<n;t++)lo[t]=hi[t]=_mm512_setzero_ps();
        const unsigned char *w=(const unsigned char*)weights+(size_t)row*stride;
        for(int k=0;k<cols;k+=32){
            __m512 a,b;unpack32(w,type,k,&a,&b);
            for(int t=0;t<n;t++){
                const float *v=x+(size_t)(first+t)*cols+k;
                lo[t]=_mm512_fmadd_ps(a,_mm512_loadu_ps(v),lo[t]);
                hi[t]=_mm512_fmadd_ps(b,_mm512_loadu_ps(v+16),hi[t]);
            }
        }
        for(int t=0;t<n;t++)out[(size_t)(first+t)*rows+row]=_mm512_reduce_add_ps(_mm512_add_ps(lo[t],hi[t]));
#else
        const float *v[TILE];float result[TILE];
        for(int t=0;t<n;t++)v[t]=x+(size_t)(first+t)*cols;
        dot_group_inner(result,(const unsigned char*)weights+(size_t)row*stride,type,cols,v,n);
        for(int t=0;t<n;t++)out[(size_t)(first+t)*rows+row]=result[t];
#endif
    }
    return 0;
}
qc_workspace *qc_create(int capacity,int threads){
    if(capacity<1||capacity>2048||threads<1||threads>28)return NULL;
    qc_workspace *w=calloc(1,sizeof(*w));if(!w)return NULL;w->capacity=capacity;w->threads=threads;
    w->input=malloc((size_t)capacity*WIDTH*4);w->gate=malloc((size_t)capacity*FF*4);
    w->up=malloc((size_t)capacity*FF*4);w->mid=malloc((size_t)capacity*FF*4);
    w->output=malloc((size_t)capacity*WIDTH*4);w->slots=malloc((size_t)capacity*TOP*WIDTH*4);
    w->map=malloc((size_t)EXPERTS*capacity*sizeof(int));
    w->small_mid=malloc(17ULL*TOP*FF*sizeof(float));
    if(!w->input||!w->gate||!w->up||!w->mid||!w->output||!w->slots||!w->map||!w->small_mid){qc_destroy(w);return NULL;}
    return w;
}
void qc_destroy(qc_workspace *w){if(!w)return;
    free(w->input);free(w->gate);free(w->up);free(w->mid);free(w->output);free(w->slots);free(w->map);free(w->small_mid);free(w);
}
int qc_set_threads(qc_workspace *w,int threads){
    if(!w||threads<1||threads>28)return -1;
    w->threads=threads;return 0;
}
int qc_threads(const qc_workspace *w){return w?w->threads:0;}
/* All active experts share one OpenMP team in short verification batches.
 * Rows, not experts, are the work units: a rare expert still uses all cores.
 * Keep the existing 32-lane accumulation and router-order output reduction. */
#if defined(QC_AVX512)
static inline __attribute__((always_inline)) void dot_group_inner(float out[TILE],const unsigned char *row,int type,int cols,
                      const float *const input[TILE],int n){
    __m512 l0=_mm512_setzero_ps(),h0=l0,l1=l0,h1=l0,l2=l0,h2=l0,l3=l0,h3=l0;
    for(int k=0;k<cols;k+=32){
        __m512 a,b;unpack32(row,type,k,&a,&b);
        l0=_mm512_fmadd_ps(a,_mm512_loadu_ps(input[0]+k),l0);
        h0=_mm512_fmadd_ps(b,_mm512_loadu_ps(input[0]+k+16),h0);
        if(n>1){l1=_mm512_fmadd_ps(a,_mm512_loadu_ps(input[1]+k),l1);h1=_mm512_fmadd_ps(b,_mm512_loadu_ps(input[1]+k+16),h1);}
        if(n>2){l2=_mm512_fmadd_ps(a,_mm512_loadu_ps(input[2]+k),l2);h2=_mm512_fmadd_ps(b,_mm512_loadu_ps(input[2]+k+16),h2);}
        if(n>3){l3=_mm512_fmadd_ps(a,_mm512_loadu_ps(input[3]+k),l3);h3=_mm512_fmadd_ps(b,_mm512_loadu_ps(input[3]+k+16),h3);}
    }
    out[0]=_mm512_reduce_add_ps(_mm512_add_ps(l0,h0));
    if(n>1)out[1]=_mm512_reduce_add_ps(_mm512_add_ps(l1,h1));
    if(n>2)out[2]=_mm512_reduce_add_ps(_mm512_add_ps(l2,h2));
    if(n>3)out[3]=_mm512_reduce_add_ps(_mm512_add_ps(l3,h3));
}
#endif
static void dot_group(float out[TILE],const unsigned char *row,int type,int cols,
                      const float *const input[TILE],int n){
    /* Dispatch outside the dot loop: Q4_K and IQ4_NL each compile to a
     * specialized unpacker; fixed accumulators stay in AVX-512 registers. */
    if(type==12){dot_group_inner(out,row,12,cols,input,n);return;}
    if(type==20){dot_group_inner(out,row,20,cols,input,n);return;}
    dot_group_inner(out,row,type,cols,input,n);
}
static int small_moe(qc_workspace *w,float *out,const float *x,const float *weights,int nt,
                     const int counts[EXPERTS],const void *gate,int gt,const void *up,int ut,const void *down,int dt){
    struct {int expert,n,slot[TILE];} jobs[17*TOP];int nj=0;
    for(int e=0;e<EXPERTS;e++)for(int first=0;first<counts[e];first+=TILE){
        int n=counts[e]-first<TILE?counts[e]-first:TILE;
        jobs[nj].expert=e;jobs[nj].n=n;
        for(int t=0;t<n;t++)jobs[nj].slot[t]=w->map[(size_t)e*w->capacity+first+t];
        nj++;
    }
    size_t gs=qc_row_bytes(gt,WIDTH),us=qc_row_bytes(ut,WIDTH),ds=qc_row_bytes(dt,FF);
    #pragma omp parallel num_threads(w->threads)
    {
        #pragma omp for schedule(static)
        for(int task=0;task<nj*FF;task++){
            int j=task/FF,row=task%FF,n=jobs[j].n,e=jobs[j].expert;
            const float *v[TILE];float g[TILE],u[TILE];
            for(int t=0;t<n;t++)v[t]=x+(size_t)(jobs[j].slot[t]/TOP)*WIDTH;
            dot_group(g,(const unsigned char*)gate+((size_t)e*FF+row)*gs,gt,WIDTH,v,n);
            dot_group(u,(const unsigned char*)up+((size_t)e*FF+row)*us,ut,WIDTH,v,n);
            for(int t=0;t<n;t++)w->small_mid[(size_t)jobs[j].slot[t]*FF+row]=(g[t]*(1.0f/(1.0f+expf(-g[t]))))*u[t];
        }
        #pragma omp for schedule(static)
        for(int task=0;task<nj*WIDTH;task++){
            int j=task/WIDTH,row=task%WIDTH,n=jobs[j].n,e=jobs[j].expert;
            const float *v[TILE];float result[TILE];
            for(int t=0;t<n;t++)v[t]=w->small_mid+(size_t)jobs[j].slot[t]*FF;
            dot_group(result,(const unsigned char*)down+((size_t)e*WIDTH+row)*ds,dt,FF,v,n);
            for(int t=0;t<n;t++){int slot=jobs[j].slot[t];w->slots[(size_t)slot*WIDTH+row]=result[t]*weights[slot];}
        }
        #pragma omp for schedule(static)
        for(int i=0;i<nt*WIDTH;i++){
            int t=i/WIDTH,d=i%WIDTH;float sum=0;
            for(int k=0;k<TOP;k++)sum+=w->slots[((size_t)t*TOP+k)*WIDTH+d];
            out[i]=sum;
        }
    }
    return 0;
}
int qc_moe(qc_workspace *w,float *out,const float *x,const int *ids,const float *weights,int nt,
           const void *gate,int gt,const void *up,int ut,const void *down,int dt){
    size_t gb=qc_row_bytes(gt,WIDTH)*FF,ub=qc_row_bytes(ut,WIDTH)*FF,db=qc_row_bytes(dt,FF)*WIDTH;
    if(!w||!out||!x||!ids||!weights||!gate||!up||!down||nt<1||nt>w->capacity||!gb||!ub||!db)return -1;
    int counts[EXPERTS]={0};
    memset(w->slots,0,(size_t)nt*TOP*WIDTH*sizeof(float));
    for(int t=0;t<nt;t++)for(int k=0;k<TOP;k++){
        int e=ids[t*TOP+k];if(e<=-2&&e>=-EXPERTS-1&&weights[t*TOP+k]==0)continue;
        if(e<0||e>=EXPERTS||!isfinite(weights[t*TOP+k])||weights[t*TOP+k]<0)return -1;
        for(int j=0;j<k;j++)if(ids[t*TOP+j]==e)return -1;
        w->map[(size_t)e*w->capacity+counts[e]++]=t*TOP+k;
    }
    if(nt<=17)return small_moe(w,out,x,weights,nt,counts,gate,gt,up,ut,down,dt);
    for(int e=0;e<EXPERTS;e++)if(counts[e]){
        int ne=counts[e];const int *map=w->map+(size_t)e*w->capacity;
        for(int j=0;j<ne;j++)memcpy(w->input+(size_t)j*WIDTH,x+(size_t)(map[j]/TOP)*WIDTH,WIDTH*4);
        if(qc_matmul(w->gate,(const unsigned char*)gate+e*gb,gt,FF,WIDTH,w->input,ne,w->threads)||
           qc_matmul(w->up,(const unsigned char*)up+e*ub,ut,FF,WIDTH,w->input,ne,w->threads))return -1;
        for(int j=0;j<ne*FF;j++)w->mid[j]=(w->gate[j]*(1.0f/(1.0f+expf(-w->gate[j]))))*w->up[j];
        if(qc_matmul(w->output,(const unsigned char*)down+e*db,dt,WIDTH,FF,w->mid,ne,w->threads))return -1;
        for(int j=0;j<ne;j++)for(int d=0;d<WIDTH;d++)w->slots[(size_t)map[j]*WIDTH+d]=w->output[(size_t)j*WIDTH+d]*weights[map[j]];
    }
    for(int t=0;t<nt;t++)for(int d=0;d<WIDTH;d++){
        float sum=0;for(int k=0;k<TOP;k++)sum+=w->slots[((size_t)t*TOP+k)*WIDTH+d];out[(size_t)t*WIDTH+d]=sum;
    }
    return 0;
}
