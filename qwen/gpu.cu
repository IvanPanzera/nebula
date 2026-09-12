/* Qwen4Exp CUDA primitives, C ABI. Quant layouts follow GGML (MIT).
 * Model equations are checked against the pinned Qwen/llama.cpp references.
 * No persistent full-size dequantized weight copies are used. */
#include "gpu.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdlib.h>

static __thread cudaStream_t capture_stream;
static int attention_shared_limit=8192, index_shared_limit=4096;
typedef struct { cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t exec; int active; } qg_graph;
extern "C" void qg_graph_destroy(void *ptr) {
    qg_graph *g=(qg_graph*)ptr;if(!g)return;
    if(g->active){cudaStreamEndCapture(g->stream,&g->graph);capture_stream=0;}
    if(g->exec)cudaGraphExecDestroy(g->exec);
    if(g->graph)cudaGraphDestroy(g->graph);
    if(g->stream)cudaStreamDestroy(g->stream);
    free(g);
}
extern "C" int qg_graph_begin(void **out) {
    if(!out||*out||capture_stream)return cudaErrorInvalidValue;
    qg_graph *g=(qg_graph*)calloc(1,sizeof(*g));if(!g)return cudaErrorMemoryAllocation;
    int rc=cudaStreamCreate(&g->stream);
    if(!rc)rc=cudaStreamBeginCapture(g->stream,cudaStreamCaptureModeThreadLocal);
    if(rc){qg_graph_destroy(g);return rc;}
    g->active=1;capture_stream=g->stream;*out=g;return 0;
}
extern "C" int qg_graph_end(void *ptr) {
    qg_graph *g=(qg_graph*)ptr;if(!g||!g->active)return cudaErrorInvalidValue;
    int rc=cudaStreamEndCapture(g->stream,&g->graph);g->active=0;capture_stream=0;
    if(!rc)rc=cudaGraphInstantiate(&g->exec,g->graph,NULL,NULL,0);
    return rc;
}
extern "C" int qg_graph_launch(void *ptr) {
    qg_graph *g=(qg_graph*)ptr;return g&&g->exec?cudaGraphLaunch(g->exec,0):cudaErrorInvalidValue;
}
extern "C" void *qg_alloc(size_t n) { void *p=0; return cudaMalloc(&p,n)==cudaSuccess?p:0; }
extern "C" void qg_free(void *p) { if(p) cudaFree(p); }
extern "C" int qg_write(void *d,const void *s,size_t n) { return cudaMemcpy(d,s,n,cudaMemcpyHostToDevice); }
extern "C" int qg_read(void *d,const void *s,size_t n) { return cudaMemcpy(d,s,n,cudaMemcpyDeviceToHost); }
extern "C" int qg_copy(void *d,const void *s,size_t n) { return cudaMemcpyAsync(d,s,n,cudaMemcpyDeviceToDevice,capture_stream); }
extern "C" int qg_zero(void *d,size_t n) { return cudaMemsetAsync(d,0,n,capture_stream); }
extern "C" int qg_sync(void) { return cudaDeviceSynchronize(); }
extern "C" int qg_memory(size_t *f,size_t *t) { return cudaMemGetInfo(f,t); }
extern "C" const char *qg_error(int e) { return cudaGetErrorString((cudaError_t)e); }

typedef struct {
    unsigned char *stage[2];cudaEvent_t begin[2],end[2];
    int next,pending[2];size_t capacity;double seconds;
} expert_uploader;
extern "C" void qg_upload_destroy(void *ptr) {
    expert_uploader *u=(expert_uploader*)ptr;if(!u)return;cudaDeviceSynchronize();
    for(int i=0;i<2;i++){if(u->stage[i])cudaFreeHost(u->stage[i]);if(u->begin[i])cudaEventDestroy(u->begin[i]);if(u->end[i])cudaEventDestroy(u->end[i]);}free(u);
}
extern "C" void *qg_upload_create(size_t bytes) {
    expert_uploader *u=(expert_uploader*)calloc(1,sizeof(*u));if(!u)return NULL;u->capacity=bytes;
    for(int i=0;i<2;i++)if(cudaHostAlloc((void**)&u->stage[i],bytes,0)!=cudaSuccess ||
        cudaEventCreate(&u->begin[i])!=cudaSuccess || cudaEventCreate(&u->end[i])!=cudaSuccess){qg_upload_destroy(u);return NULL;}
    return u;
}
static int upload_finish(expert_uploader *u,int i,int wait) {
    if(!u->pending[i])return cudaSuccess;
    cudaError_t rc=wait?cudaEventSynchronize(u->end[i]):cudaEventQuery(u->end[i]);
    if(rc==cudaErrorNotReady)return cudaSuccess;
    if(rc!=cudaSuccess)return rc;
    float ms;rc=cudaEventElapsedTime(&ms,u->begin[i],u->end[i]);if(rc!=cudaSuccess)return rc;
    u->seconds+=ms/1000.0;u->pending[i]=0;return cudaSuccess;
}
extern "C" int qg_upload_expert(void *ptr,void *dst,const void *g,size_t ng,const void *up,size_t nu,const void *down,size_t nd) {
    expert_uploader *u=(expert_uploader*)ptr;if(!u||ng>u->capacity||nu>u->capacity-ng||nd>u->capacity-ng-nu)return cudaErrorInvalidValue;
    int i=u->next,rc=upload_finish(u,i,1);if(rc)return rc;
    /* Two bounded pinned slabs permit CPU packing while the preceding expert
     * computes. Default-stream ordering also protects cache slots on eviction. */
    memcpy(u->stage[i],g,ng);memcpy(u->stage[i]+ng,up,nu);memcpy(u->stage[i]+ng+nu,down,nd);
    rc=cudaEventRecord(u->begin[i]);if(rc)return rc;
    rc=cudaMemcpyAsync(dst,u->stage[i],ng+nu+nd,cudaMemcpyHostToDevice);if(rc)return rc;
    rc=cudaEventRecord(u->end[i]);if(rc)return rc;
    u->pending[i]=1;u->next=1-i;return cudaSuccess;
}
extern "C" double qg_upload_seconds(void *ptr) {
    expert_uploader *u=(expert_uploader*)ptr;if(!u)return 0;
    for(int i=0;i<2;i++)if(upload_finish(u,i,0))return -1;
    return u->seconds;
}

static __device__ float half_at(const unsigned char *p) { return __half2float(*((const __half*)p)); }
static __device__ float sig(float x) { return 1.0f/(1.0f+expf(-x)); }
static __device__ float warp_sum(float x) {
    for(int d=16;d;d/=2) x+=__shfl_down_sync(0xffffffff,x,d);
    return x;
}
static __device__ float block_sum(float x) {
    __shared__ float partial[32];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    x=warp_sum(x);
    if(!lane) partial[warp]=x;
    __syncthreads();
    x=threadIdx.x<(blockDim.x+31)/32?partial[lane]:0;
    if(!warp) x=warp_sum(x);
    if(!threadIdx.x) partial[0]=x;
    __syncthreads();
    return partial[0];
}
static size_t row_bytes(int type,int cols) {
    switch(type) {
    case 0:return (size_t)cols*4; case 1:case 30:return (size_t)cols*2;
    case 2:case 20:return (size_t)cols/32*18; case 6:return (size_t)cols/32*22;
    case 7:return (size_t)cols/32*24; case 8:return (size_t)cols/32*34;
    case 12:return (size_t)cols/256*144; case 13:return (size_t)cols/256*176;
    case 14:return (size_t)cols/256*210; default:return 0;
    }
}
extern "C" int qg_tensor_bytes(int type,const uint64_t dims[4],uint64_t *bytes) {
    if(!dims||!bytes||dims[0]<1||dims[0]>2147483647ULL)return cudaErrorInvalidValue;
    int block=(type==12||type==13||type==14)?256:(type==2||type==6||type==7||type==8||type==20)?32:1;
    uint64_t n=row_bytes(type,(int)dims[0]);if(!n||dims[0]%block)return cudaErrorInvalidValue;
    for(int i=1;i<4;i++){if(!dims[i]||dims[i]>UINT64_MAX/n)return cudaErrorInvalidValue;n*=dims[i];}
    *bytes=n;return 0;
}
static __device__ float weight_at(const unsigned char *row,int t,int i) {
    if(t==0) return ((const float*)row)[i];
    if(t==1) return half_at(row+2*i);
    if(t==30) return __uint_as_float((unsigned)((const unsigned short*)row)[i]<<16);
    if(t==8) { const unsigned char *p=row+(i/32)*34;return half_at(p)*(float)((const signed char*)(p+2))[i%32]; }
    if(t==2 || t==20 || t==6 || t==7) {
        int bytes=t==7?24:t==6?22:18, j=i%32;
        const unsigned char *p=row+(i/32)*bytes;
        int off=t==7?8:t==6?6:2;
        int q=(p[off+j%16]>>((j/16)*4))&15;
        if(t==6 || t==7) { unsigned high;memcpy(&high,p+(t==7?4:2),4);q|=((high>>j)&1)<<4; }
        if(t==20) {
            const signed char lut[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
            return half_at(p)*lut[q];
        }
        return half_at(p)*(q-(t==2?8:t==6?16:0))+(t==7?half_at(p+2):0);
    }
    int k=i%256,g=k/32,l=k%32;
    const unsigned char *p=row+(i/256)*(t==12?144:t==13?176:210);
    if(t==14) {
        int h=k/128,s=(k%128)/32;
        int lo=(p[h*64+(s%2)*32+l]>>((s/2)*4))&15;
        int hi=(p[128+h*32+l]>>(s*2))&3;
        int scale=((const signed char*)(p+192))[h*8+s*2+l/16];
        return half_at(p+208)*scale*((lo|(hi<<4))-32);
    }
    const unsigned char *sc=p+4;
    int scale=g<4?(sc[g]&63):((sc[g+4]&15)|((sc[g-4]>>6)<<4));
    int minv=g<4?(sc[g+4]&63):((sc[g+4]>>4)|((sc[g]>>6)<<4));
    int q=(p[(t==12?16:48)+(g/2)*32+l]>>((g%2)*4))&15;
    if(t==13) q|=((p[16+l]>>g)&1)<<4;
    return (half_at(p)*scale)*q-half_at(p+2)*minv;
}
static __global__ void mm_small_kernel(float *out,const unsigned char *w,int type,int rows,int cols,
                                       const float *x,int nt,size_t stride) {
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=rows)return;
    float sum[4]={0};
    for(int k=lane;k<cols;k+=32){float a=weight_at(w+(size_t)row*stride,type,k);
        #pragma unroll
        for(int t=0;t<4;t++)if(t<nt)sum[t]+=a*x[(size_t)t*cols+k];}
    #pragma unroll
    for(int t=0;t<4;t++)if(t<nt){float s=warp_sum(sum[t]);if(!lane)out[(size_t)t*rows+row]=s;}
}
static __global__ void mm_kernel(float *out,const unsigned char *w,int type,int rows,int cols,
                                 const float *x,int nt,size_t stride) {
    int first=blockIdx.y*16;x+=(size_t)first*cols;out+=(size_t)first*rows;nt=min(nt-first,16);
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=rows) return;
    float sum[16]={0};
    for(int k=lane;k<cols;k+=32) {
        float a=weight_at(w+(size_t)row*stride,type,k);
        /* Fixed indices keep the accumulators in registers. A runtime-indexed
         * array spilled to per-thread stack on sm_89, especially at prefill. */
        #pragma unroll
        for(int t=0;t<16;t++) if(t<nt) sum[t]+=a*x[(size_t)t*cols+k];
    }
    #pragma unroll
    for(int t=0;t<16;t++) if(t<nt) { float s=warp_sum(sum[t]);if(!lane) out[(size_t)t*rows+row]=s; }
}
extern "C" int qg_matmul(float *o,const void *w,int type,int rows,int cols,const float *x,int nt) {
    size_t stride=row_bytes(type,cols);
    if(!stride || rows<1 || cols<1 || nt<1 || nt>8192 ||
       ((type==12 || type==13 || type==14) && cols%256) ||
       ((type==2 || type==6 || type==7 || type==8 || type==20) && cols%32)) return cudaErrorInvalidValue;
    if(nt<=4)mm_small_kernel<<<(rows+7)/8,256,0,capture_stream>>>(o,(const unsigned char*)w,type,rows,cols,x,nt,stride);
    else mm_kernel<<<dim3((rows+7)/8,(nt+15)/16),256,0,capture_stream>>>(o,(const unsigned char*)w,type,rows,cols,x,nt,stride);
    return cudaGetLastError();
}
static __global__ void gather_kernel(float *out,const unsigned char *w,int type,int n,size_t stride,int row) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) out[i]=weight_at(w+(size_t)row*stride,type,i);
}
extern "C" int qg_gather(float *o,const void *w,int type,int cols,int row) {
    size_t stride=row_bytes(type,cols);if(!stride || row<0) return cudaErrorInvalidValue;
    gather_kernel<<<(cols+255)/256,256,0,capture_stream>>>(o,(const unsigned char*)w,type,cols,stride,row);return cudaGetLastError();
}
static __global__ void dequant_kernel(float *o,const unsigned char *w,int type,int rows,int cols,size_t stride) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*cols)o[i]=weight_at(w+(size_t)(i/cols)*stride,type,i%cols);
}
extern "C" int qg_dequant(float *o,const void *w,int type,int rows,int cols) {
    size_t stride=row_bytes(type,cols);if(!stride)return cudaErrorInvalidValue;
    dequant_kernel<<<(rows*cols+255)/256,256,0,capture_stream>>>(o,(const unsigned char*)w,type,rows,cols,stride);return cudaGetLastError();
}
static __global__ void norm_kernel(float *o,const float *x,const float *w,int width,int groups,int wg,int l2,float eps) {
    int g=blockIdx.x,base=g*width;float ss=0;
    for(int i=threadIdx.x;i<width;i+=blockDim.x) ss+=x[base+i]*x[base+i];
    ss=block_sum(ss);
    float scale=rsqrtf(ss/(l2?1.0f:(float)width)+eps);
    for(int i=threadIdx.x;i<width;i+=blockDim.x) o[base+i]=x[base+i]*scale*(w?w[(wg?g%groups:0)*width+i]:1.0f);
}
extern "C" int qg_norm(float *o,const float *x,const float *w,int width,int groups,int nt,int wg,int l2,float eps) {
    if(width<1 || groups<1 || nt<1) return cudaErrorInvalidValue;
    norm_kernel<<<groups*nt,256,0,capture_stream>>>(o,x,w,width,groups,wg,l2,eps);return cudaGetLastError();
}
static __global__ void unary_kernel(float *o,const float *x,int n,int op,float s) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) {float a=x[i]*s;o[i]=op==0?a*sig(a):op==1?sig(a):a;}
}
extern "C" int qg_unary(float *o,const float *x,int n,int op,float s) {
    unary_kernel<<<(n+255)/256,256,0,capture_stream>>>(o,x,n,op,s);return cudaGetLastError();
}
static __global__ void binary_kernel(float *o,const float *a,const float *b,int n,int op) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n) o[i]=op==0?a[i]+b[i]:a[i]*b[i];
}
extern "C" int qg_binary(float *o,const float *a,const float *b,int n,int op) {
    binary_kernel<<<(n+255)/256,256,0,capture_stream>>>(o,a,b,n,op);return cudaGetLastError();
}
static __global__ void hc_init_kernel(float *o,const float *x,int d,int nt) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<nt*d*4) o[i]=x[(i/(d*4))*d+i%d];
}
extern "C" int qg_hc_init(float *o,const float *x,int d,int nt) {
    hc_init_kernel<<<(d*4*nt+255)/256,256,0,capture_stream>>>(o,x,d,nt);return cudaGetLastError();
}
static __global__ void hc_read_kernel(float *o,const float *x,const float *gate,int d,int nt) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<d*nt) {int base=(i/d)*4*d+i%d;float s=0;
        for(int h=0;h<4;h++) s+=x[base+h*d]*sig(gate[base+h*d]);o[i]=s*0.25f;}
}
extern "C" int qg_hc_read(float *o,const float *x,const float *g,int d,int nt) {
    hc_read_kernel<<<(d*nt+255)/256,256,0,capture_stream>>>(o,x,g,d,nt);return cudaGetLastError();
}
static __global__ void hc_write_kernel(float *r,const float *x,const float *inj,int d,int nt) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<d*4*nt) {int t=i/(d*4),h=(i/d)%4;r[i]+=x[t*d+i%d]*2.0f*sig(inj[t*4+h]*0.25f);}
}
extern "C" int qg_hc_write(float *r,const float *x,const float *g,int d,int nt) {
    hc_write_kernel<<<(d*4*nt+255)/256,256,0,capture_stream>>>(r,x,g,d,nt);return cudaGetLastError();
}
static __global__ void mtp_concat_kernel(float *o,const float *e,const float *h,int d,int nt) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<nt*8*d) {int t=i/(8*d),part=(i/d)%2,branch=(i/(2*d))%4;
        o[i]=part?h[t*4*d+branch*d+i%d]:e[t*d+i%d];}
}
extern "C" int qg_mtp_concat(float *o,const float *e,const float *h,int d,int nt) {
    mtp_concat_kernel<<<(nt*8*d+255)/256,256,0,capture_stream>>>(o,e,h,d,nt);return cudaGetLastError();
}
/* Histories are oldest-first [history_time, channel]; update only after all
 * output rows have read the old history, so chunking cannot overwrite inputs. */
static __global__ void conv_kernel(float *o,const float *x,const float *w,float *hist,int c,int k,int dil,int nt,int silu) {
    int ch=blockIdx.x*blockDim.x+threadIdx.x;if(ch>=c) return;
    int nh=(k-1)*dil;
    for(int t=0;t<nt;t++) {float s=0;
        for(int j=0;j<k;j++) {int p=t-(k-1-j)*dil;s+=w[ch*k+j]*(p<0?hist[(p+nh)*c+ch]:x[p*c+ch]);}
        o[t*c+ch]=silu?s*sig(s):s;
    }
    for(int t=0;t<nh;t++) {int src=t+nt;hist[t*c+ch]=src<nh?hist[src*c+ch]:x[(src-nh)*c+ch];}
}
extern "C" int qg_conv(float *o,const float *x,const float *w,float *hist,int c,int k,int dil,int nt,int silu) {
    if(k<1 || dil<1 || nt<1 || o==x) return cudaErrorInvalidValue;
    conv_kernel<<<(c+255)/256,256,0,capture_stream>>>(o,x,w,hist,c,k,dil,nt,silu);return cudaGetLastError();
}
/* One warp owns an output value column of S; four key entries per lane.
 * Preserve FP32 recurrent state and GGML's head modulo mapping (48 V / 16 QK). */
static __global__ void gdn_kernel(float *o,const float *qkv,const float *alpha,const float *beta,
                                   const float *a,const float *dt,float *state,int nt,float eps) {
    int h=blockIdx.x,col=blockIdx.y*4+threadIdx.x/32,lane=threadIdx.x%32;
    float s[4];for(int j=0;j<4;j++) s[j]=state[(h*128+col)*128+j*32+lane];
    for(int t=0;t<nt;t++) {
        const float *q=qkv+t*10240+(h%16)*128,*k=q+2048,*v=qkv+t*10240+4096+h*128;
        float qn=0,kn=0;for(int j=0;j<4;j++){float qq=q[j*32+lane],kk=k[j*32+lane];qn+=qq*qq;kn+=kk*kk;}
        /* FLA/Transformers use rsqrt(sum(x*x)+eps), not 1/max(norm,eps). */
        qn=rsqrtf(__shfl_sync(0xffffffff,warp_sum(qn),0)+eps);
        kn=rsqrtf(__shfl_sync(0xffffffff,warp_sum(kn),0)+eps);
        float al=alpha[t*48+h]+dt[h];
        float decay=expf(a[h]*(fmaxf(al,0.0f)+log1pf(expf(-fabsf(al)))));
        float mem=0;for(int j=0;j<4;j++) mem+=s[j]*k[j*32+lane]*kn;
        mem=__shfl_sync(0xffffffff,warp_sum(mem),0);
        float delta=(v[col]-decay*mem)*sig(beta[t*48+h]);
        float result=0;for(int j=0;j<4;j++){s[j]=decay*s[j]+k[j*32+lane]*kn*delta;result+=s[j]*q[j*32+lane]*qn;}
        result=warp_sum(result);if(!lane) o[t*6144+h*128+col]=result*0.08838834764831845f;
    }
    for(int j=0;j<4;j++) state[(h*128+col)*128+j*32+lane]=s[j];
}
extern "C" int qg_gdn(float *o,const float *qkv,const float *al,const float *be,const float *a,const float *dt,float *s,int nt,float eps) {
    gdn_kernel<<<dim3(48,32),128,0,capture_stream>>>(o,qkv,al,be,a,dt,s,nt,eps);return cudaGetLastError();
}
static __global__ void gate_kernel(float *x,const float *g,int n) {int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]*=sig(g[i]);}
extern "C" int qg_gate(float *x,const float *g,int n) {gate_kernel<<<(n+255)/256,256,0,capture_stream>>>(x,g,n);return cudaGetLastError();}
static __global__ void qsplit_kernel(float *q,float *g,const float *x,int nt) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<nt*6144){int t=i/6144,h=(i/256)%24,d=i%256;
        q[i]=x[t*12288+h*512+d];g[i]=x[t*12288+h*512+256+d];}
}
extern "C" int qg_qsplit(float *q,float *g,const float *x,int nt) {qsplit_kernel<<<(nt*6144+255)/256,256,0,capture_stream>>>(q,g,x,nt);return cudaGetLastError();}
static __device__ void rotate_pair(float *x,int i,int pos,float theta) {
    float angle=pos*powf(theta,-2.0f*i/64.0f),sn,cs;sincosf(angle,&sn,&cs);
    float a=x[i],b=x[i+32];x[i]=a*cs-b*sn;x[i+32]=a*sn+b*cs;
}
static __global__ void rope_kernel(float *x,int d,int heads,int nt,int pos,float theta) {
    int pair=threadIdx.x,head=blockIdx.x;if(pair<32)rotate_pair(x+head*d,pair,pos+head/heads,theta);
}
extern "C" int qg_rope(float *x,int d,int heads,int nt,int pos,float theta) {
    if(d<64 || pos<0)return cudaErrorInvalidValue;
    rope_kernel<<<heads*nt,32,0,capture_stream>>>(x,d,heads,nt,pos,theta);return cudaGetLastError();
}
static __global__ void kv_kernel(__half *cache,const float *x,int n,int offset) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)cache[offset+i]=__float2half_rn(x[i]);
}
extern "C" int qg_kv_store(void *cache,const float *x,int width,int nt,int pos) {
    kv_kernel<<<(width*nt+255)/256,256,0,capture_stream>>>((__half*)cache,x,width*nt,pos*width);return cudaGetLastError();
}
/* Each block computes one Q head: scores are parallel in time, followed by a
 * stable softmax and a coalesced reduction over values. Temporary scores live
 * in bounded shared memory, one context row, never a context squared matrix. */
static __global__ void attention_kernel(float *o,const float *q,const __half *k,const __half *v,
                                        const int *allowed,int nt,int pos,int cap) {
    int h=blockIdx.x,t=blockIdx.y,tid=threadIdx.x,limit=pos+t+1,kh=h/12;
    extern __shared__ float scores[];
    __shared__ float maxval,denom;
    for(int p=tid;p<limit;p+=blockDim.x) {
        bool keep=!allowed || allowed[(size_t)t*((cap+3)/4)+p/4];
        float s=-INFINITY;
        if(keep){s=0;for(int d=0;d<256;d++)s+=q[(t*24+h)*256+d]*__half2float(k[p*512+kh*256+d]);s*=0.0625f;}
        scores[p]=s;
    }
    __syncthreads();
    if(!tid){float m=-INFINITY;for(int p=0;p<limit;p++)m=fmaxf(m,scores[p]);maxval=m;}
    __syncthreads();float total=0;
    for(int p=tid;p<limit;p+=blockDim.x){float e=expf(scores[p]-maxval);scores[p]=e;total+=e;}
    float sum=block_sum(total);if(!tid)denom=sum;
    __syncthreads();
    if(tid<256){float value=0;for(int p=0;p<limit;p++)value+=scores[p]*__half2float(v[p*512+kh*256+tid]);
        o[(t*24+h)*256+tid]=value/denom;}
}
/* Long contexts use two score passes through a fixed 16 KiB tile. Preserve
 * the old reduction order (position modulo 256 for the denominator, ascending
 * positions for values), without requiring context-sized shared memory. */
static __global__ void attention_tiled_kernel(float *o,const float *q,const __half *k,const __half *v,
                                              const int *allowed,int nt,int pos,int cap) {
    const int tile=4096;
    int h=blockIdx.x,t=blockIdx.y,tid=threadIdx.x,limit=pos+t+1,kh=h/12;
    __shared__ float scores[tile],maximum,denominator;
    if(!tid)maximum=-INFINITY;
    __syncthreads();
    for(int base=0;base<limit;base+=tile){
        int size=min(tile,limit-base);
        for(int j=tid;j<size;j+=256){int p=base+j;float s=-INFINITY;
            if(!allowed||allowed[(size_t)t*((cap+3)/4)+p/4]){
                s=0;for(int d=0;d<256;d++)s+=q[(t*24+h)*256+d]*__half2float(k[p*512+kh*256+d]);s*=0.0625f;}
            scores[j]=s;}
        __syncthreads();
        if(!tid)for(int j=0;j<size;j++)maximum=fmaxf(maximum,scores[j]);
        __syncthreads();
    }
    float total=0,value=0;
    for(int base=0;base<limit;base+=tile){
        int size=min(tile,limit-base);
        for(int j=tid;j<size;j+=256){int p=base+j;float s=-INFINITY;
            if(!allowed||allowed[(size_t)t*((cap+3)/4)+p/4]){
                s=0;for(int d=0;d<256;d++)s+=q[(t*24+h)*256+d]*__half2float(k[p*512+kh*256+d]);s*=0.0625f;}
            float e=expf(s-maximum);scores[j]=e;total+=e;}
        __syncthreads();
        for(int j=0;j<size;j++)value+=scores[j]*__half2float(v[(base+j)*512+kh*256+tid]);
        __syncthreads();
    }
    float sum=block_sum(total);if(!tid)denominator=sum;
    __syncthreads();o[(t*24+h)*256+tid]=value/denominator;
}
extern "C" int qg_attention(float *o,const float *q,const void *k,const void *v,const int *allowed,int nt,int pos,int cap) {
    if(cap<1 || cap>98304 || pos<0 || nt<1 || nt>8192 || nt>cap || pos>cap-nt)return cudaErrorInvalidValue;
    if(pos+nt<=attention_shared_limit)
        attention_kernel<<<dim3(24,nt),256,(size_t)(pos+nt)*sizeof(float),capture_stream>>>(o,q,(const __half*)k,(const __half*)v,allowed,nt,pos,cap);
    else attention_tiled_kernel<<<dim3(24,nt),256,0,capture_stream>>>(o,q,(const __half*)k,(const __half*)v,allowed,nt,pos,cap);
    return cudaGetLastError();
}
/* Pooled keys are immutable once a block is complete. The incomplete tail is
 * kept separately for rollback; its members are always visible causally. */
static __global__ void index_kernel(float *keys,float *tail,int *allowed,const float *raw,const float *query,
                                    const float *kn,const float *qn,int nt,int pos,int cap,int sort_capacity,float theta,float eps,void *scratch) {
    __shared__ float q[512],pooled[128];
    extern __shared__ float shared_score[];
    float *score=scratch?(float*)scratch:shared_score;
    int *order=(int*)(score+sort_capacity);
    int tid=threadIdx.x,blocks=(cap+3)/4;
    for(int t=0;t<nt;t++) {
        int p=pos+t,filled=p%4;
        if(tid<128)tail[filled*128+tid]=raw[t*128+tid];
        __syncthreads();
        if(filled==3){
            float a=0;if(tid<128){for(int j=0;j<4;j++)a+=tail[j*128+tid];a*=0.25f;pooled[tid]=a;}
            float ss=block_sum(a*a);if(tid<128)pooled[tid]*=rsqrtf(ss/128+eps)*kn[tid];
            __syncthreads();if(tid<32)rotate_pair(pooled,tid,p-3,theta);
            __syncthreads();if(tid<128)keys[(p/4)*128+tid]=pooled[tid];
        }
        for(int h=0;h<4;h++){
            float a=tid<128?query[t*512+h*128+tid]:0;float ss=block_sum(a*a);
            if(tid<128)q[h*128+tid]=a*rsqrtf(ss/128+eps)*qn[tid];
            __syncthreads();if(tid<32)rotate_pair(q+h*128,tid,p,theta);
            __syncthreads();
        }
        int complete=(p+1)/4;
        for(int b=tid;b<blocks;b+=blockDim.x) {
            allowed[(size_t)t*blocks+b]=(b<complete && complete<=512) || (b==complete && filled!=3);
            if(b<complete && complete>512){float total=0;for(int h=0;h<4;h++){float s=0;
                for(int d=0;d<128;d++)s+=q[h*128+d]*keys[b*128+d];total+=fmaxf(s,0.0f);}score[b]=total;}
        }
        __syncthreads();
        /* Bitonic sort replaces quadratic all-pairs ranking at long context.
         * Sort descending score, then ascending original block ID, retaining
         * precisely the same causal top-512 and deterministic tie behavior. */
        if(complete>512){
            int count=1;while(count<complete)count*=2;
            for(int b=tid;b<count;b+=blockDim.x){order[b]=b;if(b>=complete)score[b]=-INFINITY;}
            __syncthreads();
            for(int width=2;width<=count;width*=2)for(int stride=width/2;stride;stride/=2){
                for(int b=tid;b<count;b+=blockDim.x){int other=b^stride;
                    if(other>b){float a=score[b],s=score[other];int ia=order[b],is=order[other];
                        int better=a>s || (a==s && ia<is);
                        if(better==((b&width)!=0)){score[b]=s;score[other]=a;order[b]=is;order[other]=ia;}}}
                __syncthreads();
            }
            for(int b=tid;b<512;b+=blockDim.x)allowed[(size_t)t*blocks+order[b]]=1;
        }
        __syncthreads();
    }
}
extern "C" int qg_index_workspace(float *keys,float *tail,int *allowed,const float *raw,const float *query,const float *kn,const float *qn,int nt,int pos,int cap,float theta,float eps,void *scratch) {
    if(cap<1 || cap>98304 || pos<0 || nt<1 || nt>8192 || nt>cap || pos>cap-nt)return cudaErrorInvalidValue;
    int count=1;while(count<(pos+nt)/4)count*=2;
    if(count<=index_shared_limit)scratch=NULL;
    else if(!scratch)return cudaErrorInvalidValue;
    index_kernel<<<1,256,scratch?0:(size_t)count*8,capture_stream>>>(keys,tail,allowed,raw,query,kn,qn,nt,pos,cap,count,theta,eps,scratch);return cudaGetLastError();
}
extern "C" int qg_index(float *keys,float *tail,int *allowed,const float *raw,const float *query,const float *kn,const float *qn,int nt,int pos,int cap,float theta,float eps) {
    return qg_index_workspace(keys,tail,allowed,raw,query,kn,qn,nt,pos,cap,theta,eps,NULL);
}
extern "C" int qg_prepare_context(int cap) {
    if(cap<1 || cap>98304)return cudaErrorInvalidValue;
    int device,shared;int rc=cudaGetDevice(&device);if(rc)return rc;
    rc=cudaDeviceGetAttribute(&shared,cudaDevAttrMaxSharedMemoryPerBlockOptin,device);if(rc)return rc;
    /* Account for each kernel's static shared allocations, including reductions. */
    cudaFuncAttributes a,b;
    rc=cudaFuncGetAttributes(&a,attention_kernel);if(rc)return rc;
    rc=cudaFuncGetAttributes(&b,index_kernel);if(rc)return rc;
    attention_shared_limit=min(cap,(shared-(int)a.sharedSizeBytes)/4);
    index_shared_limit=1;while(index_shared_limit*2<=(shared-(int)b.sharedSizeBytes)/8)index_shared_limit*=2;
    rc=cudaFuncSetAttribute(attention_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,attention_shared_limit*4);
    if(rc)return rc;
    return cudaFuncSetAttribute(index_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,index_shared_limit*8);
}
static __global__ void route_kernel(int *ids,float *weights,const float *logits,int nt) {
    __shared__ float scores[512];int t=blockIdx.x,i=threadIdx.x;
    scores[i]=logits[t*512+i];__syncthreads();
    if(!i){float largest=-INFINITY;for(int j=0;j<512;j++) {
            if(!isfinite(scores[j])){for(int k=0;k<10;k++){ids[t*10+k]=-1;weights[t*10+k]=0;}return;}
            largest=fmaxf(largest,scores[j]);}
        float total=0;for(int k=0;k<10;k++){int best=0;for(int j=1;j<512;j++)if(scores[j]>scores[best])best=j;
            ids[t*10+k]=best;weights[t*10+k]=expf(scores[best]-largest);total+=weights[t*10+k];scores[best]=-INFINITY;}
        for(int k=0;k<10;k++)weights[t*10+k]/=total;}
}
extern "C" int qg_route(int *ids,float *weights,const float *logits,int nt) {route_kernel<<<nt,512,0,capture_stream>>>(ids,weights,logits,nt);return cudaGetLastError();}
static __global__ void prune_routes_kernel(int *ids,float *weights,const unsigned char *resident,
                                            int nt,double threshold,qg_prune_stats *stats) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=nt)return;
    int missing=0;double mass=0;
    for(int k=0;k<10;k++){int e=ids[t*10+k];if(e<0||e>=512)return;
        if(!resident[e]){missing++;mass+=(double)weights[t*10+k];}}
    atomicAdd((unsigned long long*)&stats->token_layers,1ULL);
    atomicAdd((unsigned long long*)&stats->missing_selections,(unsigned long long)missing);
    /* Negative encodings <= -2 preserve the omitted expert ID. -1 remains
     * the router's invalid-distribution sentinel. No renormalization. */
    if(missing && threshold>0 && mass<threshold){
        for(int k=0;k<10;k++){int e=ids[t*10+k];if(!resident[e]){ids[t*10+k]=-e-2;weights[t*10+k]=0;}}
        atomicAdd((unsigned long long*)&stats->skipped_selections,(unsigned long long)missing);
        atomicAdd((unsigned long long*)&stats->avoided_token_handoffs,1ULL);
        atomicAdd(&stats->skipped_mass,mass);
    }
}
extern "C" int qg_prune_routes(int *ids,float *weights,const unsigned char *resident,
                                int nt,double threshold,qg_prune_stats *stats) {
    if(nt<1||nt>8192||threshold<0||threshold>0.1||!isfinite(threshold))return cudaErrorInvalidValue;
    prune_routes_kernel<<<(nt+127)/128,128,0,capture_stream>>>(ids,weights,resident,nt,threshold,stats);return cudaGetLastError();
}
static __global__ void moe_map_kernel(int *map,const int *ids,int nt) {
    __shared__ int prefix[512],carry;int e=blockIdx.x,tid=threadIdx.x;
    if(!tid)carry=0;__syncthreads();
    /* Scan bounded tiles, retaining token order without launching >512 threads. */
    for(int base=0;base<nt;base+=blockDim.x){
        int t=base+tid,selected=0;if(t<nt)for(int k=0;k<10;k++)selected|=ids[t*10+k]==e;
        prefix[tid]=selected;__syncthreads();
        for(int step=1;step<blockDim.x;step*=2){int v=tid>=step?prefix[tid-step]:0;__syncthreads();prefix[tid]+=v;__syncthreads();}
        if(selected)map[e*nt+carry+prefix[tid]-1]=t;
        __syncthreads();if(!tid)carry+=prefix[blockDim.x-1];__syncthreads();
    }
}
extern "C" int qg_moe_map(int *map,const int *ids,int nt) {
    if(nt<1||nt>8192)return cudaErrorInvalidValue;
    int threads=32;while(threads<nt && threads<512)threads*=2;
    moe_map_kernel<<<512,threads,0,capture_stream>>>(map,ids,nt);return cudaGetLastError();
}
static __global__ void moe_gather_kernel(float *out,const float *x,const int *map) {
    int rank=blockIdx.x,t=map[rank];
    for(int d=threadIdx.x;d<2560;d+=blockDim.x)out[rank*2560+d]=x[t*2560+d];
}
extern "C" int qg_moe_gather(float *o,const float *x,const int *map,int selected) {
    moe_gather_kernel<<<selected,256,0,capture_stream>>>(o,x,map);return cudaGetLastError();
}
static __global__ void moe_scatter_kernel(float *slots,const float *x,const float *w,const int *ids,const int *map,int expert) {
    __shared__ int slot;int rank=blockIdx.x,t=map?map[rank]:rank;
    if(!threadIdx.x){slot=-1;for(int k=0;k<10;k++)if(ids[t*10+k]==expert)slot=k;}__syncthreads();
    if(slot>=0)for(int d=threadIdx.x;d<2560;d+=blockDim.x)slots[(t*10+slot)*2560+d]=x[rank*2560+d]*w[t*10+slot];
}
extern "C" int qg_moe_scatter(float *o,const float *x,const float *w,const int *ids,const int *map,int expert,int selected) {
    moe_scatter_kernel<<<selected,256,0,capture_stream>>>(o,x,w,ids,map,expert);return cudaGetLastError();
}
static __global__ void moe_reduce_kernel(float *out,const float *slots,int nt) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<nt*2560){int t=i/2560,d=i%2560;float sum=0;
        for(int k=0;k<10;k++)sum+=slots[(t*10+k)*2560+d];out[i]=sum;}
}
extern "C" int qg_moe_reduce(float *out,const float *slots,int nt) {
    moe_reduce_kernel<<<(nt*2560+255)/256,256,0,capture_stream>>>(out,slots,nt);return cudaGetLastError();
}
static __global__ void mtp_gate_up_kernel(float *mid,const float *x,const int *ids,const unsigned char *g,int gt,size_t gs,const unsigned char *u,int ut,size_t us) {
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32,rank=blockIdx.y,t=blockIdx.z,e=ids[t*10+rank];
    if(row>=640)return;
    if(e<0||e>=512){if(!lane)mid[(t*10+rank)*640+row]=NAN;return;}
    const unsigned char *gw=g+((size_t)e*640+row)*gs,*uw=u+((size_t)e*640+row)*us;
    float sg=0,su=0;for(int k=lane;k<2560;k+=32){float v=x[t*2560+k];sg+=weight_at(gw,gt,k)*v;su+=weight_at(uw,ut,k)*v;}
    sg=warp_sum(sg);su=warp_sum(su);if(!lane)mid[(t*10+rank)*640+row]=(sg*sig(sg))*su;
}
static __global__ void mtp_down_kernel(float *slots,const float *mid,const int *ids,const float *weights,const unsigned char *d,int type,size_t stride) {
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32,rank=blockIdx.y,t=blockIdx.z,e=ids[t*10+rank];
    if(row>=2560)return;
    if(e<0||e>=512){if(!lane)slots[(t*10+rank)*2560+row]=NAN;return;}
    const unsigned char *w=d+((size_t)e*2560+row)*stride;float sum=0;
    for(int k=lane;k<640;k+=32)sum+=weight_at(w,type,k)*mid[(t*10+rank)*640+k];
    sum=warp_sum(sum);if(!lane)slots[(t*10+rank)*2560+row]=sum*weights[t*10+rank];
}
extern "C" int qg_mtp_moe(float *slots,float *mid,const float *x,const int *ids,const float *weights,const void *g,int gt,const void *u,int ut,const void *d,int dt,int nt) {
    size_t gs=row_bytes(gt,2560),us=row_bytes(ut,2560),ds=row_bytes(dt,640);
    /* Catchup replays the anchor and all sixteen accepted proposals. */
    if(!gs||!us||!ds||nt<1||nt>17)return cudaErrorInvalidValue;
    mtp_gate_up_kernel<<<dim3(80,10,nt),256,0,capture_stream>>>(mid,x,ids,(const unsigned char*)g,gt,gs,(const unsigned char*)u,ut,us);
    cudaError_t rc=cudaGetLastError();if(rc)return rc;
    mtp_down_kernel<<<dim3(320,10,nt),256,0,capture_stream>>>(slots,mid,ids,weights,(const unsigned char*)d,dt,ds);return cudaGetLastError();
}
static __global__ void shared_add_kernel(float *o,const float *x,const float *g,int nt) {int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<nt*2560)o[i]+=x[i]*sig(g[i/2560]);}
extern "C" int qg_shared_add(float *o,const float *x,const float *g,int nt) {shared_add_kernel<<<(nt*2560+255)/256,256,0,capture_stream>>>(o,x,g,nt);return cudaGetLastError();}
static __global__ void ple_gate_kernel(float *o,const float *key,const float *query,const float *v,int nt) {
    int branch=blockIdx.x,tid=threadIdx.x;float s=0;for(int i=tid;i<2560;i+=blockDim.x)s+=key[branch*2560+i]*query[branch*2560+i];
    s=block_sum(s)*0.01976423537605237f;
    float gate=sig(copysignf(sqrtf(fmaxf(fabsf(s),1e-6f)),s));if(s==0)gate=0.5f;
    for(int i=tid;i<2560;i+=blockDim.x)o[branch*2560+i]=v[(branch/4)*2560+i]*gate;
}
extern "C" int qg_ple_gate(float *o,const float *k,const float *q,const float *v,int nt) {ple_gate_kernel<<<nt*4,256,0,capture_stream>>>(o,k,q,v,nt);return cudaGetLastError();}
static __global__ void argmax_kernel(int *ids,const float *x,int n) {
    __shared__ float vals[256];__shared__ int idx[256],bad[256];int tid=threadIdx.x,t=blockIdx.x;float best=-INFINITY;int bi=0,invalid=0;
    for(int i=tid;i<n;i+=256){float v=x[t*n+i];invalid|=isnan(v)||v==INFINITY;if(v>best){best=v;bi=i;}}
    vals[tid]=best;idx[tid]=bi;bad[tid]=invalid;__syncthreads();
    for(int d=128;d;d/=2){if(tid<d){bad[tid]|=bad[tid+d];if(vals[tid+d]>vals[tid] || (vals[tid+d]==vals[tid] && idx[tid+d]<idx[tid])){vals[tid]=vals[tid+d];idx[tid]=idx[tid+d];}}__syncthreads();}
    if(!tid)ids[t]=(bad[0]||!isfinite(vals[0]))?-1:idx[0];
}
extern "C" int qg_argmax(int *ids,const float *x,int n,int nt) {argmax_kernel<<<nt,256,0,capture_stream>>>(ids,x,n);return cudaGetLastError();}

/* Score the actual proposed prefix. At temperature one, p(d)/p(best) is
 * exp(logit[d]-logit[best]); no softmax, sort, or vocabulary upload is needed.
 * Ties use the same lowest-ID order as argmax. The last row is the bonus token. */
static __global__ void verify_kernel(int *ids,int *eligible,const float *x,const int *proposals,
                                     int n,int count,int top_k,float max_gap) {
    __shared__ float vals[256];__shared__ int idx[256],bad[256],ranks[256];
    int tid=threadIdx.x,t=blockIdx.x,d=t<count?proposals[t]:-1;
    float score=d>=0&&d<n?x[t*n+d]:-INFINITY,best=-INFINITY;
    int bi=0,invalid=0,rank=0;
    for(int i=tid;i<n;i+=256){float v=x[t*n+i];invalid|=isnan(v)||v==INFINITY;
        if(v>best){best=v;bi=i;}
        rank+=v>score||(v==score&&i<d);
    }
    vals[tid]=best;idx[tid]=bi;bad[tid]=invalid;ranks[tid]=rank;__syncthreads();
    for(int s=128;s;s/=2){if(tid<s){bad[tid]|=bad[tid+s];ranks[tid]+=ranks[tid+s];
        if(vals[tid+s]>vals[tid]||(vals[tid+s]==vals[tid]&&idx[tid+s]<idx[tid])){vals[tid]=vals[tid+s];idx[tid]=idx[tid+s];}
    }__syncthreads();}
    if(!tid){ids[t]=(bad[0]||!isfinite(vals[0]))?-1:idx[0];
        if(t<count)eligible[t]=ids[t]>=0&&isfinite(score)&&ranks[0]<top_k&&vals[0]-score<=max_gap;
    }
}
extern "C" int qg_verify(int *ids,int *eligible,const float *x,const int *proposals,int n,int count,int top_k,float max_gap) {
    verify_kernel<<<count+1,256,0,capture_stream>>>(ids,eligible,x,proposals,n,count,top_k,max_gap);return cudaGetLastError();
}

/* Restore short causal histories from preserved inputs; no convolution or
 * index ranking is repeated. Completed index blocks remain immutable. */
static __global__ void history_prefix_kernel(float *state,const float *saved,const float *x,int c,int nh,int keep) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c*nh)return;
    int src=i/c+keep;state[i]=src<nh?saved[src*c+i%c]:x[(src-nh)*c+i%c];
}
extern "C" int qg_history_prefix(float *state,const float *saved,const float *x,int c,int nh,int keep) {
    if(!state||!saved||!x||c<1||nh<1||keep<1)return cudaErrorInvalidValue;
    history_prefix_kernel<<<(c*nh+255)/256,256>>>(state,saved,x,c,nh,keep);return cudaGetLastError();
}
static __global__ void index_prefix_kernel(float *tail,const float *saved,const float *raw,int pos,int keep) {
    int i=threadIdx.x,slot=i/128,d=i%128;
    int last=keep-1-((pos+keep-1-slot)%4+4)%4;
    tail[i]=last>=0?raw[last*128+d]:saved[i];
}
extern "C" int qg_index_prefix(float *tail,const float *saved,const float *raw,int pos,int keep) {
    if(!tail||!saved||!raw||pos<0||keep<1)return cudaErrorInvalidValue;
    index_prefix_kernel<<<1,512>>>(tail,saved,raw,pos,keep);return cudaGetLastError();
}
