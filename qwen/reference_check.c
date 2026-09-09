/* Diagnostic adapter to the pinned external llama C API. Never linked into
 * production. Reuses one loaded model across cases, saves first-token logits. */
#define _POSIX_C_SOURCE 200809L
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int top1(const float *logits,int n){
    int best=0;for(int j=0;j<n;j++){if(!isfinite(logits[j]))return -1;if(logits[j]>logits[best])best=j;}return best;
}
int main(int argc,char **argv){
    if(argc!=4){fprintf(stderr,"usage: reference-check MODEL CONTEXT OUTPUT_DIRECTORY\n");return 2;}
    int cap=atoi(argv[2]);if(cap<256||cap>8192)return 2;
    ggml_backend_load_all();llama_backend_init();
    struct llama_model_tensor_buft_override overrides[]={
        {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps",ggml_backend_cpu_buffer_type()},{NULL,NULL}};
    struct llama_model_params mp=llama_model_default_params();
    mp.n_gpu_layers=99;mp.tensor_buft_overrides=overrides;mp.use_extra_bufts=false;
    /* Bulk-read ordinary tensors, avoiding DrvFS's per-page mmap faults.
     * The pinned loader still maps lazy PLE separately in NONE load mode. */
    mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_ON;mp.load_mtp=false;
    struct llama_model *model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
    struct llama_context_params cp=llama_context_default_params();
    cp.n_ctx=cap;cp.n_batch=512;cp.n_ubatch=512;cp.n_seq_max=1;
    cp.n_threads=14;cp.n_threads_batch=14;cp.type_k=cp.type_v=GGML_TYPE_F16;
    struct llama_context *ctx=llama_init_from_model(model,cp);if(!ctx){llama_model_free(model);return 3;}
    int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_token *tokens=malloc((size_t)cap*sizeof(*tokens)),*output=malloc((size_t)cap*sizeof(*output));
    int n,limit,sequence=0,rc=0;
    if(!tokens||!output){rc=4;goto done;}
    while(scanf("%d %d",&n,&limit)==2){
        if(n<1||limit<1||n+limit>cap){rc=2;break;}
        for(int j=0;j<n;j++)if(scanf("%d",tokens+j)!=1||tokens[j]<0||tokens[j]>=vocab){rc=2;break;}
        if(rc)break;
        llama_memory_clear(llama_get_memory(ctx),true);
        double begin=now();
        for(int j=0;j<n;j+=512){int batch=n-j<512?n-j:512;
            if(llama_decode(ctx,llama_batch_get_one(tokens+j,batch))){rc=5;break;}}
        if(rc)break;
        float *logits=llama_get_logits_ith(ctx,-1);if(!logits){rc=5;break;}
        double prefill=now()-begin;int first=top1(logits,vocab);
        char path[4096];if(snprintf(path,sizeof(path),"%s/case-%d.logits.f32",argv[3],sequence)>=(int)sizeof(path)){rc=2;break;}
        FILE *f=fopen(path,"wb");if(!f){rc=6;break;}
        size_t wrote=fwrite(logits,sizeof(float),vocab,f);fclose(f);if(wrote!=(size_t)vocab){rc=6;break;}
        int count=0;begin=now();
        while(count<limit){
            int token=top1(logits,vocab);if(token<0){rc=7;break;}
            if(token==248044||token==248046)break;
            output[count++]=token;if(count==limit)break;
            llama_token next=token;if(llama_decode(ctx,llama_batch_get_one(&next,1))){rc=5;break;}
            logits=llama_get_logits_ith(ctx,-1);if(!logits){rc=5;break;}
        }
        if(rc)break;
        printf("{\"sequence\":%d,\"first_top1\":%d,\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,\"output_ids\":[",sequence++,first,prefill,now()-begin);
        for(int j=0;j<count;j++)printf("%s%d",j?",":"",output[j]);
        printf("]}\n");fflush(stdout);
    }
done:
    free(tokens);free(output);llama_free(ctx);llama_model_free(model);llama_backend_free();return rc;
}
