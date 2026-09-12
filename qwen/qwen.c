#define _POSIX_C_SOURCE 200809L
#include "qwen.h"
#include "gpu.h"
#include "cpu.h"
#include "ssd.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { D=2560, HD=10240, V=248320, NL=48, NT=8192, LOGIT_ROWS=QWEN_VERIFY_ROWS, MAX_TENSORS=1600, MAX_FILES=16,
       PLE_BLOCK=4096, PLE_SLOTS=1024 };
typedef struct {char name[128];const unsigned char *host;unsigned char *owned_host;void *device;uint64_t bytes,offset,dim[4];int type,file;} tensor;
typedef struct {int fd;unsigned char *data;uint64_t bytes;} mapped_file;
typedef struct {tensor *norm,*down,*up,*inject;void *graphs[LOGIT_ROWS+1];} mixer;
typedef struct {
    mixer attn,ffn;
    tensor *qkv,*z,*alpha,*beta,*a,*dt,*conv,*norm,*out;
    tensor *q,*k,*v,*o,*qn,*kn,*iq,*ik,*iqn,*ikn;
    tensor *router,*eg,*eu,*ed,*sg,*su,*sd,*sr;
    tensor *ple_k,*ple_v,*ple_kn,*ple_qn,*ple_cn,*ple_conv;
    tensor *enorm,*hnorm,*eh; mixer head;
    float *gdn,*gdn_save,*conv_state,*conv_save,*ple_state,*ple_save;
    float *trace_raw,*trace_qkv,*trace_alpha,*trace_beta,*trace_index,*trace_ple;
    void *shared_graphs[LOGIT_ROWS+1];
    void *kc,*vc;
    float *index_keys,*index_tail,*index_tail_save;
    unsigned char *expert_cache;
    uint64_t expert_bytes,gate_bytes,up_bytes,down_bytes;
    int *cache_ids;
    uint64_t *cache_age,clock;
} layer;
struct qwen_model {
    tensor tensors[MAX_TENSORS];int n_tensors;
    mapped_file files[MAX_FILES];int n_files,lock_fd;
    layer layers[NL+1];tensor *embedding,*output,*ngram;mixer head;
    int context,resident,batch_capacity,first_layer,last_layer,target_pos,mtp_pos,last_target_rows,last_mtp_rows,last_target_batch;
    int save_target_pos,save_mtp_pos,has_checkpoint,tracing,trace_rows;
    qwen_optimization_stats optimization;
    int *tokens;
    uint64_t multiplier[3],head_offset[16],head_vocab[16];int ple_eos;
    float *h,*hn,*gate,*low,*inject,*x,*y,*qkv,*z,*alpha,*beta,*q,*k,*v,*qgate,*iq,*ik;
    float *expert_x,*ffg,*ffu,*ffmid,*ffout,*moeout,*moe_slots,*mtp_mid,*shared,*shared_gate,*route_logits,*route_weights;
    int *route_ids,*route_map,*allowed,*device_top1,*verify_ids;
    float *ple_emb,*ple_key,*ple_query,*ple_value,*ple_gated,*ple_normal,*ple_convout;
    unsigned char *ple_packed,*host_ple_packed,*index_scratch;
    float *target_h,*target_last,*previous_h,*mtp_h,*mtp_last,*mtp_input_h,*concat;
    float *target_last_save,*previous_h_save,*mtp_last_save;
    float *target_logits,*mtp_logits;
    unsigned char *arenas[3];uint64_t arena_capacity[3],arena_used[3];
    void *uploader;
    qwen_stats stats;
    qwen_handoff_profile handoff_profile;
    FILE *route_profile;
    qc_workspace *cpu;
    float *cpu_x,*cpu_out,*cpu_weights;
    int fixed_hotlist,cpu_reduced;
    qwen_cpu_stats cpu_stats;
    unsigned char *resident_masks;
    qg_prune_stats *prune_device;
    qs_store *ssd;
    const int *storage_ranking; /* Used only during synchronous open. */
    int ram_experts,ple_on_ssd,io_threads,storage_failed;
    uint64_t ssd_buffer_bytes;
    unsigned char *ple_cache;
    uint64_t ple_cache_keys[PLE_SLOTS];
    qwen_storage_stats storage_stats;
};
static _Thread_local char last_error[512];
static int fail(const char *fmt,...) {va_list ap;va_start(ap,fmt);vsnprintf(last_error,sizeof(last_error),fmt,ap);va_end(ap);return -1;}
const char *qwen_last_error(void){return last_error;}
static double now(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
#define GPU(call) do{int rc=(call);if(rc)return fail("%s: %s",#call,qg_error(rc));}while(0)
#define RUN(call) do{if((call))return -1;}while(0)

/* Diagnostic only: preserve every selected route, including its token position.
 * The caller separates prefill/decode and excludes speculative rollback work. */
int qwen_profile_routes(qwen_model *m,const char *path){
    if(!m)return fail("route profile requires an open model");
    if(m->route_profile){FILE *old=m->route_profile;m->route_profile=NULL;if(fclose(old))return fail("close route profile: %s",strerror(errno));}
    if(!path)return 0;
    FILE *f=fopen(path,"wb");if(!f)return fail("open route profile: %s",strerror(errno));
    if(fwrite("QROUTE1\n",1,8,f)!=8){fclose(f);return fail("write route profile header");}
    m->route_profile=f;return 0;
}

static tensor *lookup(qwen_model *m,const char *name){for(int i=0;i<m->n_tensors;i++)if(!strcmp(m->tensors[i].name,name))return &m->tensors[i];return NULL;}
static tensor *at(qwen_model *m,int il,const char *suffix){char name[128];snprintf(name,sizeof(name),"blk.%d.%s",il,suffix);return lookup(m,name);}
static int shape(tensor *t,uint64_t d0,uint64_t d1,uint64_t d2){return t && t->dim[0]==d0 && t->dim[1]==d1 && t->dim[2]==d2 && t->dim[3]==1;}
static int load_mixer(qwen_model *m,mixer *mix,int il,const char *prefix){char s[100];
    snprintf(s,sizeof(s),"%s_norm.weight",prefix);mix->norm=il<0?lookup(m,s):at(m,il,s);
    snprintf(s,sizeof(s),"%s_down.weight",prefix);mix->down=il<0?lookup(m,s):at(m,il,s);
    snprintf(s,sizeof(s),"%s_up.weight",prefix);mix->up=il<0?lookup(m,s):at(m,il,s);
    snprintf(s,sizeof(s),"%s_inject.weight",prefix);mix->inject=il<0?lookup(m,s):at(m,il,s);
    if(!shape(mix->norm,HD,1,1)||!shape(mix->down,HD,320,1)||!shape(mix->up,320,HD,1))return fail("invalid mixer %d/%s",il,prefix);
    if(mix->inject && !shape(mix->inject,HD,4,1))return fail("invalid mixer injection");
    return 0;
}
static int map_index(qwen_model *m,const char *path){
    FILE *f=fopen(path,"r");if(!f)return fail("open index: %s",strerror(errno));
    char *line=NULL;size_t cap=0;ssize_t n=getline(&line,&cap,f);
    if(n<0 || strcmp(line,"COB_QWEN_INDEX_V1\n")){free(line);fclose(f);return fail("invalid Qwen index version");}
    n=getline(&line,&cap,f);
    if(n<0 || strcmp(line,"VERIFIED 1\n")){free(line);fclose(f);return fail("weights must pass SHA256 verification before loading");}
    for(int j=0;j<3;j++)if(fscanf(f,"%"SCNu64,&m->multiplier[j])!=1)goto bad;
    for(int j=0;j<16;j++)if(fscanf(f,"%"SCNu64,&m->head_offset[j])!=1)goto bad;
    for(int j=0;j<16;j++)if(fscanf(f,"%"SCNu64,&m->head_vocab[j])!=1 || !m->head_vocab[j])goto bad;
    if(fscanf(f,"%d",&m->ple_eos)!=1)goto bad;
    while((n=getline(&line,&cap,f))>=0){
        if(!strncmp(line,"FILE\t",5)){
            int id;uint64_t bytes;int used;
            if(sscanf(line,"FILE\t%d\t%"SCNu64"\t%n",&id,&bytes,&used)!=2 || id<0 || id>=MAX_FILES || m->files[id].data)goto bad;
            char *p=line+used;p[strcspn(p,"\r\n")]=0;
            mapped_file *mf=&m->files[id];mf->fd=open(p,O_RDONLY);
            struct stat st;if(mf->fd<0 || fstat(mf->fd,&st) || (uint64_t)st.st_size!=bytes){fail("weights missing/incomplete: %s",p);goto failed;}
            mf->data=mmap(NULL,bytes,PROT_READ,MAP_PRIVATE,mf->fd,0);mf->bytes=bytes;
            if(mf->data==MAP_FAILED){mf->data=NULL;fail("mmap: %s",strerror(errno));goto failed;}
            /* The PLE lookup has no useful sequential read-ahead; individual
             * expert uploads explicitly touch only their own contiguous rows. */
            posix_madvise(mf->data,bytes,POSIX_MADV_RANDOM);
            if(id>=m->n_files)m->n_files=id+1;
        }else if(!strncmp(line,"TENSOR\t",7)){
            if(m->n_tensors>=MAX_TENSORS)goto bad;
            tensor *t=&m->tensors[m->n_tensors];uint64_t offset;
            if(sscanf(line,"TENSOR\t%127[^\t]\t%d\t%"SCNu64"\t%"SCNu64"\t%d\t%"SCNu64"\t%"SCNu64"\t%"SCNu64"\t%"SCNu64,
                      t->name,&t->file,&offset,&t->bytes,&t->type,&t->dim[0],&t->dim[1],&t->dim[2],&t->dim[3])!=9)goto bad;
            if(t->file<0 || t->file>=m->n_files || !m->files[t->file].data || !t->bytes ||
               offset>m->files[t->file].bytes || t->bytes>m->files[t->file].bytes-offset || lookup(m,t->name))goto bad;
            uint64_t expected_bytes;
            if(qg_tensor_bytes(t->type,t->dim,&expected_bytes) || expected_bytes!=t->bytes)goto bad;
            t->host=m->files[t->file].data+offset;t->offset=offset;m->n_tensors++;
        }else if(line[0]!='\n' && line[0]!='\r')goto bad;
    }
    free(line);fclose(f);return 0;
bad:fail("malformed or unsupported Qwen index");
failed:free(line);fclose(f);return -1;
}
static int bind_model(qwen_model *m){
    m->embedding=lookup(m,"token_embd.weight");m->output=lookup(m,"output.weight");m->ngram=lookup(m,"per_layer_token_embd.weight");
    if((m->first_layer==0||m->first_layer==NL) && (!shape(m->embedding,D,V,1)||!shape(m->output,D,V,1)))return fail("invalid target lexical tensors");
    if(m->first_layer==0){
        if(!m->ngram || m->ngram->dim[0]!=160 || m->ngram->type!=20)return fail("invalid target PLE tensor");
        RUN(load_mixer(m,&m->head,-1,"output_hc"));
    }
    for(int il=m->first_layer;il<=m->last_layer;il++){
        layer *l=&m->layers[il];RUN(load_mixer(m,&l->attn,il,"hc_attn"));RUN(load_mixer(m,&l->ffn,il,"hc_ffn"));
        if(!l->attn.inject || !l->ffn.inject)return fail("missing layer injection");
#define B(field,name) l->field=at(m,il,name)
        B(router,"ffn_gate_inp.weight");B(eg,"ffn_gate_exps.weight");B(eu,"ffn_up_exps.weight");B(ed,"ffn_down_exps.weight");
        B(sg,"ffn_gate_shexp.weight");B(su,"ffn_up_shexp.weight");B(sd,"ffn_down_shexp.weight");B(sr,"ffn_gate_inp_shexp.weight");
        if(!shape(l->router,D,512,1)||!shape(l->eg,D,640,512)||!shape(l->eu,D,640,512)||!shape(l->ed,640,D,512)||
           !shape(l->sg,D,640,1)||!shape(l->su,D,640,1)||!shape(l->sd,640,D,1)||!shape(l->sr,D,1,1))return fail("invalid MoE layer %d",il);
        l->gate_bytes=l->eg->bytes/512;l->up_bytes=l->eu->bytes/512;l->down_bytes=l->ed->bytes/512;
        l->expert_bytes=l->gate_bytes+l->up_bytes+l->down_bytes;
        if(il<NL && il%4!=3){
            B(qkv,"attn_qkv.weight");B(z,"attn_gate.weight");B(alpha,"ssm_alpha.weight");B(beta,"ssm_beta.weight");
            B(a,"ssm_a");B(dt,"ssm_dt.bias");B(conv,"ssm_conv1d.weight");B(norm,"ssm_norm.weight");B(out,"ssm_out.weight");
            if(!shape(l->qkv,D,10240,1)||!shape(l->z,D,6144,1)||!shape(l->alpha,D,48,1)||!shape(l->beta,D,48,1)||
               !shape(l->a,48,1,1)||!shape(l->dt,48,1,1)||!shape(l->conv,4,10240,1)||!shape(l->norm,128,1,1)||!shape(l->out,6144,D,1))return fail("invalid GDN layer %d",il);
        }else{
            B(q,"attn_q.weight");B(k,"attn_k.weight");B(v,"attn_v.weight");B(o,"attn_output.weight");B(qn,"attn_q_norm.weight");B(kn,"attn_k_norm.weight");
            B(iq,"indexer.q_proj.weight");B(ik,"indexer.k_proj.weight");B(iqn,"indexer.q_norm.weight");B(ikn,"indexer.k_norm.weight");
            if(!shape(l->q,D,12288,1)||!shape(l->k,D,512,1)||!shape(l->v,D,512,1)||!shape(l->o,6144,D,1)||!shape(l->qn,256,1,1)||!shape(l->kn,256,1,1))return fail("invalid QSA layer %d",il);
            if(il<NL && (!shape(l->iq,D,512,1)||!shape(l->ik,D,128,1)||!shape(l->iqn,128,1,1)||!shape(l->ikn,128,1,1)))return fail("invalid indexer layer %d",il);
        }
        if(il==1){
            B(ple_k,"ple_key.weight");B(ple_v,"ple_value.weight");B(ple_kn,"ple_norm_key.weight");B(ple_qn,"ple_norm_query.weight");B(ple_cn,"ple_norm_conv.weight");B(ple_conv,"ple_conv1d.weight");
            if(!shape(l->ple_k,D,HD,1)||!shape(l->ple_v,D,D,1)||!shape(l->ple_kn,HD,1,1)||!shape(l->ple_qn,HD,1,1)||!shape(l->ple_cn,HD,1,1)||!shape(l->ple_conv,4,HD,1))return fail("invalid PLE tensors");
        }
        if(il==NL){B(enorm,"nextn.enorm.weight");B(hnorm,"nextn.hnorm.weight");B(eh,"nextn.eh_proj.weight");
            RUN(load_mixer(m,&l->head,il,"nextn.hc_head"));
            if(!shape(l->enorm,D,1,1)||!shape(l->hnorm,HD,1,1)||!shape(l->eh,2*D,D,1))return fail("invalid MTP input tensors");}
#undef B
    }
    return 0;
}
static int is_target_expert(const tensor *t){int il;return sscanf(t->name,"blk.%d.",&il)==1 && il<NL && strstr(t->name,"_exps.");}
static int read_tensor(qwen_model *m,tensor *t,void *out,uint64_t offset,size_t bytes){
    size_t done=0;
    while(done<bytes){ssize_t n=pread(m->files[t->file].fd,(unsigned char*)out+done,bytes-done,(off_t)(t->offset+offset+done));
        if(n<0 && errno==EINTR)continue;
        if(n<=0)return fail("read tensor %s: %s",t->name,n<0?strerror(errno):"unexpected EOF");
        done+=(size_t)n;}
    return 0;
}
static uint64_t host_weight_budget(qwen_model *m){
    uint64_t required=0;for(int i=0;i<m->n_tensors;i++)if(is_target_expert(&m->tensors[i]))required+=m->tensors[i].bytes;
    if(m->storage_ranking||m->ssd)required=required/512*m->ram_experts+m->ssd_buffer_bytes;
    if(m->first_layer==0&&m->ngram)required+=m->ple_on_ssd?4ULL*1024*1024:m->ngram->bytes;
    return required;
}
static uint64_t host_runtime_budget(qwen_model *m){
    /* qc_workspace, caller activation buffers and PLE packed staging. Reserve
     * another 2 GiB for Python, OS, allocator/pinned staging and kernel code. */
    uint64_t per_token=(2ULL*D+3*640+10*D)*4+512*4+2*D*4+10*4+16*90;
    return (uint64_t)m->batch_capacity*per_token+17ULL*10*640*4+2ULL*1024*1024*1024;
}
static int check_host_budget(qwen_model *m){
    uint64_t required=host_weight_budget(m),available=0;
    if(!required)return 0;
    FILE *mem=fopen("/proc/meminfo","r");char line[256];
    if(mem){while(fgets(line,sizeof(line),mem))if(sscanf(line,"MemAvailable: %"SCNu64" kB",&available)==1){available*=1024;break;}fclose(mem);}
    uint64_t runtime=host_runtime_budget(m);
    if(!available || required+runtime>available)return fail("configured storage needs %.3f GiB RAM plus %.3f GiB runtime; available %.3f GiB; free another %.3f GiB",required/1073741824.,runtime/1073741824.,available/1073741824.,(required+runtime>available?required+runtime-available:0)/1073741824.);
    return 0;
}
static int prepare_host_weights(qwen_model *m){
    uint64_t required=host_weight_budget(m),routed=0;
    if(!required)return 0;
    RUN(check_host_budget(m));
    double begin=now();uint64_t loaded=0;
    if(m->storage_ranking){
        qs_source sources[NL*3];
        for(int il=0;il<NL;il++){
            layer *l=&m->layers[il];tensor *ts[3]={l->eg,l->eu,l->ed};
            for(int k=0;k<3;k++)sources[il*3+k]=(qs_source){m->files[ts[k]->file].fd,ts[k]->offset,ts[k]->bytes/512};
        }
        char error[256];m->ssd=qs_create(sources,NL,m->storage_ranking,m->ram_experts,m->ssd_buffer_bytes,m->io_threads,error);
        if(!m->ssd)return fail("%s",error);
        for(int il=0;il<NL;il++){
            layer *l=&m->layers[il];tensor *ts[3]={l->eg,l->eu,l->ed};
            for(int k=0;k<3;k++){ts[k]->host=qs_tensor(m->ssd,il,k);routed+=ts[k]->bytes/512*m->ram_experts;}
        }
        loaded=routed;
    }
    if(m->ple_on_ssd&&m->first_layer==0){
        m->ple_cache=malloc(4ULL*1024*1024);if(!m->ple_cache)return fail("allocate PLE SSD cache");
    }
    /* Read once into anonymous RAM, including the complete quantized PLE.
     * Inference must not fault its random n-gram rows through WSL's DrvFS.
     * The source files remain intact and GPU allocations are unchanged. */
    for(int i=0;i<m->n_tensors;i++){
        tensor *t=&m->tensors[i];int expert=is_target_expert(t),ple=m->first_layer==0&&t==m->ngram;
        if(!expert&&!ple)continue;
        if((expert&&m->ssd)||(ple&&m->ple_on_ssd))continue;
        t->owned_host=malloc(t->bytes);if(!t->owned_host)return fail("allocate host tensor %s",t->name);
        uint64_t reported=0;
        for(uint64_t offset=0;offset<t->bytes;){size_t n=t->bytes-offset>64ULL*1024*1024?64ULL*1024*1024:(size_t)(t->bytes-offset);
            RUN(read_tensor(m,t,t->owned_host+offset,offset,n));offset+=n;
            if(ple&&(offset-reported>=1024ULL*1024*1024||offset==t->bytes)){
                fprintf(stderr,"qwen: host RAM %.2f / %.2f GiB (PLE %.2f / %.2f GiB)\n",(loaded+offset)/1073741824.,required/1073741824.,offset/1073741824.,t->bytes/1073741824.);reported=offset;}}
        t->host=t->owned_host;loaded+=t->bytes;if(expert)routed+=t->bytes;
        if(strstr(t->name,"ffn_up_exps"))fprintf(stderr,"qwen: host RAM %.2f / %.2f GiB\n",loaded/1073741824.,required/1073741824.);
    }
    m->stats.host_weight_bytes=routed;m->stats.host_load_seconds=now()-begin;return 0;
}
uint64_t qwen_ple_ram_bytes(qwen_model *m){return m&&m->first_layer==0&&m->ngram&&m->ngram->owned_host?m->ngram->bytes:0;}
static uint64_t aligned(uint64_t n){return (n+255)&~255ULL;}
static void *arena_take(qwen_model *m,int arena,size_t bytes){
    uint64_t reserved=aligned(bytes),used=m->arena_used[arena];
    if(reserved>m->arena_capacity[arena]-used){fail("GPU arena %d capacity exceeded",arena);return NULL;}
    m->arena_used[arena]+=reserved;return m->arenas[arena]+used;
}
static int workspace(qwen_model *m,int plan){
    int nt=m->batch_capacity;
    uint64_t offset=0,shared,peak=0;
#define WORK(field,count,type) do{if(!plan)m->field=(void*)(m->arenas[2]+offset);offset+=aligned((size_t)(count)*sizeof(type));}while(0)
#define PHASE() do{if(offset>peak)peak=offset;offset=shared;}while(0)
    /* Persistent across layer phases, or consumed by hc() between phases. */
    WORK(h,nt*HD,float);WORK(hn,nt*HD,float);WORK(gate,nt*HD,float);WORK(low,nt*320,float);WORK(inject,nt*4,float);
    WORK(x,nt*D,float);WORK(y,nt*D,float);
    WORK(target_h,nt*HD,float);WORK(target_last,HD,float);WORK(previous_h,HD,float);WORK(mtp_h,nt*HD,float);WORK(mtp_last,HD,float);WORK(mtp_input_h,nt*HD,float);
    WORK(target_last_save,HD,float);WORK(previous_h_save,HD,float);WORK(mtp_last_save,HD,float);
    WORK(target_logits,LOGIT_ROWS*V,float);WORK(mtp_logits,LOGIT_ROWS*V,float);WORK(device_top1,nt,int);
    WORK(verify_ids,2*LOGIT_ROWS,int);
    /* PLE, attention, MoE and the MTP input projection execute sequentially
     * on the same CUDA stream. Their scratch ranges may overlap; recurrent
     * states, speculative hidden states and logits above must never overlap.
     * Planning and pointer assignment use exactly the same aligned offsets. */
    shared=offset;
    WORK(ple_emb,nt*D,float);WORK(ple_key,nt*HD,float);WORK(ple_query,nt*HD,float);WORK(ple_value,nt*D,float);WORK(ple_gated,nt*HD,float);WORK(ple_normal,nt*HD,float);WORK(ple_convout,nt*HD,float);
    WORK(ple_packed,nt*16*90,unsigned char);
    PHASE();
    WORK(qkv,nt*12288,float);WORK(z,nt*6144,float);WORK(alpha,nt*48,float);WORK(beta,nt*48,float);
    WORK(q,nt*6144,float);WORK(k,nt*512,float);WORK(v,nt*512,float);WORK(qgate,nt*6144,float);WORK(iq,nt*512,float);WORK(ik,nt*128,float);
    WORK(allowed,nt*((m->context+3)/4),int);
    WORK(index_scratch,32768*8,unsigned char);
    PHASE();
    WORK(ffg,nt*640,float);WORK(ffu,nt*640,float);WORK(ffmid,nt*640,float);WORK(ffout,nt*D,float);WORK(moeout,nt*D,float);WORK(shared,nt*D,float);WORK(shared_gate,nt,float);
    WORK(moe_slots,nt*10*D,float);WORK(expert_x,nt*D,float);
    WORK(mtp_mid,LOGIT_ROWS*10*640,float);
    WORK(route_logits,nt*512,float);WORK(route_weights,nt*10,float);WORK(route_ids,nt*10,int);
    WORK(route_map,nt*512,int);
    PHASE();
    WORK(concat,nt*HD*2,float);
    PHASE();
#undef PHASE
#undef WORK
    if(plan)m->arena_capacity[2]=peak;
    else{
        if(peak!=m->arena_capacity[2])return fail("workspace plan changed");
        m->arena_used[2]=peak;m->stats.gpu_workspace_bytes=peak;
    }
    return 0;
}
/* Plan and assign the same individually aligned ranges. Small alpha/beta
 * traces are not multiples of 256 bytes, unlike the original state arrays. */
static int state_layout(qwen_model *m,int plan){
    uint64_t offset=0;
    for(int il=m->first_layer;il<=m->last_layer;il++){
        layer *l=&m->layers[il];
#define STATE(field,bytes) do{uint64_t size=(bytes);if(!plan){l->field=(void*)(m->arenas[1]+offset);m->stats.gpu_state_bytes+=size;}offset+=aligned(size);}while(0)
        if(l->qkv){STATE(gdn,48ULL*128*128*4);STATE(gdn_save,48ULL*128*128*4);STATE(conv_state,3ULL*10240*4);STATE(conv_save,3ULL*10240*4);
            STATE(trace_raw,LOGIT_ROWS*10240ULL*4);STATE(trace_qkv,LOGIT_ROWS*10240ULL*4);
            STATE(trace_alpha,LOGIT_ROWS*48ULL*4);STATE(trace_beta,LOGIT_ROWS*48ULL*4);}
        else{STATE(kc,(uint64_t)m->context*512*2);STATE(vc,(uint64_t)m->context*512*2);
            if(il<NL){STATE(index_keys,(uint64_t)((m->context+3)/4)*128*4);STATE(index_tail,4*128*4);STATE(index_tail_save,4*128*4);STATE(trace_index,LOGIT_ROWS*128*4);}}
        if(il==1){STATE(ple_state,9ULL*HD*4);STATE(ple_save,9ULL*HD*4);STATE(trace_ple,LOGIT_ROWS*HD*4);}
#undef STATE
    }
    if(plan)m->arena_capacity[1]=offset;
    else{if(offset!=m->arena_capacity[1])return fail("state plan changed");m->arena_used[1]=offset;}
    return 0;
}
static int prepare_device(qwen_model *m){
    GPU(qg_prepare_context(m->context));
    uint64_t weights=0;for(int i=0;i<m->n_tensors;i++)if(&m->tensors[i]!=m->ngram && !is_target_expert(&m->tensors[i]))weights+=aligned(m->tensors[i].bytes);
    for(int il=m->first_layer;il<=m->last_layer && il<NL;il++)weights+=aligned(m->resident*m->layers[il].expert_bytes);
    size_t free_bytes,total;GPU(qg_memory(&free_bytes,&total));
    RUN(state_layout(m,1));uint64_t states=m->arena_capacity[1];
    RUN(workspace(m,1));
    if(weights+states+m->arena_capacity[2]+512ULL*1024*1024>free_bytes)return fail("GPU budget exceeds available memory: weights %.3f GiB, states %.3f GiB, workspace %.3f GiB, free %.3f GiB",weights/1073741824.,states/1073741824.,m->arena_capacity[2]/1073741824.,free_bytes/1073741824.);
    fprintf(stderr,"qwen: weights/cache %.3f GiB, states estimate %.3f GiB, CUDA free %.3f GiB\n",weights/1073741824.,states/1073741824.,free_bytes/1073741824.);
    /* Three aligned arenas avoid hundreds of MiB of driver granularity waste
     * from >1300 small cudaMalloc calls. MTP lexical tensors remain aliases. */
    m->arena_capacity[0]=weights;m->arena_capacity[1]=states;
    for(int j=0;j<3;j++){
        m->arenas[j]=qg_alloc(m->arena_capacity[j]);if(!m->arenas[j])return fail("CUDA arena %d allocation failed",j);
        m->stats.gpu_arena_bytes+=m->arena_capacity[j];GPU(qg_zero(m->arenas[j],m->arena_capacity[j]));
    }
    size_t stage_bytes=16ULL*1024*1024;unsigned char *stage=malloc(stage_bytes);
    if(!stage)return fail("weight staging allocation");
    double load_start=now();int load_failed=0;
    for(int i=0;i<m->n_tensors;i++){
        tensor *t=&m->tensors[i];if(t==m->ngram || is_target_expert(t))continue;
        t->device=arena_take(m,0,t->bytes);if(!t->device){load_failed=1;break;}
        for(uint64_t offset=0;offset<t->bytes;){size_t n=t->bytes-offset>stage_bytes?stage_bytes:(size_t)(t->bytes-offset);
            if(read_tensor(m,t,stage,offset,n)){load_failed=1;break;}
            int rc=qg_write((unsigned char*)t->device+offset,stage,n);
            if(rc){fail("upload tensor %s: %s",t->name,qg_error(rc));load_failed=1;break;}offset+=n;}
        if(load_failed)break;
        m->stats.gpu_weight_bytes+=t->bytes;
    }
    free(stage);if(load_failed)return -1;
    m->stats.core_load_seconds=now()-load_start;
    for(int il=m->first_layer;il<=m->last_layer;il++){
        layer *l=&m->layers[il];
        if(il<NL){l->expert_cache=arena_take(m,0,m->resident*l->expert_bytes);if(!l->expert_cache)return -1;m->stats.gpu_weight_bytes+=m->resident*l->expert_bytes;
            l->cache_ids=malloc(m->resident*sizeof(int));l->cache_age=calloc(m->resident,sizeof(uint64_t));
            if(!l->cache_ids||!l->cache_age)return fail("expert metadata allocation");
            for(int j=0;j<m->resident;j++)l->cache_ids[j]=-1;}

    }
    RUN(state_layout(m,0));
    RUN(workspace(m,0));
    uint64_t largest=0;
    for(int il=m->first_layer;il<=m->last_layer && il<NL;il++)if(m->layers[il].expert_bytes>largest)largest=m->layers[il].expert_bytes;
    if(largest){m->uploader=qg_upload_create(largest);if(!m->uploader)return fail("pinned expert staging allocation");m->stats.host_staging_bytes=2*largest;}
    GPU(qg_sync());return prepare_host_weights(m);
}
static qwen_model *open_model(const char *index_path,int context,int resident,int first_layer,int last_layer,int batch_capacity,
        const int *ranking,int ram_experts,int ple_on_ssd,uint64_t buffer_bytes,int io_threads){
    last_error[0]=0;if(context<256||context>98304||resident<10||resident>128){fail("supported context 256..98304, expert cache 10..128 per layer");return NULL;}
    if(batch_capacity<1||batch_capacity>NT){fail("prefill capacity must be 1..8192");return NULL;}
    if(ram_experts<resident||ram_experts>512||ple_on_ssd<0||ple_on_ssd>1||io_threads<1||io_threads>4||
       (ram_experts<512&&(!ranking||first_layer!=0||last_layer!=NL))){fail("invalid host storage configuration");return NULL;}
    qwen_model *m=calloc(1,sizeof(*m));if(!m){fail("model allocation");return NULL;}
    for(int i=0;i<MAX_FILES;i++)m->files[i].fd=-1;
    m->lock_fd=-1;m->context=context;m->resident=resident;m->first_layer=first_layer;m->last_layer=last_layer;
    m->ram_experts=ram_experts;m->ple_on_ssd=ple_on_ssd;m->ssd_buffer_bytes=buffer_bytes;m->io_threads=io_threads;
    m->storage_ranking=ram_experts<512?ranking:NULL;
    m->batch_capacity=batch_capacity<LOGIT_ROWS?LOGIT_ROWS:batch_capacity;
    char lock_path[4096];if(snprintf(lock_path,sizeof(lock_path),"%s.lock",index_path)>=(int)sizeof(lock_path)){fail("index path too long");goto bad;}
    m->lock_fd=open(lock_path,O_CREAT|O_RDWR,0600);if(m->lock_fd<0 || flock(m->lock_fd,LOCK_EX|LOCK_NB)){fail("another Qwen process holds the model lock");goto bad;}
    m->tokens=calloc(context,sizeof(int));if(!m->tokens){fail("token history allocation");goto bad;}
    m->host_ple_packed=malloc((size_t)m->batch_capacity*16*90);
    if(!m->host_ple_packed){fail("PLE host staging allocation");goto bad;}
    if(map_index(m,index_path)||bind_model(m)||check_host_budget(m)||prepare_device(m))goto bad;
    m->stats.context=context;m->stats.resident_per_layer=resident;
    m->storage_ranking=NULL;
    if(first_layer==0||first_layer==NL)fprintf(stderr,"qwen: %d host experts/layer; PLE %s; MTP shares target lexical allocations\n",ram_experts,ple_on_ssd?"SSD with 4 MiB cache":"RAM");
    else fprintf(stderr,"qwen: diagnostic target layer %d, routed payloads in RAM\n",first_layer);
    return m;
bad:qwen_close(m);return NULL;
}
int qwen_validate_index(const char *index_path){
    last_error[0]=0;
    qwen_model *m=calloc(1,sizeof(*m));if(!m)return fail("metadata allocation failed");
    m->lock_fd=-1;m->last_layer=NL;
    for(int i=0;i<MAX_FILES;i++)m->files[i].fd=-1;
    int result=map_index(m,index_path);
    if(!result)result=bind_model(m);
    /* No CUDA context or physical weight allocation is required for preflight. */
    for(int i=0;i<MAX_FILES;i++){
        if(m->files[i].data)munmap(m->files[i].data,m->files[i].bytes);
        if(m->files[i].fd>=0)close(m->files[i].fd);
    }
    free(m);return result;
}
qwen_model *qwen_open(const char *index_path,int context,int resident,int batch_capacity){return open_model(index_path,context,resident,0,NL,batch_capacity,NULL,512,0,0,2);}
qwen_model *qwen_open_storage(const char *path,int context,int resident,int batch,const int *ranking,int ram,int ple_ssd,uint64_t buffer,int threads){
    return open_model(path,context,resident,0,NL,batch,ranking,ram,ple_ssd,buffer,threads);
}
void qwen_get_storage_stats(qwen_model *m,qwen_storage_stats *out){
    if(!out)return;
    memset(out,0,sizeof(*out));if(!m)return;
    *out=m->storage_stats;out->ram_experts_per_layer=m->ram_experts;out->ple_on_ssd=m->ple_on_ssd;
    out->ram_expert_bytes=m->stats.host_weight_bytes;
    if(m->ssd){qs_stats s;qs_get_stats(m->ssd,&s);out->transient_peak_bytes=s.transient_peak_bytes;
        out->expert_read_bytes=s.read_bytes;out->expert_reads=s.expert_reads;
        out->expert_io_seconds=s.io_seconds;out->expert_wait_seconds=s.wait_seconds;}
}
#ifdef QWEN_TEST_HOOKS
uint64_t qwen_probe_workspace_bytes(int context,int batch_capacity){
    if(context<256||context>98304||batch_capacity<1||batch_capacity>NT)return 0;
    qwen_model *m=calloc(1,sizeof(*m));if(!m)return 0;
    m->context=context;m->batch_capacity=batch_capacity<LOGIT_ROWS?LOGIT_ROWS:batch_capacity;
    workspace(m,1);uint64_t bytes=m->arena_capacity[2];free(m);return bytes;
}
qwen_model *qwen_probe_open(const char *index_path,int batch_capacity){return open_model(index_path,8192,10,NL,NL,batch_capacity,NULL,512,0,0,2);}
qwen_model *qwen_probe_layer_open(const char *index_path,int il,int batch_capacity){
    if(il!=12 && il!=15){fail("diagnostic supports target layers 12 and 15");return NULL;}
    return open_model(index_path,8192,16,il,il,batch_capacity,NULL,512,0,0,2);
}
int qwen_probe_hidden(qwen_model *m,const float *h){
    if(!m||m->first_layer!=NL)return fail("not an MTP probe");
    GPU(qg_write(m->target_last,h,HD*sizeof(float)));return 0;
}
int qwen_probe_get_hidden(qwen_model *m,float *h){
    if(!m||m->first_layer!=NL)return fail("not an MTP probe");
    GPU(qg_read(h,m->mtp_last,HD*sizeof(float)));return 0;
}
#endif
void qwen_close(qwen_model *m){if(!m)return;qg_sync();
    if(m->route_profile)fclose(m->route_profile);
    qc_destroy(m->cpu);free(m->cpu_x);free(m->cpu_out);free(m->cpu_weights);
    qs_destroy(m->ssd);free(m->ple_cache);free(m->host_ple_packed);
    qg_free(m->resident_masks);qg_free(m->prune_device);
    qg_upload_destroy(m->uploader);
    for(int i=0;i<m->n_tensors;i++)free(m->tensors[i].owned_host);
    for(int i=0;i<=NL;i++){
        layer *l=&m->layers[i];free(l->cache_ids);free(l->cache_age);
        for(int n=1;n<=LOGIT_ROWS;n++){
            qg_graph_destroy(l->attn.graphs[n]);qg_graph_destroy(l->ffn.graphs[n]);
            qg_graph_destroy(l->head.graphs[n]);qg_graph_destroy(l->shared_graphs[n]);}
    }
    for(int n=1;n<=LOGIT_ROWS;n++)qg_graph_destroy(m->head.graphs[n]);
    for(int i=0;i<3;i++)qg_free(m->arenas[i]);
    for(int i=0;i<MAX_FILES;i++){if(m->files[i].data)munmap(m->files[i].data,m->files[i].bytes);if(m->files[i].fd>=0)close(m->files[i].fd);}
    if(m->lock_fd>=0)close(m->lock_fd);
    free(m->tokens);free(m);
}
static int mm(float *out,tensor *w,const float *x,int nt){if(!w||!w->device)return fail("unbound GPU matrix");GPU(qg_matmul(out,w->device,w->type,w->dim[1],w->dim[0],x,nt));return 0;}
static int norm(float *out,const float *x,tensor *w,int width,int groups,int nt,int grouped){if(!w||w->type!=0)return fail("normalization weight must be F32");GPU(qg_norm(out,x,w->device,width,groups,nt,grouped,0,1e-6f));return 0;}
static int hc_direct(qwen_model *m,mixer *mix,int nt){
    RUN(norm(m->hn,m->h,mix->norm,D,4,nt,1));RUN(mm(m->low,mix->down,m->hn,nt));
    GPU(qg_unary(m->low,m->low,nt*320,0,0.25f));RUN(mm(m->gate,mix->up,m->low,nt));
    GPU(qg_hc_read(m->x,m->hn,m->gate,D,nt));if(mix->inject)RUN(mm(m->inject,mix->inject,m->hn,nt));return 0;
}
static int hc(qwen_model *m,mixer *mix,int nt){
    if(nt>LOGIT_ROWS)return hc_direct(m,mix,nt);
    void **g=&mix->graphs[nt];
    if(!*g){
        GPU(qg_graph_begin(g));int rc=hc_direct(m,mix,nt);
        if(rc){qg_graph_destroy(*g);*g=NULL;return -1;}
        int end=qg_graph_end(*g);if(end){qg_graph_destroy(*g);*g=NULL;return fail("capture HC: %s",qg_error(end));}
        m->optimization.graph_captures++;
    }
    GPU(qg_graph_launch(*g));m->optimization.graph_launches++;return 0;
}
static void ple_rows(uint64_t rows[16],const int *history,int pos,int eos,
                     const uint64_t *multiplier,const uint64_t *vocab,const uint64_t *offset){
    uint64_t ctx[3]={(uint64_t)history[pos],0,0};int cut=0;
    /* EOS cuts the preceding segment, including when it is one of the two
     * cached tokens. The current EOS still has its own within-segment history. */
    for(int j=1;j<3;j++){int p=pos-j,token=p<0?eos:history[p];cut=cut||p<0||token==eos;ctx[j]=cut?eos:token;}
    uint64_t mixed=ctx[0]*multiplier[0];
    for(int order=2;order<=3;order++){mixed^=ctx[order-1]*multiplier[order-1];
        for(int j=0;j<8;j++){int head=(order-2)*8+j;rows[head]=mixed%vocab[head]+offset[head];}}
}
#ifdef QWEN_TEST_HOOKS
void qwen_probe_ple_rows(uint64_t *rows,const int *history,int nt,int eos,
                         const uint64_t *multiplier,const uint64_t *vocab,const uint64_t *offset){
    for(int t=0;t<nt;t++)ple_rows(rows+t*16,history,t,eos,multiplier,vocab,offset);
}
#endif
/* A tiny bounded block cache also coalesces adjacent N-gram row reads. */
static int ple_read_row(qwen_model *m,unsigned char *out,uint64_t row){
    if(!m->ple_on_ssd){memcpy(out,m->ngram->host+row*90,90);return 0;}
    uint64_t offset=row*90;size_t done=0;
    while(done<90){
        uint64_t block=(offset+done)/PLE_BLOCK,base=block*PLE_BLOCK;int slot=(int)(block%PLE_SLOTS);
        if(m->ple_cache_keys[slot]!=block+1){
            uint64_t remain=m->ngram->bytes-base;size_t bytes=remain<PLE_BLOCK?(size_t)remain:PLE_BLOCK;
            if(read_tensor(m,m->ngram,m->ple_cache+(size_t)slot*PLE_BLOCK,base,bytes)){m->storage_failed=1;return -1;}
            m->ple_cache_keys[slot]=block+1;m->storage_stats.ple_disk_bytes+=bytes;m->storage_stats.ple_cache_misses++;
        }else m->storage_stats.ple_cache_hits++;
        size_t within=(size_t)((offset+done)%PLE_BLOCK),bytes=90-done;if(bytes>PLE_BLOCK-within)bytes=PLE_BLOCK-within;
        memcpy(out+done,m->ple_cache+(size_t)slot*PLE_BLOCK+within,bytes);done+=bytes;
    }
    return 0;
}
static int ple(qwen_model *m,layer *l,int nt,int pos){
    unsigned char *packed=m->host_ple_packed;double begin=now();
    for(int t=0;t<nt;t++){
        uint64_t rows[16];ple_rows(rows,m->tokens,pos+t,m->ple_eos,m->multiplier,m->head_vocab,m->head_offset);
        for(int head=0;head<16;head++){
            if(rows[head]>=m->ngram->dim[1])return fail("PLE row out of bounds");
            RUN(ple_read_row(m,packed+(t*16+head)*90,rows[head]));}
    }
    m->stats.ple_lookup_seconds+=now()-begin;
    GPU(qg_write(m->ple_packed,packed,nt*16*90));m->stats.ple_read_bytes+=nt*16*90;
    GPU(qg_dequant(m->ple_emb,m->ple_packed,20,nt*16,160));
    RUN(mm(m->ple_key,l->ple_k,m->ple_emb,nt));RUN(mm(m->ple_value,l->ple_v,m->ple_emb,nt));
    RUN(norm(m->ple_key,m->ple_key,l->ple_kn,D,4,nt,1));RUN(norm(m->ple_query,m->h,l->ple_qn,D,4,nt,1));
    GPU(qg_ple_gate(m->ple_gated,m->ple_key,m->ple_query,m->ple_value,nt));
    RUN(norm(m->ple_normal,m->ple_gated,l->ple_cn,D,4,nt,1));
    if(m->tracing)GPU(qg_copy(l->trace_ple,m->ple_normal,(size_t)nt*HD*4));
    GPU(qg_conv(m->ple_convout,m->ple_normal,l->ple_conv->device,l->ple_state,HD,4,3,nt,1));
    GPU(qg_binary(m->h,m->h,m->ple_gated,nt*HD,0));GPU(qg_binary(m->h,m->h,m->ple_convout,nt*HD,0));return 0;
}
static int attention(qwen_model *m,layer *l,int nt,int pos,int mtp){
    if(l->qkv){
        RUN(mm(m->qkv,l->qkv,m->x,nt));RUN(mm(m->z,l->z,m->x,nt));RUN(mm(m->alpha,l->alpha,m->x,nt));RUN(mm(m->beta,l->beta,m->x,nt));
        /* The convolution output needs 10240*NT floats; hn is dead after hc(). */
        if(m->tracing)GPU(qg_copy(l->trace_raw,m->qkv,(size_t)nt*10240*4));
        GPU(qg_conv(m->hn,m->qkv,l->conv->device,l->conv_state,10240,4,1,nt,1));
        if(m->tracing){
            GPU(qg_copy(l->trace_qkv,m->hn,(size_t)nt*10240*4));
            GPU(qg_copy(l->trace_alpha,m->alpha,(size_t)nt*48*4));
            GPU(qg_copy(l->trace_beta,m->beta,(size_t)nt*48*4));}
        GPU(qg_gdn(m->q,m->hn,m->alpha,m->beta,l->a->device,l->dt->device,l->gdn,nt,1e-6f));
        /* Flash-Next explicitly sets output_gate_type="sigmoid", overriding
         * the generic GDN class's SiLU default. QSA also uses sigmoid. */
        RUN(norm(m->q,m->q,l->norm,128,48,nt,0));GPU(qg_gate(m->q,m->z,nt*6144));RUN(mm(m->y,l->out,m->q,nt));
    }else{
        if(!mtp){RUN(mm(m->ik,l->ik,m->x,nt));RUN(mm(m->iq,l->iq,m->x,nt));
            if(m->tracing)GPU(qg_copy(l->trace_index,m->ik,(size_t)nt*128*4));
            GPU(qg_index_workspace(l->index_keys,l->index_tail,m->allowed,m->ik,m->iq,l->ikn->device,l->iqn->device,nt,pos,m->context,1e7f,1e-6f,m->index_scratch));}
        RUN(mm(m->qkv,l->q,m->x,nt));GPU(qg_qsplit(m->q,m->qgate,m->qkv,nt));RUN(norm(m->q,m->q,l->qn,256,24,nt,0));
        RUN(mm(m->k,l->k,m->x,nt));RUN(norm(m->k,m->k,l->kn,256,2,nt,0));RUN(mm(m->v,l->v,m->x,nt));
        GPU(qg_rope(m->q,256,24,nt,pos,1e7f));GPU(qg_rope(m->k,256,2,nt,pos,1e7f));
        GPU(qg_kv_store(l->kc,m->k,512,nt,pos));GPU(qg_kv_store(l->vc,m->v,512,nt,pos));
        GPU(qg_attention(m->q,m->q,l->kc,l->vc,mtp?NULL:m->allowed,nt,pos,m->context));
        GPU(qg_gate(m->q,m->qgate,nt*6144));RUN(mm(m->y,l->o,m->q,nt));
    }
    GPU(qg_hc_write(m->h,m->y,m->inject,D,nt));return 0;
}
static int cache_slot(qwen_model *m,layer *l,int expert,const unsigned char *protected_ids){
    for(int j=0;j<m->resident;j++)if(l->cache_ids[j]==expert){l->cache_age[j]=++l->clock;return j;}
    if(m->fixed_hotlist)return fail("fixed hotlist must never upload a missing expert");
    int slot=-1;uint64_t age=UINT64_MAX;
    for(int j=0;j<m->resident;j++)if(l->cache_ids[j]<0){slot=j;break;}else if(!protected_ids[l->cache_ids[j]] && l->cache_age[j]<age){age=l->cache_age[j];slot=j;}
    if(slot<0)return fail("expert cache has no evictable slot");
    double begin=now();unsigned char *dst=l->expert_cache+slot*l->expert_bytes;
    int rc=qg_upload_expert(m->uploader,dst,l->eg->host+expert*l->gate_bytes,l->gate_bytes,
        l->eu->host+expert*l->up_bytes,l->up_bytes,l->ed->host+expert*l->down_bytes,l->down_bytes);
    if(rc)return fail("expert handoff upload: %s",qg_error(rc));
    l->cache_ids[slot]=expert;l->cache_age[slot]=++l->clock;
    m->stats.expert_handoffs++;m->stats.expert_upload_bytes+=l->expert_bytes;m->stats.expert_handoff_host_seconds+=now()-begin;return slot;
}
int qwen_set_hotlist(qwen_model *m,const int *ids,int count,int threads){
    if(!m||!ids||count!=NL*m->resident||threads<1||threads>28||m->target_pos||m->mtp_pos||m->has_checkpoint||m->fixed_hotlist)return fail("hotlist requires a fresh target and matching capacity");
    for(int il=0;il<NL;il++){
        unsigned char seen[512]={0};for(int j=0;j<m->resident;j++){
            int e=ids[il*m->resident+j];if(e<0||e>=512||seen[e])return fail("invalid or duplicate hotlist expert in layer %d",il);seen[e]=1;
        }
    }
    if(m->cpu)return fail("CPU hotlist initialization already attempted; reopen the model");
    m->cpu=qc_create(m->batch_capacity,threads);
    m->cpu_x=malloc((size_t)m->batch_capacity*D*4);m->cpu_out=malloc((size_t)m->batch_capacity*D*4);
    m->cpu_weights=malloc((size_t)m->batch_capacity*10*4);
    if(!m->cpu||!m->cpu_x||!m->cpu_out||!m->cpu_weights)return fail("CPU activation workspace allocation");
    for(int il=m->first_layer;il<=m->last_layer && il<NL;il++){
        layer *l=&m->layers[il];unsigned char keep[512]={0};
        for(int j=0;j<m->resident;j++){
            int e=ids[il*m->resident+j];
            if(m->ssd&&!qs_is_hot(m->ssd,il,e))return fail("GPU expert %d/%d lacks its pinned RAM copy",il,e);
            keep[e]=1;
        }
        for(int j=0;j<m->resident;j++)if(cache_slot(m,l,ids[il*m->resident+j],keep)<0)return -1;
    }
    unsigned char masks[NL*512]={0};
    for(int il=0;il<NL;il++)for(int j=0;j<m->resident;j++)masks[il*512+ids[il*m->resident+j]]=1;
    m->resident_masks=qg_alloc(sizeof(masks));m->prune_device=qg_alloc(sizeof(qg_prune_stats));
    if(!m->resident_masks||!m->prune_device)return fail("hotlist routing allocation");
    GPU(qg_write(m->resident_masks,masks,sizeof(masks)));GPU(qg_zero(m->prune_device,sizeof(qg_prune_stats)));
    m->stats.gpu_workspace_bytes+=sizeof(masks)+sizeof(qg_prune_stats);
    m->stats.gpu_arena_bytes+=sizeof(masks)+sizeof(qg_prune_stats);
    GPU(qg_sync());m->fixed_hotlist=1;
    fprintf(stderr,"qwen: official CPU MoE handoff; %d fixed experts/layer; omitted mass < 0.10; %d CPU threads\n",m->resident,threads);
    return 0;
}
void qwen_get_cpu_stats(qwen_model *m,qwen_cpu_stats *out){if(m&&out)*out=m->cpu_stats;}
int qwen_set_cpu_threads(qwen_model *m,int threads){
    if(!m||!m->fixed_hotlist||qc_set_threads(m->cpu,threads))return fail("CPU thread count must be 1..28 on an initialized idle model");
    return 0;
}
int qwen_cpu_threads(qwen_model *m){return m?qc_threads(m->cpu):0;}
int qwen_handoff_version(void){return 1;}
int qwen_get_prune_stats(qwen_model *m,qwen_prune_stats *out){
    if(!m||!out||!m->prune_device)return fail("hotlist statistics unavailable");
    GPU(qg_read(out,m->prune_device,sizeof(*out)));return 0;
}

static int cpu_moe(qwen_model *m,layer *l,const int *ids,int nt){
    if(!m->ssd)return qc_moe(m->cpu,m->cpu_out,m->cpu_x,ids,m->cpu_weights,nt,
        l->eg->host,l->eg->type,l->eu->host,l->eu->type,l->ed->host,l->ed->type);
    /* Large prefill can select all 512 experts. Split only its CPU MoE into
     * token prefixes fitting the cold budget. Each token retains router-rank
     * accumulation; GPU attention/context still use the original full batch. */
    for(int start=0;start<nt;){
        int n=qs_begin(m->ssd,(int)(l-m->layers),ids+start*10,nt-start);
        if(n<0||qs_wait(m->ssd)){m->storage_failed=1;return fail("%s",qs_error(m->ssd));}
        int rc=qc_moe(m->cpu,m->cpu_out+(size_t)start*D,m->cpu_x+(size_t)start*D,
            ids+start*10,m->cpu_weights+start*10,n,l->eg->host,l->eg->type,l->eu->host,l->eu->type,l->ed->host,l->ed->type);
        int released=qs_release(m->ssd);
        if(rc)return fail("CPU streamed MoE failed");
        if(released){m->storage_failed=1;return fail("%s",qs_error(m->ssd));}
        start+=n;
    }
    return 0;
}
static int grouped_moe(qwen_model *m,layer *l,int nt,int mtp){
    uint64_t uploads_before=m->stats.expert_handoffs;
    /* Only routing IDs cross back here; hidden states/logits stay on device.
     * Cold expert payloads move exclusively through cache_slot's handoff. */
    int ids[NT*10];
    if(!mtp && m->route_profile){
        GPU(qg_read(ids,m->route_ids,nt*10*sizeof(int)));
        float weights[NT*10];GPU(qg_read(weights,m->route_weights,nt*10*sizeof(float)));
        uint32_t header[3]={(uint32_t)(l-m->layers),(uint32_t)m->target_pos,(uint32_t)nt};
        size_t n=(size_t)nt*10;
        if(fwrite(header,sizeof(uint32_t),3,m->route_profile)!=3 ||
           fwrite(ids,sizeof(int),n,m->route_profile)!=n ||
           fwrite(weights,sizeof(float),n,m->route_profile)!=n)return fail("write route profile");
    }
    if(!mtp&&m->fixed_hotlist)GPU(qg_prune_routes(m->route_ids,m->route_weights,
        m->resident_masks+(l-m->layers)*512,nt,0.10,m->prune_device));
    GPU(qg_read(ids,m->route_ids,nt*10*sizeof(int)));m->stats.metadata_read_bytes+=nt*10*sizeof(int);
    unsigned char needed[512]={0},resident[512]={0};int order[512],counts[512]={0},count=0;
    for(int j=0;j<nt*10;j++){
        if(ids[j]<=-2&&ids[j]>=-513&&m->fixed_hotlist)continue;
        if(ids[j]<0||ids[j]>=512)return fail("non-finite router distribution");
        needed[ids[j]]=1;counts[ids[j]]++;
    }
    if(!mtp){for(int j=0;j<m->resident;j++)if(l->cache_ids[j]>=0)resident[l->cache_ids[j]]=1;
        for(int j=0;j<nt*10;j++){m->stats.expert_requests++;m->stats.expert_hits+=ids[j]>=0&&resident[ids[j]];}}
    if(!mtp && m->fixed_hotlist){
        int missing=0;for(int j=0;j<nt*10;j++)if(ids[j]>=0)missing+=!resident[ids[j]];
        if(missing){
            /* Attention/GDN/KV never leave the GPU. Only the routed MoE is
             * recovered here with the current per-token omission mask. The
             * shared expert and the remaining target stay on GPU. */
            double begin=now();
            GPU(qg_read(m->cpu_x,m->x,(size_t)nt*D*4));
            GPU(qg_read(m->cpu_weights,m->route_weights,(size_t)nt*10*4));
            double compute=now();
            qs_stats before_ssd,after_ssd;qs_get_stats(m->ssd,&before_ssd);
            if(cpu_moe(m,l,ids,nt))return -1;
            qs_get_stats(m->ssd,&after_ssd);
            m->cpu_stats.compute_seconds+=now()-compute-(after_ssd.wait_seconds-before_ssd.wait_seconds);
            GPU(qg_write(m->moeout,m->cpu_out,(size_t)nt*D*4));m->cpu_reduced=1;
            m->cpu_stats.handoff_seconds+=now()-begin;m->cpu_stats.layer_handoffs++;
            m->cpu_stats.token_layers+=nt;m->cpu_stats.missing_selections+=missing;
            m->cpu_stats.activation_bytes+=(uint64_t)nt*(D*8+10*4);
            int il=(int)(l-m->layers);m->handoff_profile.layer_calls[il]++;m->handoff_profile.layer_handoffs[il]++;
            return 0;
        }
    }
    if(nt>1)GPU(qg_moe_map(m->route_map,m->route_ids,nt));
    /* Execute present experts first. This releases their slots before cold
     * experts arrive, even when a verification batch references >cache slots. */
    for(int phase=0;phase<2;phase++)for(int e=0;e<512;e++)if(needed[e] && (resident[e]?0:1)==phase)order[count++]=e;
    GPU(qg_zero(m->moe_slots,nt*10*D*sizeof(float)));
    for(int j=0;j<count;j++){
        int e=order[j],ne=counts[e];const void *wg,*wu,*wd;const float *input=m->x;const int *token_map=NULL;
        /* Compact on device: unrelated tokens must not multiply this expert's
         * matrices. This matters most during prefill with many unique routes. */
        if(ne<nt){token_map=m->route_map+e*nt;GPU(qg_moe_gather(m->expert_x,m->x,token_map,ne));input=m->expert_x;}
        if(mtp){wg=(unsigned char*)l->eg->device+e*l->gate_bytes;wu=(unsigned char*)l->eu->device+e*l->up_bytes;wd=(unsigned char*)l->ed->device+e*l->down_bytes;}
        else{int slot=cache_slot(m,l,e,needed);if(slot<0)return -1;
            wg=l->expert_cache+slot*l->expert_bytes;
            wu=(unsigned char*)wg+l->gate_bytes;wd=(unsigned char*)wu+l->up_bytes;}
        GPU(qg_matmul(m->ffg,wg,l->eg->type,640,D,input,ne));GPU(qg_matmul(m->ffu,wu,l->eu->type,640,D,input,ne));
        GPU(qg_unary(m->ffmid,m->ffg,ne*640,0,1.0f));GPU(qg_binary(m->ffmid,m->ffmid,m->ffu,ne*640,1));
        GPU(qg_matmul(m->ffout,wd,l->ed->type,D,640,m->ffmid,ne));
        GPU(qg_moe_scatter(m->moe_slots,m->ffout,m->route_weights,m->route_ids,token_map,e,ne));needed[e]=0;
    }
    if(!mtp){
        /* Count one completed layer execution with any actual upload, including
         * prefill batches and speculative replay; never count MTP's resident MoE. */
        int il=(int)(l-m->layers);
        m->handoff_profile.layer_calls[il]++;
        m->handoff_profile.layer_handoffs[il]+=m->stats.expert_handoffs!=uploads_before;
    }
    return 0;
}
static int shared_direct(qwen_model *m,layer *l,int nt){
    RUN(mm(m->ffg,l->sg,m->x,nt));RUN(mm(m->ffu,l->su,m->x,nt));
    GPU(qg_unary(m->ffmid,m->ffg,nt*640,0,1));GPU(qg_binary(m->ffmid,m->ffmid,m->ffu,nt*640,1));
    RUN(mm(m->shared,l->sd,m->ffmid,nt));RUN(mm(m->shared_gate,l->sr,m->x,nt));
    GPU(qg_shared_add(m->moeout,m->shared,m->shared_gate,nt));GPU(qg_hc_write(m->h,m->moeout,m->inject,D,nt));return 0;
}
static int moe(qwen_model *m,layer *l,int nt,int mtp){
    m->cpu_reduced=0;
    RUN(mm(m->route_logits,l->router,m->x,nt));GPU(qg_route(m->route_ids,m->route_weights,m->route_logits,nt));
    /* Short MTP steps use resident experts directly from device routing IDs:
     * no CPU routing read or per-expert host loop. Large prefill groups tokens
     * to reuse expert weights across the batch instead of rereading each one. */
    if(mtp && nt<=LOGIT_ROWS)GPU(qg_mtp_moe(m->moe_slots,m->mtp_mid,m->x,m->route_ids,m->route_weights,
        l->eg->device,l->eg->type,l->eu->device,l->eu->type,l->ed->device,l->ed->type,nt));
    else RUN(grouped_moe(m,l,nt,mtp));
    /* Fix the accumulation order to router rank, independent of cache contents
     * or scheduling. Replay must not change logits merely by warming the cache. */
    if(!m->cpu_reduced)GPU(qg_moe_reduce(m->moeout,m->moe_slots,nt));
    if(nt>LOGIT_ROWS)return shared_direct(m,l,nt);
    void **g=&l->shared_graphs[nt];
    if(!*g){
        GPU(qg_graph_begin(g));int rc=shared_direct(m,l,nt);
        if(rc){qg_graph_destroy(*g);*g=NULL;return -1;}
        int end=qg_graph_end(*g);if(end){qg_graph_destroy(*g);*g=NULL;return fail("capture shared expert: %s",qg_error(end));}
        m->optimization.graph_captures++;
    }
    GPU(qg_graph_launch(*g));m->optimization.graph_launches++;return 0;
}
static int token_embeddings(qwen_model *m,const int *tokens,int nt){for(int t=0;t<nt;t++){
    if(tokens[t]<0||tokens[t]>=V)return fail("token outside Qwen vocabulary");
    GPU(qg_gather(m->x+t*D,m->embedding->device,m->embedding->type,D,tokens[t]));}return 0;}
#ifdef QWEN_TEST_HOOKS
int qwen_probe_layer(qwen_model *m,float *out,const float *hidden,int nt){
    if(!m||m->first_layer!=m->last_layer||m->first_layer>=NL||nt<1||nt>m->batch_capacity||m->target_pos+nt>m->context)return fail("invalid target layer diagnostic");
    layer *l=&m->layers[m->first_layer];GPU(qg_write(m->h,hidden,(size_t)nt*HD*4));
    RUN(hc(m,&l->attn,nt));RUN(attention(m,l,nt,m->target_pos,0));RUN(hc(m,&l->ffn,nt));RUN(moe(m,l,nt,0));
    GPU(qg_read(out,m->h,(size_t)nt*HD*4));m->target_pos+=nt;return 0;
}
#endif
int qwen_max_draft(void){return QWEN_MAX_DRAFT;}
int qwen_target(qwen_model *m,const int *tokens,int nt,int want_logits){
    if(!m||m->first_layer!=0||nt<1||nt>m->batch_capacity||m->target_pos+nt>m->context)return fail("target batch exceeds context or capacity");
    if(m->storage_failed)return fail("storage I/O failed; reopen the model before continuing");
    if(want_logits<0||want_logits>2||(want_logits==1&&nt>LOGIT_ROWS))return fail("full logits limited to %d verification rows; use last-only prefill",LOGIT_ROWS);
    double start=now();int pos=m->target_pos;
    m->trace_rows=0;
    m->tracing=m->has_checkpoint&&pos==m->save_target_pos&&want_logits==1&&nt<=LOGIT_ROWS;
    for(int t=0;t<nt;t++)if(tokens[t]<0||tokens[t]>=V)return fail("invalid token");
    if(!m->fixed_hotlist)return fail("initialize the fixed target hotlist before inference");
    GPU(qg_copy(m->previous_h,m->target_last,HD*sizeof(float)));memcpy(m->tokens+pos,tokens,nt*sizeof(int));
    RUN(token_embeddings(m,tokens,nt));GPU(qg_hc_init(m->h,m->x,D,nt));
    for(int il=0;il<NL;il++){
        layer *l=&m->layers[il];if(il==1)RUN(ple(m,l,nt,pos));
        RUN(hc(m,&l->attn,nt));RUN(attention(m,l,nt,pos,0));RUN(hc(m,&l->ffn,nt));RUN(moe(m,l,nt,0));
    }
    m->trace_rows=m->tracing?nt:0;m->tracing=0;
    GPU(qg_copy(m->target_h,m->h,nt*HD*sizeof(float)));GPU(qg_copy(m->target_last,m->h+(nt-1)*HD,HD*sizeof(float)));
    int rows=want_logits==2?1:nt;
    if(want_logits){if(want_logits==2)GPU(qg_copy(m->h,m->target_last,HD*4));RUN(hc(m,&m->head,rows));RUN(mm(m->target_logits,m->output,m->x,rows));}
    GPU(qg_sync());m->target_pos+=nt;m->last_target_batch=nt;m->last_target_rows=want_logits?rows:0;m->stats.target_tokens+=nt;m->stats.target_seconds+=now()-start;return 0;
}
static int mtp_forward(qwen_model *m,const int *tokens,int nt,int pos,const float *hidden,int want_logits){
    if(nt<1||nt>m->batch_capacity||pos<0||pos+nt>m->context||(want_logits&&nt>LOGIT_ROWS))return fail("MTP batch exceeds context or logits capacity");
    double start=now();layer *l=&m->layers[NL];RUN(token_embeddings(m,tokens,nt));
    RUN(norm(m->x,m->x,l->enorm,D,1,nt,0));RUN(norm(m->hn,hidden,l->hnorm,D,4,nt,1));
    GPU(qg_mtp_concat(m->concat,m->x,m->hn,D,nt));
    /* eh_proj acts separately on each of the four streams; retain all streams. */
    for(int t=0;t<nt;t+=128){int n=nt-t<128?nt-t:128;RUN(mm(m->h+t*HD,l->eh,m->concat+t*HD*2,n*4));}
    RUN(hc(m,&l->attn,nt));RUN(attention(m,l,nt,pos,1));RUN(hc(m,&l->ffn,nt));RUN(moe(m,l,nt,1));
    GPU(qg_copy(m->mtp_h,m->h,nt*HD*sizeof(float)));GPU(qg_copy(m->mtp_last,m->h+(nt-1)*HD,HD*sizeof(float)));
    if(want_logits){RUN(hc(m,&l->head,nt));RUN(mm(m->mtp_logits,m->output,m->x,nt));}
    GPU(qg_sync());m->mtp_pos=pos+nt;m->last_mtp_rows=want_logits?nt:0;m->stats.mtp_tokens+=nt;m->stats.mtp_seconds+=now()-start;return 0;
}
#ifdef QWEN_TEST_HOOKS
int qwen_probe_batch(qwen_model *m,const int *tokens,int nt,int pos,const float *hidden){
    if(!m||m->first_layer!=NL||nt<1||nt>m->batch_capacity)return fail("invalid MTP diagnostic batch");
    GPU(qg_write(m->mtp_input_h,hidden,(size_t)nt*HD*sizeof(float)));
    return mtp_forward(m,tokens,nt,pos,m->mtp_input_h,nt<=LOGIT_ROWS);
}
#endif
int qwen_mtp_catchup(qwen_model *m,const int *tokens,int nt,int pos){
    if(!m||nt<1||nt>m->batch_capacity||nt!=m->last_target_batch||pos+nt!=m->target_pos)return fail("MTP catchup must follow the last target batch");
    GPU(qg_copy(m->mtp_input_h,m->previous_h,HD*sizeof(float)));
    if(nt>1)GPU(qg_copy(m->mtp_input_h+HD,m->target_h,(nt-1)*HD*sizeof(float)));
    return mtp_forward(m,tokens,nt,pos,m->mtp_input_h,0);
}
int qwen_mtp_step(qwen_model *m,int token,int pos,int target_hidden,int *next){
    if(!m||!next||pos!=m->mtp_pos||(target_hidden!=0&&target_hidden!=1))return fail("invalid MTP step or cache frontier");
    RUN(mtp_forward(m,&token,1,pos,target_hidden?m->target_last:m->mtp_last,1));return qwen_top1(m,next,1,1);
}
int qwen_logits(qwen_model *m,float *out,int row,int mtp){
    if(!m||!out||row<0||row>=(mtp?m->last_mtp_rows:m->last_target_rows))return fail("unavailable logits row");
    GPU(qg_read(out,(mtp?m->mtp_logits:m->target_logits)+(size_t)row*V,V*sizeof(float)));return 0;
}
int qwen_top1(qwen_model *m,int *out,int rows,int mtp){
    if(!m||!out||rows<1||rows>(mtp?m->last_mtp_rows:m->last_target_rows))return fail("unavailable argmax rows");
    GPU(qg_argmax(m->device_top1,mtp?m->mtp_logits:m->target_logits,V,rows));GPU(qg_read(out,m->device_top1,rows*sizeof(int)));m->stats.metadata_read_bytes+=rows*sizeof(int);
    for(int i=0;i<rows;i++)if(out[i]<0)return fail("invalid non-finite logits distribution");
    return 0;
}
int qwen_verify(qwen_model *m,const int *proposals,int count,int top_k,float ratio,int *best,int *eligible){
    if(!m||!proposals||!best||!eligible||count<1||count>=LOGIT_ROWS||m->last_target_rows!=count+1||
       top_k<1||top_k>V||!isfinite(ratio)||ratio<=0||ratio>1)return fail("invalid verification request");
    for(int i=0;i<count;i++)if(proposals[i]<0||proposals[i]>=V)return fail("invalid draft token");
    GPU(qg_write(m->verify_ids,proposals,count*sizeof(int)));
    GPU(qg_verify(m->device_top1,m->verify_ids+LOGIT_ROWS,m->target_logits,m->verify_ids,V,count,top_k,-logf(ratio)));
    GPU(qg_read(best,m->device_top1,(count+1)*sizeof(int)));
    GPU(qg_read(eligible,m->verify_ids+LOGIT_ROWS,count*sizeof(int)));
    m->stats.metadata_read_bytes+=(2*count+1)*sizeof(int);
    for(int i=0;i<=count;i++)if(best[i]<0)return fail("invalid non-finite logits distribution");
    return 0;
}
int qwen_checkpoint(qwen_model *m){if(!m)return fail("null model");
    for(int i=0;i<=NL;i++){layer *l=&m->layers[i];if(l->gdn){GPU(qg_copy(l->gdn_save,l->gdn,48ULL*128*128*4));GPU(qg_copy(l->conv_save,l->conv_state,3ULL*10240*4));}
        if(l->index_tail)GPU(qg_copy(l->index_tail_save,l->index_tail,4*128*4));
        if(l->ple_state)GPU(qg_copy(l->ple_save,l->ple_state,9ULL*HD*4));}
    GPU(qg_copy(m->target_last_save,m->target_last,HD*4));GPU(qg_copy(m->previous_h_save,m->previous_h,HD*4));GPU(qg_copy(m->mtp_last_save,m->mtp_last,HD*4));
    m->trace_rows=0;m->save_target_pos=m->target_pos;m->save_mtp_pos=m->mtp_pos;m->has_checkpoint=1;return 0;
}
int qwen_restore(qwen_model *m){if(!m||!m->has_checkpoint)return fail("no speculative checkpoint");
    for(int i=0;i<=NL;i++){layer *l=&m->layers[i];if(l->gdn){GPU(qg_copy(l->gdn,l->gdn_save,48ULL*128*128*4));GPU(qg_copy(l->conv_state,l->conv_save,3ULL*10240*4));}
        if(l->index_tail)GPU(qg_copy(l->index_tail,l->index_tail_save,4*128*4));
        if(l->ple_state)GPU(qg_copy(l->ple_state,l->ple_save,9ULL*HD*4));}
    GPU(qg_copy(m->target_last,m->target_last_save,HD*4));GPU(qg_copy(m->previous_h,m->previous_h_save,HD*4));GPU(qg_copy(m->mtp_last,m->mtp_last_save,HD*4));
    m->trace_rows=m->tracing=0;m->target_pos=m->save_target_pos;m->mtp_pos=m->save_mtp_pos;m->last_target_rows=m->last_mtp_rows=m->last_target_batch=0;return 0;
}
int qwen_commit_prefix(qwen_model *m,int keep){
    if(!m||!m->has_checkpoint||keep<1||keep>=m->trace_rows||
       m->trace_rows!=m->last_target_batch||m->target_pos!=m->save_target_pos+m->trace_rows)
        return fail("prefix requires the last checkpointed verification batch");
    double begin=now();
    for(int i=0;i<NL;i++){
        layer *l=&m->layers[i];
        if(l->gdn){
            GPU(qg_copy(l->gdn,l->gdn_save,48ULL*128*128*4));
            /* Reconstruct just the recurrence from its exact FP32 inputs.
             * No projection, convolution, routing, MoE or logits are repeated.
             * This saves ~1.8 GiB versus 17 full GDN checkpoints. */
            GPU(qg_gdn(m->q,l->trace_qkv,l->trace_alpha,l->trace_beta,l->a->device,l->dt->device,l->gdn,keep,1e-6f));
            GPU(qg_history_prefix(l->conv_state,l->conv_save,l->trace_raw,10240,3,keep));
        }
        if(l->ple_state)GPU(qg_history_prefix(l->ple_state,l->ple_save,l->trace_ple,HD,9,keep));
        if(l->index_tail)GPU(qg_index_prefix(l->index_tail,l->index_tail_save,l->trace_index,m->save_target_pos,keep));
    }
    GPU(qg_copy(m->target_last,m->target_h+(keep-1)*HD,HD*4));
    GPU(qg_copy(m->mtp_last,m->mtp_last_save,HD*4));
    m->target_pos=m->save_target_pos+keep;m->mtp_pos=m->save_mtp_pos;
    m->last_target_batch=m->last_target_rows=keep;m->last_mtp_rows=0;m->trace_rows=0;
    GPU(qg_sync());m->optimization.prefix_restores++;m->optimization.retained_tokens+=keep;
    m->optimization.prefix_restore_seconds+=now()-begin;return 0;
}
void qwen_get_optimization_stats(qwen_model *m,qwen_optimization_stats *out){if(m&&out)*out=m->optimization;}
int qwen_reset(qwen_model *m){if(!m)return fail("null model");
    if(m->storage_failed)return fail("storage I/O failed; reopen the model before continuing");
    for(int i=0;i<=NL;i++){layer *l=&m->layers[i];if(l->gdn){GPU(qg_zero(l->gdn,48ULL*128*128*4));GPU(qg_zero(l->conv_state,3ULL*10240*4));}
        if(l->index_tail){GPU(qg_zero(l->index_tail,4*128*4));GPU(qg_zero(l->index_keys,(uint64_t)((m->context+3)/4)*128*4));}
        if(l->ple_state)GPU(qg_zero(l->ple_state,9ULL*HD*4));}
    GPU(qg_zero(m->target_last,HD*4));GPU(qg_zero(m->previous_h,HD*4));GPU(qg_zero(m->mtp_last,HD*4));
    m->trace_rows=m->tracing=0;m->target_pos=m->mtp_pos=m->last_target_rows=m->last_mtp_rows=m->last_target_batch=m->has_checkpoint=0;return 0;
}
void qwen_get_stats(qwen_model *m,qwen_stats *out){if(m&&out){*out=m->stats;out->expert_dma_seconds=qg_upload_seconds(m->uploader);out->target_position=m->target_pos;out->mtp_position=m->mtp_pos;}}
void qwen_get_handoff_profile(qwen_model *m,qwen_handoff_profile *out){if(m&&out)*out=m->handoff_profile;}
