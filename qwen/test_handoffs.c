/* Exercise the real C cache scheduler with inert GPU operations. No model/GPU
 * allocation: the assertions distinguish uploads from layer-level events. */
#include "qwen.c"
#include <assert.h>

const char *qg_error(int code){(void)code;return "test GPU stub";}
int qg_read(void *dst,const void *src,size_t bytes){memcpy(dst,src,bytes);return 0;}
int qg_zero(void *dst,size_t bytes){(void)dst;(void)bytes;return 0;}
int qg_moe_map(int *map,const int *ids,int tokens){(void)map;(void)ids;(void)tokens;return 0;}
int qg_moe_gather(float *out,const float *x,const int *map,int n){(void)out;(void)x;(void)map;(void)n;return 0;}
int qg_upload_expert(void *u,void *dst,const void *g,size_t ng,const void *up,size_t nu,const void *d,size_t nd){
    (void)u;memcpy(dst,g,ng);memcpy((char*)dst+ng,up,nu);memcpy((char*)dst+ng+nu,d,nd);return 0;
}
int qg_matmul(float *out,const void *w,int type,int rows,int cols,const float *x,int n){
    (void)out;(void)w;(void)type;(void)rows;(void)cols;(void)x;(void)n;return 0;
}
int qg_unary(float *out,const float *x,int n,int op,float scale){(void)out;(void)x;(void)n;(void)op;(void)scale;return 0;}
int qg_binary(float *out,const float *a,const float *b,int n,int op){(void)out;(void)a;(void)b;(void)n;(void)op;return 0;}
int qg_moe_scatter(float *slots,const float *expert,const float *weights,const int *ids,const int *map,int e,int n){
    (void)slots;(void)expert;(void)weights;(void)ids;(void)map;(void)e;(void)n;return 0;
}

int main(void){
    static qwen_model m;
    static unsigned char host[512],cache[NL+1][24*3];
    static int cache_ids[NL+1][24],routes[20],map[512*2];
    static uint64_t ages[NL+1][24];
    tensor t={.host=host,.device=host};m.resident=24;m.route_ids=routes;m.route_map=map;
    for(int il=0;il<=NL;il++){
        layer *l=&m.layers[il];l->eg=l->eu=l->ed=&t;l->expert_cache=cache[il];
        l->gate_bytes=l->up_bytes=l->down_bytes=1;l->expert_bytes=3;
        l->cache_ids=cache_ids[il];l->cache_age=ages[il];
        for(int j=0;j<24;j++)l->cache_ids[j]=-1;
    }
    for(int i=0;i<10;i++)routes[i]=i;
    assert(grouped_moe(&m,&m.layers[12],1,0)==0); /* Ten missing experts, one event. */
    assert(m.stats.expert_handoffs==10 && m.handoff_profile.layer_handoffs[12]==1);
    assert(grouped_moe(&m,&m.layers[12],1,0)==0); /* Fully warm: no event. */
    assert(m.stats.expert_handoffs==10 && m.handoff_profile.layer_calls[12]==2);
    assert(m.handoff_profile.layer_handoffs[12]==1);
    routes[9]=50;
    assert(grouped_moe(&m,&m.layers[12],1,0)==0);
    assert(m.stats.expert_handoffs==11 && m.handoff_profile.layer_handoffs[12]==2);
    for(int i=0;i<10;i++)routes[10+i]=routes[i];
    routes[18]=100;routes[19]=101;
    assert(grouped_moe(&m,&m.layers[12],2,0)==0); /* Batch needs two new experts, one event. */
    assert(m.stats.expert_handoffs==13 && m.handoff_profile.layer_handoffs[12]==3);
    assert(m.handoff_profile.layer_calls[12]==4);
    assert(grouped_moe(&m,&m.layers[15],1,0)==0); /* Separate layer profile. */
    assert(m.handoff_profile.layer_handoffs[15]==1 && m.stats.expert_handoffs==23);
    assert(grouped_moe(&m,&m.layers[NL],1,1)==0); /* MTP never adds target events. */
    assert(m.stats.expert_handoffs==23);
    qwen_handoff_profile out; qwen_get_handoff_profile(&m,&out);
    uint64_t events=0,calls=0;for(int i=0;i<NL;i++){events+=out.layer_handoffs[i];calls+=out.layer_calls[i];}
    assert(events==4 && calls==5);
    assert(qwen_reset(&m)==0);qwen_get_handoff_profile(&m,&out);
    assert(out.layer_handoffs[12]==3); /* Work counters survive conversation reset. */
    puts("Handoff scheduler: cold, warm, partial miss, batch, per-layer, MTP exclusion and reset passed.");
    return 0;
}
