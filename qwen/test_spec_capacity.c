/* Native verification boundaries and actual workspace layout, without weights. */
#include "qwen.c"
#include <assert.h>

int main(void){
    qwen_model m={0};int ids[18]={0};
    assert(qwen_max_draft()==16 && QWEN_VERIFY_ROWS==17);
    m.context=24576;m.batch_capacity=2048;
    ids[16]=-1;
    assert(qwen_target(&m,ids,17,1)!=0);
    assert(strcmp(qwen_last_error(),"invalid token")==0); /* Passed row-capacity check. */
    assert(qwen_target(&m,ids,18,1)!=0);
    assert(strstr(qwen_last_error(),"17 verification rows")!=NULL);
    assert(workspace(&m,1)==0);
    uint64_t daily=m.arena_capacity[2];
    m.batch_capacity=17;assert(workspace(&m,1)==0);
    m.arenas[2]=malloc(m.arena_capacity[2]);assert(m.arenas[2]);
    assert(workspace(&m,0)==0);
    assert((char*)m.mtp_logits-(char*)m.target_logits>=(ptrdiff_t)(17ULL*V*sizeof(float)));
    assert((char*)m.device_top1-(char*)m.mtp_logits>=(ptrdiff_t)(17ULL*V*sizeof(float)));
    assert((char*)m.route_logits-(char*)m.mtp_mid>=(ptrdiff_t)(17ULL*10*640*sizeof(float)));
    free(m.arenas[2]);
    printf("Speculation: 17 verification rows and disjoint logits/MTP buffers passed; daily workspace %"PRIu64" bytes.\n",daily);
    return 0;
}
