/* Exercise real file reads and RAM ownership without loading a model or CUDA.
 * Cover the 64 MiB chunk boundary, source-file independence and a short file. */
#include "qwen.c"
#include <assert.h>

int main(void){
    static qwen_model m,short_model;
    char path[]="/tmp/qwen-host-weights-XXXXXX";int fd=mkstemp(path);assert(fd>=0);unlink(path);
    size_t ple_bytes=64ULL*1024*1024+257,expert_bytes=4096,total=ple_bytes+expert_bytes;
    unsigned char *expected=malloc(total);assert(expected);
    for(size_t i=0;i<total;i++)expected[i]=(unsigned char)(i*17+i/251);
    size_t written=0;while(written<total){ssize_t n=write(fd,expected+written,total-written);assert(n>0);written+=(size_t)n;}
    unsigned char *mapped=mmap(NULL,total,PROT_READ,MAP_PRIVATE,fd,0);assert(mapped!=MAP_FAILED);
    m.first_layer=0;m.n_tensors=3;m.files[0].fd=fd;m.ngram=&m.tensors[0];
    tensor *ple=m.ngram,*expert=&m.tensors[1],*core=&m.tensors[2];
    strcpy(ple->name,"per_layer_token_embd.weight");ple->bytes=ple_bytes;ple->host=mapped;
    strcpy(expert->name,"blk.0.ffn_down_exps.weight");expert->bytes=expert_bytes;expert->offset=ple_bytes;expert->host=mapped+ple_bytes;
    strcpy(core->name,"token_embd.weight");core->bytes=32;core->host=mapped;
    assert(host_weight_budget(&m)==total);
    assert(prepare_host_weights(&m)==0);
    assert(ple->host==ple->owned_host&&ple->host!=mapped);
    assert(memcmp(ple->host,expected,ple_bytes)==0);
    assert(memcmp(expert->host,expected+ple_bytes,expert_bytes)==0);
    assert(core->owned_host==NULL&&core->host==mapped);
    assert(qwen_ple_ram_bytes(&m)==ple_bytes&&m.stats.host_weight_bytes==expert_bytes);
    unsigned char changed=expected[ple_bytes-1]^255;
    assert(pwrite(fd,&changed,1,(off_t)ple_bytes-1)==1);
    assert(ple->host[ple_bytes-1]==expected[ple_bytes-1]);
    short_model.first_layer=0;short_model.n_tensors=1;short_model.files[0].fd=fd;
    short_model.ngram=&short_model.tensors[0];short_model.ngram->bytes=4096;
    strcpy(short_model.ngram->name,"per_layer_token_embd.weight");
    assert(ftruncate(fd,16)==0);assert(prepare_host_weights(&short_model)<0);
    assert(strstr(qwen_last_error(),"unexpected EOF")!=NULL);
    free(short_model.ngram->owned_host);free(ple->owned_host);free(expert->owned_host);
    munmap(mapped,total);close(fd);free(expected);
    puts("PASS: host budget includes PLE; exact full RAM copy; source independence; short-file rejection");
    return 0;
}
