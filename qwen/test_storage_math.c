/* Real CPU kernels and real storage path, with small synthetic Qwen-shaped
 * weights. No CUDA or complete model allocation. */
#include "qwen.c"
#include <assert.h>

static uint32_t rng=19;
static unsigned char random_byte(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return (unsigned char)rng;}
int main(void){
    static qwen_model m;
    char path[]="/tmp/qwen-storage-math-XXXXXX",error[256];int fd=mkstemp(path);assert(fd>=0);unlink(path);
    const size_t eb=921600,span=eb*512,total=span*3,ple_bytes=10001;
    assert(ftruncate(fd,(off_t)(total+ple_bytes))==0);
    unsigned char *packed=malloc(eb);assert(packed);
    qs_source source[3];int ranking[512];for(int e=0;e<512;e++)ranking[e]=e;
    for(int k=0;k<3;k++){
        source[k]=(qs_source){fd,k*span,eb};int block=k==2?18:144;
        for(int e=0;e<16;e++){
            for(size_t j=0;j<eb;j++)packed[j]=random_byte();
            for(size_t j=0;j<eb;j+=block){packed[j]=0x19;packed[j+1]=0x10;if(k!=2){packed[j+2]=0x8e;packed[j+3]=0x0a;}}
            assert(pwrite(fd,packed,eb,(off_t)(k*span+e*eb))==(ssize_t)eb);
        }
    }
    for(size_t i=0;i<ple_bytes;i++)packed[i]=(unsigned char)(i*7+i/17);
    assert(pwrite(fd,packed,ple_bytes,(off_t)total)==(ssize_t)ple_bytes);free(packed);
    unsigned char *baseline=mmap(NULL,total+ple_bytes,PROT_READ,MAP_PRIVATE,fd,0);assert(baseline!=MAP_FAILED);
    m.ssd=qs_create(source,1,ranking,3,eb*3*10,2,error);if(!m.ssd)fprintf(stderr,"%s\n",error);assert(m.ssd);
    m.cpu=qc_create(33,2);assert(m.cpu);
    m.cpu_x=malloc(33*D*4);m.cpu_out=malloc(33*D*4);m.cpu_weights=malloc(33*10*4);
    float *expected=malloc(33*D*4);assert(m.cpu_x&&m.cpu_out&&m.cpu_weights&&expected);
    int ids[33*10];
    for(int t=0;t<33;t++){
        for(int k=0;k<10;k++){ids[t*10+k]=(t*3+k)%16;m.cpu_weights[t*10+k]=(k+1)/55.0f;}
        if(t%3==1){ids[t*10+9]=-2;m.cpu_weights[t*10+9]=0;}
        for(int d=0;d<D;d++)m.cpu_x[t*D+d]=((int)random_byte()-128)*0.001f;
    }
    layer *l=&m.layers[0];l->eg=&m.tensors[0];l->eu=&m.tensors[1];l->ed=&m.tensors[2];
    for(int k=0;k<3;k++){m.tensors[k].host=qs_tensor(m.ssd,0,k);m.tensors[k].type=k==2?20:12;}
    const int counts[]={1,5,17,18,33};
    for(unsigned i=0;i<sizeof(counts)/sizeof(counts[0]);i++){
        int nt=counts[i];assert(qc_moe(m.cpu,expected,m.cpu_x,ids,m.cpu_weights,nt,
            baseline,12,baseline+span,12,baseline+2*span,20)==0);
        assert(cpu_moe(&m,l,ids,nt)==0);
        if(memcmp(expected,m.cpu_out,(size_t)nt*D*4)){
            float diff=0;for(int j=0;j<nt*D;j++)if(fabsf(expected[j]-m.cpu_out[j])>diff)diff=fabsf(expected[j]-m.cpu_out[j]);
            fprintf(stderr,"nt=%d maximum difference %.9g\n",nt,diff);assert(0);
        }
        printf("PASS identical RAM/SSD MoE: %d tokens, Q4_K gate/up + IQ4_NL down\n",nt);
    }
    /* PLE row crossing a block boundary, warm hits, partial final block. */
    m.ngram=&m.tensors[3];m.ngram->host=baseline+total;m.ngram->bytes=ple_bytes;
    m.ngram->offset=total;m.files[0].fd=fd;m.ple_cache=malloc(4*1024*1024);assert(m.ple_cache);m.ple_on_ssd=1;
    unsigned char row[90];const int rows[]={0,45,46,110,45};
    for(unsigned i=0;i<sizeof(rows)/sizeof(rows[0]);i++){
        assert(ple_read_row(&m,row,rows[i])==0);assert(!memcmp(row,baseline+total+rows[i]*90,90));
    }
    assert(m.storage_stats.ple_disk_bytes==ple_bytes);
    assert(m.storage_stats.ple_cache_hits>0&&m.storage_stats.ple_cache_misses==3);
    memset(m.ple_cache_keys,0,sizeof(m.ple_cache_keys));assert(ftruncate(fd,(off_t)total+100)==0);
    assert(ple_read_row(&m,row,45)<0);assert(strstr(qwen_last_error(),"unexpected EOF"));
    assert(m.storage_failed);
    qs_destroy(m.ssd);qc_destroy(m.cpu);free(m.cpu_x);free(m.cpu_out);free(m.cpu_weights);free(expected);free(m.ple_cache);
    munmap(baseline,total+ple_bytes);close(fd);
    puts("PASS PLE SSD: identical rows, block boundaries, warm cache, final block, short-read rejection");return 0;
}
