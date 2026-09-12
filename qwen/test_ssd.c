#define _GNU_SOURCE
#include "ssd.h"
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void){
    char path[]="/tmp/qwen-ssd-test-XXXXXX",error[256];
    int fd=mkstemp(path);assert(fd>=0);unlink(path);
    const int layers=2;const size_t bytes=4096,span=bytes*512;
    assert(ftruncate(fd,span*layers*3)==0);
    qs_source src[6];int ranking[1024];unsigned char page[4096];
    for(int l=0;l<layers;l++){
        for(int e=0;e<512;e++)ranking[l*512+e]=(e*13+l*7)%512;
        for(int k=0;k<3;k++){
            int i=l*3+k;src[i]=(qs_source){fd,i*span,bytes};
            for(int e=0;e<512;e++){
                memset(page,(l*31+k*17+e)%256,bytes);
                assert(pwrite(fd,page,bytes,src[i].offset+e*bytes)==(ssize_t)bytes);
            }
        }
    }
    const uint64_t budget=10*3*bytes;
    qs_store *s=qs_create(src,layers,ranking,3,budget,2,error);assert(s);
    qs_stats stats;qs_get_stats(s,&stats);assert(stats.ram_bytes==layers*3*3*bytes&&!stats.read_bytes);
    for(int l=0;l<layers;l++)for(int j=0;j<3;j++)for(int k=0;k<3;k++){
        int e=ranking[l*512+j];const unsigned char *p=qs_tensor(s,l,k);
        assert(qs_is_hot(s,l,e));assert(p[e*bytes]==(l*31+k*17+e)%256);
    }
    int ids[30];for(int i=0;i<30;i++)ids[i]=100+i;
    for(int loop=0;loop<20;loop++)for(int l=0;l<layers;l++){
        for(int start=0;start<3;){
            int n=qs_begin(s,l,ids+start*10,3-start);assert(n==1);assert(qs_wait(s)==0);
            for(int k=0;k<3;k++){
                unsigned char resident[512];const unsigned char *p=qs_tensor(s,l,k);
                assert(mincore((void*)p,span,resident)==0);
                int pages=0;for(int j=0;j<512;j++)pages+=resident[j]&1;
                assert(pages==13);
                for(int j=0;j<10;j++){int e=ids[start*10+j];assert(p[e*bytes]==(l*31+k*17+e)%256);}
            }
            assert(qs_release(s)==0);
            for(int k=0;k<3;k++){
                unsigned char resident[512];assert(mincore((void*)qs_tensor(s,l,k),span,resident)==0);
                int pages=0;for(int j=0;j<512;j++)pages+=resident[j]&1;assert(pages==3);
            }
            start+=n;
        }
    }
    qs_get_stats(s,&stats);assert(stats.transient_peak_bytes==budget);
    assert(stats.expert_reads==20*2*3*10&&stats.read_bytes==stats.expert_reads*3*bytes);
    /* Pure hot/omitted routes require no disk reads. */
    for(int i=0;i<10;i++)ids[i]=-2;
    ids[0]=ranking[0];assert(qs_begin(s,0,ids,1)==1&&qs_wait(s)==0&&qs_release(s)==0);
    qs_stats after;qs_get_stats(s,&after);assert(after.read_bytes==stats.read_bytes);
    ids[1]=ids[0];assert(qs_begin(s,0,ids,1)<0);ids[1]=512;assert(qs_begin(s,0,ids,1)<0);
    qs_store *extra=qs_create(src,layers,ranking,512,0,4,error);assert(extra);qs_destroy(extra);
    extra=qs_create(src,layers,ranking,0,budget,1,error);assert(extra);
    for(int i=0;i<10;i++)ids[i]=i;
    assert(qs_begin(extra,1,ids,1)==1);qs_destroy(extra); /* Join pending readers before unmapping. */
    assert(!qs_create(src,layers,ranking,0,budget-1,2,error));
    ranking[1]=ranking[0];assert(!qs_create(src,layers,ranking,3,budget,2,error));ranking[1]=13;
    assert(!qs_create(src,layers,ranking,3,budget,5,error));
    for(int i=0;i<10;i++)ids[i]=-2;
    ids[0]=ranking[0];
    /* Short reads fail before CPU access, with no invented/zero weights. */
    ids[1]=511;assert(ftruncate(fd,span/2)==0);
    assert(qs_begin(s,0,ids,1)==1);assert(qs_wait(s)<0);assert(strstr(qs_error(s),"SSD read"));
    qs_destroy(s);
    assert(!qs_create(src,layers,ranking,3,budget,2,error));
    close(fd);
    puts("PASS SSD: ranking, exact bytes, hot/omitted zero reads, bounded physical pages, cold release, 120 reuse cycles, invalid routes and truncated files");
    return 0;
}
