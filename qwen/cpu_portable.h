/* AVX2 and scalar dot products keep the AVX-512/CUDA 32-lane sum order. */
#if defined(QC_AVX2)
static inline __m256 q8(__m128i bytes,int upper){
    if(upper)bytes=_mm_srli_epi16(bytes,4);
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(bytes,_mm_set1_epi8(15))));
}
static inline void unpack_avx2(const unsigned char *row,int type,int k,__m256 v[4]){
    if(type==12){
        const unsigned char *p=row+(size_t)(k/256)*144,*sc=p+4;int g=(k%256)/32;
        int s=g<4?(sc[g]&63):((sc[g+4]&15)|((sc[g-4]>>6)<<4));
        int m=g<4?(sc[g+4]&63):((sc[g+4]>>4)|((sc[g]>>6)<<4));
        __m256 d=_mm256_set1_ps(half(p)*s),off=_mm256_set1_ps(-half(p+2)*m);
        for(int j=0;j<4;j++)v[j]=_mm256_fmadd_ps(d,q8(_mm_loadl_epi64((const __m128i*)(p+16+(g/2)*32+j*8)),g&1),off);
    }else if(type==20||type==2){
        const unsigned char *p=row+(k/32)*18;
        const __m128i lut=_mm_setr_epi8(-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113);
        for(int j=0;j<4;j++){
            __m128i bytes=_mm_loadl_epi64((const __m128i*)(p+2+(j%2)*8));
            if(j>=2)bytes=_mm_srli_epi16(bytes,4);
            bytes=_mm_and_si128(bytes,_mm_set1_epi8(15));
            __m256 q=type==20?_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(lut,bytes))):
                _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes)),_mm256_set1_ps(8));
            v[j]=_mm256_mul_ps(_mm256_set1_ps(half(p)),q);
        }
    }else{
        float values[32];
        for(int j=0;j<32;j++)values[j]=qc_weight_at(row,type,k+j);
        for(int j=0;j<4;j++)v[j]=_mm256_loadu_ps(values+j*8);
    }
}
#endif
static inline __attribute__((always_inline)) void dot_group_inner(float out[TILE],const unsigned char *row,int type,int cols,
                      const float *const input[TILE],int n){
#if defined(QC_AVX2)
    __m256 acc[TILE][4];
    for(int t=0;t<n;t++)for(int j=0;j<4;j++)acc[t][j]=_mm256_setzero_ps();
    for(int k=0;k<cols;k+=32){
        __m256 v[4];unpack_avx2(row,type,k,v);
        for(int t=0;t<n;t++)for(int j=0;j<4;j++)
            acc[t][j]=_mm256_fmadd_ps(v[j],_mm256_loadu_ps(input[t]+k+j*8),acc[t][j]);
    }
    for(int t=0;t<n;t++){
        __m256 s=_mm256_add_ps(_mm256_add_ps(acc[t][0],acc[t][2]),_mm256_add_ps(acc[t][1],acc[t][3]));
        __m128 a=_mm_add_ps(_mm256_castps256_ps128(s),_mm256_extractf128_ps(s,1));
        a=_mm_add_ps(a,_mm_movehl_ps(a,a));
        out[t]=_mm_cvtss_f32(_mm_add_ss(a,_mm_shuffle_ps(a,a,0x55)));
    }
#else
    float acc[TILE][32]={{0}};
    for(int k=0;k<cols;k+=32)for(int j=0;j<32;j++){
        float w=qc_weight_at(row,type,k+j);
        for(int t=0;t<n;t++)acc[t][j]=fmaf(w,input[t][k+j],acc[t][j]);
    }
    for(int t=0;t<n;t++){
        for(int stride=16;stride;stride/=2)for(int j=0;j<stride;j++)acc[t][j]+=acc[t][j+stride];
        out[t]=acc[t][0];
    }
#endif
}
