#ifndef COB_QWEN_GPU_H
#define COB_QWEN_GPU_H
#include <stddef.h>
/* Graphs contain only fixed-shape device work, never host routing decisions.
 * A capture must be ended or aborted before any host transfer/synchronization. */
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* C ABI; device arrays are token-major. CUDA implementation uses no model globals. */
void *qg_alloc(size_t bytes);
void qg_free(void *p);
int qg_write(void *dst, const void *src, size_t bytes);
int qg_read(void *dst, const void *src, size_t bytes);
int qg_copy(void *dst, const void *src, size_t bytes);
int qg_zero(void *dst, size_t bytes);
int qg_sync(void);
int qg_graph_begin(void **graph);
int qg_graph_end(void *graph);
int qg_graph_launch(void *graph);
void qg_graph_destroy(void *graph);
int qg_history_prefix(float *state,const float *saved,const float *input,int width,int history,int keep);
int qg_index_prefix(float *tail,const float *saved,const float *raw,int pos,int keep);
int qg_memory(size_t *free_bytes, size_t *total_bytes);
/* Call once before attention/index launches on the current CUDA device. */
int qg_prepare_context(int capacity);
/* scratch: 32768 * (sizeof(float)+sizeof(int)) bytes, reused across layers. */
int qg_index_workspace(float *keys,float *tail,int *allowed,const float *raw,const float *query,
    const float *kn,const float *qn,int nt,int pos,int cap,float theta,float eps,void *scratch);
const char *qg_error(int error);
int qg_tensor_bytes(int type, const uint64_t dims[4], uint64_t *bytes);
void *qg_upload_create(size_t expert_bytes);
void qg_upload_destroy(void *uploader);
int qg_upload_expert(void *uploader, void *dst, const void *gate, size_t gate_bytes,
                     const void *up, size_t up_bytes, const void *down, size_t down_bytes);
double qg_upload_seconds(void *uploader);
int qg_matmul(float *out, const void *w, int type, int rows, int cols,
              const float *x, int tokens);
int qg_gather(float *out, const void *w, int type, int cols, int row);
int qg_norm(float *out, const float *x, const float *weight, int width,
            int groups, int tokens, int weight_groups, int l2, float eps);
int qg_unary(float *out, const float *x, int count, int op, float scale);
int qg_binary(float *out, const float *a, const float *b, int count, int op);
int qg_hc_init(float *out, const float *emb, int width, int tokens);
int qg_hc_read(float *out, const float *norm, const float *gate, int width, int tokens);
int qg_hc_write(float *residual, const float *x, const float *inject, int width, int tokens);
int qg_mtp_concat(float *out, const float *e, const float *h, int width, int tokens);
int qg_conv(float *out, const float *x, const float *weight, float *history,
            int channels, int kernel, int dilation, int tokens, int silu);
int qg_gdn(float *out, const float *qkv, const float *alpha, const float *beta,
           const float *a, const float *dt, float *state, int tokens, float eps);
int qg_gate(float *x, const float *gate, int count);
int qg_qsplit(float *q, float *gate, const float *qfull, int tokens);
int qg_rope(float *x, int dim, int heads, int tokens, int position, float theta);
int qg_kv_store(void *cache, const float *x, int width, int tokens, int position);
int qg_attention(float *out, const float *q, const void *k, const void *v,
                 const int *allowed_blocks, int tokens, int position, int capacity);
int qg_index(float *block_keys, float *tail, int *allowed_blocks,
             const float *raw_keys, const float *queries,
             const float *key_norm, const float *query_norm,
             int tokens, int position, int capacity, float theta, float eps);
int qg_route(int *ids, float *weights, const float *logits, int tokens);
typedef struct {
    uint64_t token_layers,missing_selections,skipped_selections,avoided_token_handoffs;
    double skipped_mass;
} qg_prune_stats;
int qg_prune_routes(int *ids,float *weights,const unsigned char *resident,
                    int tokens,double threshold,qg_prune_stats *stats);
int qg_moe_map(int *map, const int *ids, int tokens);
int qg_moe_gather(float *out, const float *x, const int *token_map, int selected_tokens);
int qg_moe_scatter(float *slots, const float *expert, const float *weights, const int *ids,
                    const int *token_map, int expert_id, int selected_tokens);
int qg_moe_reduce(float *out, const float *slots, int tokens);
int qg_mtp_moe(float *slots, float *mid, const float *x, const int *ids, const float *weights,
                const void *gate, int gate_type, const void *up, int up_type,
                const void *down, int down_type, int tokens);
int qg_dequant(float *out, const void *packed, int type, int rows, int cols);
int qg_shared_add(float *out, const float *shared, const float *gate, int tokens);
int qg_ple_gate(float *out, const float *key, const float *query,
                const float *value, int tokens);
int qg_argmax(int *ids, const float *logits, int vocab, int tokens);
int qg_verify(int *best, int *eligible, const float *logits, const int *proposals,
              int vocab, int count, int top_k, float max_gap);
#ifdef __cplusplus
}
#endif
#endif
