#ifndef COB_QWEN_H
#define COB_QWEN_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Read-only metadata/shape preflight: maps files, never loads weight payloads. */
int qwen_validate_index(const char *index_path);
typedef struct qwen_model qwen_model;
enum { QWEN_MAX_DRAFT = 16, QWEN_VERIFY_ROWS = QWEN_MAX_DRAFT + 1 };
typedef struct {
    uint64_t gpu_weight_bytes, gpu_state_bytes, gpu_workspace_bytes, gpu_arena_bytes;
    uint64_t target_tokens, mtp_tokens, expert_handoffs, expert_upload_bytes;
    uint64_t expert_hits, expert_requests, metadata_read_bytes, ple_read_bytes;
    uint64_t host_weight_bytes, host_staging_bytes;
    double target_seconds, mtp_seconds, expert_handoff_host_seconds, expert_dma_seconds, ple_lookup_seconds, core_load_seconds, host_load_seconds;
    int target_position, mtp_position, context, resident_per_layer;
} qwen_stats;
typedef struct {
    uint64_t layer_calls[48], layer_handoffs[48];
} qwen_handoff_profile;
qwen_model *qwen_open(const char *index_path,int context,int resident_per_layer,int prefill_capacity);
/* RAM count includes host copies of GPU experts. Ranking contains all 48*512
 * IDs. NULL ranking is valid only with 512 host experts. ple_on_ssd is 0/1. */
qwen_model *qwen_open_storage(const char *index_path,int context,int resident_per_layer,int prefill_capacity,
    const int *ranking,int ram_experts,int ple_on_ssd,uint64_t buffer_bytes,int io_threads);
typedef struct {
    uint64_t ram_expert_bytes,transient_peak_bytes,expert_read_bytes,expert_reads;
    double expert_io_seconds,expert_wait_seconds;
    uint64_t ple_disk_bytes,ple_cache_hits,ple_cache_misses;
    int ram_experts_per_layer,ple_on_ssd;
} qwen_storage_stats;
void qwen_get_storage_stats(qwen_model *m,qwen_storage_stats *out);
void qwen_close(qwen_model *m);
const char *qwen_last_error(void);
int qwen_max_draft(void);
int qwen_reset(qwen_model *m);
/* Fixed residents on GPU; non-marginal routed MoE misses run on CPU.
 * Missing current-token softmax mass below 0.10 is omitted without renormalizing. */
/* Verification includes the target anchor plus up to 16 MTP proposals.
 * want_logits: 0 state only; 1 all rows (<=17); 2 last row (configured prefill). */
int qwen_target(qwen_model *m,const int *tokens,int n_tokens,int want_logits);
int qwen_mtp_catchup(qwen_model *m,const int *tokens,int n_tokens,int position);
int qwen_mtp_step(qwen_model *m,int input_token,int position,int use_target_hidden,int *next_token);
int qwen_logits(qwen_model *m,float *out,int row,int mtp);
int qwen_top1(qwen_model *m,int *out,int rows,int mtp);
/* Last target batch must contain an anchor plus count proposals. */
int qwen_verify(qwen_model *m,const int *proposals,int count,int top_k,float min_ratio,int *best,int *eligible);
int qwen_checkpoint(qwen_model *m);
int qwen_restore(qwen_model *m);
/* Retain a verified prefix of the last checkpointed batch without rerunning
 * its target projections or MoE. Reconstruct only the recurrent state. */
int qwen_commit_prefix(qwen_model *m,int tokens);
typedef struct {
    uint64_t retained_tokens, prefix_restores, graph_launches, graph_captures;
    double prefix_restore_seconds;
} qwen_optimization_stats;
void qwen_get_optimization_stats(qwen_model *m,qwen_optimization_stats *out);
void qwen_get_stats(qwen_model *m,qwen_stats *out);
/* Separate from stats.host_weight_bytes (routed experts). Source PLE bytes
 * stay quantized, fully loaded in host RAM; zero for probes without PLE. */
uint64_t qwen_ple_ram_bytes(qwen_model *m);
/* Cumulative actual target-layer executions. A batch counts once per layer;
 * handoffs counts executions recovered on CPU; initial pinning is excluded. */
void qwen_get_handoff_profile(qwen_model *m,qwen_handoff_profile *out);
/* Diagnostic trace; NULL closes/flushed the current file. No MTP routes. */
int qwen_profile_routes(qwen_model *m,const char *path);
typedef struct {
    uint64_t layer_handoffs,token_layers,missing_selections,activation_bytes;
    double compute_seconds,handoff_seconds;
} qwen_cpu_stats;
/* Official mode: pin IDs once, never replace them, recover a missing routed MoE
 * on the CPU. Shared experts and all attention/context state stay on GPU. */
int qwen_set_hotlist(qwen_model *m,const int *ids,int count,int threads);
/* Idle caller only. Performance tuning does not change weights or cached state. */
int qwen_set_cpu_threads(qwen_model *m,int threads);
int qwen_cpu_threads(qwen_model *m);
void qwen_get_cpu_stats(qwen_model *m,qwen_cpu_stats *out);
int qwen_handoff_version(void);
typedef struct {
    uint64_t token_layers,missing_selections,skipped_selections,avoided_token_handoffs;
    double skipped_mass;
} qwen_prune_stats;
int qwen_get_prune_stats(qwen_model *m,qwen_prune_stats *out);
#ifdef __cplusplus
}
#endif
#endif
