#ifndef COB_QWEN_H
#define COB_QWEN_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
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
void qwen_close(qwen_model *m);
const char *qwen_last_error(void);
int qwen_max_draft(void);
int qwen_reset(qwen_model *m);
/* Target decode retains every selected expert; cold weights move on explicit
 * expert handoff. No approximate target state is imported from another model. */
/* Verification includes the target anchor plus up to 16 MTP proposals.
 * want_logits: 0 state only; 1 all rows (<=17); 2 last row (configured prefill). */
int qwen_target(qwen_model *m,const int *tokens,int n_tokens,int want_logits);
int qwen_mtp_catchup(qwen_model *m,const int *tokens,int n_tokens,int position);
int qwen_mtp_step(qwen_model *m,int input_token,int position,int use_target_hidden,int *next_token);
int qwen_logits(qwen_model *m,float *out,int row,int mtp);
int qwen_top1(qwen_model *m,int *out,int rows,int mtp);
int qwen_checkpoint(qwen_model *m);
int qwen_restore(qwen_model *m);
void qwen_get_stats(qwen_model *m,qwen_stats *out);
/* Cumulative actual target-layer executions. A batch counts once per layer;
 * handoffs counts executions with one or more expert uploads, not uploads. */
void qwen_get_handoff_profile(qwen_model *m,qwen_handoff_profile *out);
#ifdef __cplusplus
}
#endif
#endif
