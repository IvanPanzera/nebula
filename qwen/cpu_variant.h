/* Private names for the three separately compiled instruction sets. */
#define QC_JOIN_(a,b) a##b
#define QC_JOIN(a,b) QC_JOIN_(a,b)
#define qc_create QC_JOIN(QC_PREFIX,create)
#define qc_destroy QC_JOIN(QC_PREFIX,destroy)
#define qc_set_threads QC_JOIN(QC_PREFIX,set_threads)
#define qc_threads QC_JOIN(QC_PREFIX,threads)
#define qc_row_bytes QC_JOIN(QC_PREFIX,row_bytes)
#define qc_weight_at QC_JOIN(QC_PREFIX,weight_at)
#define qc_matmul QC_JOIN(QC_PREFIX,matmul)
#define qc_moe QC_JOIN(QC_PREFIX,moe)
