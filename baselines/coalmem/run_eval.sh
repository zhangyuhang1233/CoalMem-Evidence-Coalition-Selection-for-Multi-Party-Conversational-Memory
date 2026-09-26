#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-}

CONVERSATION_JSON=${CONVERSATION_JSON:?CONVERSATION_JSON is required}
QUESTIONS_JSONL=${QUESTIONS_JSONL:?QUESTIONS_JSONL is required}
OUTPUT_JSONL=${OUTPUT_JSONL:?OUTPUT_JSONL is required}
DOMAIN="$(basename "$(dirname "${CONVERSATION_JSON}")")"

CDS_MODE=${CDS_MODE:-hybrid}
CDS_CANDIDATE_RETRIEVAL_MODE=${CDS_CANDIDATE_RETRIEVAL_MODE:-bm25}
CANDIDATE_TOP_K=${CANDIDATE_TOP_K:-48}
RETRIEVE_TOP_K=${RETRIEVE_TOP_K:-10}
CDS_SCOPE_FILTER_TOP_K=${CDS_SCOPE_FILTER_TOP_K:-0}
CDS_SCOPE_SCORE_TOP_N=${CDS_SCOPE_SCORE_TOP_N:-3}
CDS_QUERY_TEXT_MODE=${CDS_QUERY_TEXT_MODE:-expandqc}
CDS_QUERY_AFFINITY_MODE=${CDS_QUERY_AFFINITY_MODE:-candidate_bm25}
CDS_MULTIVIEW_TOP_K=${CDS_MULTIVIEW_TOP_K:-10}
CDS_MULTIVIEW_CONSENSUS_WEIGHT=${CDS_MULTIVIEW_CONSENSUS_WEIGHT:-0.05}
BM25F_CONTENT_WEIGHT=${BM25F_CONTENT_WEIGHT:-1.0}
BM25F_PARTICIPANT_WEIGHT=${BM25F_PARTICIPANT_WEIGHT:-1.0}
BM25F_EPISODE_WEIGHT=${BM25F_EPISODE_WEIGHT:-1.0}
BM25F_CONTENT_B=${BM25F_CONTENT_B:-0.75}
BM25F_PARTICIPANT_B=${BM25F_PARTICIPANT_B:-0.0}
BM25F_EPISODE_B=${BM25F_EPISODE_B:-0.0}
BM25F_K1=${BM25F_K1:-1.5}
CDS_COMPLEMENTARITY_WEIGHT=${CDS_COMPLEMENTARITY_WEIGHT:-0.0}
CDS_COMPLEMENTARITY_FIELDS=${CDS_COMPLEMENTARITY_FIELDS:-content}
CDS_COMPLEMENTARITY_NORMALIZATION=${CDS_COMPLEMENTARITY_NORMALIZATION:-raw}
CDS_DYNAMIC_COVERAGE_WEIGHT=${CDS_DYNAMIC_COVERAGE_WEIGHT:-0.0}
CDS_NATIVE_COMPLEMENTARITY_WEIGHT=${CDS_NATIVE_COMPLEMENTARITY_WEIGHT:-0.0}
CDS_NATIVE_SURPRISE_WEIGHT=${CDS_NATIVE_SURPRISE_WEIGHT:-0.0}
CDS_NATIVE_BRIDGE=${CDS_NATIVE_BRIDGE:-0}
CDS_QUERY_DYNAMIC_EDGE=${CDS_QUERY_DYNAMIC_EDGE:-0}
CDS_DYNAMIC_EDGE_FLOOR=${CDS_DYNAMIC_EDGE_FLOOR:-0.5}
CDS_EVIDENCE_ORDER=${CDS_EVIDENCE_ORDER:-cds}
CDS_SELECTION_RELEVANCE_POWER=${CDS_SELECTION_RELEVANCE_POWER:-0.0}
CDS_EPSILON=${CDS_EPSILON:-1e-3}
CDS_TOLERANCE=${CDS_TOLERANCE:-1e-8}
CDS_MAX_ITERATIONS=${CDS_MAX_ITERATIONS:-1000}
CDS_SOLVER_MODE=${CDS_SOLVER_MODE:-self_loop}
CDS_QUERY_MASS=${CDS_QUERY_MASS:-0.75}
CDS_TOPK_GRAPH_WEIGHT=${CDS_TOPK_GRAPH_WEIGHT:-0.25}
CDS_FIXED_SHARE_RATE=${CDS_FIXED_SHARE_RATE:-0.05}
CDS_IHT_INITIAL_STEP=${CDS_IHT_INITIAL_STEP:-1.0}
CDS_IHT_BACKTRACKING_FACTOR=${CDS_IHT_BACKTRACKING_FACTOR:-0.5}
CDS_IHT_ARMIJO_SIGMA=${CDS_IHT_ARMIJO_SIGMA:-1e-4}
CDS_IHT_MIN_STEP=${CDS_IHT_MIN_STEP:-1e-8}
CDS_IHT_MAX_BACKTRACKING=${CDS_IHT_MAX_BACKTRACKING:-50}
RELATION_GRAPH=${RELATION_GRAPH:-complete}
RELATION_MODEL=${RELATION_MODEL:-deepseek-v4-flash}
RELATION_MAX_TOKENS=${RELATION_MAX_TOKENS:-2048}
RELATION_PAIR_BATCH_SIZE=${RELATION_PAIR_BATCH_SIZE:-16}
NATIVE_CHANNEL_WINDOW=${NATIVE_CHANNEL_WINDOW:-5}
NATIVE_SCOPE_NEIGHBORS=${NATIVE_SCOPE_NEIGHBORS:-1}
RELATION_VERIFY=${RELATION_VERIFY:-1}
RELATION_THRESHOLD_IQR_MULTIPLIER=${RELATION_THRESHOLD_IQR_MULTIPLIER:-0.5}
RELATION_THRESHOLD_MIN_PAIRS=${RELATION_THRESHOLD_MIN_PAIRS:-0}
RELATION_THRESHOLD_MAX_PAIRS=${RELATION_THRESHOLD_MAX_PAIRS:-256}

EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH:-Qwen/Qwen3-Embedding-8B}
EMBEDDING_MODEL_SLUG=${EMBEDDING_MODEL_SLUG:-qwen3_embedding_8b}
CDS_EMBEDDING_INDEX_FIELDS=${CDS_EMBEDDING_INDEX_FIELDS:-content}
case "${CDS_EMBEDDING_INDEX_FIELDS}" in
  content) EMBEDDING_FIELD_LABEL=f0 ;;
  structured) EMBEDDING_FIELD_LABEL=f1 ;;
  *)
    echo "CDS_EMBEDDING_INDEX_FIELDS must be content or structured" >&2
    exit 2
    ;;
esac
EMBEDDING_STORE_DIR=${EMBEDDING_STORE_DIR:-"${ROOT_DIR}/stores/embedding/${EMBEDDING_FIELD_LABEL}_${EMBEDDING_MODEL_SLUG}/${DOMAIN}_store"}
CDS_PAIR_EMBEDDING_INDEX_FIELDS=${CDS_PAIR_EMBEDDING_INDEX_FIELDS:-${CDS_EMBEDDING_INDEX_FIELDS}}
case "${CDS_PAIR_EMBEDDING_INDEX_FIELDS}" in
  content) PAIR_EMBEDDING_FIELD_LABEL=f0 ;;
  structured) PAIR_EMBEDDING_FIELD_LABEL=f1 ;;
  *)
    echo "CDS_PAIR_EMBEDDING_INDEX_FIELDS must be content or structured" >&2
    exit 2
    ;;
esac
PAIR_EMBEDDING_STORE_DIR=${PAIR_EMBEDDING_STORE_DIR:-"${ROOT_DIR}/stores/embedding/${PAIR_EMBEDDING_FIELD_LABEL}_${EMBEDDING_MODEL_SLUG}/${DOMAIN}_store"}
if [[ -z "${EMBEDDING_CACHE_ONLY+x}" ]]; then
  if [[ "${CDS_CANDIDATE_RETRIEVAL_MODE}" == "embedding" ]]; then
    EMBEDDING_CACHE_ONLY=0
  else
    EMBEDDING_CACHE_ONLY=1
  fi
fi
if [[ -z "${EMBEDDING_GPU_IDS+x}" ]]; then
  if [[ "${EMBEDDING_CACHE_ONLY}" == "0" ]]; then
    EMBEDDING_GPU_IDS=0
  else
    EMBEDDING_GPU_IDS=""
  fi
fi
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-32}
EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-512}
EMBEDDING_DTYPE=${EMBEDDING_DTYPE:-bfloat16}

# The released CoalMem configuration uses BM25 and therefore requires no
# embedding model. Optional dense modes use the caller's active environment;
# set PYTHON_BIN explicitly when those dependencies live elsewhere.
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi

args=(
  --conversation-json "${CONVERSATION_JSON}"
  --questions-jsonl "${QUESTIONS_JSONL}"
  --output-jsonl "${OUTPUT_JSONL}"
  --env-file "${ENV_FILE:-.env}"
  --cds-mode "${CDS_MODE}"
  --candidate-retrieval-mode "${CDS_CANDIDATE_RETRIEVAL_MODE}"
  --candidate-top-k "${CANDIDATE_TOP_K}"
  --retrieve-top-k "${RETRIEVE_TOP_K}"
  --scope-filter-top-k "${CDS_SCOPE_FILTER_TOP_K}"
  --scope-score-top-n "${CDS_SCOPE_SCORE_TOP_N}"
  --query-text-mode "${CDS_QUERY_TEXT_MODE}"
  --query-affinity-mode "${CDS_QUERY_AFFINITY_MODE}"
  --multiview-top-k "${CDS_MULTIVIEW_TOP_K}"
  --multiview-consensus-weight "${CDS_MULTIVIEW_CONSENSUS_WEIGHT}"
  --bm25f-content-weight "${BM25F_CONTENT_WEIGHT}"
  --bm25f-participant-weight "${BM25F_PARTICIPANT_WEIGHT}"
  --bm25f-episode-weight "${BM25F_EPISODE_WEIGHT}"
  --bm25f-content-b "${BM25F_CONTENT_B}"
  --bm25f-participant-b "${BM25F_PARTICIPANT_B}"
  --bm25f-episode-b "${BM25F_EPISODE_B}"
  --bm25f-k1 "${BM25F_K1}"
  --complementarity-weight "${CDS_COMPLEMENTARITY_WEIGHT}"
  --complementarity-fields "${CDS_COMPLEMENTARITY_FIELDS}"
  --complementarity-normalization "${CDS_COMPLEMENTARITY_NORMALIZATION}"
  --dynamic-coverage-weight "${CDS_DYNAMIC_COVERAGE_WEIGHT}"
  --native-complementarity-weight "${CDS_NATIVE_COMPLEMENTARITY_WEIGHT}"
  --native-surprise-weight "${CDS_NATIVE_SURPRISE_WEIGHT}"
  --native-bridge "${CDS_NATIVE_BRIDGE}"
  --query-dynamic-edge "${CDS_QUERY_DYNAMIC_EDGE}"
  --dynamic-edge-floor "${CDS_DYNAMIC_EDGE_FLOOR}"
  --evidence-order "${CDS_EVIDENCE_ORDER}"
  --selection-relevance-power "${CDS_SELECTION_RELEVANCE_POWER}"
  --cds-epsilon "${CDS_EPSILON}"
  --cds-tolerance "${CDS_TOLERANCE}"
  --cds-max-iterations "${CDS_MAX_ITERATIONS}"
  --cds-solver-mode "${CDS_SOLVER_MODE}"
  --cds-query-mass "${CDS_QUERY_MASS}"
  --topk-graph-weight "${CDS_TOPK_GRAPH_WEIGHT}"
  --fixed-share-rate "${CDS_FIXED_SHARE_RATE}"
  --iht-initial-step "${CDS_IHT_INITIAL_STEP}"
  --iht-backtracking-factor "${CDS_IHT_BACKTRACKING_FACTOR}"
  --iht-armijo-sigma "${CDS_IHT_ARMIJO_SIGMA}"
  --iht-min-step "${CDS_IHT_MIN_STEP}"
  --iht-max-backtracking "${CDS_IHT_MAX_BACKTRACKING}"
  --relation-graph "${RELATION_GRAPH}"
  --relation-model "${RELATION_MODEL}"
  --relation-max-tokens "${RELATION_MAX_TOKENS}"
  --relation-pair-batch-size "${RELATION_PAIR_BATCH_SIZE}"
  --native-channel-window "${NATIVE_CHANNEL_WINDOW}"
  --native-scope-neighbors "${NATIVE_SCOPE_NEIGHBORS}"
  --relation-verify "${RELATION_VERIFY}"
  --relation-threshold-iqr-multiplier "${RELATION_THRESHOLD_IQR_MULTIPLIER}"
  --relation-threshold-min-pairs "${RELATION_THRESHOLD_MIN_PAIRS}"
  --relation-threshold-max-pairs "${RELATION_THRESHOLD_MAX_PAIRS}"
  --embedding-index-fields "${CDS_EMBEDDING_INDEX_FIELDS}"
  --pair-embedding-index-fields "${CDS_PAIR_EMBEDDING_INDEX_FIELDS}"
  --embedding-store-dir "${EMBEDDING_STORE_DIR}"
  --pair-embedding-store-dir "${PAIR_EMBEDDING_STORE_DIR}"
  --embedding-model-path "${EMBEDDING_MODEL_PATH}"
  --embedding-cache-only "${EMBEDDING_CACHE_ONLY}"
  --embedding-gpu-ids "${EMBEDDING_GPU_IDS}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
  --embedding-max-length "${EMBEDDING_MAX_LENGTH}"
  --embedding-dtype "${EMBEDDING_DTYPE}"
  --agent-model "${AGENT_MODEL:-gpt-5}"
  --judge-model "${JUDGE_MODEL:-gpt-5}"
  --agent-max-tokens "${AGENT_MAX_TOKENS:-512}"
  --judge-max-tokens "${JUDGE_MAX_TOKENS:-256}"
  --num-workers "${NUM_WORKERS:-1}"
)

[[ -n "${LLM_PROVIDER:-}" ]] && args+=(--llm-provider "${LLM_PROVIDER}")
[[ -n "${AGENT_THINKING:-}" ]] && args+=(--agent-thinking "${AGENT_THINKING}")
[[ -n "${JUDGE_THINKING:-}" ]] && args+=(--judge-thinking "${JUDGE_THINKING}")
[[ -n "${AGENT_REASONING_EFFORT:-}" ]] && args+=(--agent-reasoning-effort "${AGENT_REASONING_EFFORT}")
[[ -n "${JUDGE_REASONING_EFFORT:-}" ]] && args+=(--judge-reasoning-effort "${JUDGE_REASONING_EFFORT}")
[[ -n "${RELATION_THINKING:-}" ]] && args+=(--relation-thinking "${RELATION_THINKING}")
[[ "${INGEST_ONLY:-0}" == "1" ]] && args+=(--ingest-only)

exec "${PYTHON_BIN}" baselines/coalmem/eval_benchmark.py "${args[@]}"
