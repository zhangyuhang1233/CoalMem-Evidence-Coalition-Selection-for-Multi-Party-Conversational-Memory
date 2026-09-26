#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONVERSATION_JSON=${CONVERSATION_JSON:-"${ROOT_DIR}/data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json"}
QUESTIONS_JSONL=${QUESTIONS_JSONL:-""}
ENV_FILE=${ENV_FILE:-".env"}
AGENT_MODEL=${AGENT_MODEL:-"deepseek-chat"}
JUDGE_MODEL=${JUDGE_MODEL:-"deepseek-chat"}
AGENT_THINKING=${AGENT_THINKING:-}
JUDGE_THINKING=${JUDGE_THINKING:-}
AGENT_PROMPT=${AGENT_PROMPT:-"prompts/agent_system.txt"}
JUDGE_PROMPT=${JUDGE_PROMPT:-"prompts/judge_system.txt"}
OUTPUT_JSONL=${OUTPUT_JSONL:-"results/embedding_eval_results.jsonl"}
LLM_PROVIDER=${LLM_PROVIDER:-""}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /home/pkuccadm/anaconda3/envs/llama/bin/python ]]; then
    PYTHON_BIN=/home/pkuccadm/anaconda3/envs/llama/bin/python
  elif [[ -x /root/miniconda3/bin/python ]]; then
    PYTHON_BIN=/root/miniconda3/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No Python interpreter found (tried python3 and python)." >&2
    exit 127
  fi
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 127
fi

# F1 defaults. For content-only/question-only retrieval, set both explicitly.
INDEX_FIELDS=content
QUERY_FIELDS=question_only
RETRIEVE_TOP_K=${RETRIEVE_TOP_K:-10}
AGENT_MAX_TOKENS=${AGENT_MAX_TOKENS:-512}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-256}
NUM_WORKERS=${NUM_WORKERS:-1}
LOCAL_EMBEDDING_MODEL_PATH=${LOCAL_EMBEDDING_MODEL_PATH:-"/data/model/Qwen3-Embedding-8B"}
EMBEDDING_GPU_IDS=${EMBEDDING_GPU_IDS:-"0"}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-32}
EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-512}
EMBEDDING_DTYPE=${EMBEDDING_DTYPE:-bfloat16}

if [[ -z "${LOCAL_EMBEDDING_MODEL_PATH}" ]]; then
  echo "LOCAL_EMBEDDING_MODEL_PATH is required." >&2
  exit 2
fi

ARGS=(
  --conversation-json "${CONVERSATION_JSON}"
  --questions-jsonl "${QUESTIONS_JSONL}"
  --env-file "${ENV_FILE}"
  --agent-model "${AGENT_MODEL}"
  --judge-model "${JUDGE_MODEL}"
  --agent-prompt "${AGENT_PROMPT}"
  --judge-prompt "${JUDGE_PROMPT}"
  --output-jsonl "${OUTPUT_JSONL}"
  --index-fields "${INDEX_FIELDS}"
  --query-fields "${QUERY_FIELDS}"
  --retrieve-top-k "${RETRIEVE_TOP_K}"
  --agent-max-tokens "${AGENT_MAX_TOKENS}"
  --judge-max-tokens "${JUDGE_MAX_TOKENS}"
  --num-workers "${NUM_WORKERS}"
  --embedding-model-path "${LOCAL_EMBEDDING_MODEL_PATH}"
  --gpu-ids "${EMBEDDING_GPU_IDS}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
  --embedding-max-length "${EMBEDDING_MAX_LENGTH}"
  --embedding-dtype "${EMBEDDING_DTYPE}"
)
if [[ -n "${LLM_PROVIDER}" ]]; then
  ARGS+=(--llm-provider "${LLM_PROVIDER}")
fi
if [[ -n "${AGENT_THINKING}" ]]; then
  ARGS+=(--agent-thinking "${AGENT_THINKING}")
fi
if [[ -n "${JUDGE_THINKING}" ]]; then
  ARGS+=(--judge-thinking "${JUDGE_THINKING}")
fi
if [[ "${INGEST_ONLY:-0}" == "1" ]]; then
  ARGS+=(--ingest-only)
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi

PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" \
  baselines/embedding/eval_benchmark.py "${ARGS[@]}" "$@"
