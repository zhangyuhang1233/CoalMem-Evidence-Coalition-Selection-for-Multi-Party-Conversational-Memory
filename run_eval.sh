#!/usr/bin/env bash
# Evaluate CoalMem on GroupMemBench. Each failed cell is logged and the
# remaining cells continue to run; the script exits non-zero at the end when
# at least one cell failed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

REQUIRED_VARS=(
    DOMAINS QTYPES
    CDS_MODE RELATION_GRAPH
    CANDIDATE_TOP_K RETRIEVE_TOP_K
    CDS_QUERY_TEXT_MODE CDS_QUERY_AFFINITY_MODE
    CDS_COMPLEMENTARITY_WEIGHT CDS_COMPLEMENTARITY_FIELDS
    CDS_COMPLEMENTARITY_NORMALIZATION CDS_EVIDENCE_ORDER
    CDS_SOLVER_MODE CDS_EPSILON CDS_TOLERANCE CDS_MAX_ITERATIONS
    AGENT_MODEL JUDGE_MODEL AGENT_THINKING JUDGE_THINKING
    AGENT_MAX_TOKENS JUDGE_MAX_TOKENS NUM_WORKERS RESULTS_ROOT
)

missing_vars=()
for variable in "${REQUIRED_VARS[@]}"; do
    [[ -n "${!variable:-}" ]] || missing_vars+=("${variable}")
done
if [[ "${#missing_vars[@]}" -gt 0 ]]; then
    echo "Missing required configuration variables:" >&2
    printf '  %s\n' "${missing_vars[@]}" >&2
    echo "Configure the experiment before running this script." >&2
    exit 2
fi

DOMAINS_STR="${DOMAINS}"
QTYPES_STR="${QTYPES}"
PHASES_STR="${PHASES:-qa summarize}"
QUESTIONS_ROOT="${QUESTIONS_ROOT:-questions}"
DATA_ROOT="${DATA_ROOT:-data/final}"
RESULTS_ROOT="${RESULTS_ROOT}"
ENV_FILE="${ENV_FILE:-.env}"
FORCE_QA_RERUN="${FORCE_QA_RERUN:-0}"
BASELINE="coalmem"

read -r -a DOMAINS <<<"${DOMAINS_STR}"
read -r -a QTYPES <<<"${QTYPES_STR}"
read -r -a PHASES <<<"${PHASES_STR//,/ }"

has_phase() {
    local wanted="$1" phase
    for phase in "${PHASES[@]}"; do
        [[ "${phase}" == "${wanted}" ]] && return 0
    done
    return 1
}

failures=0

run_qa() {
    local domain qtype conversation questions output skipped log rc expected completed skipped_count
    echo "============================================================"
    echo "Phase: qa"
    echo "============================================================"

    for domain in "${DOMAINS[@]}"; do
        conversation="${DATA_ROOT}/${domain}/synthetic_domain_channels_rolevariants_${domain}.json"
        if [[ ! -f "${conversation}" ]]; then
            echo "[error] Missing conversation file: ${conversation}" >&2
            failures=$((failures + 1))
            continue
        fi

        mkdir -p "${RESULTS_ROOT}/${domain}/logs"
        for qtype in "${QTYPES[@]}"; do
            questions="${QUESTIONS_ROOT}/${domain}/${qtype}.jsonl"
            output="${RESULTS_ROOT}/${domain}/${BASELINE}__${qtype}.jsonl"
            skipped="${output}.skipped"
            log="${RESULTS_ROOT}/${domain}/logs/qa__${BASELINE}__${qtype}.log"

            if [[ ! -f "${questions}" ]]; then
                echo "[error] Missing question file: ${questions}" >&2
                failures=$((failures + 1))
                continue
            fi
            if [[ "${FORCE_QA_RERUN}" != "1" ]]; then
                expected="$(wc -l <"${questions}" | tr -d ' ')"
                completed=0
                skipped_count=0
                [[ -f "${output}" ]] && completed="$(wc -l <"${output}" | tr -d ' ')"
                [[ -f "${skipped}" ]] && skipped_count="$(wc -l <"${skipped}" | tr -d ' ')"
                if [[ $((completed + skipped_count)) -eq "${expected}" ]]; then
                    echo "[skip] ${domain}/${qtype}: ${completed} evaluated, ${skipped_count} skipped"
                    continue
                fi
                if [[ $((completed + skipped_count)) -gt 0 ]]; then
                    echo "[rerun] ${domain}/${qtype}: incomplete cell (${completed}+${skipped_count}/${expected})"
                fi
            fi

            echo "[run] ${domain}/${qtype} -> ${output}"
            rc=0
            env \
                CONVERSATION_JSON="${conversation}" \
                QUESTIONS_JSONL="${questions}" \
                OUTPUT_JSONL="${output}" \
                ENV_FILE="${ENV_FILE}" \
                bash baselines/coalmem/run_eval.sh >"${log}" 2>&1 || rc=$?
            if [[ "${rc}" -ne 0 ]]; then
                echo "[warn] ${domain}/${qtype} failed (rc=${rc}); see ${log}" >&2
                failures=$((failures + 1))
            fi
        done
    done
}

run_summary() {
    local domain qtypes_csv
    qtypes_csv="$(IFS=,; echo "${QTYPES[*]}")"
    echo "============================================================"
    echo "Phase: summarize"
    echo "============================================================"

    for domain in "${DOMAINS[@]}"; do
        [[ -d "${RESULTS_ROOT}/${domain}" ]] || continue
        PYTHONPATH="${REPO_ROOT}" python task_synthesis/summarize_typed_eval.py \
            --results-dir "${RESULTS_ROOT}/${domain}" \
            --baselines "${BASELINE}" \
            --question-types "${qtypes_csv}" \
            --out-markdown "${RESULTS_ROOT}/${domain}/accuracy.md" \
            --out-tsv "${RESULTS_ROOT}/${domain}/accuracy.tsv" || true
    done

    PYTHONPATH="${REPO_ROOT}" python task_synthesis/summarize_cross_domain_eval.py \
        --results-root "${RESULTS_ROOT}" \
        --baseline "${BASELINE}" \
        --domains "$(IFS=,; echo "${DOMAINS[*]}")" \
        --question-types "${qtypes_csv}" \
        --out-markdown "${RESULTS_ROOT}/accuracy_all_domains.md" \
        --out-tsv "${RESULTS_ROOT}/accuracy_all_domains.tsv" || true
}

echo "Repository : ${REPO_ROOT}"
echo "Domains    : ${DOMAINS[*]}"
echo "Qtypes     : ${QTYPES[*]}"
echo "Phases     : ${PHASES[*]}"
echo "Results    : ${RESULTS_ROOT}"
echo

has_phase qa && run_qa
has_phase summarize && run_summary

if [[ "${failures}" -gt 0 ]]; then
    echo "Completed with ${failures} failed cell(s)." >&2
    exit 1
fi
echo "Done."
