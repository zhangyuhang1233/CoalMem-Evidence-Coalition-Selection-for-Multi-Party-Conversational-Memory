# CoalMem: Evidence Coalition Selection for Multi-Party Conversational Memory

CoalMem is a training-free framework for post-retrieval evidence selection in long-term multi-party conversations. Instead of ranking messages independently, CoalMem selects a compact set of messages that is individually relevant and collectively supportive of the query.

## Method

CoalMem operates in three stages:

1. **Seed evidence retrieval.** Metadata-expanded BM25 retrieves candidate messages using message content, speaker, role, channel, topic, and discussion phase.
2. **Evidence coalition graph construction.** Query-message edges represent relevance, while message-message edges combine evidence non-redundancy with query-conditioned complementarity.
3. **Query-anchored coalition optimization.** Replicator dynamics jointly estimates coalition membership weights, after which the highest-weighted messages are selected for answer generation.

The selection stage requires no additional LLM calls and does not require parameter training or memory rewriting.


## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add an OpenAI-compatible or Azure OpenAI endpoint and API key to `.env`.

Expected data layout:

```text
data/final/<Domain>/synthetic_domain_channels_rolevariants_<Domain>.json
questions/<Domain>/<question_type>.jsonl
```

## Usage

Configure the dataset scope, retrieval and selection settings, answer and judge models, runtime options, and output directory through environment variables. The runner validates the required configuration before starting.

Then run:

```bash
bash run_eval.sh
```

The script evaluates CoalMem across the configured domains and query categories and writes predictions, logs, and aggregate accuracy summaries to the configured output directory. Refer to the paper for the experimental protocol and method settings.

## Results

Scoring is performed automatically after evaluation. Per-domain scores are written to `<RESULTS_ROOT>/<Domain>/accuracy.md`, and the micro-averaged score across all domains is written to:

```text
<RESULTS_ROOT>/accuracy_all_domains.md
```

To recompute the scores from existing prediction files without rerunning answer generation, keep the same experiment configuration and run:

```bash
PHASES="summarize" bash run_eval.sh
```

## Citation

Citation information will be added after publication.
