#!/usr/bin/env python3
"""Minimal local-embedding retrieval stage for constructive MVEM-AER ablations.

This baseline deliberately contains no episode activation, view expansion,
structural reranking, or local context. It retrieves the global cosine Top-K
messages directly from normalized local-model embeddings. The default F1
configuration uses the same structured index and structured query as MVEM-AER;
switching both field modes produces F0 without changing the evaluation path.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from baselines.common import (
    call_chat as shared_call_chat,
    format_retrieved_message,
    load_conversation_messages,
    load_env_file,
    load_questions,
    parse_judgment,
    read_text,
    split_reasoning_and_final,
)
from llm_utils import (
    create_chat_client,
    normalize_llm_provider,
    resolve_api_key,
    resolve_base_url,
)

API_VERSION_DEFAULT = "2024-02-15-preview"
AGENT_MODEL_DEFAULT = "gpt-5"
JUDGE_MODEL_DEFAULT = "gpt-5"


class EmptyModelResponseError(RuntimeError):
    """Raised when the shared chat helper returns no usable text."""


def call_chat(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    thinking_mode: Optional[str] = None,
) -> str:
    max_attempts = max(
        1,
        int(os.environ.get("EMPTY_RESPONSE_MAX_ATTEMPTS", "5")),
    )
    retry_delay = max(
        0.0,
        float(os.environ.get("EMPTY_RESPONSE_RETRY_DELAY_SECONDS", "10")),
    )

    for attempt in range(1, max_attempts + 1):
        output = shared_call_chat(
            client,
            model,
            system_prompt,
            user_prompt,
            max_tokens,
            thinking_mode=thinking_mode,
        )
        if output and output.strip():
            return output
        if attempt < max_attempts:
            print(
                f"[empty-response] model={model} attempt={attempt}/{max_attempts}; "
                f"retrying in {retry_delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(retry_delay)

    raise EmptyModelResponseError(
        f"Empty response from model {model} after {max_attempts} attempts"
    )


def build_agent_user_prompt(question: str, asker: str, docs: List[str]) -> str:
    passages = "\n\n".join(
        f"[{index}] {doc}" for index, doc in enumerate(docs, start=1)
    )
    asker_line = f"Asking user: {asker}\n\n" if asker else ""
    return (
        f"{asker_line}"
        f"Question:\n{question}\n\n"
        f"Retrieved passages:\n{passages}\n\n"
        "Answer the question using the retrieved passages."
    )


def build_judge_user_prompt(question: str, gold: str, answer: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Gold Answer:\n{gold}\n\n"
        f"Agent Answer:\n{answer}\n"
    )


def clean_content(message: Dict[str, Any]) -> str:
    value = message.get("content")
    return html.unescape(value).strip() if isinstance(value, str) else ""


def structured_index_text(message: Dict[str, Any]) -> str:
    """Use exactly the native metadata fields indexed by MVEM-AER."""
    return " ".join(
        [
            clean_content(message),
            str(message.get("_channel") or ""),
            str(message.get("phase_name") or ""),
            str(message.get("topic") or ""),
            str(message.get("author") or ""),
            str(message.get("role") or ""),
        ]
    )


@dataclass(frozen=True)
class RetrievedItem:
    msg_idx: int
    score: float


@dataclass
class EvalResult:
    qid: str
    record: Dict[str, Any]
    judgment: Optional[bool]
    skipped: bool = False


class MinimalEmbeddingRetriever:
    def __init__(
        self,
        messages: List[Dict[str, Any]],
        *,
        conversation_path: str,
        index_fields: str,
        query_fields: str,
        top_k: int,
        model_path: str,
        gpu_ids: List[str],
        batch_size: int,
        max_length: int,
        embedding_dtype: str,
        trust_remote_code: bool,
    ) -> None:
        self.messages = messages
        self.index_fields = index_fields
        self.query_fields = query_fields
        self.top_k = top_k
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_dtype = embedding_dtype
        self.pool = None
        self.query_embeddings: Dict[str, np.ndarray] = {}
        self.author_role: Dict[str, str] = {}
        for message in messages:
            author = str(message.get("author") or "")
            role = str(message.get("role") or "")
            if author and role:
                self.author_role.setdefault(author, role)

        index_text = clean_content if index_fields == "content" else structured_index_text
        self.corpus = [index_text(message) for message in messages]

        self.model_kwargs: Dict[str, Any] = {}
        self.trust_remote_code = trust_remote_code
        self.visible_devices = [f"cuda:{index}" for index in range(len(gpu_ids))]
        self.model_device = (
            "cpu" if len(self.visible_devices) != 1 else self.visible_devices[0]
        )
        self.encode_device = self.model_device
        self.model = None


        stat_result = os.stat(conversation_path)
        self.cache_metadata = {
            "conversation_path": os.path.abspath(conversation_path),
            "conversation_size": stat_result.st_size,
            "conversation_mtime_ns": stat_result.st_mtime_ns,
            "message_count": len(messages),
            "model_path": self.model_path,
            "index_fields": index_fields,
            "max_length": max_length,
            "embedding_dtype": embedding_dtype,
            "normalize_embeddings": True,
        }
        self.embeddings = self._encode(self.corpus, description="messages")

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if self.embedding_dtype != "auto":
            import torch

            self.model_kwargs["torch_dtype"] = getattr(
                torch, self.embedding_dtype
            )
        from sentence_transformers import SentenceTransformer

        print(
            f"[embedding] loading embedding model "
            f"{self.model_path}"
        )
        self.model = SentenceTransformer(
            self.model_path,
            device=self.model_device,
            trust_remote_code=self.trust_remote_code,
            model_kwargs=self.model_kwargs,
        )
        self.model.max_seq_length = self.max_length
        if len(self.visible_devices) > 1:
            self.pool = self.model.start_multi_process_pool(
                target_devices=self.visible_devices
            )

    @property
    def embedding_dims(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def _encode(self, texts: List[str], *, description: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dims), dtype=np.float32)
        self._ensure_model()
        print(
            f"[embedding] encoding {len(texts)} {description} "
            f"on {self.gpu_ids or ['cpu']}"
        )
        if self.pool is not None:
            embeddings = self.model.encode_multi_process(
                texts,
                self.pool,
                batch_size=self.batch_size,
                normalize_embeddings=True,
            )
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.encode_device,
            )
        result = np.asarray(embeddings, dtype=np.float32)
        if result.ndim != 2 or result.shape[0] != len(texts):
            raise ValueError(
                f"Unexpected embedding shape {result.shape} for {len(texts)} texts"
            )
        return result

    def build_query(self, question: str, asker: str) -> str:
        if self.query_fields == "question_only":
            return question
        parts = [question]
        if asker:
            parts.append(asker)
            if self.query_fields == "structured":
                role = self.author_role.get(asker, "")
                if role:
                    parts.append(role)
        return " ".join(parts)

    def prepare_queries(self, questions: List[Dict[str, Any]]) -> None:
        unique_queries = []
        seen = set()
        for record in questions:
            query = self.build_query(
                str(record["question"]),
                str(record.get("asking_user_id") or ""),
            )
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)

        embeddings = self._encode(unique_queries, description="queries")
        self.query_embeddings = dict(zip(unique_queries, embeddings))

    def retrieve(self, question: str, asker: str) -> Tuple[str, List[RetrievedItem]]:
        query = self.build_query(question, asker)
        if not query.strip() or self.top_k <= 0:
            return query, []
        query_embedding = self.query_embeddings.get(query)
        if query_embedding is None:
            query_embedding = self._encode([query], description="query")[0]
            self.query_embeddings[query] = query_embedding
        scores = np.asarray(self.embeddings @ query_embedding, dtype=np.float32)
        scores = np.nan_to_num(scores, nan=-np.inf)
        order = np.lexsort((np.arange(len(scores)), -scores))[: self.top_k]
        items = [
            RetrievedItem(int(index), float(scores[index])) for index in order
        ]
        return query, items

    def format_evidence(self, items: List[RetrievedItem]) -> List[str]:
        """Use the same official passage shape as the BM25 ablation."""
        return [
            format_retrieved_message(self.messages[item.msg_idx]) for item in items
        ]

    def close(self) -> None:
        if self.pool is not None:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None


def run_qa(
    *,
    questions: List[Dict[str, Any]],
    retriever: MinimalEmbeddingRetriever,
    client: Any,
    agent_model: str,
    judge_model: str,
    agent_thinking: Optional[str],
    judge_thinking: Optional[str],
    agent_max_tokens: int,
    judge_max_tokens: int,
    agent_system: str,
    judge_system: str,
    output_jsonl: str,
    num_workers: int,
) -> Tuple[int, int]:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    correct = 0
    total = 0
    skipped = 0

    config = {
        "stage": "minimal_embedding",
        "agent_model": agent_model,
        "judge_model": judge_model,
        "agent_thinking": agent_thinking,
        "judge_thinking": judge_thinking,
        "agent_max_tokens": agent_max_tokens,
        "judge_max_tokens": judge_max_tokens,
        "index_fields": retriever.index_fields,
        "query_fields": retriever.query_fields,
        "retrieve_top_k": retriever.top_k,
        "embedding_model_loaded": retriever.model is not None,
        "embedding_model_path": retriever.model_path,
        "embedding_dims": retriever.embedding_dims,
        "embedding_gpu_ids": retriever.gpu_ids,
        "embedding_max_length": retriever.max_length,
        "normalize_embeddings": True,
        "episode_activation": False,
        "expansion": "none",
        "selector": "local_embedding_cosine",
        "include_context": False,
    }

    def evaluate_one(question_record: Dict[str, Any]) -> EvalResult:
        question = question_record["question"]
        gold = question_record.get("answer", "")
        asker = question_record.get("asking_user_id") or ""
        online_started_ns = time.perf_counter_ns()
        search_query, items = retriever.retrieve(question, asker)
        docs = retriever.format_evidence(items)
        online_latency_ms = (
            time.perf_counter_ns() - online_started_ns
        ) / 1_000_000.0
        base_record = {
            "query": question,
            "asking_user_id": asker,
            "retrieval_query": search_query,
            "experiment_config": config,
            "online_latency_ms": online_latency_ms,
            "selected_message_ids": [
                retriever.messages[item.msg_idx].get("msg_node") for item in items
            ],
            "retrieval_items": [
                {
                    "rank": rank,
                    "msg_node": retriever.messages[item.msg_idx].get("msg_node"),
                    "score": item.score,
                    "source": f"{retriever.index_fields}_local_embedding",
                    "reason": (
                        "Structured local-embedding cosine retrieval"
                        if retriever.index_fields == "structured"
                        else "Content-only local-embedding cosine retrieval"
                    ),
                }
                for rank, item in enumerate(items, start=1)
            ],
            "retrieved_docs": docs,
        }
        stage = "agent"
        agent_reasoning = ""
        agent_final = ""
        try:
            agent_user = build_agent_user_prompt(question, asker, docs)
            agent_output = call_chat(
                client,
                agent_model,
                agent_system,
                agent_user,
                max_tokens=agent_max_tokens,
                thinking_mode=agent_thinking,
            )
            agent_reasoning, agent_final = split_reasoning_and_final(agent_output)

            stage = "judge"
            judge_user = build_judge_user_prompt(question, gold, agent_final)
            judge_output = call_chat(
                client,
                judge_model,
                judge_system,
                judge_user,
                max_tokens=judge_max_tokens,
                thinking_mode=judge_thinking,
            )
            judge_reasoning, judge_final = split_reasoning_and_final(judge_output)
        except EmptyModelResponseError as exc:
            return EvalResult(
                qid=str(question_record.get("id", "")),
                record={
                    **base_record,
                    "status": "skipped",
                    "skip_stage": stage,
                    "skip_reason": str(exc),
                    "agent_reasoning": agent_reasoning,
                    "agent_answer": agent_final,
                    "judge_reasoning": "",
                    "judge_answer": "",
                },
                judgment=None,
                skipped=True,
            )

        return EvalResult(
            qid=str(question_record.get("id", "")),
            record={
                **base_record,
                "agent_reasoning": agent_reasoning,
                "agent_answer": agent_final,
                "judge_reasoning": judge_reasoning,
                "judge_answer": judge_final,
            },
            judgment=parse_judgment(judge_final),
        )

    def progress_text() -> str:
        accuracy = correct / total * 100 if total else 0.0
        return f"acc={correct}/{total} ({accuracy:.1f}%) skipped={skipped}"

    def write_result(out_file: Any, skipped_file: Any, result: EvalResult) -> str:
        nonlocal correct, total, skipped
        if result.skipped:
            skipped += 1
            skipped_file.write(json.dumps(result.record, ensure_ascii=False) + "\n")
            skipped_file.flush()
            return "Skipped"
        total += 1
        if result.judgment is True:
            correct += 1
            verdict = "Correct"
        elif result.judgment is False:
            verdict = "Incorrect"
        else:
            verdict = "Unclear"
        out_file.write(json.dumps(result.record, ensure_ascii=False) + "\n")
        out_file.flush()
        return verdict

    skipped_path = output_jsonl + ".skipped"
    with open(output_jsonl, "w", encoding="utf-8") as out_file, open(
        skipped_path, "w", encoding="utf-8"
    ) as skipped_file:
        if num_workers <= 1:
            progress = tqdm(questions, desc="Evaluating", unit="q", dynamic_ncols=True)
            for question_record in progress:
                result = evaluate_one(question_record)
                verdict = write_result(out_file, skipped_file, result)
                progress.set_postfix_str(progress_text())
                label = result.qid or (
                    f"q_skipped_{skipped}" if result.skipped else f"q_{total}"
                )
                print(f"{label}: {verdict}")
        else:
            print(
                f"[embedding] evaluating with "
                f"{num_workers} workers"
            )
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(evaluate_one, question_record): str(
                        question_record.get("id", "")
                    )
                    for question_record in questions
                }
                progress = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Evaluating",
                    unit="q",
                    dynamic_ncols=True,
                )
                for future in progress:
                    result = future.result()
                    verdict = write_result(out_file, skipped_file, result)
                    progress.set_postfix_str(progress_text())
                    label = result.qid or (
                        f"q_skipped_{skipped}" if result.skipped else f"q_{total}"
                    )
                    print(f"{label}: {verdict}")

    if skipped:
        print(
            f"[embedding] skipped {skipped} questions "
            f"-> {skipped_path}"
        )
    return correct, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation-json", required=True)
    parser.add_argument("--questions-jsonl", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--agent-model", default=AGENT_MODEL_DEFAULT)
    parser.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    parser.add_argument("--agent-thinking", choices=("enabled", "disabled"), default=None)
    parser.add_argument("--judge-thinking", choices=("enabled", "disabled"), default=None)
    parser.add_argument("--agent-max-tokens", type=int, default=512)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--agent-prompt", default="prompts/agent_system.txt")
    parser.add_argument("--judge-prompt", default="prompts/judge_system.txt")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument(
        "--index-fields",
        choices=("content", "structured"),
        default="structured",
        help="Message fields encoded for retrieval. F1 defaults to structured.",
    )
    parser.add_argument(
        "--query-fields",
        choices=("question_only", "standard", "structured"),
        default="structured",
        help=(
            "question_only=question; standard=question+asker; "
            "structured also adds asker role."
        ),
    )
    parser.add_argument("--retrieve-top-k", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--embedding-model-path", required=True)
    parser.add_argument(
        "--gpu-ids",
        default="0",
        help='Physical CUDA device ids, e.g. "4" or "4,5,6,7"; use "cpu" for CPU.',
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument(
        "--embedding-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--ingest-only", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.embedding_model_path).expanduser()
    if not model_path.exists():
        parser.error(f"local embedding model path does not exist: {model_path}")
    gpu_ids = (
        []
        if args.gpu_ids.strip().lower() == "cpu"
        else [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    )
    if not gpu_ids and args.gpu_ids.strip().lower() != "cpu":
        parser.error("--gpu-ids must contain at least one id or be 'cpu'")
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    load_env_file(args.env_file)
    provider = normalize_llm_provider(args.llm_provider)
    base_url = resolve_base_url(provider)
    api_key = resolve_api_key(provider)

    messages = load_conversation_messages(args.conversation_json)
    print(f"[embedding] loaded {len(messages)} messages")
    retriever = MinimalEmbeddingRetriever(
        messages,
        conversation_path=args.conversation_json,
        index_fields=args.index_fields,
        query_fields=args.query_fields,
        top_k=max(0, args.retrieve_top_k),
        model_path=str(model_path),
        gpu_ids=gpu_ids,
        batch_size=max(1, args.embedding_batch_size),
        max_length=max(1, args.embedding_max_length),
        embedding_dtype=args.embedding_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    print(
        "[embedding] config: "
        + json.dumps(
            {
                "agent_model": args.agent_model,
                "judge_model": args.judge_model,
                "agent_thinking": args.agent_thinking,
                "judge_thinking": args.judge_thinking,
                "agent_max_tokens": max(1, args.agent_max_tokens),
                "judge_max_tokens": max(1, args.judge_max_tokens),
                "index_fields": retriever.index_fields,
                "query_fields": retriever.query_fields,
                "retrieve_top_k": retriever.top_k,
                        "embedding_model_loaded": retriever.model is not None,
                "embedding_model_path": retriever.model_path,
                "embedding_dims": retriever.embedding_dims,
                "embedding_gpu_ids": gpu_ids or ["cpu"],
                "embedding_batch_size": retriever.batch_size,
                "embedding_max_length": retriever.max_length,
                "embedding_dtype": retriever.embedding_dtype,
                "normalize_embeddings": True,
                "episode_activation": False,
                "expansion": "none",
                "selector": "local_embedding_cosine",
                "include_context": False,
            },
            sort_keys=True,
        )
    )
    print(
        f"[embedding] {retriever.index_fields} embedding "
        f"index ready (shape={retriever.embeddings.shape})"
    )
    if args.ingest_only:
        retriever.close()
        print(
            "[embedding] ingest-only: index built, "
            "exiting before QA"
        )
        return 0

    questions = load_questions(args.questions_jsonl)
    retriever.prepare_queries(questions)
    agent_system = read_text(args.agent_prompt)
    judge_system = read_text(args.judge_prompt)
    client = create_chat_client(
        provider=provider,
        azure_endpoint=base_url,
        base_url=base_url,
        api_version=API_VERSION_DEFAULT,
        api_key=api_key,
    )
    try:
        correct, total = run_qa(
            questions=questions,
            retriever=retriever,
            client=client,
            agent_model=args.agent_model,
            judge_model=args.judge_model,
            agent_thinking=args.agent_thinking,
            judge_thinking=args.judge_thinking,
            agent_max_tokens=max(1, args.agent_max_tokens),
            judge_max_tokens=max(1, args.judge_max_tokens),
            agent_system=agent_system,
            judge_system=judge_system,
            output_jsonl=args.output_jsonl,
            num_workers=max(1, args.num_workers),
        )
    finally:
        retriever.close()
    accuracy = correct / total if total else 0.0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
