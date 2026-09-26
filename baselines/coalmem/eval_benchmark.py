#!/usr/bin/env python3
"""Training-free constrained-dominant-set evidence selection.

Candidate retrieval and CDS message similarity are independently configurable.
The default path remains structured ExpandC BM25 retrieval. Embedding mode
reuses cached content embeddings and can encode/cache missing query embeddings.
Query-term complementarity always remains BM25 based.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
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

from baselines.embedding.eval_benchmark import (
    MinimalEmbeddingRetriever,
)
from baselines.coalmem.native_relation_graph import (
    BM25ThresholdRelationMaskBuilder,
    NativeRelationMaskBuilder,
)
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
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
QUERYCOV_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
    "would", "could", "should",
}


class BM25Okapi:
    """Positive-IDF BM25 used by the existing minimal BM25 baseline."""

    def __init__(self, corpus):
        self.corpus = [list(doc) for doc in corpus]
        self.n = len(self.corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / max(1, self.n)
        self.df = defaultdict(int)
        self.tf = []
        for doc in self.corpus:
            counts = Counter(doc)
            self.tf.append(counts)
            for token in counts:
                self.df[token] += 1
        self.k1 = 1.5
        self.b = 0.75

    def get_scores(self, query_tokens, document_indices=None):
        indices = range(self.n) if document_indices is None else (
            int(index) for index in document_indices
        )
        scores = []
        for index in indices:
            doc = self.corpus[index]
            counts = self.tf[index]
            dl = len(doc) or 1
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                df = self.df.get(token, 0)
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                denom = frequency + self.k1 * (
                    1 - self.b + self.b * dl / max(self.avgdl, 1e-9)
                )
                score += idf * frequency * (self.k1 + 1) / denom
            scores.append(score)
        return scores

class BM25F:
    """Training-free BM25F over separately normalized message fields."""

    def __init__(
        self,
        field_corpora: Dict[str, List[List[str]]],
        *,
        field_weights: Dict[str, float],
        field_b: Dict[str, float],
        k1: float,
    ) -> None:
        if not field_corpora:
            raise ValueError("BM25F requires at least one field")
        self.fields = tuple(field_corpora)
        sizes = {len(field_corpora[field]) for field in self.fields}
        if len(sizes) != 1:
            raise ValueError("All BM25F fields must contain the same documents")
        self.n = sizes.pop()
        self.k1 = float(k1)
        self.field_weights = {
            field: float(field_weights[field]) for field in self.fields
        }
        self.field_b = {field: float(field_b[field]) for field in self.fields}
        self.corpora = {
            field: [list(document) for document in field_corpora[field]]
            for field in self.fields
        }
        self.avgdl = {
            field: (
                sum(len(document) for document in self.corpora[field])
                / max(1, self.n)
            )
            for field in self.fields
        }
        self.tf = {
            field: [Counter(document) for document in self.corpora[field]]
            for field in self.fields
        }
        # A term's document frequency counts a message once when the term
        # occurs in any of its fields.
        self.df = defaultdict(int)
        for index in range(self.n):
            document_terms = set()
            for field in self.fields:
                document_terms.update(self.tf[field][index])
            for token in document_terms:
                self.df[token] += 1

    def get_scores(
        self,
        query_tokens: List[str],
        document_indices: Optional[np.ndarray] = None,
    ) -> List[float]:
        indices = (
            range(self.n)
            if document_indices is None
            else (int(index) for index in document_indices)
        )
        scores: List[float] = []
        for index in indices:
            score = 0.0
            for token in query_tokens:
                effective_tf = 0.0
                for field in self.fields:
                    frequency = self.tf[field][index].get(token, 0)
                    if not frequency:
                        continue
                    b = self.field_b[field]
                    field_length = len(self.corpora[field][index])
                    length_normalizer = (
                        1.0 - b
                        + b * field_length / max(self.avgdl[field], 1e-9)
                    )
                    effective_tf += (
                        self.field_weights[field]
                        * frequency
                        / max(length_normalizer, 1e-12)
                    )
                if effective_tf <= 0.0:
                    continue
                document_frequency = self.df.get(token, 0)
                idf = math.log(
                    1
                    + (self.n - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                score += (
                    idf
                    * effective_tf
                    * (self.k1 + 1.0)
                    / (effective_tf + self.k1)
                )
            scores.append(score)
        return scores


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def clean_content(message: Dict[str, Any]) -> str:
    value = message.get("content")
    return html.unescape(value).strip() if isinstance(value, str) else ""


def structured_index_text(message: Dict[str, Any]) -> str:
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

def participant_index_text(message: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(message.get("author") or ""),
            str(message.get("role") or ""),
        ]
    )


def episode_index_text(message: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(message.get("_channel") or ""),
            str(message.get("phase_name") or ""),
            str(message.get("topic") or ""),
        ]
    )


class EmptyModelResponseError(RuntimeError):
    pass


def call_chat(
    client,
    model,
    system_prompt,
    user_prompt,
    max_tokens,
    thinking_mode=None,
    reasoning_effort=None,
):
    attempts = max(1, int(os.environ.get("EMPTY_RESPONSE_MAX_ATTEMPTS", "5")))
    delay = max(0.0, float(os.environ.get("EMPTY_RESPONSE_RETRY_DELAY_SECONDS", "10")))
    for attempt in range(1, attempts + 1):
        output = shared_call_chat(
            client,
            model,
            system_prompt,
            user_prompt,
            max_tokens,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
        if output and output.strip():
            return output
        if attempt < attempts:
            print(
                f"[empty-response] model={model} attempt={attempt}/{attempts}; "
                f"retrying in {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise EmptyModelResponseError(f"Empty response from {model} after {attempts} attempts")


def build_agent_prompt(question: str, asker: str, docs: List[str]) -> str:
    passages = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(docs, 1))
    asker_line = f"Asking user: {asker}\n\n" if asker else ""
    return (
        f"{asker_line}Question:\n{question}\n\nRetrieved passages:\n{passages}\n\n"
        "Answer the question using the retrieved passages."
    )


def build_judge_prompt(question: str, gold: str, answer: str) -> str:
    return f"Question:\n{question}\n\nGold Answer:\n{gold}\n\nAgent Answer:\n{answer}\n"


@dataclass(frozen=True)
class CDSItem:
    msg_idx: int
    membership: float
    candidate_rank: int
    candidate_bm25_score: float
    query_affinity: float


@dataclass
class EvalResult:
    qid: str
    record: Dict[str, Any]
    judgment: Optional[bool]
    skipped: bool = False


class CDSRetriever:
    def __init__(
        self,
        messages: List[Dict[str, Any]],
        *,
        conversation_path: str,
        mode: str,
        candidate_top_k: int,
        retrieve_top_k: int,
        scope_filter_top_k: int,
        scope_score_top_n: int,
        query_text_mode: str,
        query_affinity_mode: str,
        multiview_top_k: int,
        multiview_consensus_weight: float,
        bm25f_content_weight: float,
        bm25f_participant_weight: float,
        bm25f_episode_weight: float,
        bm25f_content_b: float,
        bm25f_participant_b: float,
        bm25f_episode_b: float,
        bm25f_k1: float,
        complementarity_weight: float,
        complementarity_fields: str,
        complementarity_normalization: str,
        dynamic_coverage_weight: float,
        native_complementarity_weight: float,
        native_surprise_weight: float,
        native_bridge: bool,
        query_dynamic_edge: bool,
        dynamic_edge_floor: float,
        evidence_order: str,
        selection_relevance_power: float,
        cds_epsilon: float,
        cds_tolerance: float,
        cds_max_iterations: int,
        cds_solver_mode: str,
        cds_query_mass: float,
        topk_graph_weight: float,
        fixed_share_rate: float,
        iht_initial_step: float,
        iht_backtracking_factor: float,
        iht_armijo_sigma: float,
        iht_min_step: float,
        iht_max_backtracking: int,
        candidate_retrieval_mode: str,
        embedding_index_fields: str,
        embedding_store_dir: str,
        pair_embedding_index_fields: str,
        pair_embedding_store_dir: str,
        embedding_model_path: str,
        embedding_cache_only: bool,
        embedding_gpu_ids: List[str],
        embedding_batch_size: int,
        embedding_max_length: int,
        embedding_dtype: str,
        relation_graph: str,
        relation_model: str,
        relation_thinking: Optional[str],
        relation_max_tokens: int,
        relation_pair_batch_size: int,
        native_channel_window: int,
        native_scope_neighbors: int,
        relation_verify: bool,
        relation_threshold_iqr_multiplier: float,
        relation_threshold_min_pairs: int,
        relation_threshold_max_pairs: int,
    ) -> None:
        self.messages = messages
        self.mode = mode
        self.candidate_top_k = candidate_top_k
        self.retrieve_top_k = retrieve_top_k
        self.scope_filter_top_k = scope_filter_top_k
        self.scope_score_top_n = scope_score_top_n
        self.query_text_mode = query_text_mode
        self.query_affinity_mode = query_affinity_mode
        self.multiview_top_k = multiview_top_k
        self.multiview_consensus_weight = multiview_consensus_weight
        self.bm25f_content_weight = bm25f_content_weight
        self.bm25f_participant_weight = bm25f_participant_weight
        self.bm25f_episode_weight = bm25f_episode_weight
        self.bm25f_content_b = bm25f_content_b
        self.bm25f_participant_b = bm25f_participant_b
        self.bm25f_episode_b = bm25f_episode_b
        self.bm25f_k1 = bm25f_k1
        self.complementarity_weight = complementarity_weight
        self.complementarity_fields = complementarity_fields
        self.complementarity_normalization = complementarity_normalization
        self.dynamic_coverage_weight = dynamic_coverage_weight
        self.native_complementarity_weight = native_complementarity_weight
        self.native_surprise_weight = native_surprise_weight
        self.native_bridge = native_bridge
        self.query_dynamic_edge = query_dynamic_edge
        self.dynamic_edge_floor = dynamic_edge_floor
        self.evidence_order = evidence_order
        self.selection_relevance_power = selection_relevance_power
        self.cds_epsilon = cds_epsilon
        self.cds_tolerance = cds_tolerance
        self.cds_max_iterations = cds_max_iterations
        self.cds_solver_mode = cds_solver_mode
        self.cds_query_mass = cds_query_mass
        self.topk_graph_weight = topk_graph_weight
        self.fixed_share_rate = fixed_share_rate
        self.iht_initial_step = iht_initial_step
        self.iht_backtracking_factor = iht_backtracking_factor
        self.iht_armijo_sigma = iht_armijo_sigma
        self.iht_min_step = iht_min_step
        self.iht_max_backtracking = iht_max_backtracking
        self.candidate_retrieval_mode = candidate_retrieval_mode
        self.embedding_index_fields = embedding_index_fields
        self.pair_embedding_index_fields = pair_embedding_index_fields
        self.relation_graph = relation_graph
        self.native_scope_neighbors = native_scope_neighbors
        self.relation_client: Any = None
        self.relation_builder: Optional[NativeRelationMaskBuilder] = None
        builder_kwargs = dict(
            model=relation_model,
            thinking=relation_thinking,
            max_tokens=relation_max_tokens,
            pair_batch_size=relation_pair_batch_size,
            channel_window=native_channel_window,
            scope_neighbors=native_scope_neighbors,
            verify=relation_verify,
        )
        if relation_graph == "native":
            self.relation_builder = NativeRelationMaskBuilder(
                messages, **builder_kwargs
            )
        elif relation_graph == "native_candidate_scope":
            self.relation_builder = NativeRelationMaskBuilder(
                messages, model_free_candidate_scope=True, **builder_kwargs
            )
        elif relation_graph == "bm25_threshold":
            if mode not in {"bm25", "structured_bm25"}:
                raise ValueError(
                    "bm25_threshold relation graph requires cds-mode bm25 or structured_bm25"
                )
            self.relation_builder = BM25ThresholdRelationMaskBuilder(
                messages,
                threshold_iqr_multiplier=relation_threshold_iqr_multiplier,
                min_candidate_pairs=relation_threshold_min_pairs,
                max_candidate_pairs=relation_threshold_max_pairs,
                **builder_kwargs,
            )
        self.author_role: Dict[str, str] = {}
        for message in messages:
            author = str(message.get("author") or "")
            role = str(message.get("role") or "")
            if author and role:
                self.author_role.setdefault(author, role)

        # Preserve exact source-conversation positions inside strict native
        # scopes. Query-time locality must not be inferred from BM25 rank.
        scope_groups: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        for index, message in enumerate(messages):
            scope = (
                str(message.get("_channel") or ""),
                str(message.get("phase_name") or ""),
                str(message.get("topic") or ""),
            )
            if all(scope):
                scope_groups[scope].append(index)
        self.native_scope_position: Dict[
            int, Tuple[Tuple[str, str, str], int]
        ] = {}
        for scope, indices in scope_groups.items():
            indices.sort(
                key=lambda index: (
                    str(messages[index].get("timestamp") or ""),
                    index,
                )
            )
            for position, index in enumerate(indices):
                self.native_scope_position[index] = (scope, position)

        # Native-surprise relations use adjacency in complete source
        # trajectories, never adjacency induced by the retrieved Top-K list.
        topic_groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        member_topic_groups: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        for index, message in enumerate(messages):
            channel = str(message.get("_channel") or "")
            topic = str(message.get("topic") or "")
            author = str(message.get("author") or "")
            if channel and topic:
                topic_groups[(channel, topic)].append(index)
                if author:
                    member_topic_groups[(author, channel, topic)].append(index)

        def build_positions(groups):
            positions = {}
            for key, indices in groups.items():
                indices.sort(
                    key=lambda index: (
                        str(messages[index].get("timestamp") or ""),
                        index,
                    )
                )
                for position, index in enumerate(indices):
                    positions[index] = (key, position)
            return positions

        self.native_topic_position = build_positions(topic_groups)
        self.native_member_topic_position = build_positions(member_topic_groups)

        self.structured_corpus = [tokenize(structured_index_text(m)) for m in messages]
        self.candidate_bm25 = BM25Okapi(self.structured_corpus)

        self.query_bm25f: Optional[BM25F] = None
        if self.query_affinity_mode == "bm25f":
            self.query_bm25f = BM25F(
                {
                    "content": [tokenize(clean_content(m)) for m in messages],
                    "participant": [
                        tokenize(participant_index_text(m)) for m in messages
                    ],
                    "episode": [tokenize(episode_index_text(m)) for m in messages],
                },
                field_weights={
                    "content": self.bm25f_content_weight,
                    "participant": self.bm25f_participant_weight,
                    "episode": self.bm25f_episode_weight,
                },
                field_b={
                    "content": self.bm25f_content_b,
                    "participant": self.bm25f_participant_b,
                    "episode": self.bm25f_episode_b,
                },
                k1=self.bm25f_k1,
            )
        self.content_bm25: Optional[BM25Okapi] = None
        needs_content_bm25 = (
            (
                (
                    complementarity_weight > 0
                    or native_complementarity_weight > 0
                    or native_bridge
                )
                and complementarity_fields == "content"
            )
            or self.query_affinity_mode == "multiview_consensus"
        )
        if needs_content_bm25:
            self.content_bm25 = BM25Okapi(
                [tokenize(clean_content(message)) for message in messages]
            )
        self.multiview_bm25: Dict[str, BM25Okapi] = {}
        if self.query_affinity_mode == "multiview_consensus":
            if self.content_bm25 is None:
                raise RuntimeError("Multi-view query affinity requires content BM25")
            self.multiview_bm25 = {
                "content": self.content_bm25,
                "participant": BM25Okapi(
                    [tokenize(participant_index_text(message)) for message in messages]
                ),
                "episode": BM25Okapi(
                    [tokenize(episode_index_text(message)) for message in messages]
                ),
            }
        self.embedding: Optional[MinimalEmbeddingRetriever] = None
        self.pair_embedding: Optional[MinimalEmbeddingRetriever] = None
        needs_embedding = (
            mode in {"hybrid", "dense"}
            or candidate_retrieval_mode == "embedding"
        )
        if needs_embedding:
            # Preserve the historical standard query for legacy dense/hybrid
            # runs. The embedding-candidate path follows query_text_mode.
            if query_text_mode == "question_only":
                embedding_query_fields = "question_only"
            elif candidate_retrieval_mode == "embedding":
                embedding_query_fields = "structured"
            else:
                embedding_query_fields = "standard"
            self.embedding = MinimalEmbeddingRetriever(
                messages,
                conversation_path=conversation_path,
                index_fields=embedding_index_fields,
                query_fields=embedding_query_fields,
                top_k=candidate_top_k,
                cache_only=embedding_cache_only,
                model_path=embedding_model_path,
                gpu_ids=embedding_gpu_ids,
                batch_size=embedding_batch_size,
                max_length=embedding_max_length,
                embedding_dtype=embedding_dtype,
                store_dir=embedding_store_dir,
                force_rebuild=False,
                trust_remote_code=True,
            )
            if mode in {"hybrid", "dense"}:
                if (
                    pair_embedding_index_fields == embedding_index_fields
                    and pair_embedding_store_dir == embedding_store_dir
                ):
                    self.pair_embedding = self.embedding
                else:
                    self.pair_embedding = MinimalEmbeddingRetriever(
                        messages,
                        conversation_path=conversation_path,
                        index_fields=pair_embedding_index_fields,
                        query_fields=embedding_query_fields,
                        top_k=candidate_top_k,
                        cache_only=embedding_cache_only,
                        model_path=embedding_model_path,
                        gpu_ids=embedding_gpu_ids,
                        batch_size=embedding_batch_size,
                        max_length=embedding_max_length,
                        embedding_dtype=embedding_dtype,
                        store_dir=pair_embedding_store_dir,
                        force_rebuild=False,
                        trust_remote_code=True,
                    )

    def build_structured_query(self, question: str, asker: str) -> str:
        if self.query_text_mode == "question_only":
            return question
        parts = [question]
        if asker:
            parts.append(asker)
            role = self.author_role.get(asker, "")
            if role:
                parts.append(role)
        return " ".join(parts)

    def prepare_queries(self, questions: List[Dict[str, Any]]) -> None:
        if self.embedding is not None and (
            self.mode == "dense"
            or self.candidate_retrieval_mode == "embedding"
        ):
            self.embedding.prepare_queries(questions)

    @staticmethod
    def _normalize_positive(values: np.ndarray) -> np.ndarray:
        values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
        maximum = float(values.max()) if values.size else 0.0
        return values / maximum if maximum > 0 else values

    def _candidate_pool(self, question: str, asker: str):
        if self.candidate_retrieval_mode == "embedding":
            if self.embedding is None:
                raise RuntimeError(
                    "Embedding candidate retrieval requested without an "
                    "embedding retriever"
                )
            query, items = self.embedding.retrieve(question, asker)
            indices = np.asarray(
                [item.msg_idx for item in items], dtype=np.int64
            )
            scores = np.asarray(
                [item.score for item in items], dtype=np.float64
            )
            return query, indices, scores

        query = self.build_structured_query(question, asker)
        scores = np.asarray(self.candidate_bm25.get_scores(tokenize(query)), dtype=np.float64)
        order = np.lexsort((np.arange(len(scores)), -scores))[: self.candidate_top_k]
        return query, order.astype(np.int64), scores[order]

    @staticmethod
    def _scope_key(message: Dict[str, Any]) -> Tuple[str, str, str]:
        """Exact native conversation scope used only as a coarse container."""
        return (
            str(message.get("_channel") or ""),
            str(message.get("phase_name") or ""),
            str(message.get("topic") or ""),
        )

    def _filter_candidate_scopes(
        self,
        candidate_indices: np.ndarray,
        candidate_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Keep candidates from the strongest exact scopes before CDS.

        A scope is scored by the sum of its highest ``scope_score_top_n``
        candidate-retrieval scores. CDS then runs once over the union of
        selected scopes; scopes do not receive fixed evidence quotas.
        """
        original_ranks = np.arange(1, len(candidate_indices) + 1, dtype=np.int64)
        if self.scope_filter_top_k <= 0 or not len(candidate_indices):
            return candidate_indices, candidate_scores, original_ranks, {
                "scope_filter_enabled": False,
                "scope_filter_top_k": self.scope_filter_top_k,
                "scope_score_top_n": self.scope_score_top_n,
                "prefilter_candidate_count": int(len(candidate_indices)),
                "postfilter_candidate_count": int(len(candidate_indices)),
            }

        grouped_positions: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        for position, message_index in enumerate(candidate_indices):
            grouped_positions[self._scope_key(self.messages[int(message_index)])].append(position)

        ranked_scopes = []
        for scope, positions in grouped_positions.items():
            top_positions = positions[: self.scope_score_top_n]
            score = float(sum(float(candidate_scores[pos]) for pos in top_positions))
            ranked_scopes.append((scope, positions, score, positions[0]))
        ranked_scopes.sort(key=lambda item: (-item[2], item[3], item[0]))

        selected_scope_count = min(self.scope_filter_top_k, len(ranked_scopes))
        selected_positions = {
            position
            for _, positions, _, _ in ranked_scopes[:selected_scope_count]
            for position in positions
        }
        while (
            len(selected_positions) < min(self.retrieve_top_k, len(candidate_indices))
            and selected_scope_count < len(ranked_scopes)
        ):
            selected_positions.update(ranked_scopes[selected_scope_count][1])
            selected_scope_count += 1

        selected = np.asarray(sorted(selected_positions), dtype=np.int64)
        scope_rows = []
        for rank, (scope, positions, score, _) in enumerate(ranked_scopes, 1):
            scope_rows.append({
                "rank": rank,
                "channel": scope[0],
                "phase_name": scope[1],
                "topic": scope[2],
                "score": score,
                "candidate_count": len(positions),
                "best_candidate_rank": positions[0] + 1,
                "selected": rank <= selected_scope_count,
            })
        diagnostics = {
            "scope_filter_enabled": True,
            "scope_filter_top_k": self.scope_filter_top_k,
            "scope_score_top_n": self.scope_score_top_n,
            "scope_score_variant": (
                "sum_top_n_content_embedding_cosine"
                if self.candidate_retrieval_mode == "embedding"
                else "sum_top_n_structured_expandc_bm25"
            ),
            "prefilter_candidate_count": int(len(candidate_indices)),
            "postfilter_candidate_count": int(len(selected)),
            "requested_scope_count": min(self.scope_filter_top_k, len(ranked_scopes)),
            "actual_scope_count": selected_scope_count,
            "scope_budget_fallback_used": selected_scope_count > self.scope_filter_top_k,
            "scope_ranking": scope_rows,
            "prefilter_candidate_message_ids": [
                self.messages[int(index)].get("msg_node") for index in candidate_indices
            ],
        }
        return (
            candidate_indices[selected],
            candidate_scores[selected],
            original_ranks[selected],
            diagnostics,
        )

    def _bm25_pair_similarity(
        self, candidate_indices: np.ndarray, *, structured: bool
    ) -> np.ndarray:
        text_fn = structured_index_text if structured else clean_content
        corpus = [tokenize(text_fn(self.messages[int(i)])) for i in candidate_indices]
        local_bm25 = BM25Okapi(corpus)
        n = len(corpus)
        directed = np.zeros((n, n), dtype=np.float64)
        for i, tokens in enumerate(corpus):
            directed[i] = np.asarray(local_bm25.get_scores(tokens), dtype=np.float64)
        self_scores = np.maximum(np.diag(directed), 1e-12)
        normalized = directed / self_scores[:, None]
        similarity = 0.5 * (normalized + normalized.T)
        np.fill_diagonal(similarity, 1.0)
        return np.clip(similarity, 0.0, 1.0)

    def _dense_pair_similarity(self, candidate_indices: np.ndarray) -> np.ndarray:
        assert self.pair_embedding is not None
        matrix = np.asarray(
            self.pair_embedding.embeddings[candidate_indices],
            dtype=np.float32,
        )
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        similarity = np.asarray(matrix @ matrix.T, dtype=np.float64)
        np.fill_diagonal(similarity, 1.0)
        return np.clip(similarity, -1.0, 1.0)


    def _dynamic_query_coverage_features(
        self, question: str, candidate_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Build IDF-weighted query demands for dynamic residual coverage.

        Per-term BM25 contributions are normalized within the current
        candidate pool. IDF is then reintroduced exactly once as the demand
        weight, avoiding the equal-term behavior of pure term_max QueryCov.
        """
        query_terms = [
            token
            for token in dict.fromkeys(tokenize(question))
            if token not in QUERYCOV_STOPWORDS
        ]
        count = len(candidate_indices)
        if not query_terms or count == 0:
            return (
                np.zeros((count, 0), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                {
                    "dynamic_coverage_terms": query_terms,
                    "dynamic_coverage_term_count": len(query_terms),
                    "dynamic_coverage_active_term_count": 0,
                },
            )

        bm25 = (
            self.candidate_bm25
            if self.complementarity_fields == "structured"
            else self.content_bm25
        )
        if bm25 is None:
            raise RuntimeError("BM25 index required for dynamic query coverage")

        contributions = np.zeros((count, len(query_terms)), dtype=np.float64)
        term_idfs = np.zeros(len(query_terms), dtype=np.float64)
        for term_index, token in enumerate(query_terms):
            document_frequency = bm25.df.get(token, 0)
            term_idfs[term_index] = math.log(
                1
                + (bm25.n - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
        for local, global_index in enumerate(candidate_indices):
            index = int(global_index)
            counts = bm25.tf[index]
            document_length = len(bm25.corpus[index]) or 1
            for term_index, token in enumerate(query_terms):
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + bm25.k1 * (
                    1
                    - bm25.b
                    + bm25.b
                    * document_length
                    / max(bm25.avgdl, 1e-9)
                )
                contributions[local, term_index] = (
                    term_idfs[term_index]
                    * frequency
                    * (bm25.k1 + 1)
                    / denominator
                )

        term_maxima = contributions.max(axis=0)
        active = term_maxima > 0.0
        normalized = np.divide(
            contributions,
            term_maxima[None, :],
            out=np.zeros_like(contributions),
            where=term_maxima[None, :] > 0.0,
        )
        term_weights = np.where(active, term_idfs, 0.0)
        weight_total = float(term_weights.sum())
        if weight_total > 0.0:
            term_weights /= weight_total

        return normalized, term_weights, {
            "dynamic_coverage_terms": query_terms,
            "dynamic_coverage_term_count": len(query_terms),
            "dynamic_coverage_active_term_count": int(np.count_nonzero(active)),
            "dynamic_coverage_term_weights": [
                float(value) for value in term_weights
            ],
        }


    def _query_term_complementarity(
        self, question: str, candidate_indices: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Pairwise BM25-weighted adaptation of QueryCov.

        Recall Is Not Enough (arXiv:2607.00725, Eq. 1) defines QueryCov as
        set cover over distinct query content terms. CDS needs a fixed
        message-message affinity matrix, so a pair receives the normalized
        marginal query-term coverage over its better singleton:

            C_ij = (Cov(i,j) - max(R_i,R_j)) / (Cov(i,j) + eps)

        This pairwise formula is our CDS adaptation, not a source-paper
        equation.
        """
        # Distinct terms follow QueryCov's set-cover semantics. Asker/role
        # expansion stays in query-message relevance and is not added here.
        query_terms = [
            token
            for token in dict.fromkeys(tokenize(question))
            if token not in QUERYCOV_STOPWORDS
        ]
        count = len(candidate_indices)
        complementarity = np.zeros((count, count), dtype=np.float64)
        if not query_terms or count == 0:
            return complementarity, {
                "querycov_terms": query_terms,
                "querycov_term_count": len(query_terms),
                "querycov_normalization": self.complementarity_normalization,
                "querycov_nonzero_pair_count": 0,
                "querycov_pair_mean": 0.0,
                "querycov_pair_max": 0.0,
            }

        bm25 = (
            self.candidate_bm25
            if self.complementarity_fields == "structured"
            else self.content_bm25
        )
        if bm25 is None:
            raise RuntimeError("Content BM25 index required for QueryCov")

        contributions = np.zeros((count, len(query_terms)), dtype=np.float64)
        for local, global_index in enumerate(candidate_indices):
            index = int(global_index)
            counts = bm25.tf[index]
            document_length = len(bm25.corpus[index]) or 1
            for term_index, token in enumerate(query_terms):
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                document_frequency = bm25.df.get(token, 0)
                inverse_document_frequency = math.log(
                    1
                    + (bm25.n - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + bm25.k1 * (
                    1
                    - bm25.b
                    + bm25.b
                    * document_length
                    / max(bm25.avgdl, 1e-9)
                )
                contributions[local, term_index] = (
                    inverse_document_frequency
                    * frequency
                    * (bm25.k1 + 1)
                    / denominator
                )

        # raw preserves the existing behavior. term_max treats each distinct
        # query term as one demand within the current candidate pool.
        term_maxima = contributions.max(axis=0) if count else np.zeros(len(query_terms))
        if self.complementarity_normalization == "term_max":
            contributions = np.divide(
                contributions,
                term_maxima[None, :],
                out=np.zeros_like(contributions),
                where=term_maxima[None, :] > 0.0,
            )

        singleton_scores = contributions.sum(axis=1)
        epsilon = 1e-12
        for left in range(count):
            pair_coverage = np.maximum(
                contributions[left][None, :], contributions[left + 1 :]
            ).sum(axis=1)
            marginal_gain = pair_coverage - np.maximum(
                singleton_scores[left], singleton_scores[left + 1 :]
            )
            normalized = np.divide(
                np.maximum(marginal_gain, 0.0),
                pair_coverage + epsilon,
                out=np.zeros_like(pair_coverage),
                where=pair_coverage > 0,
            )
            complementarity[left, left + 1 :] = normalized
            complementarity[left + 1 :, left] = normalized

        upper = complementarity[np.triu_indices(count, k=1)]
        return complementarity, {
            "querycov_terms": query_terms,
            "querycov_term_count": len(query_terms),
            "querycov_normalization": self.complementarity_normalization,
            "querycov_covered_term_count": int(np.count_nonzero(term_maxima > 0.0)),
            "querycov_nonzero_pair_count": int(np.count_nonzero(upper > 0)),
            "querycov_pair_mean": float(upper.mean()) if upper.size else 0.0,
            "querycov_pair_max": float(upper.max()) if upper.size else 0.0,
        }


    def _native_structure_affinity(
        self, candidate_indices: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build a conservative, model-free native structure matrix.

        G_ij is binary and is activated only by a direct reply hyperedge or by
        true adjacency in the full conversation inside an exact
        channel/phase/topic scope. G never removes an edge and never claims a
        semantic relation; it only conditions positive QueryCov complementarity.
        """
        count = len(candidate_indices)
        graph = np.zeros((count, count), dtype=np.float64)
        reasons: Dict[Tuple[int, int], set[str]] = defaultdict(set)
        local_by_id = {
            str(self.messages[int(global_index)].get("msg_node") or ""): local
            for local, global_index in enumerate(candidate_indices)
            if self.messages[int(global_index)].get("msg_node")
        }

        # A reply hyperedge contains the parent and every retrieved direct
        # reply. Sibling replies may therefore share a grounded interaction.
        reply_groups: Dict[str, set[int]] = defaultdict(set)
        for local, global_index in enumerate(candidate_indices):
            reply_to = str(
                self.messages[int(global_index)].get("reply_to") or ""
            )
            if reply_to:
                reply_groups[reply_to].add(local)
        reply_hyperedge_count = 0
        for parent_id, replies in reply_groups.items():
            members = set(replies)
            parent_local = local_by_id.get(parent_id)
            if parent_local is not None:
                members.add(parent_local)
            if len(members) < 2:
                continue
            reply_hyperedge_count += 1
            ordered = sorted(members)
            for offset, left in enumerate(ordered):
                for right in ordered[offset + 1 :]:
                    reasons[(left, right)].add("reply_hyperedge")

        # Exact-scope adjacency is measured in the full source conversation,
        # not in the retrieved list. Broad same-phase/topic links are excluded.
        scope_pair_count = 0
        if self.native_scope_neighbors > 0:
            for left in range(count):
                left_position = self.native_scope_position.get(
                    int(candidate_indices[left])
                )
                if left_position is None:
                    continue
                for right in range(left + 1, count):
                    right_position = self.native_scope_position.get(
                        int(candidate_indices[right])
                    )
                    if right_position is None:
                        continue
                    if (
                        left_position[0] == right_position[0]
                        and abs(left_position[1] - right_position[1])
                        <= self.native_scope_neighbors
                    ):
                        reasons[(left, right)].add("exact_scope_local")
                        scope_pair_count += 1

        for left, right in reasons:
            graph[left, right] = 1.0
            graph[right, left] = 1.0
        upper = graph[np.triu_indices(count, k=1)]
        reply_pair_count = sum(
            "reply_hyperedge" in pair_reasons
            for pair_reasons in reasons.values()
        )
        return graph, {
            "native_conditioned_reply_hyperedge_count": reply_hyperedge_count,
            "native_conditioned_reply_pair_count": reply_pair_count,
            "native_conditioned_scope_pair_count": scope_pair_count,
            "native_conditioned_union_pair_count": len(reasons),
            "native_conditioned_edge_density": float(upper.mean())
            if upper.size
            else 0.0,
            "native_conditioned_edges": [
                {
                    "left": left,
                    "right": right,
                    "left_msg_id": self.messages[
                        int(candidate_indices[left])
                    ].get("msg_node"),
                    "right_msg_id": self.messages[
                        int(candidate_indices[right])
                    ].get("msg_node"),
                    "reasons": sorted(pair_reasons),
                }
                for (left, right), pair_reasons in sorted(reasons.items())
            ],
        }


    def _native_relation_surprise(
        self, candidate_indices: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build sparse native motifs weighted by query-local self-information."""
        count = len(candidate_indices)
        relation = np.zeros((count, count), dtype=np.float64)
        total_pairs = count * (count - 1) // 2
        motif_pairs: Dict[str, set[Tuple[int, int]]] = {
            "reply_interaction": set(),
            "topic_time_trajectory": set(),
            "member_topic_trajectory": set(),
            "cross_member_interaction": set(),
        }
        if total_pairs <= 0:
            return relation, {
                "native_surprise_total_pairs": total_pairs,
                "native_surprise_nonzero_pair_count": 0,
                "native_surprise_pair_mean": 0.0,
                "native_surprise_pair_max": 0.0,
                "native_surprise_motifs": {},
            }

        local_by_id = {
            str(self.messages[int(global_index)].get("msg_node") or ""): local
            for local, global_index in enumerate(candidate_indices)
            if self.messages[int(global_index)].get("msg_node")
        }
        reply_groups: Dict[str, set[int]] = defaultdict(set)
        for local, global_index in enumerate(candidate_indices):
            reply_to = str(self.messages[int(global_index)].get("reply_to") or "")
            if reply_to:
                reply_groups[reply_to].add(local)
        for parent_id, replies in reply_groups.items():
            members = set(replies)
            parent = local_by_id.get(parent_id)
            if parent is not None:
                members.add(parent)
            ordered = sorted(members)
            for offset, left in enumerate(ordered):
                for right in ordered[offset + 1:]:
                    motif_pairs["reply_interaction"].add((left, right))

        for left in range(count):
            left_global = int(candidate_indices[left])
            for right in range(left + 1, count):
                right_global = int(candidate_indices[right])
                left_topic = self.native_topic_position.get(left_global)
                right_topic = self.native_topic_position.get(right_global)
                if (left_topic is not None and right_topic is not None
                        and left_topic[0] == right_topic[0]
                        and abs(left_topic[1] - right_topic[1]) == 1):
                    motif_pairs["topic_time_trajectory"].add((left, right))
                left_member = self.native_member_topic_position.get(left_global)
                right_member = self.native_member_topic_position.get(right_global)
                if (left_member is not None and right_member is not None
                        and left_member[0] == right_member[0]
                        and abs(left_member[1] - right_member[1]) == 1):
                    motif_pairs["member_topic_trajectory"].add((left, right))

        for left, right in motif_pairs["reply_interaction"]:
            left_author = str(self.messages[int(candidate_indices[left])].get("author") or "")
            right_author = str(self.messages[int(candidate_indices[right])].get("author") or "")
            if left_author and right_author and left_author != right_author:
                motif_pairs["cross_member_interaction"].add((left, right))

        information: Dict[str, float] = {}
        for name, pairs in motif_pairs.items():
            if pairs:
                probability = len(pairs) / total_pairs
                information[name] = -math.log(max(probability, 1e-12))
        normalizer = sum(information.values())
        if normalizer > 0:
            for name, pairs in motif_pairs.items():
                weight = information.get(name, 0.0) / normalizer
                for left, right in pairs:
                    relation[left, right] += weight
                    relation[right, left] += weight

        np.fill_diagonal(relation, 0.0)
        upper = relation[np.triu_indices(count, k=1)]
        positive = upper[upper > 0.0]
        motifs = {
            name: {
                "pair_count": len(pairs),
                "pair_fraction": len(pairs) / total_pairs,
                "self_information": information.get(name, 0.0),
                "normalized_weight": information.get(name, 0.0) / normalizer if normalizer > 0 else 0.0,
            }
            for name, pairs in motif_pairs.items()
        }
        return relation, {
            "native_surprise_total_pairs": total_pairs,
            "native_surprise_nonzero_pair_count": int(positive.size),
            "native_surprise_pair_mean": float(positive.mean()) if positive.size else 0.0,
            "native_surprise_pair_max": float(positive.max()) if positive.size else 0.0,
            "native_surprise_motifs": motifs,
        }

    def _native_grounded_bridge(
        self,
        question: str,
        candidate_indices: np.ndarray,
        relevance: np.ndarray,
        base_dissimilarity: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Validate native candidate relations by reciprocal retrieval gain.

        Native metadata proposes a pair but never creates value by itself.
        For a proposed pair (i, j), residual content terms from i must improve
        retrieval of j (and/or vice versa). The resulting bridge value is
        additionally gated by query relevance and content non-redundancy:

            B_ij = N_ij * sqrt(R_i R_j) * D_ij * X_ij.

        B is later used only as a bounded amplifier of QueryCov; it is not an
        independent additive reward.
        """
        count = len(candidate_indices)
        bridge = np.zeros((count, count), dtype=np.float64)
        empty = {
            "native_bridge_candidate_pair_count": 0,
            "native_bridge_reply_pair_count": 0,
            "native_bridge_scope_pair_count": 0,
            "native_bridge_nonzero_pair_count": 0,
            "native_bridge_pair_mean": 0.0,
            "native_bridge_pair_max": 0.0,
        }
        if count < 2:
            return bridge, empty
        if self.content_bm25 is None:
            raise RuntimeError(
                "Native-grounded bridge requires content BM25; "
                "set CDS_COMPLEMENTARITY_FIELDS=content"
            )

        candidate_mask = np.zeros((count, count), dtype=bool)
        local_by_id = {
            str(self.messages[int(global_index)].get("msg_node") or ""): local
            for local, global_index in enumerate(candidate_indices)
            if self.messages[int(global_index)].get("msg_node")
        }

        reply_pairs: set[Tuple[int, int]] = set()
        for child, global_index in enumerate(candidate_indices):
            reply_to = str(
                self.messages[int(global_index)].get("reply_to") or ""
            )
            parent = local_by_id.get(reply_to)
            if parent is None or parent == child:
                continue
            left, right = sorted((parent, child))
            candidate_mask[left, right] = True
            candidate_mask[right, left] = True
            reply_pairs.add((left, right))

        # Native scope is a coarse proposal mechanism, not proof of a
        # semantic relation. Connect only consecutive retrieved candidates
        # inside each exact scope. Unlike full-source adjacency, this can
        # bridge over unretrieved chatter; reciprocal retrieval gain below
        # decides whether the proposed pair receives any useful strength.
        scope_candidates: Dict[
            Tuple[str, str, str], List[Tuple[int, int]]
        ] = defaultdict(list)
        for local, global_index in enumerate(candidate_indices):
            position = self.native_scope_position.get(int(global_index))
            if position is not None:
                scope_candidates[position[0]].append((position[1], local))

        scope_pairs: set[Tuple[int, int]] = set()
        for values in scope_candidates.values():
            values.sort()
            for (_, left), (_, right) in zip(values, values[1:]):
                pair = tuple(sorted((left, right)))
                candidate_mask[pair[0], pair[1]] = True
                candidate_mask[pair[1], pair[0]] = True
                scope_pairs.add(pair)

        query_terms = [
            token
            for token in dict.fromkeys(tokenize(question))
            if token not in QUERYCOV_STOPWORDS
        ]
        query_term_set = set(query_terms)
        residual_terms: List[List[str]] = []
        for global_index in candidate_indices:
            tokens = tokenize(
                clean_content(self.messages[int(global_index)])
            )
            residual_terms.append([
                token
                for token in dict.fromkeys(tokens)
                if token not in query_term_set
                and token not in QUERYCOV_STOPWORDS
            ])

        bm25 = self.content_bm25
        epsilon = 1e-12

        def term_score(global_index: int, terms: List[str]) -> float:
            counts = bm25.tf[global_index]
            document_length = len(bm25.corpus[global_index]) or 1
            score = 0.0
            for token in terms:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                document_frequency = bm25.df.get(token, 0)
                inverse_document_frequency = math.log(
                    1
                    + (bm25.n - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + bm25.k1 * (
                    1
                    - bm25.b
                    + bm25.b
                    * document_length
                    / max(bm25.avgdl, 1e-9)
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * (bm25.k1 + 1)
                    / denominator
                )
            return score

        base_scores = np.asarray([
            term_score(int(global_index), query_terms)
            for global_index in candidate_indices
        ], dtype=np.float64)
        relevance_unit = np.maximum(relevance, 0.0).astype(np.float64)
        relevance_max = (
            float(relevance_unit.max()) if relevance_unit.size else 0.0
        )
        if relevance_max > 0.0:
            relevance_unit /= relevance_max

        for left, right in zip(*np.where(np.triu(candidate_mask, k=1))):
            left_index = int(candidate_indices[left])
            right_index = int(candidate_indices[right])
            gain_left_to_right = term_score(
                right_index, residual_terms[left]
            )
            gain_right_to_left = term_score(
                left_index, residual_terms[right]
            )
            directional_left = gain_left_to_right / (
                base_scores[right] + gain_left_to_right + epsilon
            )
            directional_right = gain_right_to_left / (
                base_scores[left] + gain_right_to_left + epsilon
            )
            cross_retrieval = 0.5 * (
                directional_left + directional_right
            )
            value = (
                math.sqrt(relevance_unit[left] * relevance_unit[right])
                * float(
                    np.clip(base_dissimilarity[left, right], 0.0, 1.0)
                )
                * cross_retrieval
            )
            bridge[left, right] = value
            bridge[right, left] = value

        upper = bridge[np.triu_indices(count, k=1)]
        nonzero = upper[upper > 0.0]
        return bridge, {
            "native_bridge_candidate_pair_count": int(
                np.count_nonzero(np.triu(candidate_mask, k=1))
            ),
            "native_bridge_reply_pair_count": len(reply_pairs),
            "native_bridge_scope_pair_count": len(scope_pairs),
            "native_bridge_nonzero_pair_count": int(nonzero.size),
            "native_bridge_pair_mean": (
                float(nonzero.mean()) if nonzero.size else 0.0
            ),
            "native_bridge_pair_max": (
                float(nonzero.max()) if nonzero.size else 0.0
            ),
        }



    def _query_dynamic_edge_gate(
        self,
        question: str,
        candidate_indices: np.ndarray,
        relevance: np.ndarray,
        similarity: np.ndarray,
        complementarity: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build a query-conditioned soft gate over message-message edges.

        G_ij = sqrt(R_i R_j) * H_ij(q) * max(S_ij, C_ij), where H keeps
        exact-scope/reply pairs and otherwise requires a shared original-query
        anchor. The final gate is floor + (1-floor) * G, so no edge is deleted.
        """
        count = len(candidate_indices)
        query_terms = [
            token
            for token in dict.fromkeys(tokenize(question))
            if token not in QUERYCOV_STOPWORDS
        ]
        shared_anchor = np.zeros((count, count), dtype=np.float64)
        if query_terms and count:
            bm25 = self.content_bm25
            if bm25 is None:
                raise RuntimeError(
                    "Content BM25 index required for query-dynamic edge gating"
                )
            contributions = np.zeros((count, len(query_terms)), dtype=np.float64)
            term_idfs = np.zeros(len(query_terms), dtype=np.float64)
            for term_index, token in enumerate(query_terms):
                document_frequency = bm25.df.get(token, 0)
                term_idfs[term_index] = math.log(
                    1
                    + (bm25.n - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
            for local, global_index in enumerate(candidate_indices):
                index = int(global_index)
                counts = bm25.tf[index]
                document_length = len(bm25.corpus[index]) or 1
                for term_index, token in enumerate(query_terms):
                    frequency = counts.get(token, 0)
                    if not frequency:
                        continue
                    denominator = frequency + bm25.k1 * (
                        1
                        - bm25.b
                        + bm25.b
                        * document_length
                        / max(bm25.avgdl, 1e-9)
                    )
                    contributions[local, term_index] = (
                        term_idfs[term_index]
                        * frequency
                        * (bm25.k1 + 1)
                        / denominator
                    )
            term_maxima = contributions.max(axis=0)
            normalized = np.divide(
                contributions,
                term_maxima[None, :],
                out=np.zeros_like(contributions),
                where=term_maxima[None, :] > 0,
            )
            denominator = float(term_idfs.sum()) + 1e-12
            for left in range(count):
                values = np.minimum(normalized[left][None, :], normalized)
                shared_anchor[left] = values @ term_idfs / denominator
        np.fill_diagonal(shared_anchor, 0.0)

        native = np.zeros((count, count), dtype=np.float64)
        scopes = [
            self._scope_key(self.messages[int(index)])
            for index in candidate_indices
        ]
        message_ids = [
            str(self.messages[int(index)].get("msg_node") or "")
            for index in candidate_indices
        ]
        reply_targets = [
            str(self.messages[int(index)].get("reply_to") or "")
            for index in candidate_indices
        ]
        same_scope_pairs = 0
        reply_pairs = 0
        for left in range(count):
            for right in range(left + 1, count):
                same_scope = bool(any(scopes[left])) and scopes[left] == scopes[right]
                direct_reply = bool(
                    (reply_targets[left] and reply_targets[left] == message_ids[right])
                    or (
                        reply_targets[right]
                        and reply_targets[right] == message_ids[left]
                    )
                )
                if same_scope:
                    same_scope_pairs += 1
                if direct_reply:
                    reply_pairs += 1
                if same_scope or direct_reply:
                    native[left, right] = native[right, left] = 1.0

        context = native + (1.0 - native) * shared_anchor
        relevance_unit = self._normalize_positive(
            np.maximum(relevance, 0.0).astype(np.float64)
        )
        pair_relevance = np.sqrt(
            relevance_unit[:, None] * relevance_unit[None, :]
        )
        coherence = np.maximum(
            np.clip(similarity, 0.0, 1.0),
            np.clip(complementarity, 0.0, 1.0),
        )
        dynamic = np.clip(pair_relevance * context * coherence, 0.0, 1.0)
        np.fill_diagonal(dynamic, 0.0)
        gate = self.dynamic_edge_floor + (
            1.0 - self.dynamic_edge_floor
        ) * dynamic
        np.fill_diagonal(gate, 0.0)

        upper = dynamic[np.triu_indices(count, k=1)]
        gate_upper = gate[np.triu_indices(count, k=1)]
        return gate, {
            "query_dynamic_edge": True,
            "dynamic_edge_variant": "soft_query_conditioned_native_anchor",
            "dynamic_edge_floor": self.dynamic_edge_floor,
            "dynamic_edge_query_terms": query_terms,
            "dynamic_edge_same_scope_pair_count": same_scope_pairs,
            "dynamic_edge_reply_pair_count": reply_pairs,
            "dynamic_edge_nonzero_pair_count": int(np.count_nonzero(upper > 0.0)),
            "dynamic_edge_pair_mean": float(upper.mean()) if upper.size else 0.0,
            "dynamic_edge_pair_max": float(upper.max()) if upper.size else 0.0,
            "dynamic_gate_pair_mean": (
                float(gate_upper.mean()) if gate_upper.size else 0.0
            ),
            "dynamic_gate_pair_min": (
                float(gate_upper.min()) if gate_upper.size else 0.0
            ),
            "dynamic_gate_pair_max": (
                float(gate_upper.max()) if gate_upper.size else 0.0
            ),
        }


    def _query_affinity(
        self,
        question: str,
        asker: str,
        candidate_indices: np.ndarray,
        candidate_scores: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.query_affinity_mode == "candidate_score":
            return self._normalize_positive(candidate_scores), {
                "multiview_variant": "none",
                "query_affinity_source": self.candidate_retrieval_mode,
            }

        if self.query_affinity_mode == "bm25f":
            if self.query_bm25f is None:
                raise RuntimeError("BM25F query-affinity scorer was not initialized")
            query = self.build_structured_query(question, asker)
            scores = np.asarray(
                self.query_bm25f.get_scores(
                    tokenize(query), document_indices=candidate_indices
                ),
                dtype=np.float64,
            )
            return self._normalize_positive(scores), {"multiview_variant": "none"}

        if self.query_affinity_mode == "multiview_consensus":
            base_affinity = self._normalize_positive(candidate_scores)
            query_tokens = tokenize(self.build_structured_query(question, asker))
            support_count = np.zeros(len(candidate_indices), dtype=np.int64)
            view_support_positions: Dict[str, List[int]] = {}
            top_k = min(self.multiview_top_k, len(candidate_indices))

            for view_name in ("content", "participant", "episode"):
                scorer = self.multiview_bm25.get(view_name)
                if scorer is None:
                    raise RuntimeError(
                        f"Multi-view BM25 scorer was not initialized: {view_name}"
                    )
                scores = np.asarray(
                    scorer.get_scores(
                        query_tokens, document_indices=candidate_indices
                    ),
                    dtype=np.float64,
                )
                positive = np.flatnonzero(scores > 0.0)
                ranked = (
                    positive[np.lexsort((positive, -scores[positive]))][:top_k]
                    if positive.size and top_k > 0
                    else np.asarray([], dtype=np.int64)
                )
                support_count[ranked] += 1
                view_support_positions[view_name] = [
                    int(position) for position in ranked
                ]

            bonus = 1.0 + self.multiview_consensus_weight * (
                np.maximum(support_count - 1, 0) / 2.0
            )
            return base_affinity * bonus, {
                "multiview_variant": "qs_topl_cross_view_consensus",
                "multiview_top_k": self.multiview_top_k,
                "multiview_consensus_weight": self.multiview_consensus_weight,
                "multiview_support_counts": [
                    int(value) for value in support_count
                ],
                "multiview_consensus_bonus": [float(value) for value in bonus],
                "multiview_support_positions": view_support_positions,
            }

        if self.mode != "dense":
            return self._normalize_positive(candidate_scores), {
                "multiview_variant": "none"
            }
        assert self.embedding is not None
        query = self.embedding.build_query(question, asker)
        query_embedding = self.embedding.query_embeddings.get(query)
        if query_embedding is None:
            raise RuntimeError(
                "Dense query embedding cache miss in cache-only CDS mode: " + query
            )
        matrix = np.asarray(self.embedding.embeddings[candidate_indices], dtype=np.float32)
        affinity = np.asarray(matrix @ query_embedding, dtype=np.float64)
        return np.clip(affinity, 0.0, 1.0), {"multiview_variant": "none"}

    def _solve_cds(
        self,
        note_dissimilarity: np.ndarray,
        relevance: np.ndarray,
        dynamic_contributions: Optional[np.ndarray] = None,
        dynamic_term_weights: Optional[np.ndarray] = None,
    ):
        n = len(relevance)
        affinity = np.zeros((n + 1, n + 1), dtype=np.float64)
        affinity[1:, 1:] = note_dissimilarity
        query_edges = relevance.copy()
        affinity[0, 1:] = query_edges
        affinity[1:, 0] = query_edges
        np.fill_diagonal(affinity, 0.0)


        if self.cds_solver_mode == "query_gated_iht":
            # Query-gated Iterative Hard Thresholding with Armijo line
            # search. Unlike topk_projected, each iteration optimizes the
            # same fixed objective and accepts only sufficient ascent.
            k = min(self.retrieve_top_k, n)
            relevance_scale = (
                float(np.max(query_edges)) if query_edges.size else 0.0
            )
            normalized_relevance = (
                query_edges / relevance_scale
                if relevance_scale > 0.0
                else np.zeros_like(query_edges)
            )

            relation_scale = (
                float(np.max(note_dissimilarity))
                if note_dissimilarity.size else 0.0
            )
            normalized_relation = (
                note_dissimilarity / relation_scale
                if relation_scale > 0.0
                else np.zeros_like(note_dissimilarity)
            )
            normalized_relation = 0.5 * (
                normalized_relation + normalized_relation.T
            )
            np.fill_diagonal(normalized_relation, 0.0)

            # B_ij = (R_i + R_j) / 2 * W_ij. Relations involving two
            # weakly relevant messages cannot dominate merely by diversity.
            query_gated_relation = 0.5 * (
                normalized_relevance[:, None]
                + normalized_relevance[None, :]
            ) * normalized_relation
            query_gated_relation = 0.5 * (
                query_gated_relation + query_gated_relation.T
            )
            np.fill_diagonal(query_gated_relation, 0.0)

            def objective(values: np.ndarray) -> float:
                return float(
                    normalized_relevance @ values
                    + 0.5
                    * self.topk_graph_weight
                    * values @ query_gated_relation @ values
                )

            iht_mass_floor = min(1e-6, 0.5 / max(k, 1))

            def project_sparse_simplex(values: np.ndarray) -> np.ndarray:
                """Project onto a fixed-K simplex with a tiny mass floor.

                The floor makes all K selected messages genuine solver outputs;
                otherwise the usual <=K projection can collapse to only one or
                two positive entries and downstream Top-K would be padded by
                arbitrary zero-score candidates.
                """
                if not values.size or k <= 0:
                    return np.zeros_like(values)
                support = np.argpartition(values, -k)[-k:]
                support_values = values[support]
                free_mass = 1.0 - k * iht_mass_floor
                shifted = support_values - iht_mass_floor
                ordered = np.sort(shifted)[::-1]
                cumulative = np.cumsum(ordered)
                positive = ordered - (
                    cumulative - free_mass
                ) / np.arange(1, len(ordered) + 1) > 0.0
                if np.any(positive):
                    rho = int(np.flatnonzero(positive)[-1])
                    theta = float(cumulative[rho] - free_mass) / (rho + 1)
                else:
                    theta = float(cumulative[-1] - free_mass) / len(ordered)
                projected = np.zeros_like(values)
                projected[support] = (
                    np.maximum(shifted - theta, 0.0) + iht_mass_floor
                )
                projected[support] /= float(projected[support].sum())
                return projected

            initial_order = np.lexsort(
                (np.arange(n), -normalized_relevance)
            )
            distribution = np.zeros(n, dtype=np.float64)
            initial_support = initial_order[:k]
            initial_values = np.maximum(
                normalized_relevance[initial_support], self.cds_epsilon
            )
            distribution[initial_support] = (
                initial_values / float(initial_values.sum())
            )
            current_objective = objective(distribution)
            initial_objective = current_objective
            objective_history = [current_objective]
            step_history: List[float] = []
            backtracking_history: List[int] = []
            support_change_history: List[int] = []
            stop_reason = "max_iterations"
            iterations = 0

            for iterations in range(1, self.cds_max_iterations + 1):
                gradient = normalized_relevance + (
                    self.topk_graph_weight
                    * (query_gated_relation @ distribution)
                )
                step = self.iht_initial_step
                accepted = False
                candidate = distribution
                candidate_objective = current_objective
                backtracking_steps = 0
                while backtracking_steps <= self.iht_max_backtracking:
                    candidate = project_sparse_simplex(
                        distribution + step * gradient
                    )
                    delta = candidate - distribution
                    delta_norm_sq = float(delta @ delta)
                    candidate_objective = objective(candidate)
                    required = (
                        current_objective
                        + 0.5 * self.iht_armijo_sigma * delta_norm_sq
                    )
                    if candidate_objective + 1e-15 >= required:
                        accepted = True
                        break
                    step *= self.iht_backtracking_factor
                    backtracking_steps += 1
                    if step < self.iht_min_step:
                        break

                if not accepted:
                    stop_reason = "line_search_failed"
                    break

                old_support = set(np.flatnonzero(distribution > 0.0))
                new_support = set(np.flatnonzero(candidate > 0.0))
                support_change = len(
                    old_support.symmetric_difference(new_support)
                ) // 2
                delta_l1 = float(
                    np.linalg.norm(candidate - distribution, ord=1)
                )
                objective_gain = candidate_objective - current_objective
                step_history.append(float(step))
                backtracking_history.append(backtracking_steps)
                support_change_history.append(support_change)
                distribution = candidate
                current_objective = candidate_objective
                objective_history.append(current_objective)

                if delta_l1 <= self.cds_tolerance:
                    stop_reason = "projected_stationary"
                    break
                if (
                    support_change == 0
                    and objective_gain <= self.cds_tolerance
                ):
                    stop_reason = "fixed_support_negligible_gain"
                    break
            else:
                stop_reason = "max_iterations"

            final_gradient = normalized_relevance + (
                self.topk_graph_weight
                * (query_gated_relation @ distribution)
            )
            final_support = np.flatnonzero(distribution > 0.0)
            return (
                distribution,
                0.0,
                0.0,
                iterations,
                affinity,
                0.0,
                {
                    "dynamic_coverage_enabled": False,
                    "dynamic_coverage_weight": self.dynamic_coverage_weight,
                    "query_gated_iht_enabled": True,
                    "topk_graph_weight": self.topk_graph_weight,
                    "topk_active_count": int(len(final_support)),
                    "iht_initial_step": self.iht_initial_step,
                    "iht_backtracking_factor": self.iht_backtracking_factor,
                    "iht_armijo_sigma": self.iht_armijo_sigma,
                    "iht_min_step": self.iht_min_step,
                    "iht_max_backtracking": self.iht_max_backtracking,
                    "iht_mass_floor": iht_mass_floor,
                    "iht_stop_reason": stop_reason,
                    "iht_initial_objective": initial_objective,
                    "iht_final_objective": current_objective,
                    "iht_objective_gain": current_objective - initial_objective,
                    "iht_objective_history": objective_history,
                    "iht_step_history": step_history,
                    "iht_backtracking_history": backtracking_history,
                    "iht_support_change_history": support_change_history,
                    "iht_final_support": [int(pos) for pos in final_support],
                    "iht_final_gradient": [
                        float(value) for value in final_gradient
                    ],
                },
            )

        if self.cds_solver_mode == "topk_monotone_block":
            # Monotone block Top-K selection under one fixed-cardinality
            # quadratic objective. Unlike topk_projected, a complete Top-K
            # proposal is committed only when it strictly improves the same
            # objective. Unlike topk_swap, several mutually supporting
            # messages can enter (and leave) in one accepted update.
            k = min(self.retrieve_top_k, n)

            relevance_scale = (
                float(np.max(query_edges)) if query_edges.size else 0.0
            )
            if relevance_scale > 0.0:
                normalized_relevance = query_edges / relevance_scale
            else:
                normalized_relevance = np.zeros_like(query_edges)

            relation_scale = (
                float(np.max(note_dissimilarity))
                if note_dissimilarity.size else 0.0
            )
            if relation_scale > 0.0:
                normalized_relation = note_dissimilarity / relation_scale
            else:
                normalized_relation = np.zeros_like(note_dissimilarity)
            normalized_relation = 0.5 * (
                normalized_relation + normalized_relation.T
            )
            np.fill_diagonal(normalized_relation, 0.0)

            unary_denominator = float(max(k, 1))
            pair_denominator = float(max(k * (k - 1), 1))

            def score_set(selected_tuple: Tuple[int, ...]) -> float:
                if not selected_tuple:
                    return 0.0
                selected_array = np.asarray(
                    selected_tuple, dtype=np.int64
                )
                unary_score = float(
                    normalized_relevance[selected_array].sum()
                    / unary_denominator
                )
                pair_score = 0.0
                if len(selected_array) > 1:
                    pair_score = float(
                        normalized_relation[
                            np.ix_(selected_array, selected_array)
                        ].sum() / pair_denominator
                    )
                return unary_score + self.topk_graph_weight * pair_score

            initial_order = np.lexsort(
                (np.arange(n), -normalized_relevance)
            )
            selected_tuple = tuple(
                sorted(int(pos) for pos in initial_order[:k])
            )
            objective = score_set(selected_tuple)
            initial_objective = objective
            objective_history = [objective]
            proposal_history: List[Dict[str, Any]] = []
            stop_reason = "max_iterations"
            iterations = 0

            for iterations in range(1, self.cds_max_iterations + 1):
                selected_array = np.asarray(
                    selected_tuple, dtype=np.int64
                )
                if len(selected_array):
                    graph_marginal = normalized_relation[
                        :, selected_array
                    ].sum(axis=1)
                else:
                    graph_marginal = np.zeros(n, dtype=np.float64)
                marginal_score = (
                    normalized_relevance / unary_denominator
                    + (2.0 * self.topk_graph_weight / pair_denominator)
                    * graph_marginal
                )
                proposal_order = np.lexsort(
                    (np.arange(n), -marginal_score)
                )
                proposed_tuple = tuple(
                    sorted(int(pos) for pos in proposal_order[:k])
                )

                if proposed_tuple == selected_tuple:
                    stop_reason = "fixed_support"
                    break

                proposed_objective = score_set(proposed_tuple)
                gain = proposed_objective - objective
                accepted = gain > self.cds_tolerance
                proposal_history.append({
                    "from_positions": list(selected_tuple),
                    "to_positions": list(proposed_tuple),
                    "objective_before": objective,
                    "objective_after": proposed_objective,
                    "gain": gain,
                    "accepted": accepted,
                    "replacement_count": len(
                        set(proposed_tuple) - set(selected_tuple)
                    ),
                })
                if not accepted:
                    stop_reason = "non_improving_block"
                    break

                selected_tuple = proposed_tuple
                objective = proposed_objective
                objective_history.append(objective)
            else:
                stop_reason = "max_iterations"

            selected_array = np.asarray(
                selected_tuple, dtype=np.int64
            )
            final_graph_marginal = np.zeros(n, dtype=np.float64)
            if len(selected_array):
                final_graph_marginal = normalized_relation[
                    :, selected_array
                ].sum(axis=1)
            final_marginal_score = (
                normalized_relevance / unary_denominator
                + (2.0 * self.topk_graph_weight / pair_denominator)
                * final_graph_marginal
            )
            distribution = np.zeros(n, dtype=np.float64)
            if len(selected_array):
                selected_scores = np.maximum(
                    final_marginal_score[selected_array], self.cds_epsilon
                )
                distribution[selected_array] = (
                    selected_scores / float(selected_scores.sum())
                )

            return (
                distribution,
                0.0,
                0.0,
                iterations,
                affinity,
                0.0,
                {
                    "dynamic_coverage_enabled": False,
                    "dynamic_coverage_weight": self.dynamic_coverage_weight,
                    "topk_monotone_block_enabled": True,
                    "topk_graph_weight": self.topk_graph_weight,
                    "topk_active_count": int(
                        np.count_nonzero(distribution)
                    ),
                    "topk_initial_objective": initial_objective,
                    "topk_final_objective": objective,
                    "topk_objective_gain": objective - initial_objective,
                    "topk_objective_history": objective_history,
                    "topk_block_proposal_count": len(proposal_history),
                    "topk_block_accepted_count": sum(
                        int(row["accepted"])
                        for row in proposal_history
                    ),
                    "topk_block_stop_reason": stop_reason,
                    "topk_block_proposal_history": proposal_history,
                    "topk_final_graph_marginal": [
                        float(value) for value in final_graph_marginal
                    ],
                    "topk_final_marginal_score": [
                        float(value) for value in final_marginal_score
                    ],
                },
            )

        if self.cds_solver_mode == "topk_swap":
            # Fixed-budget coordinate ascent. Start from the K strongest
            # query-relevant messages, enumerate every one-out/one-in swap,
            # and accept only the best strictly improving move. Unlike the
            # synchronous Top-K projection, this optimizes one fixed set
            # objective and therefore cannot oscillate between two sets.
            k = min(self.retrieve_top_k, n)
            relation_scale = (
                float(note_dissimilarity.max())
                if note_dissimilarity.size else 0.0
            )
            if relation_scale > 0.0:
                normalized_relation = note_dissimilarity / relation_scale
            else:
                normalized_relation = np.zeros_like(note_dissimilarity)
            np.fill_diagonal(normalized_relation, 0.0)

            def score_set(
                selected_tuple: Tuple[int, ...],
            ) -> Tuple[float, np.ndarray, np.ndarray]:
                selected_array = np.asarray(selected_tuple, dtype=np.int64)
                if not len(selected_array):
                    return (
                        0.0,
                        np.zeros(0, dtype=np.float64),
                        np.zeros(0, dtype=np.float64),
                    )
                if len(selected_array) > 1:
                    internal_graph = normalized_relation[
                        np.ix_(selected_array, selected_array)
                    ].sum(axis=1) / float(len(selected_array) - 1)
                else:
                    internal_graph = np.zeros(1, dtype=np.float64)
                utility = query_edges[selected_array] * (
                    1.0 + self.topk_graph_weight * internal_graph
                )
                return float(utility.mean()), utility, internal_graph

            initial_order = np.lexsort((np.arange(n), -query_edges))
            selected_tuple = tuple(
                sorted(int(pos) for pos in initial_order[:k])
            )
            objective, selected_utility, selected_graph = score_set(
                selected_tuple
            )
            initial_objective = objective
            objective_history = [objective]
            swap_history: List[Dict[str, Any]] = []
            all_positions = set(range(n))

            for _ in range(self.cds_max_iterations):
                selected_set = set(selected_tuple)
                unselected = sorted(all_positions - selected_set)
                best_objective = objective
                best_tuple: Optional[Tuple[int, ...]] = None
                best_out = -1
                best_in = -1
                for outgoing in selected_tuple:
                    retained = selected_set - {outgoing}
                    for incoming in unselected:
                        trial_tuple = tuple(sorted(retained | {incoming}))
                        trial_objective, _, _ = score_set(trial_tuple)
                        if trial_objective > (
                            best_objective + self.cds_tolerance
                        ):
                            best_objective = trial_objective
                            best_tuple = trial_tuple
                            best_out = outgoing
                            best_in = incoming
                if best_tuple is None:
                    break
                previous_objective = objective
                selected_tuple = best_tuple
                objective = best_objective
                objective_history.append(objective)
                swap_history.append({
                    "outgoing_position": best_out,
                    "incoming_position": best_in,
                    "gain": objective - previous_objective,
                    "objective": objective,
                })

            objective, selected_utility, selected_graph = score_set(
                selected_tuple
            )
            selected_array = np.asarray(selected_tuple, dtype=np.int64)
            distribution = np.zeros(n, dtype=np.float64)
            positive_utility = np.maximum(
                selected_utility, self.cds_epsilon
            )
            distribution[selected_array] = (
                positive_utility / float(positive_utility.sum())
            )
            final_graph = np.zeros(n, dtype=np.float64)
            final_utility = query_edges.copy()
            if k:
                for position in range(n):
                    peers = [
                        member for member in selected_tuple
                        if member != position
                    ]
                    if peers:
                        final_graph[position] = float(
                            normalized_relation[position, peers].mean()
                        )
                final_utility = query_edges * (
                    1.0 + self.topk_graph_weight * final_graph
                )

            return (
                distribution,
                0.0,
                0.0,
                len(swap_history) + 1,
                affinity,
                0.0,
                {
                    "dynamic_coverage_enabled": False,
                    "dynamic_coverage_weight": self.dynamic_coverage_weight,
                    "topk_swap_enabled": True,
                    "topk_graph_weight": self.topk_graph_weight,
                    "topk_active_count": int(np.count_nonzero(distribution)),
                    "topk_swap_count": len(swap_history),
                    "topk_initial_objective": initial_objective,
                    "topk_final_objective": objective,
                    "topk_objective_gain": objective - initial_objective,
                    "topk_objective_history": objective_history,
                    "topk_swap_history": swap_history,
                    "topk_final_graph_payoff": [
                        float(value) for value in final_graph
                    ],
                    "topk_final_utility": [
                        float(value) for value in final_utility
                    ],
                },
            )

        if self.cds_solver_mode == "dense_query_gated":
            # Keep all candidate messages active during query-gated graph
            # fixed-point iteration. The final evidence budget is applied
            # downstream, after convergence.
            distribution = np.full(
                n, 1.0 / max(n, 1), dtype=np.float64
            )
            dense_query_edges = np.maximum(
                query_edges, self.cds_epsilon
            )
            last_graph = np.zeros(n, dtype=np.float64)
            last_utility = dense_query_edges.copy()
            last_delta = 0.0
            iterations = 0
            stop_reason = "max_iterations"

            for iterations in range(1, self.cds_max_iterations + 1):
                graph_payoff = note_dissimilarity @ distribution
                graph_max = (
                    float(graph_payoff.max())
                    if graph_payoff.size else 0.0
                )
                if graph_max > 0.0:
                    last_graph = graph_payoff / graph_max
                else:
                    last_graph = np.zeros_like(graph_payoff)

                last_utility = dense_query_edges * (
                    1.0 + self.topk_graph_weight * last_graph
                )
                utility_sum = float(last_utility.sum())
                if not np.isfinite(utility_sum) or utility_sum <= 0.0:
                    next_distribution = np.full(
                        n, 1.0 / max(n, 1), dtype=np.float64
                    )
                else:
                    next_distribution = last_utility / utility_sum

                last_delta = float(
                    np.linalg.norm(
                        next_distribution - distribution, ord=1
                    )
                )
                distribution = next_distribution
                if last_delta <= self.cds_tolerance:
                    stop_reason = "fixed_point"
                    break

            return (
                distribution,
                0.0,
                0.0,
                iterations,
                affinity,
                0.0,
                {
                    "dynamic_coverage_enabled": False,
                    "dynamic_coverage_weight": self.dynamic_coverage_weight,
                    "dense_query_gated_enabled": True,
                    "topk_graph_weight": self.topk_graph_weight,
                    "dense_active_count": int(
                        np.count_nonzero(distribution > 0.0)
                    ),
                    "dense_stop_reason": stop_reason,
                    "dense_final_delta_l1": last_delta,
                    "dense_final_graph_payoff": [
                        float(value) for value in last_graph
                    ],
                    "dense_final_utility": [
                        float(value) for value in last_utility
                    ],
                },
            )


        if self.cds_solver_mode == "query_gated_replicator":
            # CDS multiplicative update with the query-gated graph utility.
            distribution = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
            query_relevance = np.maximum(query_edges, self.cds_epsilon)
            last_graph = np.zeros(n, dtype=np.float64)
            last_utility = query_relevance.copy()
            last_delta = 0.0
            iterations = 0
            stop_reason = "max_iterations"
            for iterations in range(1, self.cds_max_iterations + 1):
                graph_payoff = note_dissimilarity @ distribution
                graph_max = float(graph_payoff.max()) if graph_payoff.size else 0.0
                last_graph = (
                    graph_payoff / graph_max
                    if graph_max > 0.0
                    else np.zeros_like(graph_payoff)
                )
                last_utility = query_relevance * (
                    1.0 + self.topk_graph_weight * last_graph
                )
                next_distribution = distribution * np.maximum(
                    last_utility, np.finfo(np.float64).tiny
                )
                mass = float(next_distribution.sum())
                if not np.isfinite(mass) or mass <= 0.0:
                    next_distribution = np.full(
                        n, 1.0 / max(n, 1), dtype=np.float64
                    )
                    stop_reason = "numerical_reset"
                else:
                    next_distribution /= mass
                last_delta = float(np.linalg.norm(
                    next_distribution - distribution, ord=1
                ))
                distribution = next_distribution
                if last_delta <= self.cds_tolerance:
                    stop_reason = "fixed_point"
                    break
            positive = distribution[distribution > 0.0]
            entropy = -float(np.sum(positive * np.log(positive)))
            return (
                distribution, 0.0, 0.0, iterations, affinity, 0.0,
                {
                    "dynamic_coverage_enabled": False,
                    "dynamic_coverage_weight": self.dynamic_coverage_weight,
                    "query_gated_replicator_enabled": True,
                    "topk_graph_weight": self.topk_graph_weight,
                    "replicator_active_count": int(np.count_nonzero(
                        distribution > 0.0
                    )),
                    "replicator_stop_reason": stop_reason,
                    "replicator_final_delta_l1": last_delta,
                    "replicator_final_entropy": entropy,
                    "replicator_max_membership": float(
                        distribution.max() if distribution.size else 0.0
                    ),
                    "replicator_final_graph_payoff": [
                        float(value) for value in last_graph
                    ],
                    "replicator_final_utility": [
                        float(value) for value in last_utility
                    ],
                },
            )

        if self.cds_solver_mode == "fixed_share_replicator":
            # Fixed-share replicator: all candidates retain positive mass,
            # and the evidence budget is applied only after convergence.
            query_relevance = np.maximum(query_edges, self.cds_epsilon)
            start_mass = float(query_relevance.sum())
            if not np.isfinite(start_mass) or start_mass <= 0.0:
                start_distribution = np.full(
                    n, 1.0 / max(n, 1), dtype=np.float64
                )
            else:
                start_distribution = query_relevance / start_mass
            distribution = start_distribution.copy()
            last_graph = np.zeros(n, dtype=np.float64)
            last_utility = query_relevance.copy()
            last_delta = 0.0
            iterations = 0
            stop_reason = "max_iterations"
            tiny = np.finfo(np.float64).tiny
            for iterations in range(1, self.cds_max_iterations + 1):
                graph_payoff = note_dissimilarity @ distribution
                graph_max = (
                    float(graph_payoff.max())
                    if graph_payoff.size else 0.0
                )
                last_graph = (
                    graph_payoff / graph_max
                    if graph_max > 0.0
                    else np.zeros_like(graph_payoff)
                )
                last_utility = query_relevance * (
                    1.0 + self.topk_graph_weight * last_graph
                )

                replicated = distribution * np.maximum(last_utility, tiny)
                replicated_mass = float(replicated.sum())
                if not np.isfinite(replicated_mass) or replicated_mass <= 0.0:
                    replicated = start_distribution.copy()
                    stop_reason = "numerical_reset"
                else:
                    replicated /= replicated_mass

                next_distribution = (
                    (1.0 - self.fixed_share_rate) * replicated
                    + self.fixed_share_rate * start_distribution
                )
                next_mass = float(next_distribution.sum())
                if not np.isfinite(next_mass) or next_mass <= 0.0:
                    next_distribution = start_distribution.copy()
                    stop_reason = "numerical_reset"
                else:
                    next_distribution /= next_mass
                last_delta = float(np.linalg.norm(
                    next_distribution - distribution, ord=1
                ))
                distribution = next_distribution
                if last_delta <= self.cds_tolerance:
                    stop_reason = "fixed_point"
                    break

            positive = distribution[distribution > 0.0]
            entropy = -float(np.sum(positive * np.log(positive)))
            return (
                distribution, 0.0, 0.0, iterations, affinity, 0.0,
                {
                    "dynamic_coverage_enabled": False,
                    "dynamic_coverage_weight": self.dynamic_coverage_weight,
                    "fixed_share_replicator_enabled": True,
                    "fixed_share_rate": self.fixed_share_rate,
                    "topk_graph_weight": self.topk_graph_weight,
                    "fixed_share_active_count": int(np.count_nonzero(
                        distribution > 0.0
                    )),
                    "fixed_share_stop_reason": stop_reason,
                    "fixed_share_final_delta_l1": last_delta,
                    "fixed_share_final_entropy": entropy,
                    "fixed_share_max_membership": float(
                        distribution.max() if distribution.size else 0.0
                    ),
                    "fixed_share_min_membership": float(
                        distribution.min() if distribution.size else 0.0
                    ),
                    "fixed_share_start_distribution": [
                        float(value) for value in start_distribution
                    ],
                    "fixed_share_final_graph_payoff": [
                        float(value) for value in last_graph
                    ],
                    "fixed_share_final_utility": [
                        float(value) for value in last_utility
                    ],
                },
            )

        if self.cds_solver_mode == "sequential_cds_payoff_growth":
            # Keep one-message-at-a-time support growth, but score each
            # remaining candidate with the original CDS payoff (M x)_i.
            k = min(self.retrieve_top_k, n)
            selected: List[int] = []
            selected_set = set()
            selection_trace: List[Dict[str, Any]] = []
            total_inner_iterations = 0

            def solve_selected_support(
                support: List[int],
            ) -> Tuple[np.ndarray, int, float]:
                local_n = len(support)
                relation = note_dissimilarity[np.ix_(support, support)]
                matrix = np.zeros(
                    (local_n + 1, local_n + 1), dtype=np.float64
                )
                if local_n:
                    matrix[1:, 1:] = relation
                    matrix[0, 1:] = query_edges[support]
                    matrix[1:, 0] = query_edges[support]
                np.fill_diagonal(matrix, 0.0)
                local_alpha = (
                    float(np.linalg.eigvalsh(relation).max())
                    + self.cds_epsilon
                    if relation.size
                    else self.cds_epsilon
                )
                matrix[0, 0] = local_alpha
                membership = np.full(
                    local_n + 1,
                    1.0 / (local_n + 1),
                    dtype=np.float64,
                )
                local_iterations = 0
                for local_iterations in range(
                    1, self.cds_max_iterations + 1
                ):
                    payoff = matrix @ membership
                    next_membership = membership * np.maximum(
                        payoff, self.cds_epsilon
                    )
                    total = float(next_membership.sum())
                    if not np.isfinite(total) or total <= 0.0:
                        raise RuntimeError(
                            "Sequential CDS payoff dynamics became "
                            "numerically unstable"
                        )
                    next_membership /= total
                    if (
                        float(np.linalg.norm(
                            next_membership - membership, ord=1
                        ))
                        <= self.cds_tolerance
                    ):
                        membership = next_membership
                        break
                    membership = next_membership
                return membership, local_iterations, local_alpha

            if k > 0:
                seed = max(
                    range(n),
                    key=lambda pos: (float(query_edges[pos]), -pos),
                )
                selected.append(seed)
                selected_set.add(seed)
                selection_trace.append({
                    "round": 1,
                    "candidate_position": int(seed),
                    "query_relevance": float(query_edges[seed]),
                    "query_component": float(query_edges[seed]),
                    "graph_component": 0.0,
                    "utility": float(query_edges[seed]),
                    "operation": "seed",
                })

            last_membership = np.zeros(1, dtype=np.float64)
            last_candidate_payoff = np.zeros(n, dtype=np.float64)
            final_alpha = 0.0
            while len(selected) < k:
                membership, inner_iterations, final_alpha = (
                    solve_selected_support(selected)
                )
                total_inner_iterations += inner_iterations
                query_mass = float(membership[0])
                selected_membership = membership[1:]
                query_component = query_mass * query_edges
                graph_component = (
                    note_dissimilarity[:, selected] @ selected_membership
                )
                last_candidate_payoff = query_component + graph_component
                unselected = [
                    pos for pos in range(n) if pos not in selected_set
                ]
                chosen = max(
                    unselected,
                    key=lambda pos: (
                        float(last_candidate_payoff[pos]),
                        float(query_edges[pos]),
                        -pos,
                    ),
                )
                selected.append(chosen)
                selected_set.add(chosen)
                selection_trace.append({
                    "round": len(selected),
                    "candidate_position": int(chosen),
                    "query_relevance": float(query_edges[chosen]),
                    "query_mass": query_mass,
                    "query_component": float(query_component[chosen]),
                    "graph_component": float(graph_component[chosen]),
                    "utility": float(last_candidate_payoff[chosen]),
                    "inner_cds_iterations": int(inner_iterations),
                    "operation": "add",
                })
                last_membership = membership

            memberships = np.zeros(n, dtype=np.float64)
            final_query_membership = 0.0
            if selected:
                final_membership, inner_iterations, final_alpha = (
                    solve_selected_support(selected)
                )
                total_inner_iterations += inner_iterations
                final_query_membership = float(final_membership[0])
                selected_scores = np.maximum(
                    final_membership[1:], self.cds_epsilon
                )
                selected_score_total = float(selected_scores.sum())
                memberships[selected] = (
                    selected_scores / selected_score_total
                )
                last_membership = final_membership

            return (
                memberships,
                final_query_membership,
                final_alpha,
                total_inner_iterations,
                affinity,
                0.0,
                {
                    "dynamic_coverage_enabled": False,
                    "dynamic_coverage_weight": self.dynamic_coverage_weight,
                    "sequential_cds_payoff_growth_enabled": True,
                    "sequential_seed_position": (
                        int(selected[0]) if selected else None
                    ),
                    "sequential_selected_positions": [
                        int(pos) for pos in selected
                    ],
                    "sequential_round_count": max(0, len(selected) - 1),
                    "sequential_stop_reason": "budget_reached",
                    "sequential_selection_trace": selection_trace,
                    "sequential_final_local_membership": [
                        float(value) for value in last_membership
                    ],
                    "sequential_last_candidate_payoff": [
                        float(value) for value in last_candidate_payoff
                    ],
                    "sequential_total_inner_cds_iterations": int(
                        total_inner_iterations
                    ),
                },
            )

        if self.cds_solver_mode == "sequential_graph_growth":
            # Query-anchored monotonic graph growth. Start from the most
            # query-relevant candidate and add exactly one message per round.
            # The selected support can only grow, so support cycles are
            # impossible. Selected-node influence is always recomputed from
            # stable query relevance rather than insertion-time utility.
            k = min(self.retrieve_top_k, n)
            selected: List[int] = []
            selected_set = set()
            selection_trace: List[Dict[str, Any]] = []

            if k > 0:
                seed = max(
                    range(n),
                    key=lambda pos: (float(query_edges[pos]), -pos),
                )
                selected.append(seed)
                selected_set.add(seed)
                selection_trace.append(
                    {
                        "round": 1,
                        "candidate_position": int(seed),
                        "query_relevance": float(query_edges[seed]),
                        "graph_score": 0.0,
                        "utility": float(query_edges[seed]),
                        "operation": "seed",
                    }
                )

            last_distribution = np.zeros(n, dtype=np.float64)
            last_graph = np.zeros(n, dtype=np.float64)
            last_utility = query_edges.copy()

            while len(selected) < k:
                selected_relevance = np.maximum(
                    query_edges[selected], self.cds_epsilon
                )
                selected_mass = float(selected_relevance.sum())
                last_distribution = np.zeros(n, dtype=np.float64)
                if not np.isfinite(selected_mass) or selected_mass <= 0.0:
                    last_distribution[selected] = 1.0 / len(selected)
                else:
                    last_distribution[selected] = (
                        selected_relevance / selected_mass
                    )

                graph_payoff = note_dissimilarity @ last_distribution
                unselected = [
                    pos for pos in range(n) if pos not in selected_set
                ]
                graph_max = max(
                    (float(graph_payoff[pos]) for pos in unselected),
                    default=0.0,
                )
                last_graph = np.zeros(n, dtype=np.float64)
                if graph_max > 0.0:
                    last_graph[unselected] = (
                        graph_payoff[unselected] / graph_max
                    )
                last_utility = query_edges * (
                    1.0 + self.topk_graph_weight * last_graph
                )
                chosen = max(
                    unselected,
                    key=lambda pos: (
                        float(last_utility[pos]),
                        float(query_edges[pos]),
                        -pos,
                    ),
                )
                selected.append(chosen)
                selected_set.add(chosen)
                selection_trace.append(
                    {
                        "round": len(selected),
                        "candidate_position": int(chosen),
                        "query_relevance": float(query_edges[chosen]),
                        "graph_score": float(last_graph[chosen]),
                        "utility": float(last_utility[chosen]),
                        "operation": "add",
                    }
                )

            # Recompute each selected message's relation to the other final
            # selected messages. This score orders the evidence but cannot
            # change the already constructed support.
            final_graph_raw = np.zeros(n, dtype=np.float64)
            if selected:
                selected_relevance = np.maximum(
                    query_edges[selected], self.cds_epsilon
                )
                total_relevance = float(selected_relevance.sum())
                for local_pos, candidate_pos in enumerate(selected):
                    denominator = total_relevance - float(
                        selected_relevance[local_pos]
                    )
                    if denominator > 0.0:
                        numerator = float(np.dot(
                            note_dissimilarity[candidate_pos, selected],
                            selected_relevance,
                        ))
                        final_graph_raw[candidate_pos] = (
                            numerator / denominator
                        )
                final_graph_max = max(
                    (float(final_graph_raw[pos]) for pos in selected),
                    default=0.0,
                )
            else:
                final_graph_max = 0.0

            final_graph = np.zeros(n, dtype=np.float64)
            if final_graph_max > 0.0:
                final_graph[selected] = (
                    final_graph_raw[selected] / final_graph_max
                )
            final_utility = np.zeros(n, dtype=np.float64)
            if selected:
                final_utility[selected] = query_edges[selected] * (
                    1.0 + self.topk_graph_weight * final_graph[selected]
                )
            memberships = np.zeros(n, dtype=np.float64)
            selected_utility_mass = float(final_utility[selected].sum())
            if selected:
                if (
                    np.isfinite(selected_utility_mass)
                    and selected_utility_mass > 0.0
                ):
                    memberships[selected] = (
                        final_utility[selected] / selected_utility_mass
                    )
                else:
                    memberships[selected] = 1.0 / len(selected)

            return (
                memberships,
                0.0,
                0.0,
                max(0, len(selected) - 1),
                affinity,
                0.0,
                {
                    "dynamic_coverage_enabled": False,
                    "dynamic_coverage_weight": self.dynamic_coverage_weight,
                    "sequential_graph_growth_enabled": True,
                    "topk_graph_weight": self.topk_graph_weight,
                    "sequential_seed_position": (
                        int(selected[0]) if selected else None
                    ),
                    "sequential_selected_positions": [
                        int(pos) for pos in selected
                    ],
                    "sequential_round_count": max(0, len(selected) - 1),
                    "sequential_stop_reason": "budget_reached",
                    "sequential_selection_trace": selection_trace,
                    "sequential_final_graph_payoff": [
                        float(value) for value in final_graph
                    ],
                    "sequential_final_utility": [
                        float(value) for value in final_utility
                    ],
                },
            )

        if self.cds_solver_mode == "topk_projected":
            # Query-gated fixed-budget graph selection. Query relevance is
            # a unary score rather than a competing graph node. Every round
            # scores all candidates, projects back to exactly K active
            # messages, and therefore allows previously inactive candidates
            # to re-enter on the next round.
            k = min(self.retrieve_top_k, n)
            distribution = np.full(
                n, 1.0 / max(n, 1), dtype=np.float64
            )
            last_graph = np.zeros(n, dtype=np.float64)
            last_utility = query_edges.copy()
            previous_selected: Optional[Tuple[int, ...]] = None
            stable_rounds = 0
            iterations = 0
            for iterations in range(1, self.cds_max_iterations + 1):
                graph_payoff = note_dissimilarity @ distribution
                graph_max = (
                    float(graph_payoff.max())
                    if graph_payoff.size else 0.0
                )
                if graph_max > 0.0:
                    last_graph = graph_payoff / graph_max
                else:
                    last_graph = np.zeros_like(graph_payoff)
                last_utility = query_edges * (
                    1.0 + self.topk_graph_weight * last_graph
                )
                order = np.lexsort((np.arange(n), -last_utility))
                selected = order[:k]
                next_distribution = np.zeros(n, dtype=np.float64)
                selected_utility = np.maximum(
                    last_utility[selected], self.cds_epsilon
                )
                next_distribution[selected] = (
                    selected_utility / float(selected_utility.sum())
                )
                selected_key = tuple(sorted(int(pos) for pos in selected))
                if selected_key == previous_selected:
                    stable_rounds += 1
                else:
                    stable_rounds = 0
                delta = float(
                    np.linalg.norm(
                        next_distribution - distribution, ord=1
                    )
                )
                distribution = next_distribution
                previous_selected = selected_key
                if stable_rounds >= 2 and delta <= self.cds_tolerance:
                    break

            return (
                distribution,
                0.0,
                0.0,
                iterations,
                affinity,
                0.0,
                {
                    "dynamic_coverage_enabled": False,
                    "dynamic_coverage_weight": self.dynamic_coverage_weight,
                    "topk_projected_enabled": True,
                    "topk_graph_weight": self.topk_graph_weight,
                    "topk_active_count": int(np.count_nonzero(distribution)),
                    "topk_stable_rounds": stable_rounds,
                    "topk_final_graph_payoff": [
                        float(value) for value in last_graph
                    ],
                    "topk_final_utility": [
                        float(value) for value in last_utility
                    ],
                },
            )

        if self.cds_solver_mode == "fixed_query_mass":
            # Keep query influence fixed instead of allowing the query node to
            # absorb the simplex. Message-message payoffs are max-normalized so
            # that they are comparable to query affinities in [0, 1].
            note_scale = (
                max(float(np.max(np.abs(note_dissimilarity))), self.cds_epsilon)
                if note_dissimilarity.size
                else 1.0
            )
            normalized_note = note_dissimilarity / note_scale
            np.fill_diagonal(normalized_note, 0.0)
            payoff_matrix = affinity.copy()
            payoff_matrix[1:, 1:] = normalized_note

            message_distribution = query_edges.copy()
            initial_total = float(message_distribution.sum())
            if not np.isfinite(initial_total) or initial_total <= 0.0:
                message_distribution = np.full(n, 1.0 / max(1, n), dtype=np.float64)
            else:
                message_distribution /= initial_total

            rho = self.cds_query_mass
            iterations = 0
            for iterations in range(1, self.cds_max_iterations + 1):
                graph_payoff = normalized_note @ message_distribution
                payoff = rho * query_edges + (1.0 - rho) * graph_payoff
                factors = np.maximum(payoff, self.cds_epsilon)
                next_distribution = message_distribution * factors
                total = float(next_distribution.sum())
                if not np.isfinite(total) or total <= 0.0:
                    raise RuntimeError(
                        "Fixed-query-mass CDS became numerically unstable"
                    )
                next_distribution /= total
                if (
                    float(
                        np.linalg.norm(
                            next_distribution - message_distribution, ord=1
                        )
                    )
                    <= self.cds_tolerance
                ):
                    message_distribution = next_distribution
                    break
                message_distribution = next_distribution

            return (
                (1.0 - rho) * message_distribution,
                rho,
                0.0,
                iterations,
                payoff_matrix,
                0.0,
                {
                    "dynamic_coverage_enabled": False,
                    "dynamic_coverage_weight": self.dynamic_coverage_weight,
                },
            )

        # The spectral bound is computed on the non-query block in both
        # original modes. ``self_loop`` preserves the historical implementation.
        # ``classical_constraint`` follows M = A - alpha * I_barS for S={query}.
        alpha = (
            float(np.linalg.eigvalsh(note_dissimilarity).max())
            + self.cds_epsilon
            if note_dissimilarity.size
            else self.cds_epsilon
        )
        payoff_matrix = affinity.copy()
        if self.cds_solver_mode == "self_loop":
            payoff_matrix[0, 0] = alpha
            payoff_shift = 0.0
        elif self.cds_solver_mode == "classical_constraint":
            if n:
                diagonal = np.arange(1, n + 1)
                payoff_matrix[diagonal, diagonal] -= alpha
            payoff_shift = max(
                0.0, -float(payoff_matrix.min())
            ) + self.cds_epsilon
        else:
            raise ValueError(
                f"Unsupported CDS solver mode: {self.cds_solver_mode}"
            )

        membership = np.full(n + 1, 1.0 / (n + 1), dtype=np.float64)
        dynamic_enabled = (
            self.dynamic_coverage_weight > 0.0
            and dynamic_contributions is not None
            and dynamic_term_weights is not None
            and dynamic_contributions.shape[0] == n
            and dynamic_contributions.shape[1] == dynamic_term_weights.shape[0]
            and dynamic_term_weights.size > 0
            and float(dynamic_term_weights.sum()) > 0.0
        )
        last_dynamic_coverage = np.zeros(
            dynamic_term_weights.shape[0]
            if dynamic_term_weights is not None else 0,
            dtype=np.float64,
        )
        last_dynamic_bonus = np.zeros(n, dtype=np.float64)
        iterations = 0
        for iterations in range(1, self.cds_max_iterations + 1):
            payoff = payoff_matrix @ membership
            if dynamic_enabled:
                message_mass = membership[1:]
                message_total = float(message_mass.sum())
                if message_total > 0.0:
                    message_distribution = message_mass / message_total
                else:
                    message_distribution = np.full(
                        n, 1.0 / max(n, 1), dtype=np.float64
                    )
                last_dynamic_coverage = (
                    message_distribution @ dynamic_contributions
                )
                residual = (
                    dynamic_contributions
                    / (
                        self.cds_epsilon
                        + last_dynamic_coverage[None, :]
                    )
                    * dynamic_term_weights[None, :]
                ).sum(axis=1)
                residual_max = float(residual.max()) if residual.size else 0.0
                if residual_max > 0.0:
                    last_dynamic_bonus = residual / residual_max
                    payoff[1:] = (
                        payoff[1:]
                        + self.dynamic_coverage_weight * last_dynamic_bonus
                    )
                else:
                    last_dynamic_bonus.fill(0.0)
            if self.cds_solver_mode == "self_loop":
                factors = np.maximum(payoff, self.cds_epsilon)
            else:
                factors = payoff + payoff_shift
                if np.any(factors <= 0.0):
                    raise RuntimeError(
                        "Classical CDS global payoff shift did not produce "
                        "positive multiplicative factors"
                    )
            next_membership = membership * factors
            total = float(next_membership.sum())
            if not np.isfinite(total) or total <= 0:
                raise RuntimeError("CDS replicator dynamics became numerically unstable")
            next_membership /= total
            if float(np.linalg.norm(next_membership - membership, ord=1)) <= self.cds_tolerance:
                membership = next_membership
                break
            membership = next_membership
        return (
            membership[1:],
            float(membership[0]),
            alpha,
            iterations,
            payoff_matrix,
            payoff_shift,
            {
                "dynamic_coverage_enabled": bool(dynamic_enabled),
                "dynamic_coverage_weight": self.dynamic_coverage_weight,
                "dynamic_coverage_final_term_coverage": [
                    float(value) for value in last_dynamic_coverage
                ],
                "dynamic_coverage_final_message_bonus": [
                    float(value) for value in last_dynamic_bonus
                ],
                "dynamic_coverage_final_bonus_mean": (
                    float(last_dynamic_bonus.mean())
                    if last_dynamic_bonus.size else 0.0
                ),
                "dynamic_coverage_final_bonus_max": (
                    float(last_dynamic_bonus.max())
                    if last_dynamic_bonus.size else 0.0
                ),
            },
        )

    def retrieve(self, question: str, asker: str):
        structured_query, candidate_indices, candidate_scores = self._candidate_pool(question, asker)
        selection_started_ns = time.perf_counter_ns()
        (
            candidate_indices,
            candidate_scores,
            original_candidate_ranks,
            scope_diagnostics,
        ) = self._filter_candidate_scopes(candidate_indices, candidate_scores)
        relevance, query_affinity_diagnostics = self._query_affinity(
            question, asker, candidate_indices, candidate_scores
        )
        if self.mode in {"bm25", "structured_bm25"}:
            similarity = self._bm25_pair_similarity(
                candidate_indices, structured=self.mode == "structured_bm25"
            )
        else:
            similarity = self._dense_pair_similarity(candidate_indices)
        dissimilarity = np.clip(1.0 - similarity, 0.0, 2.0)
        np.fill_diagonal(dissimilarity, 0.0)
        base_dissimilarity = dissimilarity.copy()
        complementarity: Optional[np.ndarray] = None
        complementarity_diagnostics: Dict[str, Any] = {}
        if (
            self.complementarity_weight > 0
            or self.native_complementarity_weight > 0
            or self.native_bridge
            or self.query_dynamic_edge
        ):
            complementarity, complementarity_diagnostics = (
                self._query_term_complementarity(question, candidate_indices)
            )
        if self.complementarity_weight > 0 and complementarity is not None:
            dissimilarity = (
                dissimilarity
                + self.complementarity_weight * complementarity
            )
            np.fill_diagonal(dissimilarity, 0.0)

        dynamic_edge_diagnostics: Dict[str, Any] = {}
        if self.query_dynamic_edge:
            if self.complementarity_weight <= 0 or complementarity is None:
                raise RuntimeError(
                    "Query-dynamic edge gating requires QueryCov complementarity"
                )
            dynamic_gate, dynamic_edge_diagnostics = (
                self._query_dynamic_edge_gate(
                    question,
                    candidate_indices,
                    relevance,
                    similarity,
                    complementarity,
                )
            )
            dissimilarity = dissimilarity * dynamic_gate
            np.fill_diagonal(dissimilarity, 0.0)

        native_bridge_diagnostics: Dict[str, Any] = {}
        if self.native_bridge:
            if self.complementarity_weight <= 0 or complementarity is None:
                raise RuntimeError(
                    "Native bridge requires QueryCov complementarity"
                )
            if self.complementarity_fields != "content":
                raise RuntimeError(
                    "Native bridge requires "
                    "CDS_COMPLEMENTARITY_FIELDS=content"
                )
            native_bridge, native_bridge_diagnostics = (
                self._native_grounded_bridge(
                    question,
                    candidate_indices,
                    relevance,
                    base_dissimilarity,
                )
            )
            bridge_amplification = (
                self.complementarity_weight
                * complementarity
                * native_bridge
            )
            dissimilarity = dissimilarity + bridge_amplification
            np.fill_diagonal(dissimilarity, 0.0)
            upper = bridge_amplification[
                np.triu_indices(len(candidate_indices), k=1)
            ]
            native_bridge_diagnostics.update({
                "native_bridge_amplification_nonzero_pair_count": int(
                    np.count_nonzero(upper > 0.0)
                ),
                "native_bridge_amplification_pair_mean": (
                    float(upper[upper > 0.0].mean())
                    if np.any(upper > 0.0)
                    else 0.0
                ),
                "native_bridge_amplification_pair_max": (
                    float(upper.max()) if upper.size else 0.0
                ),
            })

        native_complementarity_diagnostics: Dict[str, Any] = {}
        if self.native_complementarity_weight > 0:
            if complementarity is None:
                raise RuntimeError("Native-conditioned complementarity requires QueryCov")
            native_graph, native_complementarity_diagnostics = (
                self._native_structure_affinity(candidate_indices)
            )
            native_complementarity = complementarity * native_graph
            dissimilarity = (
                dissimilarity
                + self.native_complementarity_weight * native_complementarity
            )
            np.fill_diagonal(dissimilarity, 0.0)
            upper = native_complementarity[
                np.triu_indices(len(candidate_indices), k=1)
            ]
            native_complementarity_diagnostics.update({
                "native_conditioned_nonzero_pair_count": int(
                    np.count_nonzero(upper > 0)
                ),
                "native_conditioned_pair_mean": float(upper.mean())
                if upper.size
                else 0.0,
                "native_conditioned_pair_max": float(upper.max())
                if upper.size
                else 0.0,
            })
        native_surprise_diagnostics: Dict[str, Any] = {}
        if self.native_surprise_weight > 0:
            native_surprise, native_surprise_diagnostics = (
                self._native_relation_surprise(candidate_indices)
            )
            dissimilarity = dissimilarity + self.native_surprise_weight * native_surprise
            np.fill_diagonal(dissimilarity, 0.0)

        relation_diagnostics: Dict[str, Any] = {}
        if self.relation_builder is not None:
            if self.relation_client is None:
                raise RuntimeError("Relation graph requires an initialized chat client")
            relation_mask, relation_diagnostics = self.relation_builder.build(
                self.relation_client, candidate_indices, similarity
            )
            dissimilarity *= relation_mask
        dynamic_contributions: Optional[np.ndarray] = None
        dynamic_term_weights: Optional[np.ndarray] = None
        dynamic_feature_diagnostics: Dict[str, Any] = {}
        if self.dynamic_coverage_weight > 0.0:
            (
                dynamic_contributions,
                dynamic_term_weights,
                dynamic_feature_diagnostics,
            ) = self._dynamic_query_coverage_features(
                question, candidate_indices
            )

        (
            memberships,
            query_membership,
            alpha,
            iterations,
            affinity,
            payoff_shift,
            dynamic_solver_diagnostics,
        ) = self._solve_cds(
            dissimilarity,
            relevance,
            dynamic_contributions,
            dynamic_term_weights,
        )

        selection_scores = memberships.copy()
        if self.selection_relevance_power > 0:
            selection_scores = selection_scores * np.power(
                np.maximum(relevance, 1e-12),
                self.selection_relevance_power,
            )
        order = np.lexsort((np.arange(len(selection_scores)), -selection_scores))
        selected = order[: min(self.retrieve_top_k, len(order))]
        if self.evidence_order == "bm25":
            selected = np.asarray(sorted(int(pos) for pos in selected), dtype=np.int64)
        elif self.evidence_order == "chronological":
            selected = np.asarray(
                sorted(
                    (int(pos) for pos in selected),
                    key=lambda pos: (
                        str(
                            self.messages[int(candidate_indices[pos])].get("timestamp")
                            or ""
                        ),
                        int(candidate_indices[pos]),
                    ),
                ),
                dtype=np.int64,
            )
        items = [
            CDSItem(
                msg_idx=int(candidate_indices[pos]),
                membership=float(memberships[pos]),
                candidate_rank=int(original_candidate_ranks[pos]),
                candidate_bm25_score=float(candidate_scores[pos]),
                query_affinity=float(relevance[pos]),
            )
            for pos in selected
        ]
        selection_latency_ms = (
            time.perf_counter_ns() - selection_started_ns
        ) / 1_000_000.0
        diagnostics = {
            "selection_latency_ms": selection_latency_ms,
            "candidate_message_ids": [
                self.messages[int(index)].get("msg_node") for index in candidate_indices
            ],
            # Keep the legacy key for analysis-script compatibility.
            "candidate_bm25_scores": [float(value) for value in candidate_scores],
            "candidate_retrieval_scores": [
                float(value) for value in candidate_scores
            ],
            "candidate_retrieval_mode": self.candidate_retrieval_mode,
            **scope_diagnostics,
            "query_affinity_mode": self.query_affinity_mode,
            "multiview_top_k": self.multiview_top_k,
            "multiview_consensus_weight": self.multiview_consensus_weight,
            **query_affinity_diagnostics,
            "bm25f_field_weights": {
                "content": self.bm25f_content_weight,
                "participant": self.bm25f_participant_weight,
                "episode": self.bm25f_episode_weight,
            },
            "bm25f_field_b": {
                "content": self.bm25f_content_b,
                "participant": self.bm25f_participant_b,
                "episode": self.bm25f_episode_b,
            },
            "bm25f_k1": self.bm25f_k1,
            "cds_memberships": [float(value) for value in memberships],
            "cds_query_membership": query_membership,
            "cds_note_membership_sum": float(memberships.sum()),
            "cds_solver_mode": self.cds_solver_mode,
            "cds_query_mass_setting": self.cds_query_mass,
            "topk_graph_weight": self.topk_graph_weight,
            "cds_payoff_shift": float(payoff_shift),
            "cds_selection_scores": [float(value) for value in selection_scores],
            "selection_relevance_power": self.selection_relevance_power,
            "query_affinities": [float(value) for value in relevance],
            "cds_alpha": alpha,
            "cds_iterations": iterations,
            "cds_affinity_min": float(affinity.min()),
            "cds_affinity_max": float(affinity.max()),
            "relation_graph": self.relation_graph,
            "evidence_order": self.evidence_order,
            "complementarity_weight": self.complementarity_weight,
            "complementarity_fields": self.complementarity_fields,
            "complementarity_normalization": self.complementarity_normalization,
            "dynamic_coverage_weight": self.dynamic_coverage_weight,
            "dynamic_coverage_variant": (
                "idf_weighted_residual_query_coverage"
                if self.dynamic_coverage_weight > 0.0
                else "none"
            ),
            "complementarity_variant": (
                "pairwise_bm25_querycov_" + self.complementarity_normalization
                if self.complementarity_weight > 0
                else "none"
            ),
            "native_complementarity_weight": self.native_complementarity_weight,
            "native_surprise_weight": self.native_surprise_weight,
            "native_surprise_variant": (
                "query_local_self_information_native_motifs"
                if self.native_surprise_weight > 0 else "none"
            ),
            "native_bridge": self.native_bridge,
            "native_bridge_variant": (
                "native_grounded_reciprocal_retrieval_gain"
                if self.native_bridge
                else "none"
            ),
            "native_complementarity_variant": (
                "querycov_conditioned_native_structure"
                if self.native_complementarity_weight > 0
                else "none"
            ),
            "native_scope_neighbors": self.native_scope_neighbors
            if self.native_complementarity_weight > 0
            else None,
            **complementarity_diagnostics,
            **dynamic_feature_diagnostics,
            **dynamic_solver_diagnostics,
            **dynamic_edge_diagnostics,
            **native_bridge_diagnostics,
            **native_complementarity_diagnostics,
            **native_surprise_diagnostics,
            **relation_diagnostics,
        }
        return structured_query, items, diagnostics

    def format_evidence(self, items: List[CDSItem]) -> List[str]:
        return [format_retrieved_message(self.messages[item.msg_idx]) for item in items]


def run_qa(*, questions, retriever, client, args, agent_system, judge_system):
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    config = {
        "stage": "cds_evidence_selection",
        "cds_mode": retriever.mode,
        "candidate_selector": (
            retriever.embedding_index_fields
            + "_embedding_cosine_"
            + retriever.query_text_mode
            if retriever.candidate_retrieval_mode == "embedding"
            else (
                "structured_question_only_bm25"
                if retriever.query_text_mode == "question_only"
                else "structured_expandc_bm25"
            )
        ),
        "candidate_retrieval_mode": retriever.candidate_retrieval_mode,
        "candidate_top_k": retriever.candidate_top_k,
        "retrieve_top_k": retriever.retrieve_top_k,
        "scope_filter_top_k": retriever.scope_filter_top_k,
        "scope_score_top_n": retriever.scope_score_top_n,
        "embedding_index_fields": (
            retriever.embedding_index_fields if retriever.embedding else None
        ),
        "embedding_store_dir": (
            (args.embedding_store_dir or None)
            if retriever.embedding is not None
            else None
        ),
        "pair_embedding_index_fields": (
            retriever.pair_embedding_index_fields
            if retriever.pair_embedding is not None
            else None
        ),
        "pair_embedding_store_dir": (
            (args.pair_embedding_store_dir or args.embedding_store_dir or None)
            if retriever.pair_embedding is not None
            else None
        ),
        "embedding_query_fields": (
            retriever.embedding.query_fields if retriever.embedding else None
        ),
        "embedding_cache_only": (
            retriever.embedding.cache_only if retriever.embedding else None
        ),
        "embedding_gpu_ids": (
            retriever.embedding.gpu_ids if retriever.embedding else None
        ),
        "embedding_batch_size": (
            retriever.embedding.batch_size if retriever.embedding else None
        ),
        "query_text_mode": retriever.query_text_mode,
        "query_affinity_mode": retriever.query_affinity_mode,
        "multiview_top_k": retriever.multiview_top_k,
        "multiview_consensus_weight": retriever.multiview_consensus_weight,
        "bm25f_field_weights": {
            "content": retriever.bm25f_content_weight,
            "participant": retriever.bm25f_participant_weight,
            "episode": retriever.bm25f_episode_weight,
        },
        "bm25f_field_b": {
            "content": retriever.bm25f_content_b,
            "participant": retriever.bm25f_participant_b,
            "episode": retriever.bm25f_episode_b,
        },
        "bm25f_k1": retriever.bm25f_k1,
        "evidence_order": retriever.evidence_order,
        "complementarity_weight": retriever.complementarity_weight,
        "complementarity_fields": retriever.complementarity_fields,
        "complementarity_normalization": retriever.complementarity_normalization,
        "dynamic_coverage_weight": retriever.dynamic_coverage_weight,
        "dynamic_coverage_variant": (
            "idf_weighted_residual_query_coverage"
            if retriever.dynamic_coverage_weight > 0.0
            else "none"
        ),
        "complementarity_variant": (
            "pairwise_bm25_querycov_" + retriever.complementarity_normalization
            if retriever.complementarity_weight > 0
            else "none"
        ),
        "native_complementarity_weight": retriever.native_complementarity_weight,
        "native_surprise_weight": retriever.native_surprise_weight,
        "native_surprise_variant": (
            "query_local_self_information_native_motifs"
            if retriever.native_surprise_weight > 0 else "none"
        ),
        "native_bridge": retriever.native_bridge,
        "native_bridge_variant": (
            "native_grounded_reciprocal_retrieval_gain"
            if retriever.native_bridge
            else "none"
        ),
        "native_complementarity_variant": (
            "querycov_conditioned_native_structure"
            if retriever.native_complementarity_weight > 0
            else "none"
        ),
        "native_scope_neighbors_for_complementarity": (
            retriever.native_scope_neighbors
            if retriever.native_complementarity_weight > 0
            else None
        ),
        "cds_epsilon": retriever.cds_epsilon,
        "cds_tolerance": retriever.cds_tolerance,
        "cds_max_iterations": retriever.cds_max_iterations,
        "cds_solver_mode": retriever.cds_solver_mode,
        "cds_query_mass": retriever.cds_query_mass,
        "topk_graph_weight": retriever.topk_graph_weight,
        "fixed_share_rate": retriever.fixed_share_rate,
        "relation_graph": retriever.relation_graph,
        "relation_model": args.relation_model if retriever.relation_graph in {"native", "bm25_threshold"} else None,
        "relation_thinking": args.relation_thinking if retriever.relation_graph in {"native", "bm25_threshold"} else None,
        "relation_max_tokens": args.relation_max_tokens if retriever.relation_graph in {"native", "bm25_threshold"} else None,
        "relation_pair_batch_size": args.relation_pair_batch_size if retriever.relation_graph in {"native", "bm25_threshold"} else None,
        "native_channel_window": args.native_channel_window if retriever.relation_graph == "native" else None,
        "native_scope_neighbors": args.native_scope_neighbors if retriever.relation_graph in {"native", "native_candidate_scope"} else None,
        "relation_verify": bool(args.relation_verify) if retriever.relation_graph in {"native", "bm25_threshold"} else None,
        "relation_threshold_iqr_multiplier": args.relation_threshold_iqr_multiplier if retriever.relation_graph == "bm25_threshold" else None,
        "relation_threshold_min_pairs": args.relation_threshold_min_pairs if retriever.relation_graph == "bm25_threshold" else None,
        "relation_threshold_max_pairs": args.relation_threshold_max_pairs if retriever.relation_graph == "bm25_threshold" else None,
        "agent_model": args.agent_model,
        "judge_model": args.judge_model,
        "agent_thinking": args.agent_thinking,
        "judge_thinking": args.judge_thinking,
        "agent_reasoning_effort": args.agent_reasoning_effort,
        "judge_reasoning_effort": args.judge_reasoning_effort,
    }

    def evaluate_one(record):
        question = str(record["question"])
        gold = str(record.get("answer", ""))
        asker = str(record.get("asking_user_id") or "")
        online_started_ns = time.perf_counter_ns()
        search_query, items, diagnostics = retriever.retrieve(question, asker)
        docs = retriever.format_evidence(items)
        online_latency_ms = (
            time.perf_counter_ns() - online_started_ns
        ) / 1_000_000.0
        base = {
            "query": question,
            "asking_user_id": asker,
            "retrieval_query": search_query,
            "experiment_config": config,
            "selection_latency_ms": diagnostics["selection_latency_ms"],
            "online_latency_ms": online_latency_ms,
            "selected_message_ids": [retriever.messages[x.msg_idx].get("msg_node") for x in items],
            "retrieval_items": [
                {
                    "rank": rank,
                    "msg_node": retriever.messages[item.msg_idx].get("msg_node"),
                    "score": item.membership,
                    "source": f"cds_{retriever.mode}",
                    "candidate_rank": item.candidate_rank,
                    # Legacy name retained for compatibility; use the
                    # generic field for embedding-candidate experiments.
                    "candidate_bm25_score": item.candidate_bm25_score,
                    "candidate_retrieval_score": item.candidate_bm25_score,
                    "candidate_retrieval_mode": (
                        retriever.candidate_retrieval_mode
                    ),
                    "query_affinity": item.query_affinity,
                }
                for rank, item in enumerate(items, 1)
            ],
            "cds_diagnostics": diagnostics,
            "retrieved_docs": docs,
        }
        stage = "agent"
        agent_reasoning = agent_final = ""
        try:
            output = call_chat(
                client,
                args.agent_model,
                agent_system,
                build_agent_prompt(question, asker, docs),
                args.agent_max_tokens,
                args.agent_thinking,
                args.agent_reasoning_effort,
            )
            agent_reasoning, agent_final = split_reasoning_and_final(output)
            stage = "judge"
            output = call_chat(
                client,
                args.judge_model,
                judge_system,
                build_judge_prompt(question, gold, agent_final),
                args.judge_max_tokens,
                args.judge_thinking,
                args.judge_reasoning_effort,
            )
            judge_reasoning, judge_final = split_reasoning_and_final(output)
        except EmptyModelResponseError as exc:
            return EvalResult(
                str(record.get("id", "")),
                {**base, "status": "skipped", "skip_stage": stage, "skip_reason": str(exc),
                 "agent_reasoning": agent_reasoning, "agent_answer": agent_final,
                 "judge_reasoning": "", "judge_answer": ""},
                None,
                True,
            )
        return EvalResult(
            str(record.get("id", "")),
            {**base, "agent_reasoning": agent_reasoning, "agent_answer": agent_final,
             "judge_reasoning": judge_reasoning, "judge_answer": judge_final},
            parse_judgment(judge_final),
        )

    correct = total = skipped = 0
    skipped_path = args.output_jsonl + ".skipped"
    with open(args.output_jsonl, "w", encoding="utf-8") as output_file, open(
        skipped_path, "w", encoding="utf-8"
    ) as skipped_file, ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = [executor.submit(evaluate_one, record) for record in questions]
        progress = tqdm(as_completed(futures), total=len(futures), desc="Evaluating", unit="q", dynamic_ncols=True)
        for future in progress:
            result = future.result()
            if result.skipped:
                skipped += 1
                skipped_file.write(json.dumps(result.record, ensure_ascii=False) + "\n")
            else:
                total += 1
                correct += int(result.judgment is True)
                output_file.write(json.dumps(result.record, ensure_ascii=False) + "\n")
            output_file.flush()
            skipped_file.flush()
            accuracy = 100 * correct / total if total else 0.0
            progress.set_postfix_str(f"acc={correct}/{total} ({accuracy:.1f}%) skipped={skipped}")
            print(f"{result.qid}: {'Skipped' if result.skipped else ('Correct' if result.judgment is True else 'Incorrect' if result.judgment is False else 'Unclear')}")
    if skipped:
        print(f"[coalmem] skipped {skipped} questions -> {skipped_path}")
    return correct, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation-json", required=True)
    parser.add_argument("--questions-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--agent-model", default="gpt-5")
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--agent-thinking", choices=("enabled", "disabled"), default=None)
    parser.add_argument("--judge-thinking", choices=("enabled", "disabled"), default=None)
    parser.add_argument(
        "--agent-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        default=None,
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        default=None,
    )
    parser.add_argument("--agent-max-tokens", type=int, default=512)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--agent-prompt", default="prompts/agent_system.txt")
    parser.add_argument("--judge-prompt", default="prompts/judge_system.txt")
    parser.add_argument(
        "--cds-mode",
        choices=("bm25", "structured_bm25", "hybrid", "dense"),
        default="hybrid",
    )
    parser.add_argument(
        "--candidate-retrieval-mode",
        choices=("bm25", "embedding"),
        default="bm25",
        help=(
            "Retriever used to form the candidate pool. bm25 preserves the "
            "existing structured ExpandC BM25 path; embedding uses cached "
            "content embeddings and the configured query text."
        ),
    )
    parser.add_argument("--candidate-top-k", type=int, default=48)
    parser.add_argument("--retrieve-top-k", type=int, default=10)
    parser.add_argument(
        "--scope-filter-top-k",
        type=int,
        default=0,
        help=(
            "Keep candidates from this many highest-scoring exact "
            "(channel, phase, topic) scopes before running CDS; 0 disables."
        ),
    )
    parser.add_argument(
        "--scope-score-top-n",
        type=int,
        default=3,
        help="Number of highest candidate BM25 scores summed to rank each scope.",
    )
    parser.add_argument(
        "--query-text-mode",
        choices=("expandqc", "question_only"),
        default="expandqc",
        help=(
            "expandqc uses question + asking user + asker role; "
            "question_only keeps the structured message index but uses only "
            "the original question for candidate retrieval and query affinity."
        ),
    )
    parser.add_argument(
        "--query-affinity-mode",
        choices=(
            "candidate_bm25",
            "candidate_score",
            "bm25f",
            "multiview_consensus",
        ),
        default="candidate_bm25",
        help=(
            "candidate_bm25 preserves the existing structured BM25 query "
            "edge; candidate_score uses the active candidate retriever score "
            "and is the explicit choice for embedding retrieval; bm25f fuses "
            "content, participant, and episode fields; "
            "multiview_consensus keeps the existing edge and applies a "
            "small bonus when separate Qs-conditioned views agree."
        ),
    )
    parser.add_argument("--multiview-top-k", type=int, default=10)
    parser.add_argument("--multiview-consensus-weight", type=float, default=0.05)
    parser.add_argument("--bm25f-content-weight", type=float, default=1.0)
    parser.add_argument("--bm25f-participant-weight", type=float, default=1.0)
    parser.add_argument("--bm25f-episode-weight", type=float, default=1.0)
    parser.add_argument("--bm25f-content-b", type=float, default=0.75)
    parser.add_argument("--bm25f-participant-b", type=float, default=0.0)
    parser.add_argument("--bm25f-episode-b", type=float, default=0.0)
    parser.add_argument("--bm25f-k1", type=float, default=1.5)
    parser.add_argument("--complementarity-weight", type=float, default=0.0)
    parser.add_argument(
        "--complementarity-fields",
        choices=("content", "structured"),
        default="content",
        help=(
            "Message fields used for BM25 query-term contributions in the "
            "pairwise QueryCov adaptation."
        ),
    )
    parser.add_argument(
        "--complementarity-normalization",
        choices=("raw", "term_max"),
        default="raw",
        help=(
            "raw preserves the existing QueryCov formula; term_max normalizes "
            "each query term by its maximum contribution among the current "
            "candidates before computing pair complementarity."
        ),
    )
    parser.add_argument(
        "--dynamic-coverage-weight",
        type=float,
        default=0.0,
        help=(
            "Add an iteration-dependent residual query-term coverage payoff "
            "to message nodes. Zero preserves the existing CDS solver exactly."
        ),
    )
    parser.add_argument("--native-complementarity-weight", type=float, default=0.0)
    parser.add_argument(
        "--native-surprise-weight",
        type=float,
        default=0.0,
        help="Add native motifs weighted by query-local self-information.",
    )
    parser.add_argument(
        "--native-bridge",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Amplify QueryCov only for native pairs validated by reciprocal "
            "content-BM25 retrieval gain; adds no independent relation weight."
        ),
    )
    parser.add_argument(
        "--query-dynamic-edge",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Softly reweight CDS message edges using query relevance, native "
            "context or shared query anchors, and semantic/query coherence."
        ),
    )
    parser.add_argument("--dynamic-edge-floor", type=float, default=0.5)
    parser.add_argument(
        "--evidence-order",
        choices=("cds", "bm25", "chronological"),
        default="cds",
        help="Ordering applied after CDS selects the evidence set.",
    )
    parser.add_argument(
        "--selection-relevance-power",
        type=float,
        default=0.0,
        help=(
            "Post-CDS relevance calibration: rank by membership * "
            "query_affinity ** power. Zero preserves the original behavior."
        ),
    )
    parser.add_argument("--cds-epsilon", type=float, default=1e-3)
    parser.add_argument("--cds-tolerance", type=float, default=1e-8)
    parser.add_argument("--cds-max-iterations", type=int, default=1000)
    parser.add_argument(
        "--cds-solver-mode",
        choices=(
            "self_loop",
            "classical_constraint",
            "fixed_query_mass",
            "dense_query_gated",
            "query_gated_replicator",
            "fixed_share_replicator",
            "sequential_cds_payoff_growth",
            "sequential_graph_growth",
            "topk_projected",
            "topk_swap",
            "topk_monotone_block",
            "query_gated_iht",
        ),
        default="self_loop",
        help=(
            "self_loop preserves the existing positive query self-loop solver; "
            "classical_constraint uses M=A-alpha*I_barS with a global payoff shift; "
            "fixed_query_mass clamps query influence and updates only messages; "
            "dense_query_gated keeps all message weights during query-gated "
            "graph fixed-point iteration and applies the evidence budget only "
            "after convergence; "
            "query_gated_replicator keeps the same query-gated graph utility "
            "but evolves all message weights with the CDS multiplicative "
            "replicator update before applying the final evidence budget; "
            "fixed_share_replicator adds a fixed-share mutation toward the "
            "initial query-relevance prior after each replicator update, "
            "preventing premature support collapse while retaining dense "
            "global competition; "
            "sequential_cds_payoff_growth keeps monotonic one-message-at-a-"
            "time support growth but scores every remaining candidate with "
            "the original CDS payoff induced by the current query-evidence "
            "subgraph; "
            "sequential_graph_growth anchors on the highest-relevance "
            "message and adds one query-gated graph complement per round "
            "until the evidence budget is reached; "
            "topk_projected applies query-gated graph utility and explicitly "
            "maintains retrieve_top_k active messages each round; topk_swap "
            "uses monotonic one-out/one-in coordinate ascent under the same "
            "fixed evidence budget; topk_monotone_block proposes a complete "
            "fixed-budget set and accepts it only when one fixed relevance "
            "plus pairwise-relation objective strictly improves; "
            "query_gated_iht optimizes a fixed query-gated quadratic "
            "objective with K-sparse simplex projection and Armijo "
            "backtracking."
        ),
    )
    parser.add_argument(
        "--cds-query-mass",
        type=float,
        default=0.75,
        help="Fixed query contribution used by fixed_query_mass mode.",
    )
    parser.add_argument(
        "--topk-graph-weight",
        type=float,
        default=0.25,
        help=(
            "Maximum multiplicative graph bonus in query-gated solver modes: "
            "utility_i = relevance_i * (1 + weight * normalized_graph_i)."
        ),
    )
    parser.add_argument(
        "--fixed-share-rate",
        type=float,
        default=0.05,
        help=(
            "Share rate for fixed_share_replicator. After each replicator "
            "update, this fraction of mass is restored from the initial "
            "query-relevance prior. Must be in [0, 1)."
        ),
    )
    parser.add_argument("--iht-initial-step", type=float, default=1.0)
    parser.add_argument(
        "--iht-backtracking-factor", type=float, default=0.5
    )
    parser.add_argument("--iht-armijo-sigma", type=float, default=1e-4)
    parser.add_argument("--iht-min-step", type=float, default=1e-8)
    parser.add_argument("--iht-max-backtracking", type=int, default=50)
    parser.add_argument(
        "--relation-graph",
        choices=(
            "complete",
            "native",
            "native_candidate_scope",
            "bm25_threshold",
        ),
        default="complete",
    )
    parser.add_argument("--relation-model", default="deepseek-v4-flash")
    parser.add_argument("--relation-thinking", choices=("enabled", "disabled"), default=None)
    parser.add_argument("--relation-max-tokens", type=int, default=2048)
    parser.add_argument("--relation-pair-batch-size", type=int, default=16)
    parser.add_argument("--native-channel-window", type=int, default=5)
    parser.add_argument("--native-scope-neighbors", type=int, default=1)
    parser.add_argument("--relation-verify", type=int, choices=(0, 1), default=1)
    parser.add_argument("--relation-threshold-iqr-multiplier", type=float, default=0.5)
    parser.add_argument("--relation-threshold-min-pairs", type=int, default=0)
    parser.add_argument("--relation-threshold-max-pairs", type=int, default=256)
    parser.add_argument(
        "--embedding-index-fields",
        choices=("content", "structured"),
        default="content",
        help=(
            "Message text encoded for embedding retrieval and dense "
            "message-message similarity."
        ),
    )
    parser.add_argument(
        "--pair-embedding-index-fields",
        choices=("content", "structured"),
        default=None,
        help=(
            "Message text encoded only for dense message-message similarity. "
            "Defaults to --embedding-index-fields for backward compatibility."
        ),
    )
    parser.add_argument("--embedding-store-dir", default="")
    parser.add_argument("--pair-embedding-store-dir", default="")
    parser.add_argument("--embedding-model-path", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument(
        "--embedding-cache-only", type=int, choices=(0, 1), default=1
    )
    parser.add_argument(
        "--embedding-gpu-ids",
        default="",
        help="Comma-separated GPU ids used only when an embedding cache is missing.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--embedding-dtype", default="bfloat16")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--ingest-only", action="store_true")
    args = parser.parse_args()
    embedding_gpu_ids = [
        value.strip()
        for value in args.embedding_gpu_ids.split(",")
        if value.strip()
    ]
    if any(not value.isdigit() for value in embedding_gpu_ids):
        parser.error("--embedding-gpu-ids must be comma-separated integers")

    if args.dynamic_coverage_weight > 0.0 and args.cds_solver_mode != "self_loop":
        parser.error(
            "--dynamic-coverage-weight > 0 currently requires "
            "--cds-solver-mode=self_loop"
        )

    if not 0.0 <= args.fixed_share_rate < 1.0:
        parser.error("--fixed-share-rate must be in [0, 1)")

    bm25f_weights = (
        args.bm25f_content_weight,
        args.bm25f_participant_weight,
        args.bm25f_episode_weight,
    )
    if any(weight < 0.0 for weight in bm25f_weights):
        parser.error("BM25F field weights must be non-negative")
    if not any(weight > 0.0 for weight in bm25f_weights):
        parser.error("At least one BM25F field weight must be positive")
    bm25f_b_values = (
        args.bm25f_content_b,
        args.bm25f_participant_b,
        args.bm25f_episode_b,
    )
    if any(not 0.0 <= value <= 1.0 for value in bm25f_b_values):
        parser.error("BM25F field b values must be in [0, 1]")
    if args.bm25f_k1 <= 0.0:
        parser.error("--bm25f-k1 must be positive")

    if args.native_bridge:
        if args.complementarity_weight <= 0:
            parser.error(
                "--native-bridge=1 requires --complementarity-weight > 0"
            )
        if args.complementarity_fields != "content":
            parser.error(
                "--native-bridge=1 requires --complementarity-fields=content"
            )

    load_env_file(args.env_file)
    messages = load_conversation_messages(args.conversation_json)
    questions = load_questions(args.questions_jsonl)
    print(f"[coalmem] loaded {len(messages)} messages and {len(questions)} questions")
    retriever = CDSRetriever(
        messages,
        conversation_path=args.conversation_json,
        mode=args.cds_mode,
        candidate_top_k=max(1, args.candidate_top_k),
        retrieve_top_k=max(1, args.retrieve_top_k),
        scope_filter_top_k=max(0, args.scope_filter_top_k),
        scope_score_top_n=max(1, args.scope_score_top_n),
        query_text_mode=args.query_text_mode,
        query_affinity_mode=args.query_affinity_mode,
        multiview_top_k=max(1, args.multiview_top_k),
        multiview_consensus_weight=max(0.0, args.multiview_consensus_weight),
        bm25f_content_weight=args.bm25f_content_weight,
        bm25f_participant_weight=args.bm25f_participant_weight,
        bm25f_episode_weight=args.bm25f_episode_weight,
        bm25f_content_b=args.bm25f_content_b,
        bm25f_participant_b=args.bm25f_participant_b,
        bm25f_episode_b=args.bm25f_episode_b,
        bm25f_k1=args.bm25f_k1,
        complementarity_weight=max(0.0, args.complementarity_weight),
        complementarity_fields=args.complementarity_fields,
        complementarity_normalization=args.complementarity_normalization,
        dynamic_coverage_weight=max(0.0, args.dynamic_coverage_weight),
        native_complementarity_weight=max(
            0.0, args.native_complementarity_weight
        ),
        native_surprise_weight=max(0.0, args.native_surprise_weight),
        native_bridge=bool(args.native_bridge),
        query_dynamic_edge=bool(args.query_dynamic_edge),
        dynamic_edge_floor=min(max(0.0, args.dynamic_edge_floor), 1.0),
        evidence_order=args.evidence_order,
        selection_relevance_power=max(0.0, args.selection_relevance_power),
        cds_epsilon=max(1e-12, args.cds_epsilon),
        cds_tolerance=max(1e-12, args.cds_tolerance),
        cds_max_iterations=max(1, args.cds_max_iterations),
        cds_solver_mode=args.cds_solver_mode,
        cds_query_mass=min(max(args.cds_query_mass, 1e-6), 1.0 - 1e-6),
        topk_graph_weight=max(0.0, args.topk_graph_weight),
        fixed_share_rate=args.fixed_share_rate,
        iht_initial_step=max(1e-12, args.iht_initial_step),
        iht_backtracking_factor=min(
            max(args.iht_backtracking_factor, 1e-6), 1.0 - 1e-6
        ),
        iht_armijo_sigma=max(0.0, args.iht_armijo_sigma),
        iht_min_step=max(1e-16, args.iht_min_step),
        iht_max_backtracking=max(0, args.iht_max_backtracking),
        candidate_retrieval_mode=args.candidate_retrieval_mode,
        embedding_index_fields=args.embedding_index_fields,
        embedding_store_dir=args.embedding_store_dir,
        pair_embedding_index_fields=(
            args.pair_embedding_index_fields or args.embedding_index_fields
        ),
        pair_embedding_store_dir=(
            args.pair_embedding_store_dir or args.embedding_store_dir
        ),
        embedding_model_path=args.embedding_model_path,
        embedding_cache_only=bool(args.embedding_cache_only),
        embedding_gpu_ids=embedding_gpu_ids,
        embedding_batch_size=max(1, args.embedding_batch_size),
        embedding_max_length=args.embedding_max_length,
        embedding_dtype=args.embedding_dtype,
        relation_graph=args.relation_graph,
        relation_model=args.relation_model,
        relation_thinking=args.relation_thinking,
        relation_max_tokens=max(128, args.relation_max_tokens),
        relation_pair_batch_size=max(1, args.relation_pair_batch_size),
        native_channel_window=max(0, args.native_channel_window),
        native_scope_neighbors=max(0, args.native_scope_neighbors),
        relation_verify=bool(args.relation_verify),
        relation_threshold_iqr_multiplier=max(0.0, args.relation_threshold_iqr_multiplier),
        relation_threshold_min_pairs=max(0, args.relation_threshold_min_pairs),
        relation_threshold_max_pairs=max(0, args.relation_threshold_max_pairs),
    )
    retriever.prepare_queries(questions)
    config = {
        "cds_mode": args.cds_mode,
        "candidate_retrieval_mode": args.candidate_retrieval_mode,
        "candidate_top_k": args.candidate_top_k,
        "retrieve_top_k": args.retrieve_top_k,
        "scope_filter_top_k": args.scope_filter_top_k,
        "scope_score_top_n": args.scope_score_top_n,
        "evidence_order": args.evidence_order,
        "query_affinity_mode": args.query_affinity_mode,
        "multiview_top_k": args.multiview_top_k,
        "multiview_consensus_weight": args.multiview_consensus_weight,
        "bm25f_field_weights": {
            "content": args.bm25f_content_weight,
            "participant": args.bm25f_participant_weight,
            "episode": args.bm25f_episode_weight,
        },
        "bm25f_field_b": {
            "content": args.bm25f_content_b,
            "participant": args.bm25f_participant_b,
            "episode": args.bm25f_episode_b,
        },
        "bm25f_k1": args.bm25f_k1,
        "complementarity_weight": args.complementarity_weight,
        "complementarity_fields": args.complementarity_fields,
        "complementarity_normalization": args.complementarity_normalization,
        "dynamic_coverage_weight": args.dynamic_coverage_weight,
        "dynamic_coverage_variant": (
            "idf_weighted_residual_query_coverage"
            if args.dynamic_coverage_weight > 0.0
            else "none"
        ),
        "complementarity_variant": (
            "pairwise_bm25_querycov_" + args.complementarity_normalization
            if args.complementarity_weight > 0
            else "none"
        ),
        "native_complementarity_weight": args.native_complementarity_weight,
        "native_surprise_weight": args.native_surprise_weight,
        "native_surprise_variant": (
            "query_local_self_information_native_motifs"
            if args.native_surprise_weight > 0 else "none"
        ),
        "native_bridge": bool(args.native_bridge),
        "query_dynamic_edge": bool(args.query_dynamic_edge),
        "dynamic_edge_floor": args.dynamic_edge_floor,
        "query_dynamic_edge_variant": (
            "query_conditioned_soft_relation_gate"
            if args.query_dynamic_edge
            else "none"
        ),
        "native_bridge_variant": (
            "native_grounded_reciprocal_retrieval_gain"
            if args.native_bridge
            else "none"
        ),
        "native_complementarity_variant": (
            "querycov_conditioned_native_structure"
            if args.native_complementarity_weight > 0
            else "none"
        ),
        "native_scope_neighbors_for_complementarity": (
            args.native_scope_neighbors
            if args.native_complementarity_weight > 0
            else None
        ),
        "candidate_retrieval": (
            args.embedding_index_fields
            + "_embedding_cosine_"
            + args.query_text_mode
            if args.candidate_retrieval_mode == "embedding"
            else "structured_expandc_bm25_" + args.query_text_mode
        ),
        "topk_graph_weight": args.topk_graph_weight,
        "fixed_share_rate": args.fixed_share_rate,
        "iht_initial_step": args.iht_initial_step,
        "iht_backtracking_factor": args.iht_backtracking_factor,
        "iht_armijo_sigma": args.iht_armijo_sigma,
        "iht_min_step": args.iht_min_step,
        "iht_max_backtracking": args.iht_max_backtracking,
        "embedding_fields": (
            args.embedding_index_fields + "/" + (
                "question_only"
                if args.query_text_mode == "question_only"
                else (
                    "structured"
                    if args.candidate_retrieval_mode == "embedding"
                    else "standard"
                )
            )
        ),
        "embedding_cache_only": bool(args.embedding_cache_only),
        "embedding_gpu_ids": embedding_gpu_ids or ["cpu"],
        "embedding_batch_size": args.embedding_batch_size,
        "pair_embedding_index_fields": (
            (args.pair_embedding_index_fields or args.embedding_index_fields)
            if args.cds_mode in {"hybrid", "dense"}
            else None
        ),
        "pair_embedding_store_dir": (
            (args.pair_embedding_store_dir or args.embedding_store_dir or None)
            if args.cds_mode in {"hybrid", "dense"}
            else None
        ),
        "embedding_store_dir": args.embedding_store_dir or None,
        "relation_graph": args.relation_graph,
        "relation_model": args.relation_model if args.relation_graph in {"native", "bm25_threshold"} else None,
        "relation_pair_batch_size": args.relation_pair_batch_size if args.relation_graph in {"native", "bm25_threshold"} else None,
        "native_channel_window": args.native_channel_window if args.relation_graph == "native" else None,
        "native_scope_neighbors": args.native_scope_neighbors if args.relation_graph in {"native", "native_candidate_scope"} else None,
        "relation_verify": bool(args.relation_verify) if args.relation_graph in {"native", "bm25_threshold"} else None,
        "relation_threshold_iqr_multiplier": args.relation_threshold_iqr_multiplier if args.relation_graph == "bm25_threshold" else None,
        "relation_threshold_min_pairs": args.relation_threshold_min_pairs if args.relation_graph == "bm25_threshold" else None,
        "relation_threshold_max_pairs": args.relation_threshold_max_pairs if args.relation_graph == "bm25_threshold" else None,
    }
    print("[coalmem] config: " + json.dumps(config, sort_keys=True))
    if args.ingest_only:
        print("[coalmem] ingest-only: indexes/caches validated")
        return 0

    provider = normalize_llm_provider(args.llm_provider)
    base_url = resolve_base_url(provider)
    client = create_chat_client(
        provider=provider,
        azure_endpoint=base_url,
        base_url=base_url,
        api_version=API_VERSION_DEFAULT,
        api_key=resolve_api_key(provider),
    )
    retriever.relation_client = client
    correct, total = run_qa(
        questions=questions,
        retriever=retriever,
        client=client,
        args=args,
        agent_system=read_text(args.agent_prompt),
        judge_system=read_text(args.judge_prompt),
    )
    print(f"Accuracy: {correct / total if total else 0.0:.4f} ({correct}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
