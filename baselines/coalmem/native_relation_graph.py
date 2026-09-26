"""Sparse edge proposal and LLM verification for CDS.

Native metadata or an adaptive BM25 neighborhood proposes sparse message pairs.
The model only verifies whether each pair has a direct evidence-bearing link;
it never ranks or drops messages.
"""
from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from baselines.common import call_chat

RELATION_BASES = {"DEPENDENCY", "STATE", "TERMINOLOGY", "NONE"}
CLASSIFIER_SYSTEM = """You verify direct evidence-bearing links between pairs of group-chat messages.
You do not rank, select, summarize, or answer the user question.

A LINK must have one of exactly three grounded bases:
- DEPENDENCY: the messages must be jointly read because one directly replies to,
  references, explains, causes, or supplies a necessary missing part of the other.
- STATE: the messages concern the same concrete object/decision and express
  equivalent, complementary, competing, or temporally changing states.
- TERMINOLOGY: different expressions in the two messages refer to the same
  concrete concept or practice.

Broad topical similarity, shared author/role/channel/phase/topic, chronological
proximity, or merely mentioning related projects is not a LINK.  When uncertain,
return link=false.  For every accepted link, quote a short exact content span
from both endpoints.  Return JSON only and one decision for every supplied pair."""

VERIFIER_SYSTEM = """You conservatively verify proposed links between pairs of group-chat messages.
Accept only when the quoted original messages directly support the proposed
DEPENDENCY, STATE, or TERMINOLOGY basis.  Shared metadata or broad topical
similarity is insufficient.  Do not rank messages and do not answer any user
question.  Return JSON only and one verdict for every supplied pair."""


def _content(message: Dict[str, Any]) -> str:
    value = message.get("content")
    return html.unescape(value).strip() if isinstance(value, str) else ""


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _grounded(span: Any, message: Dict[str, Any]) -> bool:
    needle = _norm(str(span or ""))
    return len(needle) >= 3 and needle in _norm(_content(message))


def _json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _message_text(label: str, message: Dict[str, Any]) -> str:
    return (
        f"{label}\n"
        f"msg_id: {message.get('msg_node') or ''}\n"
        f"author: {message.get('author') or ''}\n"
        f"role: {message.get('role') or ''}\n"
        f"timestamp: {message.get('timestamp') or ''}\n"
        f"channel: {message.get('_channel') or ''}\n"
        f"phase: {message.get('phase_name') or ''}\n"
        f"topic: {message.get('topic') or ''}\n"
        f"reply_to: {message.get('reply_to') or ''}\n"
        f"content: {_content(message)}"
    )


class NativeRelationMaskBuilder:
    def __init__(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        model: str,
        thinking: str | None,
        max_tokens: int,
        pair_batch_size: int,
        channel_window: int,
        scope_neighbors: int,
        verify: bool,
        model_free_candidate_scope: bool = False,
    ) -> None:
        self.messages = messages
        self.model = model
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.pair_batch_size = pair_batch_size
        self.channel_window = channel_window
        self.scope_neighbors = scope_neighbors
        self.verify = verify
        self.model_free_candidate_scope = model_free_candidate_scope
        self.id_to_index = {
            str(message.get("msg_node") or ""): index
            for index, message in enumerate(messages)
            if message.get("msg_node")
        }
        self.channel_positions: Dict[int, int] = {}
        by_channel: Dict[str, List[int]] = defaultdict(list)
        for index, message in enumerate(messages):
            by_channel[str(message.get("_channel") or "")].append(index)
        for indices in by_channel.values():
            for position, index in enumerate(indices):
                self.channel_positions[index] = position

    @staticmethod
    def _pair(i: int, j: int) -> Tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def propose(
        self,
        candidate_indices: np.ndarray,
        candidate_similarity: np.ndarray | None = None,
    ) -> Tuple[
        List[Tuple[int, int]],
        Dict[Tuple[int, int], List[str]],
        Dict[str, Any],
    ]:
        count = len(candidate_indices)
        local_by_global = {int(global_index): local for local, global_index in enumerate(candidate_indices)}
        reasons: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

        # Direct reply links are the strongest native proposal.
        for local, global_index in enumerate(candidate_indices):
            reply_id = str(self.messages[int(global_index)].get("reply_to") or "")
            target_global = self.id_to_index.get(reply_id)
            target_local = local_by_global.get(target_global) if target_global is not None else None
            if target_local is not None and target_local != local:
                reasons[self._pair(local, target_local)].add("reply")

        # Query-conditioned temporal neighborhood over the retrieved
        # subsequence. Exact-scope candidates are ordered by their original
        # timestamp, then connected to the next k retrieved candidates.
        if self.model_free_candidate_scope:
            groups: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
            for local, global_index in enumerate(candidate_indices):
                message = self.messages[int(global_index)]
                scope = (
                    str(message.get("_channel") or ""),
                    str(message.get("phase_name") or ""),
                    str(message.get("topic") or ""),
                )
                if all(scope):
                    groups[scope].append(local)
            for locals_ in groups.values():
                locals_.sort(key=lambda local: (
                    str(self.messages[int(candidate_indices[local])].get("timestamp") or ""),
                    int(candidate_indices[local]),
                ))
                for position, local in enumerate(locals_):
                    stop = min(len(locals_), position + self.scope_neighbors + 1)
                    for neighbor_position in range(position + 1, stop):
                        reasons[self._pair(local, locals_[neighbor_position])].add(
                            "candidate_scope_window"
                        )
            pairs = sorted(reasons)
            return pairs, {pair: sorted(values) for pair, values in reasons.items()}, {
                "relation_variant": "model_free_candidate_scope_window",
                "relation_candidate_scope_window": self.scope_neighbors,
                "relation_exact_scope_group_count": len(groups),
            }

        # Analogue of page/chunk proximity: actual positions in the full channel,
        # never BM25-rank adjacency.
        for left in range(count):
            gi = int(candidate_indices[left]); mi = self.messages[gi]
            channel = str(mi.get("_channel") or "")
            pi = self.channel_positions.get(gi)
            if pi is None:
                continue
            for right in range(left + 1, count):
                gj = int(candidate_indices[right]); mj = self.messages[gj]
                if channel != str(mj.get("_channel") or ""):
                    continue
                pj = self.channel_positions.get(gj)
                if pj is not None and abs(pi - pj) <= self.channel_window:
                    reasons[(left, right)].add("channel_local")

        # Scope fields only propose the nearest preceding/following candidates;
        # matching metadata by itself never becomes an edge.
        groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for local, global_index in enumerate(candidate_indices):
            message = self.messages[int(global_index)]
            scope = (str(message.get("phase_name") or ""), str(message.get("topic") or ""))
            if any(scope):
                groups[scope].append(local)
        for locals_ in groups.values():
            locals_.sort(key=lambda local: (str(self.messages[int(candidate_indices[local])].get("timestamp") or ""), local))
            for position, local in enumerate(locals_):
                for offset in range(1, self.scope_neighbors + 1):
                    if position - offset >= 0:
                        reasons[self._pair(local, locals_[position - offset])].add("scope_local")
                    if position + offset < len(locals_):
                        reasons[self._pair(local, locals_[position + offset])].add("scope_local")

        pairs = sorted(reasons)
        return pairs, {pair: sorted(values) for pair, values in reasons.items()}, {}

    def _classify_batch(self, client: Any, candidate_indices: np.ndarray, batch: Sequence[Tuple[int, int]], offset: int) -> Tuple[List[Dict[str, Any]], str]:
        unique = sorted({local for pair in batch for local in pair})
        messages_text = "\n\n".join(
            _message_text(f"C{local + 1}", self.messages[int(candidate_indices[local])])
            for local in unique
        )
        pair_lines = "\n".join(
            f"P{offset + index + 1}: C{left + 1} <-> C{right + 1}"
            for index, (left, right) in enumerate(batch)
        )
        prompt = f"""MESSAGES:\n{messages_text}\n\nPAIRS TO CLASSIFY:\n{pair_lines}\n\nReturn:\n{{\n  \"decisions\": [\n    {{\"pair\": \"P<number>\", \"link\": true, \"basis\": \"DEPENDENCY|STATE|TERMINOLOGY|NONE\", \"span_a\": \"exact span from first endpoint\", \"span_b\": \"exact span from second endpoint\", \"reason\": \"brief grounded reason\"}}\n  ]\n}}"""
        raw = call_chat(client, self.model, CLASSIFIER_SYSTEM, prompt, max_tokens=self.max_tokens, thinking_mode=self.thinking)
        payload = _json_object(raw)
        parsed: Dict[str, Dict[str, Any]] = {}
        for item in payload.get("decisions") or []:
            if isinstance(item, dict):
                parsed[str(item.get("pair") or "").upper()] = item
        decisions = []
        for index, (left, right) in enumerate(batch):
            pair_id = f"P{offset + index + 1}"
            item = parsed.get(pair_id, {})
            basis = str(item.get("basis") or "NONE").upper()
            link = bool(item.get("link")) and basis in RELATION_BASES - {"NONE"}
            span_a = str(item.get("span_a") or "")
            span_b = str(item.get("span_b") or "")
            link = link and _grounded(span_a, self.messages[int(candidate_indices[left])]) and _grounded(span_b, self.messages[int(candidate_indices[right])])
            decisions.append({
                "pair": pair_id, "left": left, "right": right,
                "link": bool(link), "basis": basis if link else "NONE",
                "span_a": span_a if link else "", "span_b": span_b if link else "",
                "reason": str(item.get("reason") or "")[:500],
            })
        return decisions, raw

    def _verify_batch(self, client: Any, candidate_indices: np.ndarray, decisions: Sequence[Dict[str, Any]]) -> Tuple[Set[str], str]:
        unique = sorted({int(item[key]) for item in decisions for key in ("left", "right")})
        messages_text = "\n\n".join(
            _message_text(f"C{local + 1}", self.messages[int(candidate_indices[local])])
            for local in unique
        )
        proposals = "\n".join(
            f"{item['pair']}: C{item['left'] + 1} <-> C{item['right'] + 1}; basis={item['basis']}; span_a={json.dumps(item['span_a'])}; span_b={json.dumps(item['span_b'])}"
            for item in decisions
        )
        prompt = f"""MESSAGES:\n{messages_text}\n\nPROPOSED LINKS:\n{proposals}\n\nReturn:\n{{\"verdicts\": [{{\"pair\": \"P<number>\", \"verdict\": \"SUPPORT|REJECT\", \"reason\": \"brief reason\"}}]}}"""
        raw = call_chat(client, self.model, VERIFIER_SYSTEM, prompt, max_tokens=self.max_tokens, thinking_mode=self.thinking)
        payload = _json_object(raw)
        supported = {
            str(item.get("pair") or "").upper()
            for item in payload.get("verdicts") or []
            if isinstance(item, dict) and str(item.get("verdict") or "").upper() == "SUPPORT"
        }
        return supported, raw

    def build(
        self,
        client: Any,
        candidate_indices: np.ndarray,
        candidate_similarity: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        pairs, reasons, proposal_diagnostics = self.propose(
            candidate_indices, candidate_similarity
        )
        if self.model_free_candidate_scope:
            count = len(candidate_indices)
            mask = np.zeros((count, count), dtype=np.float64)
            for left, right in pairs:
                mask[left, right] = mask[right, left] = 1.0
            degrees = np.count_nonzero(mask, axis=1) if count else np.zeros(0)
            possible_pairs = count * (count - 1) // 2
            return mask, {
                **proposal_diagnostics,
                "relation_candidate_pair_count": len(pairs),
                "relation_classifier_link_count": None,
                "relation_verified_edge_count": len(pairs),
                "relation_isolated_node_count": int(np.count_nonzero(degrees == 0)),
                "relation_edge_density": (
                    len(pairs) / possible_pairs if possible_pairs else 0.0
                ),
                "relation_candidate_pairs": [
                    {
                        "left": left,
                        "right": right,
                        "left_msg_id": self.messages[
                            int(candidate_indices[left])
                        ].get("msg_node"),
                        "right_msg_id": self.messages[
                            int(candidate_indices[right])
                        ].get("msg_node"),
                        "proposal_reasons": reasons[(left, right)],
                    }
                    for left, right in pairs
                ],
                "relation_decisions": [],
                "relation_edges": [],
                "relation_classifier_raw_outputs": [],
                "relation_verifier_raw_outputs": [],
                "additional_model_calls": 0,
            }
        decisions: List[Dict[str, Any]] = []
        classifier_raw: List[str] = []
        for offset in range(0, len(pairs), self.pair_batch_size):
            batch = pairs[offset : offset + self.pair_batch_size]
            parsed, raw = self._classify_batch(client, candidate_indices, batch, offset)
            decisions.extend(parsed); classifier_raw.append(raw)

        linked = [item for item in decisions if item["link"]]
        verifier_raw: List[str] = []
        supported: Set[str] = {item["pair"] for item in linked}
        if self.verify and linked:
            supported = set()
            for offset in range(0, len(linked), self.pair_batch_size):
                batch = linked[offset : offset + self.pair_batch_size]
                accepted, raw = self._verify_batch(client, candidate_indices, batch)
                supported.update(accepted); verifier_raw.append(raw)

        mask = np.zeros((len(candidate_indices), len(candidate_indices)), dtype=np.float64)
        accepted_edges = []
        for item in linked:
            if item["pair"] not in supported:
                continue
            left, right = int(item["left"]), int(item["right"])
            mask[left, right] = mask[right, left] = 1.0
            accepted_edges.append({
                **item,
                "left_msg_id": self.messages[int(candidate_indices[left])].get("msg_node"),
                "right_msg_id": self.messages[int(candidate_indices[right])].get("msg_node"),
                "proposal_reasons": reasons.get((left, right), []),
            })
        degrees = mask.sum(axis=1)
        diagnostics = {
            **proposal_diagnostics,
            "relation_candidate_pair_count": len(pairs),
            "relation_classifier_link_count": len(linked),
            "relation_verified_edge_count": len(accepted_edges),
            "relation_isolated_node_count": int(np.sum(degrees == 0)),
            "relation_edge_density": float(len(accepted_edges) / max(1, len(candidate_indices) * (len(candidate_indices) - 1) / 2)),
            "relation_basis_counts": {
                basis: sum(edge["basis"] == basis for edge in accepted_edges)
                for basis in sorted(RELATION_BASES - {"NONE"})
            },
            "relation_candidate_pairs": [
                {
                    "left": left, "right": right,
                    "left_msg_id": self.messages[int(candidate_indices[left])].get("msg_node"),
                    "right_msg_id": self.messages[int(candidate_indices[right])].get("msg_node"),
                    "proposal_reasons": reasons[(left, right)],
                }
                for left, right in pairs
            ],
            "relation_decisions": decisions,
            "relation_edges": accepted_edges,
            "relation_classifier_raw_outputs": classifier_raw,
            "relation_verifier_raw_outputs": verifier_raw,
        }
        return mask, diagnostics



class BM25ThresholdRelationMaskBuilder(NativeRelationMaskBuilder):
    """Propose an adaptive upper-tail BM25 neighborhood for verification."""

    def __init__(
        self,
        *args: Any,
        threshold_iqr_multiplier: float,
        min_candidate_pairs: int,
        max_candidate_pairs: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.threshold_iqr_multiplier = threshold_iqr_multiplier
        self.min_candidate_pairs = min_candidate_pairs
        self.max_candidate_pairs = max_candidate_pairs

    def propose(
        self,
        candidate_indices: np.ndarray,
        candidate_similarity: np.ndarray | None = None,
    ) -> Tuple[
        List[Tuple[int, int]],
        Dict[Tuple[int, int], List[str]],
        Dict[str, Any],
    ]:
        if candidate_similarity is None:
            raise ValueError("BM25 threshold relation graph requires pair similarity")
        similarity = np.asarray(candidate_similarity, dtype=np.float64)
        count = len(candidate_indices)
        if similarity.shape != (count, count):
            raise ValueError(
                "Candidate similarity shape mismatch: "
                f"expected {(count, count)}, got {similarity.shape}"
            )

        scored = [
            (float(similarity[left, right]), left, right)
            for left in range(count)
            for right in range(left + 1, count)
        ]
        if not scored:
            return [], {}, {
                "relation_threshold_strategy": "adaptive_iqr",
                "relation_similarity_threshold": None,
                "relation_threshold_raw_pair_count": 0,
            }

        values = np.asarray([item[0] for item in scored], dtype=np.float64)
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = float(q3 - q1)
        threshold = float(q3 + self.threshold_iqr_multiplier * iqr)
        ranked = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
        raw = [item for item in ranked if item[0] >= threshold]

        total_pairs = len(ranked)
        minimum = min(max(0, self.min_candidate_pairs), total_pairs)
        maximum = (
            total_pairs
            if self.max_candidate_pairs <= 0
            else min(max(minimum, self.max_candidate_pairs), total_pairs)
        )
        selected = ranked[:minimum] if len(raw) < minimum else raw[:maximum]

        pairs = [(left, right) for _, left, right in selected]
        reasons = {
            (left, right): [
                "adaptive_bm25_threshold",
                f"bm25_similarity={score:.8f}",
            ]
            for score, left, right in selected
        }
        diagnostics = {
            "relation_threshold_strategy": "adaptive_iqr",
            "relation_threshold_q1": float(q1),
            "relation_threshold_q3": float(q3),
            "relation_threshold_iqr": iqr,
            "relation_threshold_iqr_multiplier": self.threshold_iqr_multiplier,
            "relation_similarity_threshold": threshold,
            "relation_threshold_raw_pair_count": len(raw),
            "relation_threshold_min_candidate_pairs": minimum,
            "relation_threshold_max_candidate_pairs": maximum,
            "relation_threshold_budget_adjusted": len(selected) != len(raw),
            "relation_candidate_similarity_min": float(selected[-1][0]) if selected else None,
            "relation_candidate_similarity_max": float(selected[0][0]) if selected else None,
        }
        return pairs, reasons, diagnostics
