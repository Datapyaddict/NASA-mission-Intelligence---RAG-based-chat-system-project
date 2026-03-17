"""greedy_set_cover_dedup.py

Greedy document selection to maximise covered accounting rules with as few documents as possible,
with hash-based deduplication of identical rule-vectors.

Why hashing helps
-----------------
If two documents have identical binary vectors, they cover exactly the same accounting rules.
For coverage optimisation, those documents are interchangeable *unless* you have extra constraints
(e.g. pick at least one doc per legal entity) or preferences (trusted source, preferred portfolio, etc.).

So we:
  1) Hash each vector (or sparse rule-index set) into a stable digest.
  2) Group documents by hash.
  3) Keep the best representative per hash-group (or top-N) using a user-defined preference score.
  4) Run greedy selection on the deduplicated candidates.

This gives you fewer candidates to score at each greedy step (often much faster) without changing
coverage results, assuming you only keep one representative per identical vector.

Example
-------
>>> docs = [
...   ContractDoc("A", {"source":"S1"}, {0,1,2}),
...   ContractDoc("B", {"source":"S2"}, {0,1,2}),  # identical rules => same hash
...   ContractDoc("C", {"source":"S1"}, {3,4}),
... ]
>>>
>>> # Prefer source S1 over S2 among identical vectors
>>> def pref(doc):
...     return 10.0 if doc.metadata.get("source") == "S1" else 1.0
>>>
>>> deduped, groups = deduplicate_by_vector_hash(docs, d=5, preference_score=pref)
>>> [d.doc_id for d in deduped]
['A', 'C']

Notes
-----
- The greedy core uses *marginal gain*: new_rules = doc.rules ∩ uncovered.
- Metadata can be injected as a multiplicative weight or as a preference when choosing a representative.

"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class ContractDoc:
    """A single contract document with metadata and a set of covered rule indices."""

    doc_id: str
    metadata: Dict[str, Any]
    rules: Set[int]  # indices where the vector has 1


# -------------------------
# Rule encoding utilities
# -------------------------

def build_rule_index(rule_labels: List[str]) -> Dict[str, int]:
    """Build a stable mapping from rule label -> column index.

    You said you have an *ordered* master list of all possible rule labels.
    That ordering defines the vector coordinates.

    Unit breakdown:
      - A rule label is a string identifier, e.g. "IFRS9_ECL_STAGE1".
      - The index is an integer position in [0, d-1].
      - d = len(rule_labels) is the shared vector dimensionality for all documents.

    Args:
        rule_labels: ordered list of all rule labels.

    Returns:
        dict mapping each label to its index.

    Example:
        >>> labels = ["R1", "R2", "R3"]
        >>> build_rule_index(labels)
        {'R1': 0, 'R2': 1, 'R3': 2}
    """

    return {label: i for i, label in enumerate(rule_labels)}


def ruleset_from_triggered_labels(
    triggered_labels: Iterable[str],
    rule_index: Dict[str, int],
    *,
    strict: bool = True,
) -> Set[int]:
    """Convert a document's triggered rule labels into a sparse set of rule indices.

    This is usually the best internal representation for set-cover: you only store the 1s.

    Args:
        triggered_labels: the rule labels triggered by the document.
        rule_index: mapping from label -> index (from build_rule_index).
        strict:
            - If True, raise KeyError when a triggered label is unknown (not in the master list).
            - If False, silently ignore unknown labels.

    Returns:
        Set of integer indices where the binary vector would have value 1.

    Example:
        >>> idx = build_rule_index(["R1","R2","R3"])
        >>> ruleset_from_triggered_labels(["R2","R3"], idx)
        {1, 2}
    """

    out: Set[int] = set()
    for label in triggered_labels:
        if label in rule_index:
            out.add(rule_index[label])
        elif strict:
            raise KeyError(f"Unknown rule label: {label!r}")
    return out


def dense_vector_from_ruleset(rules: Set[int], d: int) -> List[int]:
    """Convert a sparse rule-index set into a dense 0/1 vector.

    You only need this if you must export a full vector (e.g. for storage or matrix ops).
    The greedy algorithm here works directly with the sparse set.

    Args:
        rules: indices where v[j] = 1.
        d: vector dimension (number of total possible rules).

    Returns:
        A list of length d containing 0/1 integers.

    Example:
        >>> dense_vector_from_ruleset({1,3}, d=5)
        [0, 1, 0, 1, 0]
    """

    v = [0] * d
    for j in rules:
        if j < 0 or j >= d:
            raise ValueError(f"Rule index {j} is out of bounds for d={d}")
        v[j] = 1
    return v


# -------------------------
# Loading utilities
# -------------------------

def load_contract_docs_from_json(folder: str | Path) -> List[ContractDoc]:
    """Load raw documents from a folder of JSON files.

    Expected JSON shape (example):
        {
          "metadata": {"portfolio": "P1", "entity": "E2", "source": "systemA"},
          "triggered_rules": ["R2", "R7", "R42"]
        }

    This function loads metadata + triggered rule labels, but it does *not* yet convert labels into
    vector indices (because it doesn't know your master ordered rule list).

    Use `load_contract_docs_from_json_with_rule_labels(...)` for the fully encoded form.

    Returns:
        A list of ContractDoc where:
          - doc.metadata contains an extra key "_triggered_rules" with the raw labels.
          - doc.rules is empty.
    """

    folder = Path(folder)
    docs: List[ContractDoc] = []

    for fp in folder.rglob("*.json"):
        with fp.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        metadata = dict(obj.get("metadata", {}))
        triggered = obj.get("triggered_rules", [])

        metadata["_triggered_rules"] = triggered
        docs.append(ContractDoc(doc_id=str(fp), metadata=metadata, rules=set()))

    return docs


def load_contract_docs_from_json_with_rule_labels(
    folder: str | Path,
    rule_labels: List[str],
    *,
    strict: bool = True,
) -> List[ContractDoc]:
    """Load documents and convert triggered rule labels into rule-index sets.

    This is the function you want if:
      - All docs share the same vector dimensionality d.
      - You have a master ordered list of all possible rule labels.
      - Each doc stores an ordered/unordered list of triggered rule labels.

    The resulting ContractDoc.rules is a sparse set of indices:
        rules = {j | the doc triggers rule_labels[j]}

    Unit breakdown:
      - rule_labels: defines the coordinate system (length d).
      - rule_index: dict(label -> j).
      - triggered_rules: list of labels for one doc.
      - ruleset_from_triggered_labels(...) maps labels -> indices.

    Args:
        folder: folder containing JSON documents.
        rule_labels: master ordered list of all possible rules.
        strict: raise if a doc contains an unknown rule label.

    Returns:
        List[ContractDoc] ready for deduplication and greedy selection.

    Example:
        >>> master = ["R1","R2","R3","R4"]
        >>> docs = load_contract_docs_from_json_with_rule_labels("/data/contracts", master)
        >>> d = len(master)
        >>> # Optional: export dense vectors
        >>> dense_vector_from_ruleset(docs[0].rules, d)
    """

    rule_index = build_rule_index(rule_labels)
    raw_docs = load_contract_docs_from_json(folder)

    encoded: List[ContractDoc] = []
    for doc in raw_docs:
        triggered = doc.metadata.get("_triggered_rules", [])
        rules = ruleset_from_triggered_labels(triggered, rule_index, strict=strict)

        # Remove the temporary field to keep metadata clean
        md = dict(doc.metadata)
        md.pop("_triggered_rules", None)

        encoded.append(ContractDoc(doc_id=doc.doc_id, metadata=md, rules=rules))

    return encoded


# -------------------------
# Hashing & deduplication
# -------------------------

def vector_hash_from_rules(rules_set: Set[int], d: int) -> str:
    """Return a stable hash for a binary vector represented as a set of 1-indices.

    We hash a canonical serialisation of the vector:
        f"{d}|{i1,i2,...}"  where indices are sorted.

    Why include `d`?
        Two vectors with the same 1-indices but different dimensionality should be treated as different
        (in practice, this avoids accidental collisions across datasets).

    Hash choice:
        BLAKE2 is fast and stable. `digest_size=16` gives a 128-bit digest.

    Args:
        rules_set: indices j where v_j = 1.
        d: dimensionality of the original vector.

    Returns:
        Hex digest string.

    Example:
        >>> vector_hash_from_rules({0,2,10}, d=12)
        '...'
    """

    payload = f"{d}|{','.join(map(str, sorted(rules_set)))}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def deduplicate_by_vector_hash(
    docs: List[ContractDoc],
    d: int,
    preference_score: Optional[Callable[[ContractDoc], float]] = None,
    keep_top_n_per_hash: int = 1,
) -> Tuple[List[ContractDoc], Dict[str, List[ContractDoc]]]:
    """Deduplicate documents by identical rule coverage using hashing.

    This groups documents by their vector hash (i.e. identical rule sets).

    If keep_top_n_per_hash == 1:
        Keep a single representative per hash-group.
        If `preference_score` is provided, pick the representative with the highest score.

    If keep_top_n_per_hash > 1:
        Keep the top-N docs per hash-group ranked by preference_score (or arbitrary if None).
        This is useful if you anticipate later constraints (e.g. must pick different entities)
        and want alternatives available.

    Args:
        docs: list of documents.
        d: vector dimensionality.
        preference_score: higher is better; used to choose representative(s) within a hash-group.
        keep_top_n_per_hash: number of representatives to keep per hash.

    Returns:
        deduped_docs: list of kept documents.
        groups: mapping hash -> all docs in that group (full groups, not truncated).

    Example:
        >>> def pref(doc):
        ...     return 10.0 if doc.metadata.get('source') == 'S1' else 1.0
        >>> deduped, groups = deduplicate_by_vector_hash(docs, d=100, preference_score=pref)
    """

    if keep_top_n_per_hash < 1:
        raise ValueError("keep_top_n_per_hash must be >= 1")

    if preference_score is None:
        # Neutral score if you don't care which representative is chosen.
        def preference_score(_: ContractDoc) -> float:  # type: ignore[misc]
            return 0.0

    groups: Dict[str, List[ContractDoc]] = {}

    for doc in docs:
        h = vector_hash_from_rules(doc.rules, d=d)
        groups.setdefault(h, []).append(doc)

    deduped: List[ContractDoc] = []

    for h, bucket in groups.items():
        # Sort descending by preference score
        bucket_sorted = sorted(bucket, key=lambda x: float(preference_score(x)), reverse=True)
        deduped.extend(bucket_sorted[:keep_top_n_per_hash])

    return deduped, groups


# -------------------------
# Coverage reporting
# -------------------------

def compute_rule_coverage(
    docs: List[ContractDoc],
    *,
    rule_labels: Optional[List[str]] = None,
    d: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute overall rule coverage of a corpus of docs.

    This answers: "Across all docs, what fraction of the rule-label matrix is ever triggered?"

    You typically run this *before* dedup/greedy search to confirm that your input files can
    plausibly reach the desired coverage target.

    Requirements:
      - `docs` must already be encoded, i.e. `ContractDoc.rules` contains rule *indices*.
      - You must provide either `rule_labels` (preferred) or `d` so we know the full universe.

    Args:
        docs: encoded documents.
        rule_labels: master ordered rule labels (defines the full universe and enables reporting
            uncovered label names).
        d: dimensionality (used if rule_labels is not provided).

    Returns:
        A report dict with:
          - coverage: covered/total as a float in [0,1]
          - covered_rules / uncovered_rules: sets of indices
          - covered_rule_count / uncovered_rule_count / total_rules
          - rule_trigger_counts: list[int] of length d (how many docs triggered each rule)
          - uncovered_rule_labels: list[str] (only if rule_labels provided)
    """

    if rule_labels is not None:
        d_inferred = len(rule_labels)
        if d is not None and d != d_inferred:
            raise ValueError(f"d ({d}) does not match len(rule_labels) ({d_inferred})")
        d = d_inferred

    if d is None:
        raise ValueError("Provide rule_labels or d to compute coverage against the full rule universe")
    if d < 0:
        raise ValueError("d must be >= 0")

    universe: Set[int] = set(range(d))
    covered: Set[int] = set()
    rule_trigger_counts: List[int] = [0] * d

    for doc in docs:
        for j in doc.rules:
            if j < 0 or j >= d:
                raise ValueError(
                    f"Doc {doc.doc_id!r} contains rule index {j} which is out of bounds for d={d}"
                )
            rule_trigger_counts[j] += 1
        covered |= doc.rules

    uncovered = universe - covered
    total_rules = len(universe)
    covered_count = len(covered)
    coverage = (covered_count / total_rules) if total_rules > 0 else 1.0

    report: Dict[str, Any] = {
        "coverage": coverage,
        "covered_rules": covered,
        "uncovered_rules": uncovered,
        "covered_rule_count": covered_count,
        "uncovered_rule_count": len(uncovered),
        "total_rules": total_rules,
        "num_docs": len(docs),
        "rule_trigger_counts": rule_trigger_counts,
    }

    if rule_labels is not None:
        report["uncovered_rule_labels"] = [rule_labels[i] for i in sorted(uncovered)]

    return report


def compute_rule_coverage_from_folder(
    folder: str | Path,
    rule_labels: List[str],
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper: load JSON docs, encode them, then compute coverage.

    Args:
        folder: folder containing JSON docs with `triggered_rules`.
        rule_labels: master ordered rule labels.
        strict: whether to raise on unknown triggered rule labels.

    Returns:
        Coverage report as per `compute_rule_coverage`.
    """

    docs = load_contract_docs_from_json_with_rule_labels(folder, rule_labels, strict=strict)
    return compute_rule_coverage(docs, rule_labels=rule_labels)


# -------------------------
# Greedy selection
# -------------------------

def default_metadata_weight(_: ContractDoc) -> float:
    """Default metadata weight: no bias."""

    return 1.0


def make_triggered_event_count_weight(
    *,
    event_count_key: str = "triggered_event_count",
    alpha: float = 0.25,
    cap: float = 2.0,
) -> Callable[[ContractDoc], float]:
    """Create a `metadata_weight` function based on a triggered-event count in metadata.

    This is a safe default that:
      - grows sublinearly with event count via log1p
      - is capped to avoid overwhelming the marginal-coverage signal

    Weight formula:
        w = min(cap, 1 + alpha * log(1 + n))

    Args:
        event_count_key: metadata field name containing the count.
        alpha: strength of the boost.
        cap: maximum multiplier.

    Returns:
        A function suitable to pass as `metadata_weight` to `greedy_select_docs`.
    """

    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    if cap < 0:
        raise ValueError("cap must be >= 0")

    def _weight(doc: ContractDoc) -> float:
        raw = doc.metadata.get(event_count_key, 0)
        try:
            n = int(raw)
        except (TypeError, ValueError):
            n = 0
        n = max(0, n)

        w = 1.0 + alpha * math.log1p(n)
        if cap == 0:
            return 0.0
        return min(cap, w)

    return _weight


def make_metadata_weight_from_fields(
    *,
    categorical_multipliers: Optional[Dict[str, Dict[Any, float]]] = None,
    numeric_log_boosts: Optional[Dict[str, Tuple[float, float]]] = None,
    default: float = 1.0,
) -> Callable[[ContractDoc], float]:
    """Build a `metadata_weight` function from multiple metadata fields.

    This is a convenience utility to combine many "meta heuristics" into a single multiplicative
    weight used by `greedy_select_docs`.

    Two feature types are supported:
      - Categorical multipliers (e.g. source / portfolio): exact lookup in a mapping.
      - Numeric log boosts (e.g. counts): sublinear boost with a cap.

    Overall weight is the product of all configured terms:
        weight = default
               * Π categorical_multiplier_k(doc)
               * Π numeric_log_boost_k(doc)

    Numeric log boost formula (for a field value n >= 0):
        term = min(cap, 1 + alpha * log(1 + n))

    Args:
        categorical_multipliers:
            Mapping: metadata_key -> {metadata_value -> multiplier}.
            Missing values fall back to multiplier 1.0.
        numeric_log_boosts:
            Mapping: metadata_key -> (alpha, cap).
            The metadata value is coerced to int; invalid/missing treated as 0.
        default:
            Base multiplier. Typically 1.0.

    Returns:
        Function suitable to pass as `metadata_weight`.
    """

    categorical_multipliers = categorical_multipliers or {}
    numeric_log_boosts = numeric_log_boosts or {}

    if default < 0:
        raise ValueError("default must be >= 0")
    for key, (alpha, cap) in numeric_log_boosts.items():
        if alpha < 0:
            raise ValueError(f"alpha for {key!r} must be >= 0")
        if cap < 0:
            raise ValueError(f"cap for {key!r} must be >= 0")

    def _weight(doc: ContractDoc) -> float:
        w = float(default)

        for key, value_to_mult in categorical_multipliers.items():
            v = doc.metadata.get(key, None)
            w *= float(value_to_mult.get(v, 1.0))

        for key, (alpha, cap) in numeric_log_boosts.items():
            raw = doc.metadata.get(key, 0)
            try:
                n = int(raw)
            except (TypeError, ValueError):
                n = 0
            n = max(0, n)

            term = 1.0 + float(alpha) * math.log1p(n)
            if cap == 0:
                term = 0.0
            else:
                term = min(float(cap), term)
            w *= float(term)

        return w

    return _weight


def greedy_select_docs(
    docs: List[ContractDoc],
    universe: Optional[Set[int]] = None,
    target_coverage: Optional[float] = 0.95,
    max_docs: Optional[int] = None,
    metadata_weight: Callable[[ContractDoc], float] = default_metadata_weight,
    diversification_key: Optional[str | List[str] | Tuple[str, ...]] = None,
    diversification_penalty: float | Dict[str, float] = 0.0,
) -> Tuple[List[ContractDoc], Dict[str, Any]]:
    """Greedily select documents to maximise covered rules while keeping the selection small.

    Score at step t is:
        score_i(t) = |S_i ∩ U_t| * metadata_weight(doc) * diversification_factor(doc)

        Metadata weight (optional)
        --------------------------
        `metadata_weight(doc)` is a user-supplied function that returns a multiplicative weight derived
        from `doc.metadata`. It lets you bias the greedy choice toward certain documents *without*
        changing the underlying coverage objective.

        Typical uses:
            - Prefer higher-quality sources (e.g. audited vs unaudited) by returning a larger weight.
            - Prefer certain portfolios/entities for business reasons.
            - Prefer cheaper/faster-to-review documents.

        Because the score multiplies by `metadata_weight`, a document with fewer new rules can beat one
        with more new rules if its weight is sufficiently larger.

        Recommended properties:
            - Return non-negative values (negative weights invert the meaning of the score).
            - Return 1.0 for "neutral" docs.
            - Keep weights in a reasonable range (e.g. 0.5–2.0) unless you intentionally want a strong bias.

        Example:
            - Doc A adds 10 new rules with weight 1.0 -> score 10
            - Doc B adds 8 new rules with weight 1.5 -> score 12 (so B is picked)

    Diversification (optional)
    -------------------------
        If you set `diversification_key`, the algorithm applies a *soft* penalty to repeatedly selecting
        documents from the same group.

        You can pass either:
            - a single key (e.g. "portfolio"), or
            - multiple keys (e.g. ["portfolio", "file_type"]).

        With multiple keys, the overall diversification factor is the product of the per-key factors,
        so the scorer discourages repeated picks from the same portfolio *and* the same file type.

    For a single key k, let g_k(doc) = doc.metadata.get(k). During greedy selection we keep a counter:
        group_counts[k][g_k] = how many already-selected docs had group value g_k

    If you pass a single float `diversification_penalty`, it is applied to all keys.
    Alternatively, you can pass a dict mapping each key to its own penalty.

    The diversification factor is:
        diversification_factor(doc) = Π_k 1 / (1 + penalty_k * group_counts[k][g_k(doc)])

    Interpretation:
      - `diversification_penalty = 0.0` or `diversification_key is None` disables diversification.
      - Larger `diversification_penalty` => stronger preference for "new" groups.
      - This is NOT a hard constraint: a repeat group can still be selected if its marginal coverage
        gain is high enough.

    Example (diversification_penalty = 0.25):
      - First pick from a group: factor = 1 / (1 + 0.25 * 0) = 1.0
      - Second pick from same group: factor = 1 / (1 + 0.25 * 1) = 0.8
      - Third pick from same group: factor = 1 / (1 + 0.25 * 2) = 0.666...

    Note: if a document is missing the key in metadata, it is treated as group `None`. That means
    all such documents share the same group for diversification purposes.

    where:
      - S_i is the doc's rule set
      - U_t is the set of uncovered rules at step t

    Args:
        docs: candidate documents.
        universe: rules you want to cover. If None, uses union of all rules in docs.
        target_coverage: stop when covered/total >= target_coverage. If None, stop when no improvement.
        max_docs: hard cap on number selected.
        metadata_weight:
            Function returning a multiplicative heuristic weight based on metadata. This scales the
            marginal-gain term |S_i ∩ U_t|; default returns 1.0 for all docs.
        diversification_key:
            Metadata field name(s) used to define one or more "group" dimensions.
            Examples: "portfolio" or ["portfolio", "file_type"].

            If set, selection is softly discouraged from picking multiple docs from the same group
            in each dimension.

        diversification_penalty:
            Non-negative coefficient(s) controlling the strength of the penalty.
            - If float: used for all diversification keys.
            - If dict[str, float]: per-key penalty.

    Returns:
        selected_docs: chosen docs in order.
        report: summary + step-by-step progress.
    """

    if not docs:
        return [], {"coverage": 0.0, "covered_rules": 0, "total_rules": 0, "steps": []}

    if universe is None:
        universe = set().union(*(d.rules for d in docs))

    total_rules = len(universe)
    if total_rules == 0:
        return [], {"coverage": 1.0, "covered_rules": 0, "total_rules": 0, "steps": []}

    uncovered = set(universe)
    selected: List[ContractDoc] = []
    steps: List[Dict[str, Any]] = []

    # Track group counts for diversification (e.g. portfolio counts, file type counts).
    group_counts: Dict[str, Dict[Any, int]] = {}

    def _normalise_diversification_keys(
        key: Optional[str | List[str] | Tuple[str, ...]],
    ) -> List[str]:
        if key is None:
            return []
        if isinstance(key, str):
            return [key]
        # Defensive: filter falsy/empty keys
        return [k for k in list(key) if isinstance(k, str) and k]

    diversification_keys = _normalise_diversification_keys(diversification_key)

    def _penalty_for_key(key: str) -> float:
        if isinstance(diversification_penalty, dict):
            return float(diversification_penalty.get(key, 0.0))
        return float(diversification_penalty)

    # Validate penalties early
    for k in diversification_keys:
        p = _penalty_for_key(k)
        if p < 0.0:
            raise ValueError(f"diversification_penalty for key {k!r} must be >= 0")

    def diversification_factor(doc: ContractDoc) -> float:
        if not diversification_keys:
            return 1.0

        factor = 1.0
        for k in diversification_keys:
            p = _penalty_for_key(k)
            if p <= 0.0:
                continue

            group = doc.metadata.get(k, None)
            c = group_counts.get(k, {}).get(group, 0)
            factor *= 1.0 / (1.0 + p * c)

        return factor

    while True:
        current_covered = total_rules - len(uncovered)
        current_coverage = current_covered / total_rules

        # Stop conditions
        if target_coverage is not None and current_coverage >= target_coverage:
            break
        if max_docs is not None and len(selected) >= max_docs:
            break

        best_doc: Optional[ContractDoc] = None
        best_score: float = 0.0
        best_new_rules: Set[int] = set()

        for doc in docs:
            if doc in selected:
                continue

            # Marginal gain: how many *new* rules this doc would add now
            new_rules = doc.rules & uncovered
            new_count = len(new_rules)
            if new_count == 0:
                continue

            score = new_count * float(metadata_weight(doc)) * float(diversification_factor(doc))

            if score > best_score:
                best_score = score
                best_doc = doc
                best_new_rules = new_rules

        # No improvement possible
        if best_doc is None:
            break

        selected.append(best_doc)
        uncovered -= best_new_rules

        # Update diversification counts
        for k in diversification_keys:
            group = best_doc.metadata.get(k, None)
            bucket = group_counts.setdefault(k, {})
            bucket[group] = bucket.get(group, 0) + 1

        steps.append(
            {
                "picked": best_doc.doc_id,
                "new_rules_added": len(best_new_rules),
                "covered_rules_so_far": total_rules - len(uncovered),
                "coverage_so_far": (total_rules - len(uncovered)) / total_rules,
                "metadata": best_doc.metadata,
            }
        )

    final_covered = total_rules - len(uncovered)
    report = {
        "covered_rules": final_covered,
        "total_rules": total_rules,
        "coverage": final_covered / total_rules,
        "num_selected_docs": len(selected),
        "steps": steps,
    }
    return selected, report


# -------------------------
# End-to-end helper
# -------------------------

def select_docs_with_hash_dedup(
    folder: str | Path,
    d: Optional[int] = None,
    rule_labels: Optional[List[str]] = None,
    target_coverage: float = 0.95,
    max_docs: Optional[int] = None,
    min_input_coverage: Optional[float] = None,
    # Dedup controls
    keep_top_n_per_hash: int = 1,
    representative_preference: Optional[Callable[[ContractDoc], float]] = None,
    # Greedy heuristic controls
    metadata_weight: Callable[[ContractDoc], float] = default_metadata_weight,
    diversification_key: Optional[str] = None,
    diversification_penalty: float = 0.0,
) -> Tuple[List[ContractDoc], Dict[str, Any]]:
    """One-call pipeline:

    1) Load docs from folder.
    2) Deduplicate by vector hash (keep one or top-N reps per identical vector).
    3) Run greedy selection.

    Args:
        folder: path containing JSON docs.
        d: vector dimensionality. If rule_labels is provided, this must match len(rule_labels).
           If loading from JSON with triggered rule labels, rule_labels is required.
        target_coverage: stop once this fraction is reached.
        max_docs: optional cap on number of docs.
        min_input_coverage:
            Optional fail-fast check run *before* dedup/greedy. If provided, raises ValueError when
            the union of triggered rules across all input docs covers less than this fraction of the
            rule-label universe.
        keep_top_n_per_hash: keep 1 representative per identical vector by default.
        representative_preference: score used to pick reps within a hash-group.
        metadata_weight: weight used during greedy scoring.
        diversification_key/diversification_penalty: optional diversification.

    Returns:
        selected docs, and report dict.
    """

    if rule_labels is None:
        raise ValueError(
            "rule_labels is required when loading documents from JSON, because triggered rule labels "
            "must be mapped into vector indices before hashing/greedy selection."
        )

    d_inferred = len(rule_labels)
    if d is not None and d != d_inferred:
        raise ValueError(f"d ({d}) does not match len(rule_labels) ({d_inferred})")

    docs = load_contract_docs_from_json_with_rule_labels(folder, rule_labels, strict=True)

    input_coverage_report = compute_rule_coverage(docs, rule_labels=rule_labels)
    if min_input_coverage is not None:
        if not (0.0 <= float(min_input_coverage) <= 1.0):
            raise ValueError("min_input_coverage must be between 0 and 1")
        if float(input_coverage_report["coverage"]) < float(min_input_coverage):
            uncovered = input_coverage_report.get("uncovered_rule_labels", [])
            preview = uncovered[:20]
            suffix = "" if len(uncovered) <= 20 else f" (+{len(uncovered) - 20} more)"
            raise ValueError(
                "Input rule coverage is below the required threshold. "
                f"coverage={input_coverage_report['coverage']:.4f}, "
                f"min_required={float(min_input_coverage):.4f}. "
                f"Uncovered labels preview: {preview}{suffix}"
            )

    deduped, groups = deduplicate_by_vector_hash(
        docs=docs,
        d=d_inferred,
        preference_score=representative_preference,
        keep_top_n_per_hash=keep_top_n_per_hash,
    )

    selected, report = greedy_select_docs(
        docs=deduped,
        universe=None,
        target_coverage=target_coverage,
        max_docs=max_docs,
        metadata_weight=metadata_weight,
        diversification_key=diversification_key,
        diversification_penalty=diversification_penalty,
    )

    # Add dedup stats
    report["dedup"] = {
        "original_docs": len(docs),
        "unique_vectors": len(groups),
        "kept_after_dedup": len(deduped),
        "keep_top_n_per_hash": keep_top_n_per_hash,
        "d": d_inferred,
    }

    report["input_coverage"] = {
        "coverage": input_coverage_report["coverage"],
        "covered_rule_count": input_coverage_report["covered_rule_count"],
        "uncovered_rule_count": input_coverage_report["uncovered_rule_count"],
        "total_rules": input_coverage_report["total_rules"],
    }

    return selected, report


# -------------------------
# Example usage (toy data)
# -------------------------

if __name__ == "__main__":
    # Toy in-memory example
    docs = [
        ContractDoc(
            "A",
            {
                "portfolio": "P1",
                "source": "S1",
                "num_distinct_products": 2,
                "num_distinct_legal_entities": 1,
                "num_contracts_triggered_events": 3,
                "num_cashflows_triggered_events": 8,
            },
            {0, 1, 2},
        ),
        ContractDoc(
            "B",
            {
                "portfolio": "P9",
                "source": "S2",
                "num_distinct_products": 5,
                "num_distinct_legal_entities": 2,
                "num_contracts_triggered_events": 30,
                "num_cashflows_triggered_events": 120,
            },
            {0, 1, 2},
        ),  # duplicate vector
        ContractDoc(
            "C",
            {
                "portfolio": "P2",
                "source": "S1",
                "num_distinct_products": 3,
                "num_distinct_legal_entities": 1,
                "num_contracts_triggered_events": 10,
                "num_cashflows_triggered_events": 40,
            },
            {3, 4, 5},
        ),
        ContractDoc(
            "D",
            {
                "portfolio": "P3",
                "source": "S3",
                "num_distinct_products": 1,
                "num_distinct_legal_entities": 1,
                "num_contracts_triggered_events": 1,
                "num_cashflows_triggered_events": 2,
            },
            {5, 6},
        ),
    ]

    d = 7

    # Prefer source S1 over others *when choosing the representative inside a hash group*
    def rep_preference(doc: ContractDoc) -> float:
        src = doc.metadata.get("source")
        return 100.0 if src == "S1" else 1.0

    deduped, groups = deduplicate_by_vector_hash(docs, d=d, preference_score=rep_preference)

    print("Original docs:", [x.doc_id for x in docs])
    print("Unique vectors:", len(groups))
    print("Kept after dedup:", [x.doc_id for x in deduped])

    # During greedy selection, you can inject metadata heuristics.
    # This example combines categorical boosts (source/portfolio) and numeric boosts (counts).
    weight = make_metadata_weight_from_fields(
        categorical_multipliers={
            "source": {"S1": 1.15, "S2": 1.00, "S3": 0.95},
            "portfolio": {"P2": 1.05},
        },
        numeric_log_boosts={
            "num_distinct_products": (0.10, 1.30),
            "num_distinct_legal_entities": (0.10, 1.30),
            "num_contracts_triggered_events": (0.08, 1.40),
            "num_cashflows_triggered_events": (0.06, 1.40),
        },
        default=1.0,
    )

    selected, report = greedy_select_docs(
        docs=deduped,
        target_coverage=1.0,
        metadata_weight=weight,
        diversification_key="portfolio",
        diversification_penalty=0.25,
    )

    print("Selected:", [d.doc_id for d in selected])
    print("Coverage:", report["coverage"])
    print("Steps:")
    for step in report["steps"]:
        print(" ", step)
