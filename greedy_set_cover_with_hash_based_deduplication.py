"""greedy_set_cover_dedup.py

Greedy document selection to maximise covered test cases with as few TNR_file docs as possible,
with hash-based deduplication of identical test_case-vectors.

Why hashing helps
-----------------
If two docs have identical binary vectors, they cover exactly the same test cases.
For coverage optimisation, those documents are interchangeable *unless* you have extra constraints
(e.g. pick at least one doc per legal entity) or preferences (trusted source, preferred portfolio, etc.).

So we:
    1) Hash each vector (or sparse test_case-index set) into a stable digest.
  2) Group documents by hash.
  3) Keep the best representative per hash-group (or top-N) using a user-defined preference score.
  4) Run greedy selection on the deduplicated candidates.

This gives you fewer candidates to score at each greedy step (often much faster) without changing
coverage results, assuming you only keep one representative per identical vector.

Example
-------
>>> docs = [
...   TNRFileDoc("A", {"source":"S1"}, {"TC1","TC2"}),
...   TNRFileDoc("B", {"source":"S2"}, {"TC1","TC2"}),  # identical test_cases => same hash
...   TNRFileDoc("C", {"source":"S1"}, {"TC3"}),
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
- The greedy core uses *marginal gain*: new_test_cases = doc.test_cases ∩ uncovered.
- Metadata can be injected as a multiplicative weight or as a preference when choosing a representative.

"""

from __future__ import annotations

import json
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class TNRFileDoc:
    """A single TNR_file doc with metadata and a set of covered test_case identifiers."""

    doc_id: str
    metadata: Dict[str, Any]
    test_cases: Set[str]


# -------------------------
# Test case encoding utilities
# -------------------------

def build_test_case_index(test_case_ids: List[str]) -> Dict[str, int]:
    """Build a stable mapping from test_case_id -> column index.

    You said you have an *ordered* master list of all possible test_case IDs.
    That ordering defines the vector coordinates.

    Unit breakdown:
    - A test_case ID is a string identifier, e.g. "954deaa705".
      - The index is an integer position in [0, d-1].
    - d = len(test_case_ids) is the shared vector dimensionality for all docs.

    Args:
        test_case_ids: ordered list of all test_case IDs.

    Returns:
        dict mapping each test_case_id to its index.

    Example:
        >>> ids = ["TC1", "TC2", "TC3"]
        >>> build_test_case_index(ids)
        {'TC1': 0, 'TC2': 1, 'TC3': 2}
    """

    return {tc_id: i for i, tc_id in enumerate(test_case_ids)}


def test_case_index_set_from_ids(
    test_case_ids: Iterable[str],
    test_case_index: Dict[str, int],
    *,
    strict: bool = True,
) -> Set[int]:
    """Convert a doc's triggered test_case IDs into a sparse set of test_case indices.

    This is usually the best internal representation for set-cover: you only store the 1s.

    Args:
        test_case_ids: the test_case IDs triggered by the doc.
        test_case_index: mapping from test_case_id -> index.
        strict:
            - If True, raise KeyError when a test_case_id is unknown (not in the master list).
            - If False, silently ignore unknown labels.

    Returns:
        Set of integer indices where the binary vector would have value 1.

    Example:
        >>> idx = build_test_case_index(["TC1","TC2","TC3"])
        >>> test_case_index_set_from_ids(["TC2","TC3"], idx)
        {1, 2}
    """

    out: Set[int] = set()
    for tc_id in test_case_ids:
        if tc_id in test_case_index:
            out.add(test_case_index[tc_id])
        elif strict:
            raise KeyError(f"Unknown test_case_id: {tc_id!r}")
    return out


def dense_vector_from_test_case_index_set(test_case_indices: Set[int], d: int) -> List[int]:
    """Convert a sparse test_case-index set into a dense 0/1 vector.

    You only need this if you must export a full vector (e.g. for storage or matrix ops).
    The greedy algorithm here works directly with the sparse set.

    Args:
        test_case_indices: indices where v[j] = 1.
        d: vector dimension (number of total possible test cases).

    Returns:
        A list of length d containing 0/1 integers.

    Example:
        >>> dense_vector_from_test_case_index_set({1,3}, d=5)
        [0, 1, 0, 1, 0]
    """

    v = [0] * d
    for j in test_case_indices:
        if j < 0 or j >= d:
            raise ValueError(f"Test_case index {j} is out of bounds for d={d}")
        v[j] = 1
    return v


# -------------------------
# Loading utilities
# -------------------------

def load_test_case_universe_from_json(file_path: str | Path) -> List[str]:
    """Load the universe of all possible test cases.

    Expected JSON shape (example from your `test_cases_universe.json`):
        [
          {"test_case_id": "954deaa705", "criteria": {...}, "expected_tests": [...]},
          ...
        ]

    Only the `test_case_id` field is used.

    Returns:
        Ordered list of test_case IDs as strings.
    """

    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Universe JSON must be a list, got {type(data).__name__}")

    ids: List[str] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(f"Universe entry {i} must be an object, got {type(obj).__name__}")
        tc_id = obj.get("test_case_id")
        if tc_id is None:
            raise ValueError(f"Universe entry {i} missing 'test_case_id'")
        ids.append(str(tc_id))

    return ids


def load_tnr_file_doc_from_json(file_path: str | Path) -> TNRFileDoc:
    """Load a single TNR_file doc from JSON.

    Expected JSON shape (example from your `tnr_file.json`):
        {
          "metadata": { ... },
          "test_cases_ids": ["1e9e960695", "00ec029e53", ...]
        }

    Returns:
        TNRFileDoc where:
          - doc_id is taken from metadata['filename'] when available, else the JSON path.
          - metadata is the JSON 'metadata' object (copied).
          - test_cases is a set of test_case_id strings.
    """

    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"TNR_file JSON must be an object, got {type(obj).__name__}")

    metadata = dict(obj.get("metadata", {}))
    raw_ids = obj.get("test_cases_ids", None)
    if raw_ids is None:
        raise ValueError("Missing required key 'test_cases_ids'")
    if not isinstance(raw_ids, list):
        raise ValueError(f"'test_cases_ids' must be a list, got {type(raw_ids).__name__}")

    test_cases = {str(x) for x in raw_ids if x is not None}
    doc_id = str(metadata.get("filename") or file_path)
    return TNRFileDoc(doc_id=doc_id, metadata=metadata, test_cases=test_cases)


def load_tnr_file_docs_from_folder(folder: str | Path) -> List[TNRFileDoc]:
    """Load all TNR_file docs from a folder recursively.

    Notes:
        The folder may contain other JSON files (e.g. a test_case universe JSON). This loader only
        keeps files that match the expected TNR_file shape (i.e. contain `test_cases_ids`).
    """

    folder = Path(folder)
    docs: List[TNRFileDoc] = []

    for fp in folder.rglob("*.json"):
        try:
            docs.append(load_tnr_file_doc_from_json(fp))
        except ValueError:
            # Not a TNR_file doc JSON; skip.
            continue

    return docs


# -------------------------
# Hashing & deduplication
# -------------------------

def vector_hash_from_test_cases(test_cases: Set[str], *, d: Optional[int] = None) -> str:
    """Return a stable hash for a binary vector represented as a set of test_case IDs.

    Canonical serialisation:
        f"{d}|{id1,id2,...}" where IDs are sorted lexicographically.

    Why include `d`?
        It prevents accidental collisions across datasets with different universes.
        Pass `d=len(universe)` when you have a global test_case universe.

    Hash choice:
        BLAKE2 is fast and stable. `digest_size=16` gives a 128-bit digest.
    """

    d_payload = "?" if d is None else str(int(d))
    payload = f"{d_payload}|{','.join(sorted(test_cases))}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def deduplicate_by_vector_hash(
    docs: List[TNRFileDoc],
    d: Optional[int] = None,
    preference_score: Optional[Callable[[TNRFileDoc], float]] = None,
    keep_top_n_per_hash: int = 1,
) -> Tuple[List[TNRFileDoc], Dict[str, List[TNRFileDoc]]]:
    """Deduplicate docs by identical test_case coverage using hashing.

    This groups documents by their vector hash (i.e. identical test_case sets).

    If keep_top_n_per_hash == 1:
        Keep a single representative per hash-group.
        If `preference_score` is provided, pick the representative with the highest score.

    If keep_top_n_per_hash > 1:
        Keep the top-N docs per hash-group ranked by preference_score (or arbitrary if None).
        This is useful if you anticipate later constraints (e.g. must pick different entities)
        and want alternatives available.

    Args:
        docs: list of documents.
        d: vector dimensionality (typically len(universe)); used only for hashing stability.
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
        def preference_score(_: TNRFileDoc) -> float:  # type: ignore[misc]
            return 0.0

    groups: Dict[str, List[TNRFileDoc]] = {}

    for doc in docs:
        h = vector_hash_from_test_cases(doc.test_cases, d=d)
        groups.setdefault(h, []).append(doc)

    deduped: List[TNRFileDoc] = []

    for h, bucket in groups.items():
        # Sort descending by preference score
        bucket_sorted = sorted(bucket, key=lambda x: float(preference_score(x)), reverse=True)
        deduped.extend(bucket_sorted[:keep_top_n_per_hash])

    return deduped, groups


# -------------------------
# Coverage reporting
# -------------------------

def compute_test_case_coverage(
    docs: List[TNRFileDoc],
    *,
    test_case_universe: Optional[Set[str]] = None,
    test_case_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute overall test_case coverage of a corpus of TNR_file docs.

    This answers: "Across all docs, what fraction of the global test_case universe is ever triggered?"

    You typically run this *before* dedup/selection to confirm that your input files can plausibly
    reach the desired coverage target.

    Provide the universe via either:
      - `test_case_universe`: a set[str] of all possible test_case IDs, or
      - `test_case_ids`: an ordered list[str] of all possible test_case IDs (allows stable ordering).

    Returns:
        A report dict with:
          - coverage: covered/total as float in [0,1]
          - covered_test_cases / uncovered_test_cases: sets[str]
          - covered_test_case_count / uncovered_test_case_count / total_test_cases
          - test_case_trigger_counts: dict[str, int]
          - num_docs
    """

    if test_case_ids is not None:
        inferred = set(map(str, test_case_ids))
        if test_case_universe is not None and set(map(str, test_case_universe)) != inferred:
            raise ValueError("test_case_universe does not match test_case_ids")
        test_case_universe = inferred

    if test_case_universe is None:
        raise ValueError("Provide test_case_universe or test_case_ids")

    universe = set(map(str, test_case_universe))
    covered: Set[str] = set()
    trigger_counts: Dict[str, int] = {}

    for doc in docs:
        for tc_id in doc.test_cases:
            tc_id = str(tc_id)
            if tc_id in universe:
                trigger_counts[tc_id] = trigger_counts.get(tc_id, 0) + 1
        covered |= (doc.test_cases & universe)

    uncovered = universe - covered
    total = len(universe)
    covered_count = len(covered)
    coverage = (covered_count / total) if total > 0 else 1.0

    report: Dict[str, Any] = {
        "coverage": coverage,
        "covered_test_cases": covered,
        "uncovered_test_cases": uncovered,
        "covered_test_case_count": covered_count,
        "uncovered_test_case_count": len(uncovered),
        "total_test_cases": total,
        "num_docs": len(docs),
        "test_case_trigger_counts": trigger_counts,
    }

    if test_case_ids is not None:
        report["uncovered_test_case_ids"] = [tc for tc in test_case_ids if str(tc) in uncovered]

    return report


def compute_test_case_coverage_from_folder(
    folder: str | Path,
    test_case_universe_json: str | Path,
) -> Dict[str, Any]:
    """Convenience wrapper: load TNR_file JSON docs + universe JSON, then compute coverage."""

    ids = load_test_case_universe_from_json(test_case_universe_json)
    docs = load_tnr_file_docs_from_folder(folder)
    return compute_test_case_coverage(docs, test_case_ids=ids)


def compute_rule_coverage_from_folder(
    folder: str | Path,
    test_case_universe_json: str | Path,
) -> Dict[str, Any]:
    """Backward-compatible alias for `compute_test_case_coverage_from_folder`."""

    return compute_test_case_coverage_from_folder(folder, test_case_universe_json)


# -------------------------
# Greedy selection
# -------------------------

def default_metadata_weight(_: TNRFileDoc) -> float:
    """Default metadata weight: no bias."""

    return 1.0


def make_triggered_event_count_weight(
    *,
    event_count_key: str = "triggered_event_count",
    alpha: float = 0.25,
    cap: float = 2.0,
) -> Callable[[TNRFileDoc], float]:
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
        A function suitable to pass as `metadata_weight` to `epsilon_neighbour_select_docs`.
    """

    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    if cap < 0:
        raise ValueError("cap must be >= 0")

    def _weight(doc: TNRFileDoc) -> float:
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
) -> Callable[[TNRFileDoc], float]:
    """Build a `metadata_weight` function from multiple metadata fields.

    This is a convenience utility to combine many "meta heuristics" into a single multiplicative
    weight used by `epsilon_neighbour_select_docs`.

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

    def _weight(doc: TNRFileDoc) -> float:
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

def epsilon_neighbour_select_docs(
    docs: List[TNRFileDoc],
    *,
    test_case_universe: Set[str],
    target_coverage: float = 0.95,
    metadata_weight: Callable[[TNRFileDoc], float] = default_metadata_weight,
    epsilon: float = 0.7,
    neighbour_count: int = 25,
    max_steps: int = 250,
    no_improve_patience: int = 10,
    epsilon_decay: float = 0.9,
    min_epsilon: float = 0.1,
    random_seed: Optional[int] = None,
) -> Tuple[List[TNRFileDoc], Dict[str, Any]]:
    """Epsilon-greedy + neighbour-window greedy set cover for TNR_file docs.

    This implements the approach you described:

    1) **Score and rank all candidate docs by a meta-heuristic**.
       The meta-heuristic score is computed by `metadata_weight(doc)` and is *independent* of
       coverage; it encodes preferences such as trusted sources or larger files.

       The algorithm maintains a list of available docs sorted by decreasing meta score.

    2) **Epsilon-based anchor selection** at each step.
       With probability `epsilon`, the anchor is the top-ranked doc (greedy on the meta score).
       With probability `1 - epsilon`, the anchor is chosen uniformly at random from the remaining
       available docs (exploration).

    3) **Neighbourhood restriction**.
       Let the chosen anchor be at position `i` in the meta-ranked list. We consider the subset:

           window = docs[i : i + 1 + neighbour_count]

       (the anchor plus the next `neighbour_count` neighbours).

    4) **Within the window, pick the doc with the best marginal gain**.
       Marginal gain is the number of *new* uncovered test cases the doc would cover:

           gain(doc) = |doc.test_cases ∩ uncovered|

       The step chooses the doc with maximum `gain` within the window (ties broken by meta score).

    5) **Prune zero-gain docs**.
       Any evaluated doc with zero marginal gain is removed from the available list.

    6) **Adaptive exploration when stuck**.
       If the algorithm performs `no_improve_patience` consecutive steps with no coverage improvement
       (i.e. no selected doc because the sampled window had no positive-gain doc), reduce epsilon to
       encourage exploration:

           epsilon <- max(min_epsilon, epsilon * epsilon_decay)

    7) **Early stopping**.
       Stop when one of these happens:
         - coverage reaches `target_coverage`
         - `max_steps` is reached
         - no available docs remain

    Important note:
        This is *not* brute force over all subsets (which would be exponential), but it does still
        require computing marginal gain for docs in the sampled windows. The neighbourhood heuristic
        is what reduces per-step work compared to scanning all docs.

    Returns:
        selected_docs, report where report includes:
          - steps: list of per-step dicts
          - marginal_gain_per_step: list[int]
          - coverage_per_step: list[float]
          - final_coverage
          - epsilon_final
    """

    if not (0.0 <= float(epsilon) <= 1.0):
        raise ValueError("epsilon must be between 0 and 1")
    if neighbour_count < 0:
        raise ValueError("neighbour_count must be >= 0")
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")
    if no_improve_patience < 1:
        raise ValueError("no_improve_patience must be >= 1")
    if not (0.0 < float(epsilon_decay) <= 1.0):
        raise ValueError("epsilon_decay must be in (0, 1]")
    if not (0.0 <= float(min_epsilon) <= 1.0):
        raise ValueError("min_epsilon must be between 0 and 1")

    rng = random.Random(random_seed)

    universe = set(map(str, test_case_universe))
    uncovered: Set[str] = set(universe)
    total = len(universe)

    # Pre-score and sort docs by meta-heuristic.
    available: List[Tuple[TNRFileDoc, float]] = [
        (doc, float(metadata_weight(doc))) for doc in docs
    ]
    available.sort(key=lambda x: x[1], reverse=True)

    selected: List[TNRFileDoc] = []
    steps: List[Dict[str, Any]] = []
    marginal_gain_per_step: List[int] = []
    coverage_per_step: List[float] = []

    consecutive_no_improve = 0

    def _coverage() -> float:
        if total == 0:
            return 1.0
        return (total - len(uncovered)) / total

    for step_idx in range(int(max_steps)):
        current_cov = _coverage()
        if current_cov >= float(target_coverage):
            break
        if not available:
            break

        # 1) epsilon-greedy anchor selection on the meta-ranked list
        greedy_pick = rng.random() < float(epsilon)
        anchor_idx = 0 if greedy_pick else rng.randrange(len(available))

        window_end = min(len(available), anchor_idx + 1 + int(neighbour_count))
        window_indices = list(range(anchor_idx, window_end))

        best_idx: Optional[int] = None
        best_gain = 0
        best_meta = float("-inf")
        zero_gain_indices: List[int] = []

        for idx in window_indices:
            doc, meta = available[idx]
            gain = len(doc.test_cases & uncovered)
            if gain <= 0:
                zero_gain_indices.append(idx)
                continue

            if (gain > best_gain) or (gain == best_gain and meta > best_meta):
                best_gain = gain
                best_meta = meta
                best_idx = idx

        # 2) prune zero-gain docs that we evaluated
        removed_zero_gain = 0
        for idx in sorted(zero_gain_indices, reverse=True):
            # Defensive: avoid removing the chosen best doc (should not happen)
            if best_idx is not None and idx == best_idx:
                continue
            available.pop(idx)
            removed_zero_gain += 1
            if best_idx is not None and idx < best_idx:
                best_idx -= 1

        picked_doc: Optional[TNRFileDoc] = None
        picked_meta: Optional[float] = None

        if best_idx is not None and best_gain > 0:
            picked_doc, picked_meta = available.pop(best_idx)
            selected.append(picked_doc)
            uncovered -= (picked_doc.test_cases & universe)
            consecutive_no_improve = 0
        else:
            # No improvement from this sampled window (but we still pruned some zero-gain docs).
            consecutive_no_improve += 1

        # 3) Adaptive epsilon decrease to encourage exploration
        epsilon_before = float(epsilon)
        if consecutive_no_improve >= int(no_improve_patience):
            epsilon = max(float(min_epsilon), float(epsilon) * float(epsilon_decay))
            consecutive_no_improve = 0

        cov_after = _coverage()
        marginal_gain_per_step.append(int(best_gain) if picked_doc is not None else 0)
        coverage_per_step.append(float(cov_after))

        steps.append(
            {
                "step": step_idx,
                "epsilon_used": epsilon_before,
                "epsilon_after": float(epsilon),
                "anchor_strategy": "greedy" if greedy_pick else "random",
                "anchor_index": anchor_idx,
                "window_size": len(window_indices),
                "removed_zero_gain_docs": removed_zero_gain,
                "picked": None if picked_doc is None else picked_doc.doc_id,
                "picked_meta_score": picked_meta,
                "marginal_gain": int(best_gain) if picked_doc is not None else 0,
                "coverage": float(cov_after),
                "uncovered_remaining": len(uncovered),
                "available_remaining": len(available),
            }
        )

    report = {
        "final_coverage": _coverage(),
        "total_test_cases": total,
        "uncovered_remaining": len(uncovered),
        "num_selected_docs": len(selected),
        "epsilon_final": float(epsilon),
        "steps": steps,
        "marginal_gain_per_step": marginal_gain_per_step,
        "coverage_per_step": coverage_per_step,
    }
    return selected, report


def plot_selection_report(
    report: Dict[str, Any],
    *,
    output_dir: Optional[str | Path] = None,
    show: bool = False,
) -> Dict[str, Path]:
    """Plot marginal gain and coverage per step.

    Produces two graphs:
      - marginal_gain_per_step
      - coverage_per_step

    Args:
        report: output of `epsilon_neighbour_select_docs`.
        output_dir: directory to save PNGs; if None, only shows (if show=True).
        show: whether to display the plots interactively.

    Returns:
        Dict with keys 'marginal_gain' and 'coverage' mapping to saved PNG paths (if output_dir set).
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    gains = list(map(int, report.get("marginal_gain_per_step", [])))
    cov = list(map(float, report.get("coverage_per_step", [])))

    saved: Dict[str, Path] = {}
    out_dir_path: Optional[Path] = None
    if output_dir is not None:
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

    # Plot marginal gain per step
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(gains)), gains, marker="o", linewidth=1)
    plt.title("Marginal Gain per Step")
    plt.xlabel("Step")
    plt.ylabel("New test_cases covered")
    plt.grid(True, alpha=0.3)
    if out_dir_path is not None:
        fp = out_dir_path / "marginal_gain_per_step.png"
        plt.tight_layout()
        plt.savefig(fp)
        saved["marginal_gain"] = fp
    if show:
        plt.show()
    plt.close()

    # Plot coverage per step
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(cov)), cov, marker="o", linewidth=1)
    plt.title("Coverage per Step")
    plt.xlabel("Step")
    plt.ylabel("Coverage")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    if out_dir_path is not None:
        fp = out_dir_path / "coverage_per_step.png"
        plt.tight_layout()
        plt.savefig(fp)
        saved["coverage"] = fp
    if show:
        plt.show()
    plt.close()

    return saved


# -------------------------
# End-to-end helper
# -------------------------

def select_docs_with_hash_dedup(
    tnr_files_folder: str | Path,
    test_case_universe_json: str | Path,
    *,
    target_coverage: float = 0.95,
    min_input_coverage: Optional[float] = None,
    # Dedup controls
    keep_top_n_per_hash: int = 1,
    representative_preference: Optional[Callable[[TNRFileDoc], float]] = None,
    # Meta-heuristic ranking & neighbourhood-search controls
    metadata_weight: Callable[[TNRFileDoc], float] = default_metadata_weight,
    epsilon: float = 0.7,
    neighbour_count: int = 25,
    max_steps: int = 250,
    no_improve_patience: int = 10,
    epsilon_decay: float = 0.9,
    min_epsilon: float = 0.1,
    random_seed: Optional[int] = None,
    plot_output_dir: Optional[str | Path] = None,
    show_plots: bool = False,
) -> Tuple[List[TNRFileDoc], Dict[str, Any]]:
    """End-to-end pipeline for TNR_file doc selection.

    Inputs
    ------
    - `tnr_files_folder`: folder containing many TNR_file JSON files.
      Each file must have keys:
        - `metadata`: object
        - `test_cases_ids`: list[str]
    - `test_case_universe_json`: a JSON list where each entry has `test_case_id`.

    Pipeline
    --------
    1) Load the universe of test cases.
    2) Load all TNR_file docs.
    3) (Optional) fail fast if the input corpus cannot reach `min_input_coverage`.
    4) Deduplicate by vector hash (identical `test_cases` sets).
    5) Run `epsilon_neighbour_select_docs`.
    6) (Optional) save plots.

    Returns:
        selected_docs, report dict.
    """

    test_case_ids = load_test_case_universe_from_json(test_case_universe_json)
    universe = set(test_case_ids)
    docs = load_tnr_file_docs_from_folder(tnr_files_folder)

    input_coverage_report = compute_test_case_coverage(docs, test_case_ids=test_case_ids)
    if min_input_coverage is not None:
        if not (0.0 <= float(min_input_coverage) <= 1.0):
            raise ValueError("min_input_coverage must be between 0 and 1")
        if float(input_coverage_report["coverage"]) < float(min_input_coverage):
            uncovered = input_coverage_report.get("uncovered_test_case_ids", [])
            preview = uncovered[:20]
            suffix = "" if len(uncovered) <= 20 else f" (+{len(uncovered) - 20} more)"
            raise ValueError(
                "Input test_case coverage is below the required threshold. "
                f"coverage={input_coverage_report['coverage']:.4f}, "
                f"min_required={float(min_input_coverage):.4f}. "
                f"Uncovered test_case_id preview: {preview}{suffix}"
            )

    deduped, groups = deduplicate_by_vector_hash(
        docs=docs,
        d=len(test_case_ids),
        preference_score=representative_preference,
        keep_top_n_per_hash=keep_top_n_per_hash,
    )

    selected, report = epsilon_neighbour_select_docs(
        docs=deduped,
        test_case_universe=universe,
        target_coverage=target_coverage,
        metadata_weight=metadata_weight,
        epsilon=epsilon,
        neighbour_count=neighbour_count,
        max_steps=max_steps,
        no_improve_patience=no_improve_patience,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        random_seed=random_seed,
    )

    report["dedup"] = {
        "original_docs": len(docs),
        "unique_vectors": len(groups),
        "kept_after_dedup": len(deduped),
        "keep_top_n_per_hash": keep_top_n_per_hash,
        "d": len(test_case_ids),
    }

    report["input_coverage"] = {
        "coverage": input_coverage_report["coverage"],
        "covered_test_case_count": input_coverage_report["covered_test_case_count"],
        "uncovered_test_case_count": input_coverage_report["uncovered_test_case_count"],
        "total_test_cases": input_coverage_report["total_test_cases"],
        "num_docs": input_coverage_report["num_docs"],
    }

    if plot_output_dir is not None or show_plots:
        saved = plot_selection_report(report, output_dir=plot_output_dir, show=show_plots)
        report["plots"] = {k: str(v) for k, v in saved.items()}

    return selected, report


# -------------------------
# Example usage (toy data)
# -------------------------

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    tnr_folder = base_dir / "output_json"
    universe_json = base_dir / "test_cases_universe.json"

    if tnr_folder.exists() and universe_json.exists():
        # End-to-end run on synthesized JSON docs in ./output_json
        def rep_preference(doc: TNRFileDoc) -> float:
            # Prefer S1 when multiple docs have identical test_case vectors
            return 10.0 if doc.metadata.get("source") == "S1" else 1.0

        selected, report = select_docs_with_hash_dedup(
            tnr_files_folder=tnr_folder,
            test_case_universe_json=universe_json,
            target_coverage=1.0,
            keep_top_n_per_hash=1,
            representative_preference=rep_preference,
            epsilon=0.7,
            neighbour_count=5,
            max_steps=50,
            no_improve_patience=5,
            epsilon_decay=0.9,
            min_epsilon=0.1,
            random_seed=42,
            plot_output_dir=None,
            show_plots=False,
        )

        print("Selected:", [d.doc_id for d in selected])
        print("Final coverage:", report["final_coverage"])
        print("Dedup:", report.get("dedup"))
        print("Input coverage:", report.get("input_coverage"))
    else:
        print("No ./output_json and/or ./test_cases_universe.json found; nothing to run.")
