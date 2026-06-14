import os
import re
import math
import json
import time
import pickle
import sqlite3
import hashlib
import smtplib
import numpy as np
import anthropic
import scipy.sparse as sp
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from common import (
    pull_db, push_db, SOURCE_PRIORS, USER_PROFILE, KEY_ENTITIES,
    pipeline_lock, now_iso, LOCAL_DB,
    SECTORS, SECTOR_ORDER, SOURCE_SECTOR, DEFAULT_SECTOR,
    CATEGORY_NAMES, CATEGORY_PATTERNS, DEFAULT_CATEGORY,
    CATEGORIES
)

EMBED_MODEL = "all-MiniLM-L6-v2"
_EMBED_BODY_CHARS = 500

_embedder_cache = None
_embedder_loaded = False


def _embedder():
    global _embedder_cache, _embedder_loaded
    if _embedder_loaded:
        return _embedder_cache
    _embedder_loaded = True
    try:
        from sentence_transformers import SentenceTransformer
        _embedder_cache = SentenceTransformer(EMBED_MODEL)
    except Exception:
        _embedder_cache = None
    return _embedder_cache


def _embed(texts):
    """Encode a list of strings to a normalized (n, d) float32 ndarray, or None."""
    model = _embedder()
    if model is None or not texts:
        return None
    try:
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        return None


_aspect_emb_cache = {}


def _embed_aspects(aspects, cache_key):
    """Encode aspect descriptors; cache by profile hash so we encode once per run."""
    if cache_key in _aspect_emb_cache:
        return _aspect_emb_cache[cache_key]
    if _embedder() is None:
        return None
    texts = [desc for _, desc in aspects]
    embs = _embed(texts)
    _aspect_emb_cache[cache_key] = embs
    return embs

_nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
# Rule-based sentence boundaries (cheap; parser is still disabled). Needed by
# the sentence-level MMR medoid summary and the lexical-cohesion richness term.
if "sentencizer" not in _nlp.pipe_names:
    _nlp.add_pipe("sentencizer")
_NER_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT", "MONEY", "PERCENT", "DATE"}
_ENTITY_SIGNAL_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT"}
_SOURCE_NAMES = frozenset(k.lower() for k in SOURCE_SECTOR)

# Default scorer weights (before any adaptive tuning).
# Anchor invariant: USER_PROFILE defines the discourse space; the relevance term
# (profile-BM25) is the link to that space and its weight is floored at
# RELEVANCE_FLOOR so adaptive tuning can never let the system drift away from it.
DEFAULT_WEIGHTS = {
    "coverage":       0.14,
    "prior":          0.07,
    "novelty":        0.09,
    "relevance":      0.20,
    "entity_signal":  0.06,
    "trend":          0.05,
    "richness":       0.06,
    "coverage_gap":   0.06,
    "persistence":    0.10,
    "source_breadth": 0.07,
    "recency":        0.10,
}
RELEVANCE_FLOOR = 0.20
DUMP_RELEVANCE_FLOOR = 0.12
WEIGHT_KEYS = list(DEFAULT_WEIGHTS.keys())


# Age parsing for the recency-decay term. feedparser emits RFC 822 strings;
# fall back to None when unparseable so the cluster gets the per-week median.
from email.utils import parsedate_to_datetime as _parse_rfc822  # noqa: E402

def _item_age_days(ts_str, now):
    if not ts_str:
        return None
    try:
        dt = _parse_rfc822(ts_str)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:
        return None


# ── text utilities ────────────────────────────────────────────────────────────

def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s.lower())).strip()


def _shingles(s, k=3):
    t = _norm(s).replace(" ", "")
    return {t[i:i + k] for i in range(len(t) - k + 1)} if len(t) >= k else {t}


# ── sentence-level helpers (TREC Novelty / lexical cohesion) ──────────────────

def _split_sentences(text):
    """Rule-based sentence boundaries via spaCy's sentencizer.

    Cheap (no parser, no transformer) and good enough for journalistic prose.
    Returns a list of stripped, non-empty sentence strings."""
    if not text:
        return []
    try:
        doc = _nlp(text)
    except Exception:
        return []
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def _mmr_sentence(body, title, query, max_chars=240, lam=0.7):
    """Pick the single most information-dense sentence from `body` via MMR.

    Replaces the heuristic body[:240] (which assumes inverted-pyramid lede
    structure). MMR objective per TREC Novelty:

        score(s) = lam * cos(s, query) - (1 - lam) * cos(s, title)

    Relevance = aligned with the user's discourse query (USER_PROFILE).
    Diversity penalty = aligned with the title, which the LLM already sees
    — we want the body excerpt to ADD information, not echo the headline.

    Returns the picked sentence trimmed to `max_chars`. Falls back to
    body[:max_chars] for very short bodies or vectoriser failures."""
    sentences = _split_sentences(body)
    if len(sentences) <= 1:
        return (body or "")[:max_chars].strip()

    try:
        v = TfidfVectorizer(
            stop_words="english", sublinear_tf=True,
            ngram_range=(1, 2), min_df=1, norm="l2",
        )
        # Order: [sent_0, ..., sent_{n-1}, title, query]
        X = v.fit_transform(sentences + [_norm(title or ""), _norm(query or "")])
    except ValueError:
        return sentences[0][:max_chars].strip()

    n = len(sentences)
    title_vec = X[n]
    query_vec = X[n + 1]
    rel = np.asarray(X[:n].dot(query_vec.T).todense()).ravel()
    redundancy = np.asarray(X[:n].dot(title_vec.T).todense()).ravel()
    scores = lam * rel - (1.0 - lam) * redundancy
    best = int(np.argmax(scores))
    return sentences[best][:max_chars].strip()


def _lexical_cohesion(text):
    """Average adjacent-sentence cosine similarity in `text`.

    Proxy for local coherence (Entity Grid–lite): high cohesion means
    adjacent sentences reuse vocabulary, characteristic of sustained
    argument; low cohesion is the listicle / link-dump pattern. Returns
    a value in [0, 1] (0 if the document has fewer than two sentences
    or the vectoriser cannot fit on it)."""
    sents = _split_sentences(text)
    if len(sents) < 2:
        return 0.0
    try:
        v = TfidfVectorizer(stop_words="english", sublinear_tf=True, norm="l2")
        X = v.fit_transform(sents)
    except ValueError:
        return 0.0
    sims = []
    for i in range(len(sents) - 1):
        sims.append(float(X[i].dot(X[i + 1].T).toarray()[0, 0]))
    if not sims:
        return 0.0
    return float(np.clip(np.mean(sims), 0.0, 1.0))


# ── NLP enrichment ────────────────────────────────────────────────────────────

def _is_signal_entity(text, label):
    return label in _ENTITY_SIGNAL_LABELS and text.lower() not in _SOURCE_NAMES


def enrich(rows):
    items = []
    for id_, source, title, url, body, ts in rows:
        text = f"{title} {(body or '')[:800]}"
        doc = _nlp(text)
        entities = {
            ent.text.strip(): ent.label_
            for ent in doc.ents if ent.label_ in _NER_LABELS
        }
        items.append({
            "id": id_, "source": source, "title": title,
            "url": url, "body": body or "", "ts": ts,
            "entities": entities,
        })
    return items


# ── deduplication & clustering ────────────────────────────────────────────────
#
# Refactored away from MinHash LSH + union-find (single-linkage). At weekly
# corpus scales (thousands of items, ~20k vocab) the exact cosine matrix
# X·X^T over sparse TF-IDF is faster than an LSH proposal step plus pairwise
# verification, and removes a probabilistic hyperparameter (Jaccard threshold)
# that was always going to drift from the cosine threshold it gated.
#
# Clustering moved to average-linkage agglomerative with a hard cosine-distance
# diameter — same intent as the old sim_threshold but enforced over the whole
# cluster rather than one transitive edge at a time. This kills the chaining
# pathology where A-B and B-C linked clustered A and C with zero overlap.


def _pairs_above(S, threshold):
    """Yield (i, j) with i < j whose entries in the (sparse) similarity matrix
    S exceed `threshold`. S is the output of X·X^T for L2-normalised X."""
    S_coo = S.tocoo() if sp.issparse(S) else sp.coo_matrix(S)
    for i, j, v in zip(S_coo.row, S_coo.col, S_coo.data):
        if i < j and v >= threshold:
            yield int(i), int(j)


def build_tfidf(items, sim_threshold=0.20, title_sim_threshold=0.40):
    """Build augmented TF-IDF and the two-lane candidate edge set.

    Two lanes preserved from the LSH design (different cosine thresholds
    on different vocabulary spaces — editorial structure separates headline
    semantics from body semantics):

      • Body+title augmented TF-IDF, threshold `sim_threshold` (0.20).
      • Title-only TF-IDF, threshold `title_sim_threshold` (0.40, stricter
        because short text is noisier per term).

    `edges` is the union of both lanes — diagnostic only now; clustering
    runs from the augmented matrix directly (see cluster_average_linkage).

    `min_df=2` (was 1): drops hapaxes, URLs, typos that would otherwise
    inflate dimensionality on persistent topics across weeks."""
    def augmented(a):
        ent_tokens = " ".join(
            f"ent_{e.lower().replace(' ', '_')}" for e in a["entities"]
        )
        return f"{_norm(a['title'])} {_norm(a['body'])} {ent_tokens}"

    def _fit(texts, min_df):
        v = TfidfVectorizer(
            stop_words="english", sublinear_tf=True,
            ngram_range=(1, 2), min_df=min_df, norm="l2",
        )
        return v, v.fit_transform(texts)

    aug_texts = [augmented(a) for a in items]
    try:
        vec, X = _fit(aug_texts, min_df=2)
    except ValueError:
        # Tiny corpus where every term appears once — fall back so the
        # weekly job doesn't crash on a slow ingest day.
        vec, X = _fit(aug_texts, min_df=1)

    # Pass 1: body+title augmented cosine.
    edges = set(_pairs_above(X.dot(X.T), sim_threshold))

    # Pass 2: title-only cosine on its own vocabulary. Skip silently if the
    # title-only corpus has too few repeating terms to survive min_df=2.
    title_texts = [_norm(a["title"]) for a in items]
    try:
        _, X_title = _fit(title_texts, min_df=2)
        edges.update(_pairs_above(X_title.dot(X_title.T), title_sim_threshold))
    except ValueError:
        pass

    return X, vec, edges


def cluster_average_linkage(X, max_diameter=0.80):
    """Average-linkage agglomerative clustering with a cosine-distance
    diameter ceiling.

    Replaces single-linkage union-find. Single-linkage chains: an edge A-B
    and an edge B-C produce {A, B, C} even when A·C ≈ 0. Average-linkage
    uses mean inter-cluster distance, and the diameter cut at `max_diameter`
    enforces a strict semantic radius.

    `max_diameter` is cosine distance (= 1 - cosine similarity). 0.80
    corresponds roughly to "average inter-member cosine ≥ 0.20" — the
    same target as the old pairwise threshold, but enforced over the
    whole cluster.

    Returns list of [item_index, ...] groups."""
    n = X.shape[0]
    if n <= 1:
        return [[i] for i in range(n)]

    # Cosine distance from L2-normalised rows: 1 - X·X^T.
    S = X.dot(X.T)
    if sp.issparse(S):
        S = S.toarray()
    D = np.clip(1.0 - S, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    # Sparse arithmetic can leave tiny asymmetries that squareform rejects.
    D = (D + D.T) * 0.5

    Y = squareform(D, checks=False)
    Z = linkage(Y, method="average")
    labels = fcluster(Z, t=max_diameter, criterion="distance")

    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[int(lab)].append(i)
    return list(groups.values())


def cluster_medoid(idxs, X):
    if len(idxs) == 1:
        return idxs[0]
    sub = X[idxs]
    return idxs[int(np.asarray(sub.dot(sub.T).sum(axis=1)).ravel().argmax())]


# ── novelty projection ────────────────────────────────────────────────────────

def _load_last_digest(conn):
    row = conn.execute(
        "SELECT centroids, vocab FROM digests ORDER BY week DESC LIMIT 1"
    ).fetchone()
    return (pickle.loads(row[0]), pickle.loads(row[1])) if row and row[0] else (None, None)


def _project_centroids(old_c, old_vocab, new_vocab):
    if old_c is None:
        return None
    rows_, cols, data = [], [], []
    for j, term in enumerate(old_vocab):
        if term is None:
            continue
        jn = new_vocab.get(term)
        if jn is not None:
            rows_.append(j)
            cols.append(jn)
            data.append(1.0)
    if not data:
        return None
    P = sp.csr_matrix((data, (rows_, cols)), shape=(len(old_vocab), len(new_vocab)))
    M = old_c.dot(P)
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms).dot(M)


# ── entity trend analysis ─────────────────────────────────────────────────────

def load_entity_history(conn, n_weeks=4):
    rows = conn.execute(
        "SELECT entity, count FROM entity_history "
        "WHERE week IN (SELECT DISTINCT week FROM entity_history ORDER BY week DESC LIMIT ?) "
        "ORDER BY week DESC",
        (n_weeks,),
    ).fetchall()
    hist = defaultdict(list)
    for entity, count in rows:
        hist[entity].append(count)
    return hist


def compute_velocities(entity_counts, history):
    result = {}
    for ent, cnt in entity_counts.items():
        past = history.get(ent, [])
        avg = sum(past) / len(past) if past else 0.0
        result[ent] = (cnt - avg) / max(avg, 1.0)
    return result


def save_entity_history(conn, entity_counts, week):
    conn.executemany(
        "INSERT OR REPLACE INTO entity_history(entity, week, count) VALUES(?,?,?)",
        [(e, week, c) for e, c in entity_counts.items()],
    )


# ── §D longitudinal context — entity & topic memory fed to the LLM ────────────

def _topic_label(topic):
    """Derive a short human-readable label for a topic_bank entry from the
    top tokens of its centroid (skipping the `ent_…` tokens which are
    entity-resolved markers from build_tfidf's augmented text)."""
    centroid = topic.get("centroid")
    vocab = topic.get("vocab") or []
    if centroid is None or not vocab:
        return f"topic#{topic.get('topic_id', '?')}"
    arr = (centroid.toarray().ravel() if hasattr(centroid, "toarray")
           else np.asarray(centroid).ravel())
    if arr.size == 0:
        return f"topic#{topic.get('topic_id', '?')}"
    order = np.argsort(-arr)
    tokens = []
    for idx in order:
        if idx >= len(vocab):
            continue
        tok = vocab[idx]
        if not tok or tok.startswith("ent_"):
            continue
        tokens.append(tok)
        if len(tokens) >= 3:
            break
    return " · ".join(tokens) if tokens else f"topic#{topic.get('topic_id', '?')}"


def build_longitudinal_context(conn, week, this_week_entity_counts,
                               top_clusters, bank,
                               streak_min=3, dormant_min=4, top_n=5):
    """Compact text block of multi-week patterns for the summarise() prompt.

    Returns "" when entity_history has < 2 distinct weeks.

    Captures four signal classes — all derived from existing tables, no new
    persistence required:
      1. Entity streaks: entities present every week for the last N weeks.
      2. Returning entities: entities present this week + absent K+ weeks
         immediately prior.
      3. Topic streaks: bank topics matched by this week's top clusters that
         have weeks_seen >= streak_min.
      4. Returning topics: bank topics matched this week whose previous
         last_week was >= dormant_min weeks ago.
    """
    n_weeks_total = conn.execute(
        "SELECT COUNT(DISTINCT week) FROM entity_history"
    ).fetchone()[0] or 0
    if n_weeks_total < 2:
        return ""

    lines = []

    # --- entity streaks
    recent_weeks = [r[0] for r in conn.execute(
        "SELECT DISTINCT week FROM entity_history ORDER BY week DESC LIMIT ?",
        (streak_min,),
    ).fetchall()]
    if len(recent_weeks) >= streak_min and this_week_entity_counts:
        placeholders = ",".join("?" * len(recent_weeks))
        rows = conn.execute(
            f"SELECT entity, COUNT(DISTINCT week) AS n "
            f"FROM entity_history WHERE week IN ({placeholders}) "
            f"GROUP BY entity HAVING n >= ?",
            (*recent_weeks, streak_min),
        ).fetchall()
        # also require entity to be in this week's stream
        this_week = {e for e in this_week_entity_counts.keys()}
        on_streak = sorted(
            [(e, n) for e, n in rows if e in this_week],
            key=lambda x: (-x[1], x[0]),
        )[:top_n]
        for ent, n in on_streak:
            lines.append(f"- {ent}: appearing for {n} consecutive weeks (incl. this week)")

    # --- returning entities
    if this_week_entity_counts:
        prior_weeks = [r[0] for r in conn.execute(
            "SELECT DISTINCT week FROM entity_history WHERE week != ? "
            "ORDER BY week DESC LIMIT ?", (week, dormant_min + 2),
        ).fetchall()]
        if prior_weeks:
            # last appearance per entity (if any) within the recent window
            last_seen = {}
            for wk in prior_weeks:
                ents_in_wk = {r[0] for r in conn.execute(
                    "SELECT entity FROM entity_history WHERE week=?", (wk,),
                ).fetchall()}
                for e in ents_in_wk:
                    if e not in last_seen:
                        last_seen[e] = wk
            returning = []
            for ent in this_week_entity_counts:
                lw = last_seen.get(ent)
                if not lw:
                    continue
                gap = _weeks_between(week, lw)
                if gap >= dormant_min:
                    returning.append((ent, gap))
            returning.sort(key=lambda x: (-x[1], x[0]))
            for ent, gap in returning[:top_n]:
                lines.append(f"- {ent}: returns this week after {gap}-week gap")

    # --- topic streaks + returns
    matched_ids = {
        c.get("matched_topic_id") for c in top_clusters
        if c.get("matched_topic_id") is not None
    }
    if matched_ids and bank:
        bank_by_id = {t["topic_id"]: t for t in bank}
        topic_streaks = []
        topic_returns = []
        for tid in matched_ids:
            t = bank_by_id.get(tid)
            if not t:
                continue
            if t.get("weeks_seen", 0) >= streak_min:
                topic_streaks.append((t, t["weeks_seen"]))
            gap_before_this_week = _weeks_between(week, t.get("last_week", ""))
            # Note: t["last_week"] was the last_week BEFORE this run updated it
            # for matched topics. We treat gap >= dormant_min as "returning".
            if gap_before_this_week >= dormant_min:
                topic_returns.append((t, gap_before_this_week))
        topic_streaks.sort(key=lambda x: -x[1])
        for t, n in topic_streaks[:3]:
            lines.append(f"- Topic '{_topic_label(t)}': matched again, {n} weeks of total coverage")
        topic_returns.sort(key=lambda x: -x[1])
        for t, gap in topic_returns[:3]:
            lines.append(f"- Topic '{_topic_label(t)}': returns to coverage after {gap}-week dormancy")

    if not lines:
        return ""
    return ("Longitudinal context (multi-week entity & topic patterns; cite "
            "explicitly when relevant — do not invent streaks not listed):\n"
            + "\n".join(lines))


# ── discourse-learning signals ────────────────────────────────────────────────

def load_entity_idf(conn):
    """Entity IDF across past weeks of entity_history. Each week = a document.

    idf(e) = log((1 + n_weeks_total) / (1 + n_weeks_with_e))
    Entities never seen receive the maximum IDF (n_weeks_with_e treated as 0).
    Returns ({entity: idf}, max_idf) so callers can default unknown entities.
    """
    n_weeks = conn.execute(
        "SELECT COUNT(DISTINCT week) FROM entity_history"
    ).fetchone()[0] or 0
    if n_weeks == 0:
        return {}, math.log((1 + 0) / (1 + 0) + 1.0)  # placeholder positive value
    rows = conn.execute(
        "SELECT entity, COUNT(DISTINCT week) FROM entity_history GROUP BY entity"
    ).fetchall()
    idf = {e: math.log((1 + n_weeks) / (1 + df)) for e, df in rows}
    max_idf = math.log((1 + n_weeks) / 1.0)
    return idf, max_idf


def _information_richness(idxs, items, entity_idf, max_idf):
    """Per-cluster signal combining entity-type diversity, factual density,
    entity specificity (mean IDF), and lexical cohesion of the longest body.

    Cohesion (new, TREC structural-retrieval refactor): average cosine
    similarity between adjacent sentences in the most-content article of
    the cluster. High cohesion = sustained argument / analytical piece;
    low cohesion = listicle / link-dump. Computed once per cluster on the
    longest body (most diagnostic).

    Returns a scalar in [0,1] (each sub-component is in [0,1]; mean is
    later renormalised across clusters in score_clusters)."""
    labels = set()
    factual = 0
    idfs = []
    for i in idxs:
        for ent, label in items[i]["entities"].items():
            labels.add(label)
            if label in ("MONEY", "PERCENT", "DATE"):
                factual += 1
            idfs.append(entity_idf.get(ent, max_idf))
    type_diversity = len(labels & {"ORG", "PERSON", "GPE", "PRODUCT", "MONEY", "PERCENT"}) / 6.0
    factual_density = min(factual, 3) / 3.0
    mean_idf = (sum(idfs) / len(idfs)) if idfs else 0.0
    # Pre-normalise specificity by max_idf so each component is in [0,1] before
    # the per-week renorm in score_clusters.
    specificity = mean_idf / max_idf if max_idf > 0 else 0.0

    # Lexical cohesion of the longest body in the cluster (most likely to
    # be the analytical piece if one exists).
    longest_body = ""
    for i in idxs:
        body = items[i].get("body") or ""
        if len(body) > len(longest_body):
            longest_body = body
    cohesion = _lexical_cohesion(longest_body)

    return (type_diversity + factual_density + specificity + cohesion) / 4.0


# Anchor invariant: aspects are LLM-derived FROM USER_PROFILE. They sharpen
# coverage within the profile's space, never redirect it.
def _profile_hash():
    return hashlib.sha256(USER_PROFILE.strip().encode()).hexdigest()[:16]


def get_profile_aspects(conn):
    """5-9 short aspect labels covering USER_PROFILE's discourse space.

    Cached in profile_aspects keyed by hash(USER_PROFILE); regenerated only when
    the profile text changes."""
    h = _profile_hash()
    rows = conn.execute(
        "SELECT aspect, descriptor, profile_hash FROM profile_aspects"
    ).fetchall()
    if rows and all(r[2] == h for r in rows):
        return [(r[0], r[1]) for r in rows]

    client = anthropic.Anthropic()
    msg = _retry(lambda: client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=(
            "You decompose a user's interest profile into 5-9 distinct aspects "
            "that together cover the discourse space the profile describes. "
            "Each aspect has a short label and a one-line descriptor of search "
            "terms / concepts that signal its presence in news articles. "
            "Output strict JSON: {\"aspects\": [{\"label\": \"...\", "
            "\"descriptor\": \"...\"}, ...]}. No prose."
        ),
        messages=[{"role": "user", "content": (
            "Decompose this profile into 5-9 aspects:\n\n" + USER_PROFILE.strip()
        )}],
    ))
    text = msg.content[0].text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(m.group(0) if m else text)
    aspects = [(a["label"].strip(), a["descriptor"].strip())
               for a in data.get("aspects", []) if a.get("label") and a.get("descriptor")]
    if not aspects:
        raise RuntimeError("aspect decomposition returned no aspects")

    # Wipe old cache and write new (hash changed or first run).
    conn.execute("DELETE FROM profile_aspects")
    conn.executemany(
        "INSERT INTO profile_aspects(aspect, descriptor, profile_hash, created_at) VALUES(?,?,?,?)",
        [(label, desc, h, now_iso()) for label, desc in aspects],
    )
    print(f"profile_aspects regenerated: {len(aspects)} aspects (hash {h})")
    return aspects


# Anchor invariant: exclusion aspects are LLM-derived FROM USER_PROFILE's
# "Not relevant" clause. They sharpen what to exclude WITHIN the profile's
# defined space; they cannot redirect the discourse model away from
# USER_PROFILE itself. The persistence layer (in run()) adds an unsupervised
# content-derived signal once history is deep enough.
def get_exclusion_aspects(conn):
    """4-6 short aspect labels covering what USER_PROFILE rules out.

    Cached in profile_exclusions keyed by hash(USER_PROFILE); regenerated only
    when the profile text changes. Symmetric to get_profile_aspects but
    derived from the 'Not relevant' clause."""
    h = _profile_hash()
    rows = conn.execute(
        "SELECT aspect, descriptor, profile_hash FROM profile_exclusions"
    ).fetchall()
    if rows and all(r[2] == h for r in rows):
        return [(r[0], r[1]) for r in rows]

    client = anthropic.Anthropic()
    msg = _retry(lambda: client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=(
            "You decompose a user's interest profile into 4-6 distinct EXCLUSION "
            "aspects that describe the kinds of news stories the profile rules "
            "out as off-topic. Use the 'Not relevant' clause and any other "
            "exclusionary themes you can infer. Each aspect has a short label "
            "(2-4 words) and a one-line descriptor of search terms / concepts "
            "that signal its presence in news articles. "
            "Output strict JSON: {\"aspects\": [{\"label\": \"...\", "
            "\"descriptor\": \"...\"}, ...]}. No prose."
        ),
        messages=[{"role": "user", "content": (
            "From this profile, list 4-6 exclusion aspects (what should NOT be "
            "covered):\n\n" + USER_PROFILE.strip()
        )}],
    ))
    text = msg.content[0].text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(m.group(0) if m else text)
    aspects = [(a["label"].strip(), a["descriptor"].strip())
               for a in data.get("aspects", []) if a.get("label") and a.get("descriptor")]
    if not aspects:
        raise RuntimeError("exclusion-aspect decomposition returned no aspects")

    conn.execute("DELETE FROM profile_exclusions")
    conn.executemany(
        "INSERT INTO profile_exclusions(aspect, descriptor, profile_hash, created_at) VALUES(?,?,?,?)",
        [(label, desc, h, now_iso()) for label, desc in aspects],
    )
    print(f"profile_exclusions regenerated: {len(aspects)} aspects (hash {h})")
    return aspects


def save_coverage_ledger(conn, week, aspect_coverage):
    conn.executemany(
        "INSERT OR REPLACE INTO coverage_ledger(week, aspect, coverage) VALUES(?,?,?)",
        [(week, a, float(v)) for a, v in aspect_coverage.items()],
    )


def compute_coverage_debt(conn, aspects, decay=0.7, window=6, threshold=0.4):
    """Decayed count of recent weeks per aspect that fell below `threshold`.

    Older weeks contribute less (decay^age). A high debt means the aspect has
    been systematically under-covered and should boost any cluster that
    finally touches it."""
    weeks = [r[0] for r in conn.execute(
        "SELECT DISTINCT week FROM coverage_ledger ORDER BY week DESC LIMIT ?",
        (window,),
    ).fetchall()]
    if not weeks:
        return {a: 0.0 for a, _ in aspects}
    debt = {a: 0.0 for a, _ in aspects}
    for age, wk in enumerate(weeks):
        rows = conn.execute(
            "SELECT aspect, coverage FROM coverage_ledger WHERE week=?",
            (wk,),
        ).fetchall()
        seen = {a: c for a, c in rows}
        for a, _ in aspects:
            cov = seen.get(a, 0.0)
            if cov < threshold:
                debt[a] += decay ** age
    return debt


# ── persistent topic bank (A4) ────────────────────────────────────────────────

def load_topic_bank(conn):
    """Return list of bank topics with metadata. Each entry:
    {topic_id, centroid (sparse), vocab (list), mass, first_week, last_week,
    weeks_seen}."""
    rows = conn.execute(
        "SELECT topic_id, centroid, vocab, mass, first_week, last_week, weeks_seen "
        "FROM topic_bank"
    ).fetchall()
    out = []
    for tid, cent_b, vocab_b, mass, fw, lw, ws in rows:
        out.append({
            "topic_id": tid,
            "centroid": pickle.loads(cent_b),
            "vocab": pickle.loads(vocab_b),
            "mass": mass,
            "first_week": fw,
            "last_week": lw,
            "weeks_seen": ws,
        })
    return out


def _week_to_date(week_str):
    """Parse '%Y-W%V' to a Monday datetime (ISO week)."""
    try:
        return datetime.strptime(week_str + "-1", "%G-W%V-%u")
    except ValueError:
        return None


def _weeks_between(later, earlier):
    a, b = _week_to_date(later), _week_to_date(earlier)
    if a is None or b is None:
        return 0
    return max(0, int((a - b).days / 7))


def _project_bank(bank, new_vocab):
    """Project each bank topic's centroid into new_vocab. Returns
    (matrix, [bank_indices_present]) — matrix has one row per topic with at
    least one overlapping term; bank entries with zero overlap are dropped."""
    if not bank:
        return None, []
    rows, present = [], []
    for k, t in enumerate(bank):
        c = _project_centroids(t["centroid"], t["vocab"], new_vocab)
        if c is None:
            continue
        rows.append(c)
        present.append(k)
    if not rows:
        return None, []
    return sp.vstack(rows), present


def _truncate_centroid(centroid, top_n=200):
    """Sparsity constraint after the alpha-blend in update_topic_bank.

    Exponential moving averages of sparse TF-IDF vectors accumulate tiny
    nonzeros on every term that has ever appeared in any blended week.
    Over time this blurs the topic centroid into a near-uniform distribution
    ("grey noise"), defeating both novelty and persistence comparisons.

    Truncating to the top-N highest-magnitude terms and re-normalising to
    L2=1 preserves the topic's semantic core while bounding the support.
    `top_n=200` is large enough to cover headline+body bigrams of a coherent
    multi-week topic and small enough to keep the centroid recognisably
    sparse."""
    if sp.issparse(centroid):
        arr = np.asarray(centroid.todense()).ravel()
    else:
        arr = np.asarray(centroid).ravel()
    if arr.size <= top_n:
        return centroid if sp.issparse(centroid) else sp.csr_matrix(arr.reshape(1, -1))
    # argpartition is O(n) — much cheaper than full argsort at this scale.
    keep = np.argpartition(-np.abs(arr), top_n)[:top_n]
    truncated = np.zeros_like(arr)
    truncated[keep] = arr[keep]
    norm = float(np.sqrt(np.dot(truncated, truncated)))
    if norm > 0:
        truncated = truncated / norm
    return sp.csr_matrix(truncated.reshape(1, -1))


def update_topic_bank(conn, top, vec, week, alpha=0.7, match_threshold=0.3,
                      decay=0.8, mass_floor=0.05, stale_weeks=12,
                      centroid_top_n=200):
    """Merge or spawn topics in topic_bank from this week's top clusters.

    Each cluster in `top` already carries a tentative `matched_topic_id` and
    `matched_sim` set by score_clusters; this function commits the matches,
    spawns rows for unmatched clusters, decays all topics, and prunes stale
    rows. Writes the final `topic_id` back onto each cluster dict."""
    inv_vocab = [None] * len(vec.vocabulary_)
    for term, idx in vec.vocabulary_.items():
        inv_vocab[idx] = term

    for c in top:
        mid = c.get("matched_topic_id")
        sim = c.get("matched_sim", 0.0)
        if mid is not None and sim >= match_threshold:
            row = conn.execute(
                "SELECT centroid, vocab, mass, weeks_seen FROM topic_bank WHERE topic_id=?",
                (mid,),
            ).fetchone()
            if row is None:
                mid = None
            else:
                old_cent, old_vocab, mass, weeks_seen = (
                    pickle.loads(row[0]), pickle.loads(row[1]), row[2], row[3]
                )
                projected_old = _project_centroids(old_cent, old_vocab, vec.vocabulary_)
                if projected_old is None:
                    merged = c["vec"]
                else:
                    merged = alpha * projected_old + (1 - alpha) * c["vec"]
                # Sparsity constraint — see _truncate_centroid docstring.
                merged = _truncate_centroid(merged, top_n=centroid_top_n)
                conn.execute(
                    "UPDATE topic_bank SET centroid=?, vocab=?, mass=?, "
                    "last_week=?, weeks_seen=? WHERE topic_id=?",
                    (pickle.dumps(merged), pickle.dumps(inv_vocab),
                     mass + 1.0, week, weeks_seen + 1, mid),
                )
                c["topic_id"] = mid
                continue
        # spawn new topic
        cur = conn.execute(
            "INSERT INTO topic_bank(centroid, vocab, mass, first_week, last_week, weeks_seen) "
            "VALUES(?,?,?,?,?,?)",
            (pickle.dumps(c["vec"]), pickle.dumps(inv_vocab),
             1.0, week, week, 1),
        )
        c["topic_id"] = cur.lastrowid

    # decay all and prune
    conn.execute("UPDATE topic_bank SET mass = mass * ?", (decay,))
    # delete stale (haven't appeared in stale_weeks weeks) or below mass floor
    rows = conn.execute(
        "SELECT topic_id, last_week FROM topic_bank"
    ).fetchall()
    to_delete = [
        (tid,) for tid, lw in rows
        if _weeks_between(week, lw) > stale_weeks
    ]
    if to_delete:
        conn.executemany("DELETE FROM topic_bank WHERE topic_id=?", to_delete)
    conn.execute("DELETE FROM topic_bank WHERE mass < ?", (mass_floor,))


# ── adaptive weights (A5) ─────────────────────────────────────────────────────

def load_weights(conn):
    row = conn.execute(
        "SELECT weights FROM scorer_weights ORDER BY week DESC LIMIT 1"
    ).fetchone()
    if not row:
        return dict(DEFAULT_WEIGHTS)
    try:
        w = json.loads(row[0])
        # backfill any missing keys with defaults
        return {k: float(w.get(k, DEFAULT_WEIGHTS[k])) for k in WEIGHT_KEYS}
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def _enforce_floor(weights):
    """Floor relevance, renormalise the rest to keep sum == 1.0.

    Anchor invariant: relevance is the link from learned scoring to USER_PROFILE.
    Floored so adaptive tuning can never redirect the discourse model."""
    w = dict(weights)
    rel = max(w.get("relevance", 0.0), RELEVANCE_FLOOR)
    rest_keys = [k for k in WEIGHT_KEYS if k != "relevance"]
    rest_sum = sum(max(w.get(k, 0.0), 0.0) for k in rest_keys)
    remainder = max(1.0 - rel, 0.0)
    if rest_sum <= 0:
        # degenerate; fall back to defaults for the non-relevance terms
        rest = {k: DEFAULT_WEIGHTS[k] for k in rest_keys}
        rest_sum = sum(rest.values())
        for k in rest_keys:
            w[k] = rest[k] * remainder / rest_sum
    else:
        for k in rest_keys:
            w[k] = max(w.get(k, 0.0), 0.0) * remainder / rest_sum
    w["relevance"] = rel
    return w


_TUNE_KEYS = [
    "coverage", "prior", "novelty", "relevance",
    "entity_signal", "trend", "richness", "coverage_gap",
]

def tune_weights(conn, min_weeks=10):
    """Fit logistic regression of signals → topic-persistence and blend the
    learned weights with the current weights. Skips if history is shallow or
    the persistence label is degenerate. Writes a row to scorer_weights and
    logs old vs new.

    Only the 8 signals stored in cluster_signals are fitted (_TUNE_KEYS).
    persistence/source_breadth/recency are carried over from current weights
    unchanged — persistence would leak (it is bank-derived, same axis as the
    label), and the other two are recency heuristics without historical rows.

    Persistence label is leak-free: a row at (week=wk, topic_id=t) "persisted"
    iff that same topic_id appears in cluster_signals for any week > wk."""
    rows = conn.execute(
        "SELECT cs.week, cs.topic_id, cs.coverage, cs.prior, cs.novelty, "
        "cs.relevance, cs.entity_signal, cs.trend, cs.richness, cs.coverage_gap "
        "FROM cluster_signals cs "
        "WHERE cs.topic_id IS NOT NULL"
    ).fetchall()
    weeks_in_data = {r[0] for r in rows}
    if len(weeks_in_data) < min_weeks:
        print(f"tune_weights: skip ({len(weeks_in_data)} weeks < {min_weeks})")
        return None

    later_weeks = {}
    for r in rows:
        tid = r[1]
        wk = r[0]
        if tid not in later_weeks:
            later_weeks[tid] = set()
        later_weeks[tid].add(wk)

    X, y = [], []
    for r in rows:
        wk, tid = r[0], r[1]
        signals = list(r[2:10])
        persisted = int(any(w > wk for w in later_weeks.get(tid, set())))
        X.append(signals)
        y.append(persisted)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    if len(set(y.tolist())) < 2:
        print(f"tune_weights: skip (degenerate label, {y.sum()}/{len(y)} positive)")
        return None

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=500)
    lr.fit(X, y)
    coefs = lr.coef_[0]
    pos = np.clip(coefs, 0, None)
    if pos.sum() <= 0:
        print("tune_weights: skip (no positive coefficients)")
        return None

    current = load_weights(conn)
    learned_partial = {k: float(pos[i] / pos.sum()) for i, k in enumerate(_TUNE_KEYS)}
    frozen_keys = [k for k in WEIGHT_KEYS if k not in learned_partial]
    frozen_total = sum(current.get(k, 0.0) for k in frozen_keys)
    fit_total = 1.0 - frozen_total
    learned = {k: learned_partial.get(k, 0.0) * max(fit_total, 1e-9) for k in WEIGHT_KEYS}
    for k in frozen_keys:
        learned[k] = current.get(k, DEFAULT_WEIGHTS[k])

    blended = {k: 0.7 * current[k] + 0.3 * learned[k] for k in WEIGHT_KEYS}
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    final = _enforce_floor(blended)

    week = datetime.now(timezone.utc).strftime("%Y-W%V")
    conn.execute(
        "INSERT OR REPLACE INTO scorer_weights(week, weights, created_at) VALUES(?,?,?)",
        (week, json.dumps(final), now_iso()),
    )
    print(f"tune_weights: old={current} -> new={final}")
    return final


def save_cluster_signals(conn, week, top):
    """Persist per-cluster signal vector + score so A5 can fit retrospectively.
    Keyed by (week, topic_id); topic_id is set by update_topic_bank earlier."""
    rows = [
        (week, c.get("topic_id"),
         float(c["signals"].get("coverage", 0.0)),
         float(c["signals"].get("prior", 0.0)),
         float(c["signals"].get("novelty", 0.0)),
         float(c["signals"].get("relevance", 0.0)),
         float(c["signals"].get("entity_signal", 0.0)),
         float(c["signals"].get("trend", 0.0)),
         float(c["signals"].get("richness", 0.0)),
         float(c["signals"].get("coverage_gap", 0.0)),
         float(c["signals"].get("profile_exclusion", 0.0)),
         float(c["signals"].get("persistence_rate", -1.0)),
         float(c["signals"].get("persistence", 0.0)),
         float(c["signals"].get("source_breadth", 0.0)),
         float(c["signals"].get("recency", 0.0)),
         float(c["score"]))
        for c in top if c.get("topic_id") is not None
    ]
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO cluster_signals(week, topic_id, coverage, prior, "
            "novelty, relevance, entity_signal, trend, richness, coverage_gap, "
            "profile_exclusion, persistence_rate, persistence, source_breadth, "
            "recency_decay, score) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )


# ── sector / category classification ───────────────────────────────────────────

def classify_sector(idxs, items):
    """Dominant sector across a cluster's items (majority of sources)."""
    counts = defaultdict(int)
    for i in idxs:
        counts[SOURCE_SECTOR.get(items[i]["source"], DEFAULT_SECTOR)] += 1
    # tie-break by SECTOR_ORDER so output is stable
    return max(counts, key=lambda s: (counts[s], -SECTOR_ORDER.index(s)
               if s in SECTOR_ORDER else 0))


def classify_category(idxs, items):
    """Kind of development, by keyword cues over the cluster's title+body text.

    Falls back to DEFAULT_CATEGORY ('market') when no cue fires."""
    text = " ".join(
        f"{items[i]['title']} {(items[i]['body'] or '')[:300]}" for i in idxs
    ).lower()
    best, best_hits = DEFAULT_CATEGORY, 0
    for cat, patterns in CATEGORY_PATTERNS.items():
        hits = sum(text.count(p) for p in patterns)
        if hits > best_hits:
            best, best_hits = cat, hits
    return best


# ── cluster scoring ───────────────────────────────────────────────────────────

def _normalize(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


def _cluster_top_entities(idxs, items, velocities, n=5):
    counts = defaultdict(int)
    for i in idxs:
        for e, label in items[i]["entities"].items():
            if _is_signal_entity(e, label) and label in {"ORG", "PERSON", "PRODUCT"}:
                counts[e] += 1
    return sorted(
        counts.items(),
        key=lambda x: (x[1], velocities.get(x[0], 0.0)),
        reverse=True,
    )[:n]


def score_clusters(clusters, items, X, vec, velocities, bank, aspects, debt,
                   entity_idf, max_idf, weights, week, exclusion_aspects=None,
                   E=None):
    """Score clusters with the dict-weighted linear model.

    Returns (scored_clusters, aspect_coverage) where aspect_coverage is the
    per-aspect max coverage this week, normalised to [0,1] for the
    coverage_ledger.

    `exclusion_aspects` (optional): list of (label, descriptor) decomposed
    from USER_PROFILE's 'Not relevant' clause. When present, each cluster
    gets a `profile_exclusion` signal used by the pre-MMR off-profile filter.

    `E` (optional): (n_items, d) normalized dense embedding matrix computed
    once by run(). When present, a dense lane is RRF-fused with the lexical
    BM25 lane for both relevance and exclusion (§6.12 hybrid recipe). When
    absent, falls back to pure lexical behavior."""
    medoids = [cluster_medoid(c, X) for c in clusters]
    medoid_vecs = sp.vstack([X[m] for m in medoids])

    corpus = [_norm(f"{a['title']} {a['body']}").split() for a in items]
    bm25 = BM25Okapi(corpus)

    ph = _profile_hash()
    _RRF_K = 60
    n_cl = len(clusters)

    # ── relevance: per-aspect BM25 RRF (lexical lane) ────────────────────────
    rrf_lex = np.zeros(n_cl)
    if aspects:
        for _a, desc in aspects:
            desc_tokens = _norm(desc).split() or _norm(_a).split()
            if not desc_tokens:
                continue
            scores = bm25.get_scores(desc_tokens)
            raw = np.array([scores[m] for m in medoids])
            order = np.argsort(-raw)
            for rank, idx in enumerate(order):
                rrf_lex[idx] += 1.0 / (_RRF_K + rank + 1)
    else:
        user_q = _norm(USER_PROFILE).split()
        raw_bm25 = bm25.get_scores(user_q)
        rrf_lex = np.array([raw_bm25[m] for m in medoids])

    # ── relevance: dense lane (aspect cosine via RRF), fused with lexical ────
    rrf_dense_rel = np.zeros(n_cl)
    aspect_embs = _embed_aspects(aspects, ph) if (E is not None and aspects) else None
    if aspect_embs is not None and E is not None:
        medoid_embs = E[medoids]                       # (n_cl, d)
        for a_idx in range(len(aspects)):
            q_vec = aspect_embs[a_idx]                 # (d,)
            sims = medoid_embs @ q_vec                 # (n_cl,)
            order = np.argsort(-sims)
            for rank, idx in enumerate(order):
                rrf_dense_rel[idx] += 1.0 / (_RRF_K + rank + 1)
        rrf_scores = rrf_lex + rrf_dense_rel
    else:
        rrf_scores = rrf_lex

    relevance = _normalize(rrf_scores)

    sizes = np.array([len(c) for c in clusters])
    coverage = np.log1p(sizes) / np.log1p(max(sizes.max(), 1))

    prior = _normalize(np.array([
        np.mean([SOURCE_PRIORS.get(items[i]["source"], 1.0) for i in c])
        for c in clusters
    ]))

    # Novelty against the persistent topic bank (A4) — replaces the previous
    # one-week diff. Also attach matched_topic_id + matched_sim to each cluster
    # so update_topic_bank can commit merges/spawns after MMR selection.
    matched_ids = [None] * len(clusters)
    matched_sims = np.zeros(len(clusters))
    dormant_bonus = np.zeros(len(clusters))
    bank_matrix, bank_present = _project_bank(bank, vec.vocabulary_)
    if bank_matrix is not None:
        sims = medoid_vecs.dot(bank_matrix.T).toarray()  # (n_clusters, n_topics_present)
        best_idx = sims.argmax(axis=1)
        best_sim = sims.max(axis=1)
        for i in range(len(clusters)):
            t = bank[bank_present[best_idx[i]]]
            matched_ids[i] = t["topic_id"]
            matched_sims[i] = float(best_sim[i])
            # Partial novelty if matched topic has been dormant >= 4 weeks
            if _weeks_between(week, t["last_week"]) >= 4:
                dormant_bonus[i] = 0.3
    novelty = np.clip(1.0 - matched_sims + dormant_bonus, 0.0, 1.0)

    # Entity signal with IDF reweighting (A1) and richness (A2)
    ent_signals, cluster_trends, richness_raw = [], [], []
    for c in clusters:
        ent_counts = defaultdict(int)
        for i in c:
            for e, lbl in items[i]["entities"].items():
                if _is_signal_entity(e, lbl):
                    ent_counts[e] += 1
        signal = 0.0
        for e, cnt in ent_counts.items():
            idf_w = entity_idf.get(e, max_idf)
            boost = 2 if e in KEY_ENTITIES else 1
            signal += cnt * idf_w * boost
        ent_signals.append(signal)
        top5 = sorted(ent_counts, key=ent_counts.get, reverse=True)[:5]
        vels = [velocities.get(e, 0.0) for e in top5]
        cluster_trends.append(sum(vels) / max(len(vels), 1))
        richness_raw.append(_information_richness(c, items, entity_idf, max_idf))

    entity_signal = _normalize(np.array(ent_signals, dtype=float))
    trend_norm = _normalize(np.clip(np.array(cluster_trends, dtype=float), -1, 3))
    richness = _normalize(np.array(richness_raw, dtype=float))

    # Aspect coverage and coverage_gap (A3)
    # Anchor invariant: aspects are derived FROM USER_PROFILE; coverage_gap
    # sharpens *within* the profile's space, never redirects it.
    coverage_gap = np.zeros(len(clusters))
    aspect_coverage = {}
    if aspects:
        per_aspect_medoid_scores = {}
        for a, desc in aspects:
            desc_tokens = _norm(desc).split() or _norm(a).split()
            scores = bm25.get_scores(desc_tokens)
            medoid_scores = np.array([scores[m] for m in medoids])
            per_aspect_medoid_scores[a] = medoid_scores
            if medoid_scores.size:
                # per-aspect coverage this week = max across medoids, normalised
                mx = float(medoid_scores.max())
                aspect_coverage[a] = mx
        # normalise aspect_coverage across aspects to [0,1] for the ledger
        if aspect_coverage:
            mx_all = max(aspect_coverage.values()) or 1.0
            aspect_coverage = {a: v / mx_all for a, v in aspect_coverage.items()}
        # for each cluster, find its best-matching aspect and use that aspect's debt
        gap_raw = np.zeros(len(clusters))
        if per_aspect_medoid_scores:
            stack = np.stack(list(per_aspect_medoid_scores.values()))  # (n_aspects, n_clusters)
            aspect_keys = list(per_aspect_medoid_scores.keys())
            best_aspect_idx = stack.argmax(axis=0)
            for i in range(len(clusters)):
                gap_raw[i] = float(debt.get(aspect_keys[best_aspect_idx[i]], 0.0))
        coverage_gap = _normalize(gap_raw) if gap_raw.max() > 0 else gap_raw

    # Profile-exclusion signal (Layer A of §2.4). Hybrid: lexical BM25 max
    # across exclusion-aspect descriptors, RRF-fused with a dense cosine lane
    # when E is available. The dense lane is what catches celebrity/consumer-
    # tech leakage where BM25 vocabulary fails (evaluation §1). Used as a
    # filter input only — NOT a positive score term.
    profile_exclusion = np.zeros(n_cl)
    if exclusion_aspects:
        excl_lex = np.zeros(n_cl)
        for a, desc in exclusion_aspects:
            desc_tokens = _norm(desc).split() or _norm(a).split()
            if not desc_tokens:
                continue
            scores_excl = bm25.get_scores(desc_tokens)
            arr = np.array([scores_excl[m] for m in medoids])
            excl_lex = np.maximum(excl_lex, arr)

        excl_dense = np.zeros(n_cl)
        excl_embs = _embed_aspects(exclusion_aspects, ph + "_excl") if E is not None else None
        if excl_embs is not None and E is not None:
            medoid_embs = E[medoids]
            for a_idx in range(len(exclusion_aspects)):
                q_vec = excl_embs[a_idx]
                sims = medoid_embs @ q_vec
                excl_dense = np.maximum(excl_dense, sims)

        combined = excl_lex / (excl_lex.max() + 1e-9) + excl_dense / (excl_dense.max() + 1e-9)
        if combined.max() > 0:
            profile_exclusion = _normalize(combined)

    # Persistence rate (Layer B precursor of §2.4). For each cluster's
    # matched topic in the bank, the empirical recurrence rate
    # (weeks_seen / weeks_since_first_seen). Used by the run() filter once
    # ≥ 10 weeks of history exist. -1.0 means "no matched topic yet".
    persistence_rate = -np.ones(len(clusters))
    if bank:
        bank_by_id = {t["topic_id"]: t for t in bank}
        for i in range(len(clusters)):
            mid = matched_ids[i]
            if mid is None or mid not in bank_by_id:
                continue
            t = bank_by_id[mid]
            span = max(_weeks_between(week, t["first_week"]), 1)
            persistence_rate[i] = float(t["weeks_seen"]) / span

    # §B.1 — persistence as a positive score term. The signal is the same
    # weeks_seen / weeks_since_first_seen ratio used by Layer B of the
    # off-profile filter, but flipped into a positive contribution: a cluster
    # whose matched bank topic has earned multiple appearances gets a boost.
    # Net effect with novelty:  flash-in-pan matches lose; multi-week ongoing
    # matches win. Unmatched clusters get 0 (no boost, no penalty).
    persistence_signal = np.clip(persistence_rate, 0.0, 1.0)
    if persistence_signal.max() > 0:
        persistence_signal = _normalize(persistence_signal)

    # §E.1 — source_breadth: # of distinct sources covering this cluster.
    # Wire-service heuristic: the more independent outlets, the more real.
    sb_raw = np.array([
        float(len({items[i]["source"] for i in c})) for c in clusters
    ], dtype=float)
    source_breadth = _normalize(np.log1p(sb_raw))

    # §E.7 — recency: exp(-age_days/τ) where age is the freshest item in the
    # cluster. τ ≈ 4 days inside the 7-day window. Prevents stale items that
    # made it in once from re-surfacing past their welcome.
    now = datetime.now(timezone.utc)
    rec_raw = []
    for c in clusters:
        ages = []
        for i in c:
            age = _item_age_days(items[i].get("ts"), now)
            if age is not None:
                ages.append(age)
        rec_raw.append(np.exp(-(min(ages) if ages else 0.0) / 4.0))
    recency = _normalize(np.array(rec_raw, dtype=float))

    signal_arrays = {
        "coverage": coverage,
        "prior": prior,
        "novelty": novelty,
        "relevance": relevance,
        "entity_signal": entity_signal,
        "trend": trend_norm,
        "richness": richness,
        "coverage_gap": coverage_gap,
        "persistence": persistence_signal,
        "source_breadth": source_breadth,
        "recency": recency,
    }
    score = np.zeros(len(clusters))
    for k, arr in signal_arrays.items():
        score = score + weights.get(k, 0.0) * arr

    order = np.argsort(-score)
    scored = []
    for i in order:
        scored.append({
            "idxs": clusters[i],
            "medoid": medoids[i],
            "score": float(score[i]),
            "vec": medoid_vecs[i],
            "entities": _cluster_top_entities(clusters[i], items, velocities),
            "sector": classify_sector(clusters[i], items),
            "category": classify_category(clusters[i], items),
            "matched_topic_id": matched_ids[i],
            "matched_sim": float(matched_sims[i]),
            "signals": {
                **{k: float(arr[i]) for k, arr in signal_arrays.items()},
                "profile_exclusion": float(profile_exclusion[i]),
                "persistence_rate": float(persistence_rate[i]),
            },
        })
    return scored, aspect_coverage


def mmr_select(scored, k=15, lam=0.65):
    """Fixed-k MMR. Retained for compatibility; run() now uses
    mmr_dynamic_select to avoid an arbitrary cap."""
    if not scored:
        return []
    V = sp.vstack([c["vec"] for c in scored])
    S = V.dot(V.T).toarray()
    base = np.array([c["score"] for c in scored])
    selected, mask = [0], np.ones(len(scored), dtype=bool)
    mask[0] = False
    while len(selected) < min(k, len(scored)):
        pool = np.where(mask)[0]
        max_sim = S[np.ix_(pool, selected)].max(axis=1)
        pick = pool[int(np.argmax(lam * base[mask] - (1 - lam) * max_sim))]
        selected.append(pick)
        mask[pick] = False
    return [scored[i] for i in selected]


def mmr_dynamic_select(scored, lam=0.65, min_k=8, max_k=40, relevance_floor=0.15):
    """MMR over the whole qualifying pool, then cut at the score-knee.

    Floors at min_k (don't render an empty-looking digest on a quiet week),
    ceilings at max_k (cost / readability guard, rarely binds), and unconditionally
    keeps any cluster whose relevance signal exceeds relevance_floor — that
    preserves the USER_PROFILE link even when the knee falls early."""
    if not scored:
        return []
    n = len(scored)
    V = sp.vstack([c["vec"] for c in scored])
    S = V.dot(V.T).toarray()
    base = np.array([c["score"] for c in scored])

    # Walk MMR over the entire pool (no early termination).
    selected, mask = [0], np.ones(n, dtype=bool)
    mask[0] = False
    while mask.any():
        pool = np.where(mask)[0]
        max_sim = S[np.ix_(pool, selected)].max(axis=1)
        pick = pool[int(np.argmax(lam * base[mask] - (1 - lam) * max_sim))]
        selected.append(pick)
        mask[pick] = False

    ordered = [scored[i] for i in selected]
    scores = np.array([c["score"] for c in ordered], dtype=float)

    # Knee: index of the largest drop in the (smoothed) score curve.
    if len(scores) >= 3:
        diffs = -np.diff(scores)  # positive when score falls
        # 3-point smoothing
        if len(diffs) >= 3:
            kernel = np.array([0.25, 0.5, 0.25])
            smooth = np.convolve(diffs, kernel, mode="same")
        else:
            smooth = diffs
        cut = int(np.argmax(smooth)) + 1
    else:
        cut = len(scores)

    cut = max(min_k, min(cut, max_k, len(scores)))

    # Always keep clusters above the relevance floor, even past the knee — they
    # are on-profile signal we don't want to drop just because the curve broke.
    keep_idxs = set(range(cut))
    for i, c in enumerate(ordered[cut:max_k], start=cut):
        if c.get("signals", {}).get("relevance", 0.0) >= relevance_floor:
            keep_idxs.add(i)

    return [ordered[i] for i in sorted(keep_idxs)]


def _filter_off_profile(scored, history_weeks, relevance_floor=0.10,
                        persistence_floor=0.15):
    """Drop clusters whose profile_exclusion signal exceeds relevance AND
    whose relevance falls below the floor — Layer A of §2.4.

    Once history_weeks >= 10, the filter tightens with Layer B: a
    high-exclusion cluster also has to clear the persistence floor (its
    matched topic must have a non-trivial recurrence rate) to survive.
    Topics with no match in the bank (persistence_rate == -1) are treated
    as failing the persistence test only once history is deep enough.

    Additionally, clusters below DUMP_RELEVANCE_FLOOR with non-trivial
    exclusion are dropped regardless of the other gates — these are the
    low-signal off-profile items that leak into the ALL STORIES dump.
    On-profile clusters (profile_exclusion == 0.0) are never dropped by
    this secondary gate so the rule cannot suppress genuine coverage.
    """
    layer_b = history_weeks >= 10
    out = []
    for c in scored:
        s = c.get("signals", {})
        excl = s.get("profile_exclusion", 0.0)
        rel = s.get("relevance", 0.0)
        if excl > rel and rel < relevance_floor:
            if not layer_b:
                continue
            pers = s.get("persistence_rate", -1.0)
            if pers < persistence_floor:
                continue
        if excl > 0.0 and rel < DUMP_RELEVANCE_FLOOR:
            continue
        out.append(c)
    return out


def _title_signature(title):
    """A coarse content-only signature used by _dedupe_top.

    Strips: source/date prefixes ("Cynopsis 05/30/26:"), countdown / deadline
    phrases ("Final 24 hours", "Early Bird ends May 29"), bracketed source
    tags, and trailing "[Source]" attributions. Returns the first 8 normalised
    tokens joined; two headlines with the same prefix tokens collapse to the
    same signature.
    """
    t = title or ""
    # strip leading "Source 05/30/26:" prefix
    t = re.sub(r"^[A-Za-z][A-Za-z0-9 ]{0,30}\s+\d{1,2}/\d{1,2}/\d{2,4}\s*:\s*", "", t)
    # strip bracketed source tags at start
    t = re.sub(r"^\[[^\]]{1,40}\]\s*", "", t)
    # strip countdown / deadline tails
    t = re.sub(r"[—\-:|]\s*(?:final|last|only|just)\s+\d+\s*(?:hours?|days?|hrs?)\s+left.*$", "", t, flags=re.I)
    t = re.sub(r"[—\-:|]\s*(?:early\s+bird|deadline|register|save\s+\$?\d+).*$", "", t, flags=re.I)
    t = re.sub(r"\d+\s*(?:hours?|days?|hrs?)\s+left.*$", "", t, flags=re.I)
    tokens = _norm(t).split()
    return " ".join(tokens[:8])


def _dedupe_top(top, items):
    """Cross-cluster dedup. Each URL and each title signature (see
    _title_signature) appears in at most one cluster — the highest-scoring
    one that holds it (top is already in descending-score order). Clusters
    left with zero items are dropped entirely."""
    seen_urls, seen_sigs = set(), set()
    out = []
    for c in top:
        kept = []
        for i in c["idxs"]:
            url = items[i]["url"]
            sig = _title_signature(items[i]["title"])
            if url in seen_urls or (sig and sig in seen_sigs):
                continue
            seen_urls.add(url)
            if sig:
                seen_sigs.add(sig)
            kept.append(i)
        if kept:
            c["idxs"] = kept
            out.append(c)
    return out


# ── Claude summarization ──────────────────────────────────────────────────────

def _retry(fn, attempts=3, base=2.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(base ** i)


# ── macro-trend memory ────────────────────────────────────────────────────────

def load_macro_context(conn):
    row = conn.execute(
        "SELECT month, summary FROM macro_trends ORDER BY month DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    try:
        age_weeks = (datetime.now() - datetime.strptime(row[0], "%Y-%m")).days / 7
        if age_weeks > 10:
            return None
    except Exception:
        pass
    return row[1]


def generate_macro_trends(conn):
    last = conn.execute(
        "SELECT weeks FROM macro_trends ORDER BY month DESC LIMIT 1"
    ).fetchone()
    last_weeks = set(json.loads(last[0])) if last else set()

    all_week_rows = conn.execute(
        "SELECT week, summary FROM digests WHERE summary IS NOT NULL ORDER BY week DESC LIMIT 8"
    ).fetchall()

    new_rows = [r for r in all_week_rows if r[0] not in last_weeks]
    if len(new_rows) < 4:
        return None

    rows_to_use = all_week_rows[:min(5, len(all_week_rows))]
    context = "\n\n---\n\n".join(f"Week {r[0]}:\n{r[1]}" for r in rows_to_use)

    client = anthropic.Anthropic()
    msg = _retry(lambda: client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=(
            "You are a media industry researcher synthesising longitudinal industry data. "
            "Identify structural shifts, market concentration dynamics, and regulatory "
            "trajectories that span multiple observation periods — not individual events. "
            "Write for an academic and industry professional audience. "
            "Dense, precise prose. Name companies and cite patterns. No hedging."
        ),
        messages=[{"role": "user", "content": (
            "Analyse these consecutive weekly media industry observations. "
            "Write 2 paragraphs identifying the dominant structural trends, "
            "competitive dynamics, and regulatory or technological pressures "
            "that have persisted, intensified, or shifted across these weeks:\n\n"
            + context
        )}],
    ))

    macro_text = msg.content[0].text
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    week_ids = [r[0] for r in rows_to_use]
    conn.execute(
        "INSERT OR REPLACE INTO macro_trends(month, summary, weeks, created_at) VALUES(?,?,?,?)",
        (month, macro_text, json.dumps(week_ids), now_iso()),
    )
    print(f"macro trend generated for {month} covering weeks {week_ids}")
    return macro_text


def get_archive_links(conn, n=8):
    site_bucket = os.environ.get("GCS_SITE_BUCKET", "")
    if not site_bucket:
        return []
    rows = conn.execute(
        "SELECT week FROM digests ORDER BY week DESC LIMIT ?", (n,)
    ).fetchall()
    return [
        (r[0], f"https://storage.googleapis.com/{site_bucket}/digest-{r[0]}.html")
        for r in rows
    ]


# ── Claude summarization ──────────────────────────────────────────────────────

def sectors_present(top):
    """[(sector_key, [clusters])] in SECTOR_ORDER, only sectors with clusters."""
    grouped = defaultdict(list)
    for c in top:
        grouped[c.get("sector", DEFAULT_SECTOR)].append(c)
    return [(s, grouped[s]) for s in SECTOR_ORDER if grouped.get(s)]


def summarise(top, items, macro_context=None, history_depth="warmup",
              longitudinal_context=None):
    """Editorial spotlight: ask the LLM to pick up to 5 most meaningful
    clusters per (sector, category), each summarised in 1-3 sentences,
    referenced by the cluster's `Cn` index so the renderer can attach
    the real article links.

    Returns ``(summary_text, cluster_index)`` where cluster_index maps
    "C1", "C2", … to the cluster dicts the LLM saw. The renderer uses
    this to look picks back up.

    ``history_depth`` is retained for symmetry but no longer drives the
    output format — longitudinal_context being non-empty is what unlocks
    multi-week framing in the LLM's prose.
    """
    client = anthropic.Anthropic()

    # Group clusters by (sector, category) and number them C1, C2, … in
    # sector → category → score-descending order. The numbering is global
    # so a Cn never collides across categories.
    by_sector_cat = defaultdict(lambda: defaultdict(list))
    for c in top:
        sk = c.get("sector", DEFAULT_SECTOR)
        ck = c.get("category", DEFAULT_CATEGORY)
        by_sector_cat[sk][ck].append(c)

    cluster_index = {}
    blocks = []
    counter = 0
    for sector_key in SECTOR_ORDER:
        if sector_key not in by_sector_cat:
            continue
        section_lines = [f"## {SECTORS[sector_key]}"]
        for cat_key, _ in CATEGORIES:
            if cat_key not in by_sector_cat[sector_key]:
                continue
            cat_clusters = sorted(
                by_sector_cat[sector_key][cat_key],
                key=lambda c: -c.get("score", 0.0),
            )
            section_lines.append(f"### {CATEGORY_NAMES[cat_key]}")
            for c in cat_clusters:
                counter += 1
                cid = f"C{counter}"
                cluster_index[cid] = c
                title = items[c["medoid"]]["title"]
                # Sentence-level MMR (replaces blind [:240] lede assumption).
                # Picks the body sentence that maximises relevance to
                # USER_PROFILE while penalising overlap with the title the
                # LLM already sees.
                body = _mmr_sentence(
                    items[c["medoid"]].get("body") or "",
                    title,
                    USER_PROFILE,
                )
                ents = " · ".join(e for e, _ in c["entities"][:4])
                srcs = sorted({items[i]["source"].upper() for i in c["idxs"][:5]})
                section_lines.append(
                    f"{cid} (score {c.get('score', 0.0):.2f}, sources "
                    f"{', '.join(srcs)}): {title}"
                )
                if body:
                    section_lines.append(f"   {body}")
                if ents:
                    section_lines.append(f"   entities: {ents}")
        blocks.append("\n".join(section_lines))

    long_section = (
        f"\n\n{longitudinal_context}\n" if longitudinal_context else ""
    )
    macro_section = (
        "\n\nMacro context from prior weeks (structural trends; cite when "
        f"relevant):\n{macro_context}\n" if macro_context else ""
    )

    system_prompt = (
        "You are a media industry researcher curating an editorial spotlight "
        "for an audience of industry professionals and academics.\n\n"
        "STRICT anti-patterns:\n"
        "- Do NOT open with abstract claims about markets shifting, "
        "industries transforming, or sectors evolving. Lead with the named "
        "entity, deal, figure, or regulator.\n"
        "- Do NOT invent streaks, returns, or multi-week patterns. Only "
        "cite longitudinal context that is explicitly listed in the prompt.\n"
        "- Do NOT reference a Cn that you did not pick, and do not "
        "invent a Cn that does not appear in the source clusters.\n\n"
        "Each pick: 1–3 sentences. Lead with the concrete observable, then "
        "the structural implication. Cite companies and figures precisely. "
        "No hedging. No filler."
    )

    body_instructions = (
        "Below are clustered stories grouped by sector (## heading) and "
        "category (### heading). Each cluster is numbered C{n} with its "
        "score, source list, headline, body snippet, and top entities. "
        "Clusters within each category are listed in descending score "
        "order — the first is the highest-scoring.\n\n"
        "For each (sector, category) you find meaningful, pick UP TO 5 "
        "clusters to spotlight, ordered by your editorial judgement of "
        "importance. Skip a category entirely if nothing rises above noise. "
        "Skip a sector entirely if all its categories are skipped.\n\n"
        "Output strictly:\n\n"
        "## SectorName (display name as shown below)\n"
        "### CategoryName (display name as shown below)\n"
        "- C{n}: 1–3 sentence editorial summary.\n"
        "- C{m}: 1–3 sentence editorial summary.\n\n"
        "Repeat per sector and per category. Preserve each Cn reference "
        "exactly — the renderer parses it to attach links."
        + long_section
        + macro_section
        + "\n\nClusters this week:\n\n"
        + "\n\n".join(blocks)
    )

    msg = _retry(lambda: client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2400,
        system=system_prompt,
        messages=[{"role": "user", "content": body_instructions}],
    ))
    return msg.content[0].text, cluster_index


_CN_RE = re.compile(r"^-\s*(C\d+)\s*:\s*(.+)$", re.IGNORECASE)


def parse_spotlight(summary, cluster_index):
    """Parse the summarise() output back into the per-(sector, category)
    picks the LLM made. Returns a dict:

        {sector_key: {category_key: [(cluster_obj, summary_text), ...]}}

    Robust to a few drift cases: an unmapped sector or category name is
    dropped; a Cn that isn't in cluster_index is dropped; pure prose lines
    between bullets are ignored. The cluster ordering within each category
    is preserved (the LLM's editorial sequence)."""
    out = defaultdict(lambda: defaultdict(list))
    sector_key = None
    category_key = None
    seen_cids = set()
    for line in summary.splitlines():
        s = line.rstrip()
        if not s:
            continue
        if s.startswith("## "):
            sector_key = _sector_key_for_title(s[3:].strip())
            category_key = None
            continue
        if s.startswith("### "):
            cat_name = s[4:].strip()
            category_key = None
            for ck, name in CATEGORIES:
                if name.lower() == cat_name.lower():
                    category_key = ck
                    break
            continue
        m = _CN_RE.match(s.lstrip())
        if not m or sector_key is None or category_key is None:
            continue
        cid = m.group(1).upper()
        if cid in seen_cids:
            continue
        cluster = cluster_index.get(cid)
        if cluster is None:
            continue
        seen_cids.add(cid)
        out[sector_key][category_key].append((cluster, m.group(2).strip()))
    return out


def split_summary_sections(summary):
    """Legacy parser kept for callers that just want '## Sector' chunks
    (currently unused after the spotlight refactor)."""
    sections, title, body = [], None, []
    for line in summary.splitlines():
        if line.startswith("## "):
            if title or body:
                sections.append((title, "\n".join(body).strip()))
            title, body = line[3:].strip(), []
        else:
            body.append(line)
    if title or body:
        sections.append((title, "\n".join(body).strip()))
    return [(t, b) for t, b in sections if b] or [(None, summary.strip())]


# Reverse-lookup from a sector display name back to its SECTOR_ORDER key, so
# render_static/render_email can match an LLM-emitted "## SectorName" heading
# to the cluster bucket for that sector. Case-insensitive, ignores whitespace.
_SECTOR_NAME_TO_KEY = {v.strip().lower(): k for k, v in SECTORS.items()}


def _sector_key_for_title(title):
    if not title:
        return None
    return _SECTOR_NAME_TO_KEY.get(title.strip().lower())


def _bullet_lines(body):
    """Pull out lines that look like list bullets (`- ...` / `* ...` /
    `• ...`), one per element. Strips the bullet marker. Returns [] if the
    body is paragraph-shaped instead."""
    out = []
    for raw in (body or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ", "• ")):
            out.append(line[2:].strip())
        elif line[:2] in ("-\t", "*\t"):
            out.append(line[2:].strip())
    return out


def _summary_section_html(body, *, ul_class="sector-summary",
                          paragraph_class="ssec-para"):
    """Render a parsed summary section body as either a bullet <ul> (preferred,
    matches the new bullet-format prompts) or — if the LLM ignored the bullet
    instruction — a paragraph fallback so the digest still renders."""
    bullets = _bullet_lines(body)
    if bullets:
        items = "".join(f"<li>{b}</li>" for b in bullets)
        return f'<ul class="{ul_class}">{items}</ul>'
    paras = "".join(
        f'<p class="{paragraph_class}">{p.strip()}</p>'
        for p in (body or "").split("\n\n") if p.strip()
    )
    return paras


def _summary_by_sector_key(summary):
    """{sector_key: html_block} from the LLM summary, keyed by SECTOR_ORDER key.

    Unmatched section titles (LLM emitted a name that doesn't map to SECTORS)
    are dropped silently — the renderer falls back to "no summary for this
    sector" rather than rendering an orphaned block above the wrong stories."""
    out = {}
    for title, body in split_summary_sections(summary):
        key = _sector_key_for_title(title)
        if key is None or not body:
            continue
        out[key] = _summary_section_html(body)
    return out


def _cluster_distinct_links(c, items, cap=5):
    """Distinct (url, source, title) tuples within a cluster, preserving
    original ordering of c['idxs'] and capped at `cap`. Used by render_*
    to decide singleton vs multi-link layout AND to render the actual link
    list — single source of truth."""
    seen = set()
    out = []
    for i in c["idxs"]:
        url = items[i]["url"]
        if url in seen:
            continue
        seen.add(url)
        out.append((url, items[i]["source"], items[i]["title"]))
        if len(out) >= cap:
            break
    return out


# ── email output ──────────────────────────────────────────────────────────────

def _sector_breakdown(sector_clusters, picks_for_sector):
    """Helper: split a sector's clusters into spotlit (with LLM summary)
    and dump (by category, score-descending). Used by both renderers."""
    spotlit_ids = set()
    spotlight_by_cat = defaultdict(list)  # {cat_key: [(cluster, summary_text)]}
    for cat_key, picks in (picks_for_sector or {}).items():
        for cluster, summary_text in picks:
            if id(cluster) in spotlit_ids:
                continue
            spotlit_ids.add(id(cluster))
            spotlight_by_cat[cat_key].append((cluster, summary_text))
    dump_by_cat = defaultdict(list)
    for c in sector_clusters:
        if id(c) in spotlit_ids:
            continue
        dump_by_cat[c.get("category", DEFAULT_CATEGORY)].append(c)
    for cat_key in list(dump_by_cat.keys()):
        dump_by_cat[cat_key].sort(key=lambda c: -c.get("score", 0.0))
    return spotlight_by_cat, dump_by_cat


def render_email(summary, cluster_index, top, items, macro=None):
    date_str = datetime.now().strftime("%b %d, %Y")
    parts = [f"<h2>Weekly Media Industry News — {date_str}</h2>"]

    picks = parse_spotlight(summary, cluster_index)

    for sector_key, sector_clusters in sectors_present(top):
        parts.append(
            f"<h3 style='border-bottom:1px solid #ddd;padding-bottom:4px;"
            f"margin-top:28px;color:#f59e0b'>{SECTORS[sector_key]} "
            f"<small style='color:#888;font-weight:400'>"
            f"({len(sector_clusters)} stories)</small></h3>"
        )

        spotlight_by_cat, dump_by_cat = _sector_breakdown(
            sector_clusters, picks.get(sector_key, {})
        )

        # SPOTLIGHT — LLM-picked, grouped by category
        for cat_key, _ in CATEGORIES:
            if cat_key not in spotlight_by_cat:
                continue
            parts.append(
                f"<h4 style='color:#60a5fa;font-size:12px;text-transform:"
                f"uppercase;letter-spacing:.05em;margin:14px 0 6px'>"
                f"{CATEGORY_NAMES[cat_key]}</h4>"
            )
            parts.append("<ul style='margin:0 0 8px;padding-left:18px'>")
            for cluster, summary_text in spotlight_by_cat[cat_key]:
                links = _cluster_distinct_links(cluster, items, cap=5)
                if not links:
                    continue
                parts.append(
                    f"<li style='margin-bottom:10px'>"
                    f"<div style='color:#111;font-size:14px'>{summary_text}</div>"
                    f"<div style='font-size:12px;margin-top:4px'>"
                )
                url, src, title = links[0]
                parts.append(
                    f'↳ <a href="{url}"><b>{src.upper()}:</b> {title}</a>'
                )
                for u, s, t in links[1:]:
                    parts.append(
                        f'<br>↳ <a href="{u}" style="color:#666"><b>{s.upper()}:</b> {t}</a>'
                    )
                parts.append("</div></li>")
            parts.append("</ul>")

        # SEPARATOR between spotlight and dump (only if both present)
        if spotlight_by_cat and dump_by_cat:
            parts.append(
                "<hr style='border:none;border-top:1px dashed #ccc;margin:14px 0'>"
            )

        # DUMP — mechanical, grouped by category, score-descending
        if dump_by_cat:
            parts.append(
                "<div style='color:#888;font-size:11px;text-transform:uppercase;"
                "letter-spacing:.05em;margin-bottom:6px'>All stories</div>"
            )
            for cat_key, _ in CATEGORIES:
                if cat_key not in dump_by_cat:
                    continue
                cat_clusters = dump_by_cat[cat_key]
                parts.append(
                    f"<div style='color:#444;font-size:11px;margin:10px 0 4px;"
                    f"font-weight:600'>{CATEGORY_NAMES[cat_key]} "
                    f"<span style='color:#888;font-weight:400'>"
                    f"({len(cat_clusters)})</span></div>"
                )
                parts.append("<ul style='margin:0 0 6px;padding-left:18px'>")
                for c in cat_clusters:
                    links = _cluster_distinct_links(c, items, cap=4)
                    if not links:
                        continue
                    url, src, title = links[0]
                    parts.append(
                        f'<li style="margin-bottom:3px;font-size:13px">'
                        f'<a href="{url}"><b>{src.upper()}:</b> {title}</a>'
                    )
                    for u, s, t in links[1:]:
                        parts.append(
                            f'<br><span style="font-size:12px;color:#666">↳ '
                            f'<a href="{u}" style="color:#666"><b>{s.upper()}:</b> '
                            f'{t}</a></span>'
                        )
                    parts.append("</li>")
                parts.append("</ul>")

    if macro:
        month_label = datetime.now().strftime("%B %Y")
        parts.append(
            f"<h3 style='color:#888;font-size:13px;margin-top:24px'>"
            f"Macro Trends — {month_label}</h3>"
        )
        parts.append(
            "<p style='color:#aaa;font-size:13px;border-left:2px solid #444;"
            "padding-left:12px'>"
            + macro.replace("\n\n", "</p><p style='color:#aaa;font-size:13px;"
                            "border-left:2px solid #444;padding-left:12px'>")
            + "</p>"
        )

    return "\n".join(parts)


def send_email(html):
    msg = MIMEText(html, "html")
    msg["Subject"] = f"Weekly Media Industry News — {datetime.now().strftime('%b %d')}"
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = os.environ["DIGEST_TO"]
    with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], 465) as s:
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        s.send_message(msg)


# ── static site ───────────────────────────────────────────────────────────────

def render_static(summary, cluster_index, top, items, week, macro=None,
                  archive_links=None):
    date_str = datetime.now().strftime("%B %d, %Y")

    picks = parse_spotlight(summary, cluster_index)

    sector_blocks_html = []
    for sector_key, sector_clusters in sectors_present(top):
        spotlight_by_cat, dump_by_cat = _sector_breakdown(
            sector_clusters, picks.get(sector_key, {})
        )
        block = [
            f'<section class="sec">'
            f'<h2 class="sector">{SECTORS[sector_key]} '
            f'<span class="sn">{len(sector_clusters)} stories</span></h2>'
        ]

        # SPOTLIGHT
        if spotlight_by_cat:
            spot_html = ['<div class="spotlight">']
            for cat_key, _ in CATEGORIES:
                if cat_key not in spotlight_by_cat:
                    continue
                spot_html.append(
                    f'<h3 class="cat-block">{CATEGORY_NAMES[cat_key]}</h3>'
                )
                spot_html.append('<ul class="spotlight-list">')
                for cluster, summary_text in spotlight_by_cat[cat_key]:
                    links = _cluster_distinct_links(cluster, items, cap=5)
                    if not links:
                        continue
                    primary_url, primary_src, primary_title = links[0]
                    spot_html.append(
                        f'<li class="spot-item">'
                        f'<div class="spot-summary">{summary_text}</div>'
                        f'<div class="spot-links">'
                        f'<a href="{primary_url}" target="_blank">'
                        f'<span class="src">{primary_src.upper()}</span> '
                        f'{primary_title}</a>'
                    )
                    for u, s, t in links[1:]:
                        spot_html.append(
                            f'<div class="also">↳ <a href="{u}" target="_blank">'
                            f'<span class="src">{s.upper()}</span> {t}</a></div>'
                        )
                    spot_html.append('</div></li>')
                spot_html.append('</ul>')
            spot_html.append('</div>')
            block.append("".join(spot_html))

        # SEPARATOR
        if spotlight_by_cat and dump_by_cat:
            block.append('<hr class="sep">')

        # DUMP
        if dump_by_cat:
            dump_html = [
                '<div class="dump">',
                '<h4 class="dump-header">All stories</h4>',
            ]
            for cat_key, _ in CATEGORIES:
                if cat_key not in dump_by_cat:
                    continue
                cat_clusters = dump_by_cat[cat_key]
                dump_html.append(
                    f'<h5 class="dump-cat">{CATEGORY_NAMES[cat_key]} '
                    f'<span class="sn">{len(cat_clusters)}</span></h5>'
                )
                dump_html.append('<ul class="dump-list">')
                for c in cat_clusters:
                    links = _cluster_distinct_links(c, items, cap=4)
                    if not links:
                        continue
                    primary_url, primary_src, primary_title = links[0]
                    dump_html.append(
                        f'<li>'
                        f'<a href="{primary_url}" target="_blank">'
                        f'<span class="src">{primary_src.upper()}</span> '
                        f'{primary_title}</a>'
                    )
                    for u, s, t in links[1:]:
                        dump_html.append(
                            f'<div class="also">↳ <a href="{u}" target="_blank">'
                            f'<span class="src">{s.upper()}</span> {t}</a></div>'
                        )
                    dump_html.append('</li>')
                dump_html.append('</ul>')
            dump_html.append('</div>')
            block.append("".join(dump_html))

        block.append('</section>')
        sector_blocks_html.append("".join(block))

    macro_section = ""
    if macro:
        month_label = datetime.now().strftime("%B %Y")
        macro_html = "".join(
            f"<p>{p.strip()}</p>" for p in macro.split("\n\n") if p.strip()
        )
        macro_section = (
            f'<section class="macro">'
            f'<h2>Macro Trends &mdash; {month_label}</h2>'
            f'<div class="macro-body">{macro_html}</div>'
            f'</section>'
        )

    archive_section = ""
    if archive_links:
        links_html = "".join(
            f'<li><a href="{url}">{wk}</a></li>' for wk, url in archive_links
        )
        archive_section = f'<nav class="archive"><h2>Archive</h2><ul>{links_html}</ul></nav>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weekly Media Industry News — {date_str}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#e5e5e5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:15px;line-height:1.65;padding:0 16px 64px;max-width:860px;margin:0 auto}}
a{{color:#f59e0b;text-decoration:none}}a:hover{{text-decoration:underline}}
header{{border-bottom:2px solid #f59e0b;padding:24px 0 14px;margin-bottom:28px}}
header h1{{font-size:clamp(18px,4vw,28px);letter-spacing:.02em;color:#f59e0b}}
.wk{{color:#555;font-size:12px;margin-top:4px}}
.macro{{background:#0e0e14;border-left:3px solid #6366f1;padding:16px 20px;border-radius:4px;margin:28px 0}}
.macro h2{{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#6366f1;margin-bottom:12px}}
.macro-body p{{color:#9ca3af;font-size:14px;margin-bottom:8px;line-height:1.6}}
.macro-body p:last-child{{margin-bottom:0}}
h2{{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#444;margin:28px 0 14px}}
h2.sector{{font-size:18px;text-transform:none;letter-spacing:.01em;color:#f59e0b;border-bottom:1px solid #2a2a2a;padding-bottom:6px;margin:36px 0 12px;font-weight:600}}
h2.sector .sn{{color:#555;font-size:12px;font-weight:400;margin-left:8px}}
/* SPOTLIGHT — LLM editorial picks, grouped by category */
.spotlight{{margin:6px 0 12px}}
.spotlight h3.cat-block{{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#60a5fa;margin:14px 0 6px;border:none;padding:0;font-weight:600}}
ul.spotlight-list{{list-style:none;padding:0;margin:0 0 10px}}
ul.spotlight-list li.spot-item{{padding:8px 0;border-top:1px solid #1a1a1a}}
ul.spotlight-list li.spot-item:first-child{{border-top:none}}
.spot-summary{{color:#e5e5e5;font-size:14.5px;line-height:1.55;margin-bottom:6px}}
.spot-links{{font-size:13px;color:#aaa;padding-left:2px}}
.spot-links a{{color:#e5e5e5}}
.spot-links .also{{font-size:12.5px;color:#888;margin-top:2px;padding-left:8px}}
.spot-links .also a{{color:#bbb}}

/* SEPARATOR between spotlight and dump */
hr.sep{{border:none;border-top:1px dashed #2a2a2a;margin:14px 0 10px}}

/* DUMP — mechanical, category-grouped, score-descending */
.dump h4.dump-header{{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#666;margin:4px 0 8px;font-weight:600;border:none;padding:0}}
.dump h5.dump-cat{{font-size:11px;color:#888;margin:10px 0 4px;font-weight:600}}
.dump h5.dump-cat .sn{{color:#555;font-weight:400;font-size:10.5px;margin-left:4px}}
ul.dump-list{{list-style:none;padding:0;margin:0 0 6px}}
ul.dump-list li{{border-top:1px solid #161616;padding:5px 0;font-size:12.5px;line-height:1.45;color:#999}}
ul.dump-list li:first-child{{border-top:none}}
ul.dump-list li a{{color:#cfcfcf}}
ul.dump-list li .also{{font-size:11.5px;color:#666;padding-left:8px;margin-top:1px}}
ul.dump-list li .also a{{color:#999}}

.src{{color:#f59e0b;font-size:10px;font-weight:700;margin-right:6px;letter-spacing:.02em}}
.archive{{margin-top:48px;padding-top:24px;border-top:1px solid #1a1a1a}}
.archive h2{{margin-bottom:10px}}
.archive ul{{display:flex;flex-wrap:wrap;gap:8px}}
.archive li{{border:none;padding:0}}
.archive a{{font-size:12px;color:#555;border:1px solid #222;padding:3px 10px;border-radius:4px}}
.archive a:hover{{color:#f59e0b;border-color:#f59e0b}}
footer{{margin-top:32px;color:#333;font-size:11px;text-align:center}}
@media(max-width:580px){{.ch{{flex-direction:column}}}}
</style>
</head>
<body>
<header>
  <h1>Weekly Media Industry News</h1>
  <div class="wk">Week {week} &nbsp;·&nbsp; {date_str} &nbsp;·&nbsp; {len(top)} stories</div>
</header>
{"".join(sector_blocks_html)}
{macro_section}
{archive_section}
<footer>Weekly Media Industry News &nbsp;·&nbsp; <a href="coverage.html">Coverage Analysis</a> &nbsp;·&nbsp; Generated {date_str}</footer>
</body>
</html>"""


def publish_static(html, week):
    site_bucket = os.environ.get("GCS_SITE_BUCKET", "")
    if not site_bucket:
        return
    from google.cloud import storage as gcs
    bucket = gcs.Client().bucket(site_bucket)
    content = html.encode("utf-8")
    for name, max_age in (("digest.html", 3600), (f"digest-{week}.html", 86400)):
        blob = bucket.blob(name)
        blob.upload_from_string(content, content_type="text/html; charset=utf-8")
        blob.cache_control = f"public, max-age={max_age}"
        blob.patch()
    print(f"site: https://storage.googleapis.com/{site_bucket}/digest.html")


# ── persistence ───────────────────────────────────────────────────────────────

def load_week_items(conn):
    """The full rolling 7-day window. Previously this also filtered out
    items that had already appeared in an earlier digest (used_in_digest=0),
    but that single-shot gate caused important ongoing stories to vanish at
    the next run — the rolling window is the right unit, and re-surfacing a
    persistent topic across two digests is a feature, not a bug. The
    used_in_digest column is still written by save_digest for informational
    purposes but no longer gates retrieval here."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    return conn.execute(
        "SELECT id,source,title,url,body,ts FROM items "
        "WHERE ingested_at>=?",
        (cutoff,),
    ).fetchall()


def save_digest(conn, top, items, summary, vec):
    centroids = sp.vstack([c["vec"] for c in top])
    inv_vocab = [None] * len(vec.vocabulary_)
    for term, idx in vec.vocabulary_.items():
        inv_vocab[idx] = term
    week = datetime.now(timezone.utc).strftime("%Y-W%V")
    conn.execute(
        "INSERT OR REPLACE INTO digests(week,centroids,vocab,summary,created_at) VALUES(?,?,?,?,?)",
        (week, pickle.dumps(centroids), pickle.dumps(inv_vocab), summary, now_iso()),
    )
    ids = [items[i]["id"] for c in top for i in c["idxs"]]
    conn.executemany("UPDATE items SET used_in_digest=1 WHERE id=?", [(i,) for i in ids])


# ── orchestration ─────────────────────────────────────────────────────────────

def run():
    with pipeline_lock():
        conn = pull_db()
        rows = load_week_items(conn)
        if len(rows) < 10:
            print(f"too few items: {len(rows)}")
            return

        items = enrich(rows)
        X, vec, edges = build_tfidf(items)
        clusters = cluster_average_linkage(X)

        item_texts = [
            f"{a['title']} {(a['body'] or '')[:_EMBED_BODY_CHARS]}" for a in items
        ]
        E = _embed(item_texts)
        if E is not None:
            print(f"embeddings: {E.shape[0]} items × {E.shape[1]} dims")
        else:
            print("embeddings: unavailable, using lexical-only fallback")

        all_ent_counts = defaultdict(int)
        for a in items:
            for e, lbl in a["entities"].items():
                if _is_signal_entity(e, lbl):
                    all_ent_counts[e] += 1
        ent_history = load_entity_history(conn)
        velocities = compute_velocities(all_ent_counts, ent_history)

        week = datetime.now(timezone.utc).strftime("%Y-W%V")

        # Discourse-learning context: IDF, aspects, debt, topic bank, weights.
        entity_idf, max_idf = load_entity_idf(conn)
        aspects = get_profile_aspects(conn)
        exclusion_aspects = get_exclusion_aspects(conn)
        debt = compute_coverage_debt(conn, aspects)
        bank = load_topic_bank(conn)
        # tune_weights runs before scoring so this week uses the latest blend;
        # it no-ops until cluster_signals has accumulated >= 10 weeks.
        try:
            tune_weights(conn)
        except Exception as _tw_exc:
            print(f"tune_weights failed, continuing with existing weights: {_tw_exc}")
        weights = load_weights(conn)

        # History depth gates both the off-profile filter's Layer B
        # (persistence override) and the summary prose mode (warmup vs full).
        history_weeks = conn.execute(
            "SELECT COUNT(DISTINCT week) FROM cluster_signals"
        ).fetchone()[0] or 0

        scored, aspect_coverage = score_clusters(
            clusters, items, X, vec, velocities, bank, aspects, debt,
            entity_idf, max_idf, weights, week,
            exclusion_aspects=exclusion_aspects,
            E=E,
        )

        # §2.4 — off-profile filter (Layer A always, Layer B at >= 10 weeks).
        before_filter = len(scored)
        scored = _filter_off_profile(scored, history_weeks)
        print(f"off-profile filter: {before_filter} -> {len(scored)} clusters "
              f"(history_weeks={history_weeks})")

        # §2.5 — dynamic, score-knee-aware selection (replaces k=15).
        top = mmr_dynamic_select(scored)

        # §2.3 — cross-cluster URL + title-signature dedup BEFORE persistence
        # writes so the topic bank and cluster_signals reflect the deduped set.
        top = _dedupe_top(top, items)

        # §D — Longitudinal context for the LLM, computed from the bank state
        # BEFORE update_topic_bank mutates it (we want the pre-this-week
        # last_week values for "returns after X-week gap" framing).
        longitudinal_context = build_longitudinal_context(
            conn, week, all_ent_counts, top, bank,
        )

        # Commit topic-bank state from the selected top clusters, then persist
        # this week's per-cluster signal vectors (keyed by topic_id) for A5
        # retrospective tuning.
        update_topic_bank(conn, top, vec, week)
        save_cluster_signals(conn, week, top)
        save_coverage_ledger(conn, week, aspect_coverage)

        macro_context = load_macro_context(conn)
        history_depth = "full" if history_weeks >= 10 else "warmup"
        summary, cluster_index = summarise(
            top, items, macro_context, history_depth=history_depth,
            longitudinal_context=longitudinal_context,
        )

        save_digest(conn, top, items, summary, vec)
        save_entity_history(conn, dict(all_ent_counts), week)

        macro_new = generate_macro_trends(conn)
        conn.commit()

        display_macro = load_macro_context(conn)
        archive_links = get_archive_links(conn)
        # Build coverage HTML before closing the connection so evaluate.py can
        # read the same DB state we just committed.
        from evaluate import report as eval_report, render_html as eval_render, publish_coverage
        try:
            coverage_html = eval_render(eval_report(conn, weeks=12))
        except Exception as e:
            print(f"coverage render failed: {e}")
            coverage_html = None
        conn.close()
        push_db()

        send_email(render_email(summary, cluster_index, top, items, display_macro))
        publish_static(
            render_static(summary, cluster_index, top, items, week,
                          display_macro, archive_links),
            week,
        )
        if coverage_html:
            publish_coverage(coverage_html, week)

        # Monthly self-learning metrics email — only on the first Sunday of the
        # month. Re-open the DB read-only since we already closed/pushed above;
        # we don't mutate state here, so this stays a clean side-effect.
        try:
            from metrics_email import is_first_sunday_of_month, send_metrics_email
            if is_first_sunday_of_month():
                with sqlite3.connect(LOCAL_DB) as mconn:
                    send_metrics_email(mconn, week)
                print(f"monthly metrics email sent for {week}")
        except Exception as e:
            print(f"metrics email failed (non-fatal): {e}")

        print(
            f"digest done: {len(top)} clusters from {len(rows)} items "
            f"({len(clusters)} total), {len(all_ent_counts)} entities tracked"
            + (f", macro updated ({week})" if macro_new else "")
        )


if __name__ == "__main__":
    run()
