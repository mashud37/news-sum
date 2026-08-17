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
from functools import partial

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

_NER_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT", "MONEY", "PERCENT", "DATE"}
_ENTITY_SIGNAL_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT"}
_SPOTLIGHT_LABELS = {"ORG", "PERSON", "PRODUCT"}
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

TFIDF_SIM_THRESHOLD = 0.20
TITLE_SIM_THRESHOLD = 0.40
MIN_DOCUMENT_FREQUENCY = 2
SPARSE_DOCUMENT_FREQUENCY = 1
MIN_ITEMS_FOR_DIGEST = 10
FULL_HISTORY_WEEKS = 10
COVERAGE_REPORT_WEEKS = 12

STEP_PLAN = [
    "Pull DB",
    "Load & enrich items",
    "Build TF-IDF + cluster",
    "Embed items",
    "Load discourse context",
    "Score clusters",
    "MMR select + dedup",
    "Update topic bank",
    "LLM summarise",
    "Save digest state",
    "Push DB + publish",
    "Send email",
]

RRF_K = 60
DORMANT_WEEKS = 4
DORMANT_NOVELTY_BONUS = 0.3
TOP_ENTITIES_PER_CLUSTER = 5
TREND_CLIP_LOW = -1.0
TREND_CLIP_HIGH = 3.0
RECENCY_DECAY_DAYS = 4.0
KEY_ENTITY_BOOST = 2

SHINGLE_SIZE = 3
MMR_SENTENCE_MAX_CHARS = 240
MMR_SENTENCE_LAMBDA = 0.7
CLUSTER_MAX_DIAMETER = 0.80
ENTITY_HISTORY_WEEKS = 4
COVERAGE_DEBT_DECAY = 0.7
COVERAGE_DEBT_WINDOW = 6
COVERAGE_DEBT_THRESHOLD = 0.4
TOPIC_CENTROID_TOP_N = 200
TOPIC_BANK_ALPHA = 0.7
TOPIC_BANK_MATCH_THRESHOLD = 0.3
TOPIC_BANK_DECAY = 0.8
TOPIC_BANK_MASS_FLOOR = 0.05
TOPIC_BANK_STALE_WEEKS = 12
TUNE_MIN_WEEKS = 10
TUNE_BLEND_CURRENT = 0.7
TUNE_BLEND_LEARNED = 0.3
LONGITUDINAL_STREAK_MIN = 3
LONGITUDINAL_DORMANT_MIN = 4
LONGITUDINAL_TOP_N = 5
LONGITUDINAL_TOPIC_DISPLAY_MAX = 3
MMR_SELECT_K = 15
MMR_LAMBDA = 0.65
MMR_MIN_K = 8
MMR_MAX_K = 40
MMR_RELEVANCE_FLOOR = 0.15
OFF_PROFILE_RELEVANCE_FLOOR = 0.10
OFF_PROFILE_PERSISTENCE_FLOOR = 0.15
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2.0
ARCHIVE_LINKS_COUNT = 8
CLUSTER_LINKS_CAP = 5
SECTION_DUMP_LINKS_CAP = 4

SUMMARISE_SYSTEM_PROMPT = (
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
    "Each pick: 1 to 3 sentences. Lead with the concrete observable, then "
    "the structural implication. Cite companies and figures precisely. "
    "No hedging. No filler."
)

SUMMARISE_BODY_PREAMBLE = (
    "Below are clustered stories grouped by sector (## heading) and "
    "category (### heading). Each cluster is numbered C{n} with its "
    "score, source list, headline, body snippet, and top entities. "
    "Clusters within each category are listed in descending score "
    "order: the first is the highest-scoring.\n\n"
    "For each (sector, category) you find meaningful, pick UP TO 5 "
    "clusters to spotlight, ordered by your editorial judgement of "
    "importance. Skip a category entirely if nothing rises above noise. "
    "Skip a sector entirely if all its categories are skipped.\n\n"
    "Output strictly:\n\n"
    "## SectorName (display name as shown below)\n"
    "### CategoryName (display name as shown below)\n"
    "- C{n}: 1 to 3 sentence editorial summary.\n"
    "- C{m}: 1 to 3 sentence editorial summary.\n\n"
    "Repeat per sector and per category. Preserve each Cn reference "
    "exactly: the renderer parses it to attach links."
)

STATIC_PAGE_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#ffffff;color:#202124;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:15px;line-height:1.65;padding:0 16px 64px;max-width:860px;margin:0 auto}
a{color:#1a73e8;text-decoration:none}a:hover{text-decoration:underline}
header{border-bottom:2px solid #1a73e8;padding:24px 0 14px;margin-bottom:28px}
header h1{font-size:clamp(18px,4vw,28px);letter-spacing:.02em;color:#1a73e8}
.wk{color:#5f6368;font-size:12px;margin-top:4px}
.macro{background:#e8f0fe;border-left:3px solid #1a73e8;padding:16px 20px;border-radius:4px;margin:28px 0}
.macro h2{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#1967d2;margin-bottom:12px}
.macro-body p{color:#3c4043;font-size:14px;margin-bottom:8px;line-height:1.6}
.macro-body p:last-child{margin-bottom:0}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#5f6368;margin:28px 0 14px}
h2.sector{font-size:18px;text-transform:none;letter-spacing:.01em;color:#1a73e8;border-bottom:1px solid #dadce0;padding-bottom:6px;margin:36px 0 12px;font-weight:600}
h2.sector .sn{color:#5f6368;font-size:12px;font-weight:400;margin-left:8px}
/* SPOTLIGHT: LLM editorial picks, grouped by category */
.spotlight{margin:6px 0 12px}
.spotlight h3.cat-block{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#1967d2;margin:14px 0 6px;border:none;padding:0;font-weight:600}
ul.spotlight-list{list-style:none;padding:0;margin:0 0 10px}
ul.spotlight-list li.spot-item{padding:8px 0;border-top:1px solid #dadce0}
ul.spotlight-list li.spot-item:first-child{border-top:none}
.spot-summary{color:#202124;font-size:14.5px;line-height:1.55;margin-bottom:6px}
.spot-links{font-size:13px;color:#5f6368;padding-left:2px}
.spot-links a{color:#1a73e8}
.spot-links .also{font-size:12.5px;color:#80868b;margin-top:2px;padding-left:8px}
.spot-links .also a{color:#1967d2}

/* SEPARATOR between spotlight and dump */
hr.sep{border:none;border-top:1px dashed #dadce0;margin:14px 0 10px}

/* DUMP: mechanical, category-grouped, score-descending */
.dump h4.dump-header{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:#5f6368;margin:4px 0 8px;font-weight:600;border:none;padding:0}
.dump h5.dump-cat{font-size:11px;color:#3c4043;margin:10px 0 4px;font-weight:600}
.dump h5.dump-cat .sn{color:#80868b;font-weight:400;font-size:10.5px;margin-left:4px}
ul.dump-list{list-style:none;padding:0;margin:0 0 6px}
ul.dump-list li{border-top:1px solid #f1f3f4;padding:5px 0;font-size:12.5px;line-height:1.45;color:#5f6368}
ul.dump-list li:first-child{border-top:none}
ul.dump-list li a{color:#1a73e8}
ul.dump-list li .also{font-size:11.5px;color:#80868b;padding-left:8px;margin-top:1px}
ul.dump-list li .also a{color:#1a73e8}

.src{color:#185abc;font-size:10px;font-weight:700;margin-right:6px;letter-spacing:.02em}
.archive{margin-top:48px;padding-top:24px;border-top:1px solid #dadce0}
.archive h2{margin-bottom:10px}
.archive ul{display:flex;flex-wrap:wrap;gap:8px}
.archive li{border:none;padding:0}
.archive a{font-size:12px;color:#3c4043;border:1px solid #dadce0;padding:3px 10px;border-radius:4px}
.archive a:hover{color:#1a73e8;border-color:#1a73e8}
footer{margin-top:32px;color:#80868b;font-size:11px;text-align:center}
@media(max-width:580px){.ch{flex-direction:column}}
"""

_TUNE_KEYS = [
    "coverage",
    "prior",
    "novelty",
    "relevance",
    "entity_signal",
    "trend",
    "richness",
    "coverage_gap",
]

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


# ---- Text utilities ----

def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s.lower())).strip()


def _shingles(s, k=SHINGLE_SIZE):
    t = _norm(s).replace(" ", "")
    return {t[i:i + k] for i in range(len(t) - k + 1)} if len(t) >= k else {t}


# ---- Sentence-level helpers (TREC Novelty / lexical cohesion) ----

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


def _mmr_sentence(body, title, query, max_chars=MMR_SENTENCE_MAX_CHARS,
                  lam=MMR_SENTENCE_LAMBDA):
    """Pick the single most information-dense sentence from `body` via MMR.

    Replaces the heuristic body[:240] (which assumes inverted-pyramid lede
    structure). MMR objective per TREC Novelty:

        score(s) = lam * cos(s, query) - (1 - lam) * cos(s, title)

    Relevance = aligned with the user's discourse query (USER_PROFILE).
    Diversity penalty = aligned with the title, which the LLM already sees.
    We want the body excerpt to ADD information, not echo the headline.

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


# ---- NLP enrichment ----

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
            "id": id_,
            "source": source,
            "title": title,
            "url": url,
            "body": body or "",
            "ts": ts,
            "entities": entities,
        })
    return items


# ---- Deduplication & clustering ----
#
# Refactored away from MinHash LSH + union-find (single-linkage). At weekly
# corpus scales (thousands of items, ~20k vocab) the exact cosine matrix
# X·X^T over sparse TF-IDF is faster than an LSH proposal step plus pairwise
# verification, and removes a probabilistic hyperparameter (Jaccard threshold)
# that was always going to drift from the cosine threshold it gated.
#
# Clustering moved to average-linkage agglomerative with a hard cosine-distance
# diameter: same intent as the old sim_threshold but enforced over the whole
# cluster rather than one transitive edge at a time. This kills the chaining
# pathology where A-B and B-C linked clustered A and C with zero overlap.


def _pairs_above(S, threshold):
    """Return (i, j) with i < j whose entries in the (sparse) similarity matrix
    S exceed `threshold`. S is the output of X·X^T for L2-normalised X."""
    S_coo = S.tocoo() if sp.issparse(S) else sp.coo_matrix(S)
    found = []
    for i, j, v in zip(S_coo.row, S_coo.col, S_coo.data):
        if i < j and v >= threshold:
            found.append((int(i), int(j)))
    return found


def _augmented_text(item):
    """Join an item's title, body, and entity tokens into one string to vectorise."""
    entity_tokens = []
    for entity in item["entities"]:
        entity_tokens.append(f"ent_{entity.lower().replace(' ', '_')}")
    joined = " ".join(entity_tokens)
    return f"{_norm(item['title'])} {_norm(item['body'])} {joined}"


def _fit_tfidf(texts, min_df):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=min_df,
        norm="l2",
    )
    return {"vectorizer": vectorizer, "matrix": vectorizer.fit_transform(texts)}


def build_tfidf(items, sim_threshold=TFIDF_SIM_THRESHOLD,
                title_sim_threshold=TITLE_SIM_THRESHOLD):
    """Build the augmented TF-IDF matrix and the two-lane candidate edge set.

    Two lanes run on different vocabulary spaces, because editorial structure
    separates headline semantics from body semantics: body+title augmented
    TF-IDF at `sim_threshold`, and title-only TF-IDF at the stricter
    `title_sim_threshold`, short text being noisier per term.

    Args:
        items: this week's enriched articles.
        sim_threshold: cosine floor for the body+title lane.
        title_sim_threshold: cosine floor for the title-only lane.

    Returns:
        {"matrix": augmented TF-IDF matrix, "vectorizer": its fitted vectorizer,
         "edges": union of both lanes, diagnostic only since clustering runs
         from the matrix directly}
    """
    augmented_texts = []
    for item in items:
        augmented_texts.append(_augmented_text(item))
    try:
        fitted = _fit_tfidf(augmented_texts, min_df=MIN_DOCUMENT_FREQUENCY)
    except ValueError:
        # A slow ingest day can leave every term appearing once, which min_df=2
        # rejects outright; falling back keeps the weekly job alive.
        fitted = _fit_tfidf(augmented_texts, min_df=SPARSE_DOCUMENT_FREQUENCY)
    X = fitted["matrix"]

    edges = set(_pairs_above(X.dot(X.T), sim_threshold))

    title_texts = []
    for item in items:
        title_texts.append(_norm(item["title"]))
    try:
        title_fitted = _fit_tfidf(title_texts, min_df=MIN_DOCUMENT_FREQUENCY)
        title_matrix = title_fitted["matrix"]
        edges.update(_pairs_above(title_matrix.dot(title_matrix.T), title_sim_threshold))
    except ValueError:
        pass

    return {"matrix": X, "vectorizer": fitted["vectorizer"], "edges": edges}


def cluster_average_linkage(X, max_diameter=CLUSTER_MAX_DIAMETER):
    """Average-linkage agglomerative clustering with a cosine-distance
    diameter ceiling.

    Replaces single-linkage union-find. Single-linkage chains: an edge A-B
    and an edge B-C produce {A, B, C} even when A·C ≈ 0. Average-linkage
    uses mean inter-cluster distance, and the diameter cut at `max_diameter`
    enforces a strict semantic radius.

    `max_diameter` is cosine distance (= 1 - cosine similarity). 0.80
    corresponds roughly to "average inter-member cosine ≥ 0.20": the
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


# ---- Novelty projection ----

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


# ---- Entity trend analysis ----

def load_entity_history(conn, n_weeks=ENTITY_HISTORY_WEEKS):
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


# ---- §D longitudinal context: entity & topic memory fed to the LLM ----

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


def _entity_streak_lines(conn, this_week_entity_counts):
    """Entities present in entity_history every one of the last
    LONGITUDINAL_STREAK_MIN weeks, and present again this week."""
    if not this_week_entity_counts:
        return []
    recent_weeks = [r[0] for r in conn.execute(
        "SELECT DISTINCT week FROM entity_history ORDER BY week DESC LIMIT ?",
        (LONGITUDINAL_STREAK_MIN,),
    ).fetchall()]
    if len(recent_weeks) < LONGITUDINAL_STREAK_MIN:
        return []
    placeholders = ",".join("?" * len(recent_weeks))
    rows = conn.execute(
        f"SELECT entity, COUNT(DISTINCT week) AS n "
        f"FROM entity_history WHERE week IN ({placeholders}) "
        f"GROUP BY entity HAVING n >= ?",
        (*recent_weeks, LONGITUDINAL_STREAK_MIN),
    ).fetchall()
    this_week = set(this_week_entity_counts.keys())
    on_streak = sorted(
        [(entity, count) for entity, count in rows if entity in this_week],
        key=lambda pair: (-pair[1], pair[0]),
    )[:LONGITUDINAL_TOP_N]
    lines = []
    for entity, count in on_streak:
        lines.append(f"- {entity}: appearing for {count} consecutive weeks (incl. this week)")
    return lines


def _last_seen_before_week(conn, week):
    """Most recent week (other than `week`) each entity appeared in, over
    the window needed to detect a LONGITUDINAL_DORMANT_MIN-week gap."""
    window = LONGITUDINAL_DORMANT_MIN + 2
    prior_weeks = [r[0] for r in conn.execute(
        "SELECT DISTINCT week FROM entity_history WHERE week != ? "
        "ORDER BY week DESC LIMIT ?", (week, window),
    ).fetchall()]
    last_seen = {}
    for wk in prior_weeks:
        entities_in_week = {r[0] for r in conn.execute(
            "SELECT entity FROM entity_history WHERE week=?", (wk,),
        ).fetchall()}
        for entity in entities_in_week:
            if entity not in last_seen:
                last_seen[entity] = wk
    return last_seen


def _returning_entity_lines(conn, week, this_week_entity_counts):
    """Entities present this week after a gap of LONGITUDINAL_DORMANT_MIN
    or more weeks."""
    if not this_week_entity_counts:
        return []
    last_seen = _last_seen_before_week(conn, week)
    if not last_seen:
        return []
    returning = []
    for entity in this_week_entity_counts:
        last_week = last_seen.get(entity)
        if not last_week:
            continue
        gap = _weeks_between(week, last_week)
        if gap >= LONGITUDINAL_DORMANT_MIN:
            returning.append((entity, gap))
    returning.sort(key=lambda pair: (-pair[1], pair[0]))
    lines = []
    for entity, gap in returning[:LONGITUDINAL_TOP_N]:
        lines.append(f"- {entity}: returns this week after {gap}-week gap")
    return lines


def _topic_pattern_lines(top_clusters, bank, week):
    """Bank topics this week's clusters matched that are on a streak or
    returning from dormancy."""
    matched_ids = {
        c.get("matched_topic_id") for c in top_clusters
        if c.get("matched_topic_id") is not None
    }
    if not matched_ids or not bank:
        return []
    bank_by_id = {topic["topic_id"]: topic for topic in bank}
    topic_streaks = []
    topic_returns = []
    for topic_id in matched_ids:
        topic = bank_by_id.get(topic_id)
        if not topic:
            continue
        if topic.get("weeks_seen", 0) >= LONGITUDINAL_STREAK_MIN:
            topic_streaks.append((topic, topic["weeks_seen"]))
        # last_week is the value BEFORE this run updated it for matched topics.
        gap_before_this_week = _weeks_between(week, topic.get("last_week", ""))
        if gap_before_this_week >= LONGITUDINAL_DORMANT_MIN:
            topic_returns.append((topic, gap_before_this_week))
    topic_streaks.sort(key=lambda pair: -pair[1])
    topic_returns.sort(key=lambda pair: -pair[1])
    lines = []
    for topic, weeks_seen in topic_streaks[:LONGITUDINAL_TOPIC_DISPLAY_MAX]:
        lines.append(f"- Topic '{_topic_label(topic)}': matched again, "
                     f"{weeks_seen} weeks of total coverage")
    for topic, gap in topic_returns[:LONGITUDINAL_TOPIC_DISPLAY_MAX]:
        lines.append(f"- Topic '{_topic_label(topic)}': returns to coverage "
                     f"after {gap}-week dormancy")
    return lines


def build_longitudinal_context(conn, week, this_week_entity_counts, top_clusters, bank):
    """Compact text block of multi-week patterns for the summarise() prompt.

    Returns "" when entity_history has fewer than 2 distinct weeks. Combines
    entity streaks, returning entities, and topic streaks or returns from
    the persistent topic bank, all derived from existing tables.
    """
    n_weeks_total = conn.execute(
        "SELECT COUNT(DISTINCT week) FROM entity_history"
    ).fetchone()[0] or 0
    if n_weeks_total < 2:
        return ""

    lines = []
    lines.extend(_entity_streak_lines(conn, this_week_entity_counts))
    lines.extend(_returning_entity_lines(conn, week, this_week_entity_counts))
    lines.extend(_topic_pattern_lines(top_clusters, bank, week))

    if not lines:
        return ""
    return ("Longitudinal context (multi-week entity & topic patterns; cite "
            "explicitly when relevant: do not invent streaks not listed):\n"
            + "\n".join(lines))


# ---- Discourse-learning signals ----

def load_entity_idf(conn):
    """Entity IDF across past weeks of entity_history. Each week = a document.

    idf(e) = log((1 + n_weeks_total) / (1 + n_weeks_with_e))
    Entities never seen receive the maximum IDF (n_weeks_with_e treated as 0).

    Returns:
        {"idf": {entity: idf}, "max_idf": the ceiling callers use for
        unknown entities}.
    """
    n_weeks = conn.execute(
        "SELECT COUNT(DISTINCT week) FROM entity_history"
    ).fetchone()[0] or 0
    if n_weeks == 0:
        placeholder_max_idf = math.log((1 + 0) / (1 + 0) + 1.0)
        return {"idf": {}, "max_idf": placeholder_max_idf}
    rows = conn.execute(
        "SELECT entity, COUNT(DISTINCT week) FROM entity_history GROUP BY entity"
    ).fetchall()
    idf = {e: math.log((1 + n_weeks) / (1 + df)) for e, df in rows}
    max_idf = math.log((1 + n_weeks) / 1.0)
    return {"idf": idf, "max_idf": max_idf}


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
    request = partial(
        client.messages.create,
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
    )
    msg = _retry(request)
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
    request = partial(
        client.messages.create,
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
    )
    msg = _retry(request)
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


def compute_coverage_debt(conn, aspects, decay=COVERAGE_DEBT_DECAY,
                          window=COVERAGE_DEBT_WINDOW,
                          threshold=COVERAGE_DEBT_THRESHOLD):
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


# ---- Persistent topic bank (A4) ----

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
    """Project every bank topic's centroid into new_vocab.

    Returns:
        {"matrix": one row per topic with at least one overlapping term, or
        None if no topic overlapped; "present": the bank indices those rows
        came from, in the same order}.
    """
    if not bank:
        return {"matrix": None, "present": []}
    rows, present = [], []
    for index, topic in enumerate(bank):
        projected = _project_centroids(topic["centroid"], topic["vocab"], new_vocab)
        if projected is None:
            continue
        rows.append(projected)
        present.append(index)
    if not rows:
        return {"matrix": None, "present": []}
    return {"matrix": sp.vstack(rows), "present": present}


def _truncate_centroid(centroid, top_n=TOPIC_CENTROID_TOP_N):
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
    # argpartition is O(n): much cheaper than full argsort at this scale.
    keep = np.argpartition(-np.abs(arr), top_n)[:top_n]
    truncated = np.zeros_like(arr)
    truncated[keep] = arr[keep]
    norm = float(np.sqrt(np.dot(truncated, truncated)))
    if norm > 0:
        truncated = truncated / norm
    return sp.csr_matrix(truncated.reshape(1, -1))


def _merge_matched_topic(conn, cluster, inv_vocab, vocabulary, week):
    """Blend a cluster's centroid into its matched topic_bank row.

    Returns the matched topic_id on success, or None when the cluster has
    no confident match, so the caller spawns a new topic instead."""
    topic_id = cluster.get("matched_topic_id")
    similarity = cluster.get("matched_sim", 0.0)
    if topic_id is None or similarity < TOPIC_BANK_MATCH_THRESHOLD:
        return None
    row = conn.execute(
        "SELECT centroid, vocab, mass, weeks_seen FROM topic_bank WHERE topic_id=?",
        (topic_id,),
    ).fetchone()
    if row is None:
        return None

    old_centroid, old_vocab, mass, weeks_seen = (
        pickle.loads(row[0]), pickle.loads(row[1]), row[2], row[3]
    )
    projected_old = _project_centroids(old_centroid, old_vocab, vocabulary)
    if projected_old is None:
        merged = cluster["vec"]
    else:
        merged = TOPIC_BANK_ALPHA * projected_old + (1 - TOPIC_BANK_ALPHA) * cluster["vec"]
    # Sparsity constraint: see _truncate_centroid docstring.
    merged = _truncate_centroid(merged)

    conn.execute(
        "UPDATE topic_bank SET centroid=?, vocab=?, mass=?, "
        "last_week=?, weeks_seen=? WHERE topic_id=?",
        (
            pickle.dumps(merged),
            pickle.dumps(inv_vocab),
            mass + 1.0,
            week,
            weeks_seen + 1,
            topic_id,
        ),
    )
    return topic_id


def _spawn_topic(conn, cluster, inv_vocab, week):
    cursor = conn.execute(
        "INSERT INTO topic_bank(centroid, vocab, mass, first_week, last_week, weeks_seen) "
        "VALUES(?,?,?,?,?,?)",
        (pickle.dumps(cluster["vec"]), pickle.dumps(inv_vocab), 1.0, week, week, 1),
    )
    return cursor.lastrowid


def _decay_and_prune_bank(conn, week):
    """Age every topic's mass, then drop rows that went stale or too light."""
    conn.execute("UPDATE topic_bank SET mass = mass * ?", (TOPIC_BANK_DECAY,))
    rows = conn.execute("SELECT topic_id, last_week FROM topic_bank").fetchall()
    stale_ids = []
    for topic_id, last_week in rows:
        if _weeks_between(week, last_week) > TOPIC_BANK_STALE_WEEKS:
            stale_ids.append((topic_id,))
    if stale_ids:
        conn.executemany("DELETE FROM topic_bank WHERE topic_id=?", stale_ids)
    conn.execute("DELETE FROM topic_bank WHERE mass < ?", (TOPIC_BANK_MASS_FLOOR,))


def update_topic_bank(conn, top, vec, week):
    """Merge or spawn topics in topic_bank from this week's top clusters.

    Each cluster in `top` already carries a tentative `matched_topic_id` and
    `matched_sim` set by score_clusters; this commits the matches, spawns
    rows for unmatched clusters, decays all topics, and prunes stale rows.
    Writes the final `topic_id` back onto each cluster dict.
    """
    inv_vocab = [None] * len(vec.vocabulary_)
    for term, index in vec.vocabulary_.items():
        inv_vocab[index] = term

    for cluster in top:
        topic_id = _merge_matched_topic(conn, cluster, inv_vocab, vec.vocabulary_, week)
        if topic_id is None:
            topic_id = _spawn_topic(conn, cluster, inv_vocab, week)
        cluster["topic_id"] = topic_id

    _decay_and_prune_bank(conn, week)


# ---- Adaptive weights (A5) ----

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


def _load_signal_rows(conn):
    return conn.execute(
        "SELECT cs.week, cs.topic_id, cs.coverage, cs.prior, cs.novelty, "
        "cs.relevance, cs.entity_signal, cs.trend, cs.richness, cs.coverage_gap "
        "FROM cluster_signals cs "
        "WHERE cs.topic_id IS NOT NULL"
    ).fetchall()


def _persistence_labels(rows):
    """Build the (signals, persisted) training set from cluster_signals rows.

    A row at (week=wk, topic_id=t) "persisted" iff that same topic_id
    appears in cluster_signals for any later week. This label is leak-free:
    it never reads the current week's own outcome.

    Returns:
        {"signals": float array, "labels": int array}, one row per input row.
    """
    later_weeks = defaultdict(set)
    for row in rows:
        week, topic_id = row[0], row[1]
        later_weeks[topic_id].add(week)

    signals, labels = [], []
    for row in rows:
        week, topic_id = row[0], row[1]
        signals.append(list(row[2:10]))
        persisted = int(any(later_week > week for later_week in later_weeks[topic_id]))
        labels.append(persisted)
    return {
        "signals": np.array(signals, dtype=float),
        "labels": np.array(labels, dtype=int),
    }


def _fit_positive_coefficients(X, y):
    """Fit logistic regression on the signal matrix and return only its
    positive coefficients, clipped to zero elsewhere.

    Returns None when the label is degenerate (all one class) or every
    coefficient came out non-positive."""
    if len(set(y.tolist())) < 2:
        print(f"tune_weights: skip (degenerate label, {y.sum()}/{len(y)} positive)")
        return None
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    positive = np.clip(model.coef_[0], 0, None)
    if positive.sum() <= 0:
        print("tune_weights: skip (no positive coefficients)")
        return None
    return positive


def _blend_learned_weights(conn, positive_coefficients):
    """Blend logistic-regression-learned weights with the current weights.

    Only the _TUNE_KEYS signals are fitted. persistence, source_breadth, and
    recency carry over from the current weights unchanged: persistence would
    leak (it is bank-derived, the same axis as the label), and the other two
    are recency heuristics with no historical rows to fit on.

    Returns:
        {"current": weights before tuning, "final": blended and
        floor-enforced weights}.
    """
    current = load_weights(conn)
    learned_partial = {
        key: float(positive_coefficients[i] / positive_coefficients.sum())
        for i, key in enumerate(_TUNE_KEYS)
    }
    frozen_keys = [key for key in WEIGHT_KEYS if key not in learned_partial]
    frozen_total = sum(current.get(key, 0.0) for key in frozen_keys)
    fit_total = 1.0 - frozen_total
    learned = {
        key: learned_partial.get(key, 0.0) * max(fit_total, 1e-9)
        for key in WEIGHT_KEYS
    }
    for key in frozen_keys:
        learned[key] = current.get(key, DEFAULT_WEIGHTS[key])

    blended = {
        key: TUNE_BLEND_CURRENT * current[key] + TUNE_BLEND_LEARNED * learned[key]
        for key in WEIGHT_KEYS
    }
    total = sum(blended.values())
    if total > 0:
        blended = {key: value / total for key, value in blended.items()}
    return {"current": current, "final": _enforce_floor(blended)}


def _save_tuned_weights(conn, weights):
    week = datetime.now(timezone.utc).strftime("%Y-W%V")
    conn.execute(
        "INSERT OR REPLACE INTO scorer_weights(week, weights, created_at) VALUES(?,?,?)",
        (week, json.dumps(weights), now_iso()),
    )


def tune_weights(conn, min_weeks=TUNE_MIN_WEEKS):
    """Fit logistic regression of signals to topic persistence and blend the
    learned weights with the current weights.

    Skips if history is shallow or the persistence label is degenerate.
    Writes a row to scorer_weights and logs old vs new.
    """
    rows = _load_signal_rows(conn)
    weeks_in_data = {row[0] for row in rows}
    if len(weeks_in_data) < min_weeks:
        print(f"tune_weights: skip ({len(weeks_in_data)} weeks < {min_weeks})")
        return None

    training = _persistence_labels(rows)
    positive_coefficients = _fit_positive_coefficients(
        training["signals"], training["labels"]
    )
    if positive_coefficients is None:
        return None

    blend = _blend_learned_weights(conn, positive_coefficients)
    current, final = blend["current"], blend["final"]
    _save_tuned_weights(conn, final)
    print(f"tune_weights: old={current} -> new={final}")
    return final


def save_cluster_signals(conn, week, top):
    """Persist per-cluster signal vector + score so A5 can fit retrospectively.
    Keyed by (week, topic_id); topic_id is set by update_topic_bank earlier."""
    rows = [
        (
            week,
            c.get("topic_id"),
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
            float(c["score"]),
        )
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


# ---- Sector / category classification ----

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


# ---- Cluster scoring ----

def _normalize(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


def _medoid_scores(bm25, medoids, tokens):
    scores = bm25.get_scores(tokens)
    return np.array([scores[m] for m in medoids])


def _add_rrf_ranks(totals, values):
    order = np.argsort(-values)
    fused = totals.copy()
    for rank, idx in enumerate(order):
        fused[idx] += 1.0 / (RRF_K + rank + 1)
    return fused


def _lexical_relevance(inputs, medoids, bm25):
    """Fuse one BM25 ranking per profile aspect by reciprocal rank."""
    aspects = inputs["aspects"]
    if not aspects:
        tokens = _norm(USER_PROFILE).split()
        return _medoid_scores(bm25, medoids, tokens)
    totals = np.zeros(len(medoids))
    for label, descriptor in aspects:
        tokens = _norm(descriptor).split() or _norm(label).split()
        if not tokens:
            continue
        totals = _add_rrf_ranks(totals, _medoid_scores(bm25, medoids, tokens))
    return totals


def _dense_relevance(inputs, medoids):
    """Fuse one embedding-cosine ranking per profile aspect by reciprocal rank."""
    embeddings = inputs["embeddings"]
    aspects = inputs["aspects"]
    totals = np.zeros(len(medoids))
    if embeddings is None or not aspects:
        return totals
    aspect_embs = _embed_aspects(aspects, _profile_hash())
    if aspect_embs is None:
        return totals
    medoid_embs = embeddings[medoids]
    for position in range(len(aspects)):
        sims = medoid_embs @ aspect_embs[position]
        totals = _add_rrf_ranks(totals, sims)
    return totals


def _novelty_against_bank(inputs, medoid_vecs):
    """Novelty as distance from the persistent topic bank, plus a dormancy bonus.

    Args:
        inputs: the scoring inputs dictionary.
        medoid_vecs: sparse matrix of one medoid vector per cluster.

    Returns:
        {"novelty": array, "matched_ids": list, "matched_sims": array}, where a
        matched id is None when the cluster matched no topic in the bank.
    """
    clusters = inputs["clusters"]
    bank = inputs["bank"]
    week = inputs["week"]
    matched_ids = [None] * len(clusters)
    matched_sims = np.zeros(len(clusters))
    dormant_bonus = np.zeros(len(clusters))
    projection = _project_bank(bank, inputs["vec"].vocabulary_)
    bank_matrix, bank_present = projection["matrix"], projection["present"]
    if bank_matrix is not None:
        sims = medoid_vecs.dot(bank_matrix.T).toarray()
        best_idx = sims.argmax(axis=1)
        best_sim = sims.max(axis=1)
        for i in range(len(clusters)):
            topic = bank[bank_present[best_idx[i]]]
            matched_ids[i] = topic["topic_id"]
            matched_sims[i] = float(best_sim[i])
            if _weeks_between(week, topic["last_week"]) >= DORMANT_WEEKS:
                dormant_bonus[i] = DORMANT_NOVELTY_BONUS
    return {
        "novelty": np.clip(1.0 - matched_sims + dormant_bonus, 0.0, 1.0),
        "matched_ids": matched_ids,
        "matched_sims": matched_sims,
    }


def _signal_entity_counts(cluster, items):
    counts = defaultdict(int)
    for i in cluster:
        for entity, label in items[i]["entities"].items():
            if _is_signal_entity(entity, label):
                counts[entity] += 1
    return counts


def _entity_signals(inputs):
    """Entity weight, trend velocity, and information richness, one value per cluster."""
    items = inputs["items"]
    entity_idf = inputs["entity_idf"]
    max_idf = inputs["max_idf"]
    velocities = inputs["velocities"]
    weight_raw = []
    trend_raw = []
    richness_raw = []
    for cluster in inputs["clusters"]:
        counts = _signal_entity_counts(cluster, items)
        weight = 0.0
        for entity, count in counts.items():
            idf = entity_idf.get(entity, max_idf)
            boost = KEY_ENTITY_BOOST if entity in KEY_ENTITIES else 1
            weight += count * idf * boost
        weight_raw.append(weight)
        loudest = sorted(counts, key=counts.get, reverse=True)[:TOP_ENTITIES_PER_CLUSTER]
        speeds = [velocities.get(entity, 0.0) for entity in loudest]
        trend_raw.append(sum(speeds) / max(len(speeds), 1))
        richness_raw.append(_information_richness(cluster, items, entity_idf, max_idf))
    trend = np.clip(np.array(trend_raw, dtype=float), TREND_CLIP_LOW, TREND_CLIP_HIGH)
    return {
        "entity_signal": _normalize(np.array(weight_raw, dtype=float)),
        "trend": _normalize(trend),
        "richness": _normalize(np.array(richness_raw, dtype=float)),
    }


def _coverage_gap(inputs, medoids, bm25):
    """How far behind each cluster's best-matching aspect is on the coverage ledger.

    Anchor invariant: aspects are derived FROM USER_PROFILE, so this sharpens
    within the profile's space and never redirects it.

    Args:
        inputs: the scoring inputs dictionary.
        medoids: index of the representative item of each cluster.
        bm25: fitted BM25 index over every item this week.

    Returns:
        {"coverage_gap": array, "aspect_coverage": per-aspect max normalised to [0, 1]}
    """
    clusters = inputs["clusters"]
    aspects = inputs["aspects"]
    if not aspects:
        return {"coverage_gap": np.zeros(len(clusters)), "aspect_coverage": {}}
    per_aspect = {}
    coverage = {}
    for label, descriptor in aspects:
        tokens = _norm(descriptor).split() or _norm(label).split()
        medoid_scores = _medoid_scores(bm25, medoids, tokens)
        per_aspect[label] = medoid_scores
        if medoid_scores.size:
            coverage[label] = float(medoid_scores.max())
    if coverage:
        highest = max(coverage.values()) or 1.0
        coverage = {label: value / highest for label, value in coverage.items()}
    if not per_aspect:
        return {"coverage_gap": np.zeros(len(clusters)), "aspect_coverage": coverage}
    stack = np.stack(list(per_aspect.values()))
    labels = list(per_aspect.keys())
    best_aspect = stack.argmax(axis=0)
    gap_raw = np.zeros(len(clusters))
    for i in range(len(clusters)):
        gap_raw[i] = float(inputs["debt"].get(labels[best_aspect[i]], 0.0))
    if gap_raw.max() > 0:
        return {"coverage_gap": _normalize(gap_raw), "aspect_coverage": coverage}
    return {"coverage_gap": gap_raw, "aspect_coverage": coverage}


def _dense_exclusion(inputs, medoids):
    """Highest embedding cosine to any exclusion descriptor, per cluster."""
    embeddings = inputs["embeddings"]
    aspects = inputs["exclusion_aspects"]
    dense = np.zeros(len(medoids))
    if embeddings is None:
        return dense
    aspect_embs = _embed_aspects(aspects, _profile_hash() + "_excl")
    if aspect_embs is None:
        return dense
    medoid_embs = embeddings[medoids]
    for position in range(len(aspects)):
        dense = np.maximum(dense, medoid_embs @ aspect_embs[position])
    return dense


def _profile_exclusion(inputs, medoids, bm25):
    """How strongly each cluster matches the profile's 'not relevant' clause.

    The dense lane is what catches celebrity and consumer-tech leakage where
    BM25 vocabulary fails. This is a filter input only, never a positive term.
    """
    aspects = inputs["exclusion_aspects"]
    blank = np.zeros(len(medoids))
    if not aspects:
        return blank
    lexical = np.zeros(len(medoids))
    for label, descriptor in aspects:
        tokens = _norm(descriptor).split() or _norm(label).split()
        if not tokens:
            continue
        lexical = np.maximum(lexical, _medoid_scores(bm25, medoids, tokens))
    dense = _dense_exclusion(inputs, medoids)
    combined = lexical / (lexical.max() + 1e-9) + dense / (dense.max() + 1e-9)
    if combined.max() > 0:
        return _normalize(combined)
    return blank


def _persistence(inputs, matched_ids):
    """Recurrence rate of each cluster's matched bank topic, and its score term.

    Args:
        inputs: the scoring inputs dictionary.
        matched_ids: matched bank topic id per cluster, None where unmatched.

    Returns:
        {"rate": array where -1 means unmatched, "signal": array in [0, 1]}
    """
    clusters = inputs["clusters"]
    bank = inputs["bank"]
    week = inputs["week"]
    rate = -np.ones(len(clusters))
    if not bank:
        return {"rate": rate, "signal": np.clip(rate, 0.0, 1.0)}
    by_id = {topic["topic_id"]: topic for topic in bank}
    for i in range(len(clusters)):
        topic = by_id.get(matched_ids[i])
        if topic is None:
            continue
        span = max(_weeks_between(week, topic["first_week"]), 1)
        rate[i] = float(topic["weeks_seen"]) / span
    signal = np.clip(rate, 0.0, 1.0)
    if signal.max() > 0:
        signal = _normalize(signal)
    return {"rate": rate, "signal": signal}


def _source_prior(inputs):
    items = inputs["items"]
    means = []
    for cluster in inputs["clusters"]:
        priors = [SOURCE_PRIORS.get(items[i]["source"], 1.0) for i in cluster]
        means.append(np.mean(priors))
    return _normalize(np.array(means))


def _source_breadth(inputs):
    """Count the distinct outlets covering each cluster, log-scaled."""
    items = inputs["items"]
    counts = []
    for cluster in inputs["clusters"]:
        sources = {items[i]["source"] for i in cluster}
        counts.append(float(len(sources)))
    return _normalize(np.log1p(np.array(counts, dtype=float)))


def _recency(inputs):
    """Decay each cluster by the age of its freshest item, so stale items fade."""
    items = inputs["items"]
    now = datetime.now(timezone.utc)
    freshness = []
    for cluster in inputs["clusters"]:
        ages = []
        for i in cluster:
            age = _item_age_days(items[i].get("ts"), now)
            if age is not None:
                ages.append(age)
        youngest = min(ages) if ages else 0.0
        freshness.append(np.exp(-youngest / RECENCY_DECAY_DAYS))
    return _normalize(np.array(freshness, dtype=float))


def _spotlight_entities(cluster, items, velocities):
    """Pick the loudest named entities in a cluster for the digest spotlight."""
    counts = defaultdict(int)
    for i in cluster:
        for entity, label in items[i]["entities"].items():
            if _is_signal_entity(entity, label) and label in _SPOTLIGHT_LABELS:
                counts[entity] += 1
    ranked = sorted(
        counts.items(),
        key=lambda pair: (pair[1], velocities.get(pair[0], 0.0)),
        reverse=True,
    )
    return ranked[:TOP_ENTITIES_PER_CLUSTER]


def _signal_row(signals, exclusion, persistence_rate, position):
    row = {name: float(arr[position]) for name, arr in signals.items()}
    row["profile_exclusion"] = float(exclusion[position])
    row["persistence_rate"] = float(persistence_rate[position])
    return row


def score_clusters(inputs):
    """Score every cluster with the weighted linear model.

    Args:
        inputs: one dictionary holding clusters, items, X, vec, velocities,
            bank, aspects, debt, entity_idf, max_idf, weights, week,
            exclusion_aspects, and embeddings. embeddings is a normalized dense
            matrix, or None to score on the lexical lane alone.

    Returns:
        {"scored": cluster rows sorted best first,
         "aspect_coverage": per-aspect coverage this week, normalised to [0, 1]}
    """
    clusters = inputs["clusters"]
    items = inputs["items"]
    medoids = [cluster_medoid(cluster, inputs["X"]) for cluster in clusters]
    medoid_vecs = sp.vstack([inputs["X"][m] for m in medoids])
    corpus = [_norm(f"{item['title']} {item['body']}").split() for item in items]
    bm25 = BM25Okapi(corpus)

    relevance = _lexical_relevance(inputs, medoids, bm25)
    relevance = relevance + _dense_relevance(inputs, medoids)
    novelty = _novelty_against_bank(inputs, medoid_vecs)
    entities = _entity_signals(inputs)
    gap = _coverage_gap(inputs, medoids, bm25)
    persistence = _persistence(inputs, novelty["matched_ids"])
    exclusion = _profile_exclusion(inputs, medoids, bm25)

    sizes = np.array([len(cluster) for cluster in clusters])
    signals = {
        "coverage": np.log1p(sizes) / np.log1p(max(sizes.max(), 1)),
        "prior": _source_prior(inputs),
        "novelty": novelty["novelty"],
        "relevance": _normalize(relevance),
        "entity_signal": entities["entity_signal"],
        "trend": entities["trend"],
        "richness": entities["richness"],
        "coverage_gap": gap["coverage_gap"],
        "persistence": persistence["signal"],
        "source_breadth": _source_breadth(inputs),
        "recency": _recency(inputs),
    }
    score = np.zeros(len(clusters))
    for name, arr in signals.items():
        score = score + inputs["weights"].get(name, 0.0) * arr

    scored = []
    for i in np.argsort(-score):
        scored.append({
            "idxs": clusters[i],
            "medoid": medoids[i],
            "score": float(score[i]),
            "vec": medoid_vecs[i],
            "entities": _spotlight_entities(clusters[i], items, inputs["velocities"]),
            "sector": classify_sector(clusters[i], items),
            "category": classify_category(clusters[i], items),
            "matched_topic_id": novelty["matched_ids"][i],
            "matched_sim": float(novelty["matched_sims"][i]),
            "signals": _signal_row(signals, exclusion, persistence["rate"], i),
        })
    return {"scored": scored, "aspect_coverage": gap["aspect_coverage"]}


def mmr_select(scored, k=MMR_SELECT_K, lam=MMR_LAMBDA):
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


def mmr_dynamic_select(scored, lam=MMR_LAMBDA, min_k=MMR_MIN_K, max_k=MMR_MAX_K,
                       relevance_floor=MMR_RELEVANCE_FLOOR):
    """MMR over the whole qualifying pool, then cut at the score-knee.

    Floors at min_k (don't render an empty-looking digest on a quiet week),
    ceilings at max_k (cost / readability guard, rarely binds), and unconditionally
    keeps any cluster whose relevance signal exceeds relevance_floor: that
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

    # Always keep clusters above the relevance floor, even past the knee, they
    # are on-profile signal we don't want to drop just because the curve broke.
    keep_idxs = set(range(cut))
    for i, c in enumerate(ordered[cut:max_k], start=cut):
        if c.get("signals", {}).get("relevance", 0.0) >= relevance_floor:
            keep_idxs.add(i)

    return [ordered[i] for i in sorted(keep_idxs)]


def _filter_off_profile(scored, history_weeks, relevance_floor=OFF_PROFILE_RELEVANCE_FLOOR,
                        persistence_floor=OFF_PROFILE_PERSISTENCE_FLOOR):
    """Drop clusters whose profile_exclusion signal exceeds relevance AND
    whose relevance falls below the floor. Layer A of §2.4.

    Once history_weeks >= 10, the filter tightens with Layer B: a
    high-exclusion cluster also has to clear the persistence floor (its
    matched topic must have a non-trivial recurrence rate) to survive.
    Topics with no match in the bank (persistence_rate == -1) are treated
    as failing the persistence test only once history is deep enough.

    Additionally, clusters below DUMP_RELEVANCE_FLOOR with non-trivial
    exclusion are dropped regardless of the other gates: these are the
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
    t = re.sub(r"[\u2014\-:|]\s*(?:final|last|only|just)\s+\d+\s*(?:hours?|days?|hrs?)\s+left.*$", "", t, flags=re.I)
    t = re.sub(r"[\u2014\-:|]\s*(?:early\s+bird|deadline|register|save\s+\$?\d+).*$", "", t, flags=re.I)
    t = re.sub(r"\d+\s*(?:hours?|days?|hrs?)\s+left.*$", "", t, flags=re.I)
    tokens = _norm(t).split()
    return " ".join(tokens[:8])


def _dedupe_top(top, items):
    """Cross-cluster dedup. Each URL and each title signature (see
    _title_signature) appears in at most one cluster, the highest-scoring
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


# ---- Claude summarization ----

def _retry(fn, attempts=RETRY_ATTEMPTS, base=RETRY_BACKOFF_BASE):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(base ** i)


# ---- Macro-trend memory ----

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
    request = partial(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=(
            "You are a media industry researcher synthesising longitudinal industry data. "
            "Identify structural shifts, market concentration dynamics, and regulatory "
            "trajectories that span multiple observation periods, not individual events. "
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
    )
    msg = _retry(request)

    macro_text = msg.content[0].text
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    week_ids = [r[0] for r in rows_to_use]
    conn.execute(
        "INSERT OR REPLACE INTO macro_trends(month, summary, weeks, created_at) VALUES(?,?,?,?)",
        (month, macro_text, json.dumps(week_ids), now_iso()),
    )
    print(f"macro trend generated for {month} covering weeks {week_ids}")
    return macro_text


def get_archive_links(conn, n=ARCHIVE_LINKS_COUNT):
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


# ---- Claude summarization ----

def sectors_present(top):
    """[(sector_key, [clusters])] in SECTOR_ORDER, only sectors with clusters."""
    grouped = defaultdict(list)
    for c in top:
        grouped[c.get("sector", DEFAULT_SECTOR)].append(c)
    return [(s, grouped[s]) for s in SECTOR_ORDER if grouped.get(s)]


def _new_category_bucket():
    return defaultdict(list)


def _group_by_sector_category(top):
    grouped = defaultdict(_new_category_bucket)
    for cluster in top:
        sector_key = cluster.get("sector", DEFAULT_SECTOR)
        category_key = cluster.get("category", DEFAULT_CATEGORY)
        grouped[sector_key][category_key].append(cluster)
    return grouped


def _cluster_block_lines(cid, cluster, items):
    """The prompt lines for one numbered cluster: header, body snippet,
    entities."""
    title = items[cluster["medoid"]]["title"]
    # Sentence-level MMR: picks the body sentence that maximises relevance
    # to USER_PROFILE while penalising overlap with the title the LLM
    # already sees, instead of a blind body[:240] lede assumption.
    body = _mmr_sentence(items[cluster["medoid"]].get("body") or "", title, USER_PROFILE)
    entity_names = " · ".join(entity for entity, _ in cluster["entities"][:4])
    sources = sorted({items[i]["source"].upper() for i in cluster["idxs"][:5]})
    lines = [
        f"{cid} (score {cluster.get('score', 0.0):.2f}, sources "
        f"{', '.join(sources)}): {title}"
    ]
    if body:
        lines.append(f"   {body}")
    if entity_names:
        lines.append(f"   entities: {entity_names}")
    return lines


def _build_cluster_blocks(top, items):
    """Number every cluster C1, C2, … in sector, then category, then
    score-descending order, and render one prompt-ready text block per
    sector.

    Returns:
        {"blocks": one text block per sector, "cluster_index": {"C1":
        cluster, ...} so the parser can look picks back up}.
    """
    grouped = _group_by_sector_category(top)
    cluster_index = {}
    blocks = []
    counter = 0
    for sector_key in SECTOR_ORDER:
        if sector_key not in grouped:
            continue
        section_lines = [f"## {SECTORS[sector_key]}"]
        for category_key, _ in CATEGORIES:
            if category_key not in grouped[sector_key]:
                continue
            category_clusters = sorted(
                grouped[sector_key][category_key],
                key=lambda cluster: -cluster.get("score", 0.0),
            )
            section_lines.append(f"### {CATEGORY_NAMES[category_key]}")
            for cluster in category_clusters:
                counter += 1
                cid = f"C{counter}"
                cluster_index[cid] = cluster
                section_lines.extend(_cluster_block_lines(cid, cluster, items))
        blocks.append("\n".join(section_lines))
    return {"blocks": blocks, "cluster_index": cluster_index}


def _summarise_body_instructions(blocks, longitudinal_context, macro_context):
    long_section = f"\n\n{longitudinal_context}\n" if longitudinal_context else ""
    macro_section = (
        "\n\nMacro context from prior weeks (structural trends; cite when "
        f"relevant):\n{macro_context}\n" if macro_context else ""
    )
    return (
        SUMMARISE_BODY_PREAMBLE
        + long_section
        + macro_section
        + "\n\nClusters this week:\n\n"
        + "\n\n".join(blocks)
    )


def summarise(top, items, macro_context=None, history_depth="warmup",
              longitudinal_context=None):
    """Editorial spotlight: ask the LLM to pick up to 5 most meaningful
    clusters per (sector, category), each summarised in 1-3 sentences,
    referenced by the cluster's `Cn` index so the renderer can attach
    the real article links.

    ``history_depth`` is retained for symmetry but no longer drives the
    output format: longitudinal_context being non-empty is what unlocks
    multi-week framing in the LLM's prose.

    Returns:
        {"summary": the LLM's raw text, "cluster_index": {"C1": cluster,
        ...} mapping every Cn the LLM saw back to its cluster dict}.
    """
    built = _build_cluster_blocks(top, items)
    body_instructions = _summarise_body_instructions(
        built["blocks"], longitudinal_context, macro_context
    )

    client = anthropic.Anthropic()
    request = partial(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=2400,
        system=SUMMARISE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": body_instructions}],
    )
    msg = _retry(request)
    return {"summary": msg.content[0].text, "cluster_index": built["cluster_index"]}


_CN_RE = re.compile(r"^-\s*(C\d+)\s*:\s*(.+)$", re.IGNORECASE)


def _match_category(cat_name):
    for category_key, name in CATEGORIES:
        if name.lower() == cat_name.lower():
            return category_key
    return None


def parse_spotlight(summary, cluster_index):
    """Parse the summarise() output back into the per-(sector, category)
    picks the LLM made. Returns a dict:

        {sector_key: {category_key: [(cluster_obj, summary_text), ...]}}

    Robust to a few drift cases: an unmapped sector or category name is
    dropped; a Cn that isn't in cluster_index is dropped; pure prose lines
    between bullets are ignored. The cluster ordering within each category
    is preserved (the LLM's editorial sequence)."""
    out = defaultdict(_new_category_bucket)
    sector_key = None
    category_key = None
    seen_cids = set()
    for line in summary.splitlines():
        s = line.rstrip()
        if not s:
            continue
        if s.startswith("## "):
            title = s[3:].strip()
            sector_key = _SECTOR_NAME_TO_KEY.get(title.lower()) if title else None
            category_key = None
            continue
        if s.startswith("### "):
            category_key = _match_category(s[4:].strip())
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


# Reverse-lookup from a sector display name back to its SECTOR_ORDER key, so
# render_static/render_email can match an LLM-emitted "## SectorName" heading
# to the cluster bucket for that sector. Case-insensitive, ignores whitespace.
_SECTOR_NAME_TO_KEY = {v.strip().lower(): k for k, v in SECTORS.items()}


def _cluster_distinct_links(c, items, cap=CLUSTER_LINKS_CAP):
    """Distinct (url, source, title) tuples within a cluster, preserving
    original ordering of c['idxs'] and capped at `cap`. Used by render_*
    to decide singleton vs multi-link layout AND to render the actual link
    list, the single source of truth."""
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


# ---- Email output ----

def _sector_breakdown(sector_clusters, picks_for_sector):
    """Split a sector's clusters into spotlit (with an LLM summary) and
    dump (by category, score-descending). Shared by both renderers.

    Returns:
        {"spotlight": {category_key: [(cluster, summary_text), ...]},
        "dump": {category_key: [cluster, ...]}}.
    """
    spotlit_ids = set()
    spotlight_by_cat = defaultdict(list)
    for cat_key, picks in (picks_for_sector or {}).items():
        for cluster, summary_text in picks:
            if id(cluster) in spotlit_ids:
                continue
            spotlit_ids.add(id(cluster))
            spotlight_by_cat[cat_key].append((cluster, summary_text))
    dump_by_cat = defaultdict(list)
    for cluster in sector_clusters:
        if id(cluster) in spotlit_ids:
            continue
        dump_by_cat[cluster.get("category", DEFAULT_CATEGORY)].append(cluster)
    for cat_key in list(dump_by_cat.keys()):
        dump_by_cat[cat_key].sort(key=lambda cluster: -cluster.get("score", 0.0))
    return {"spotlight": spotlight_by_cat, "dump": dump_by_cat}


def _email_link_line(url, source, title, also=False):
    if also:
        return f'<br>↳ <a href="{url}" style="color:#666"><b>{source.upper()}:</b> {title}</a>'
    return f'↳ <a href="{url}"><b>{source.upper()}:</b> {title}</a>'


def _render_email_spotlight_item(cluster, summary_text, items):
    links = _cluster_distinct_links(cluster, items, cap=CLUSTER_LINKS_CAP)
    if not links:
        return ""
    parts = [
        "<li style='margin-bottom:10px'>"
        f"<div style='color:#111;font-size:14px'>{summary_text}</div>"
        "<div style='font-size:12px;margin-top:4px'>"
    ]
    url, source, title = links[0]
    parts.append(_email_link_line(url, source, title))
    for url, source, title in links[1:]:
        parts.append(_email_link_line(url, source, title, also=True))
    parts.append("</div></li>")
    return "".join(parts)


def _render_email_spotlight(spotlight_by_cat, items):
    """SPOTLIGHT section: LLM-picked stories, grouped by category."""
    parts = []
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
            parts.append(_render_email_spotlight_item(cluster, summary_text, items))
        parts.append("</ul>")
    return "\n".join(parts)


def _render_email_dump_item(cluster, items):
    links = _cluster_distinct_links(cluster, items, cap=SECTION_DUMP_LINKS_CAP)
    if not links:
        return ""
    parts = []
    url, source, title = links[0]
    parts.append(
        f'<li style="margin-bottom:3px;font-size:13px">'
        f'<a href="{url}"><b>{source.upper()}:</b> {title}</a>'
    )
    for url, source, title in links[1:]:
        parts.append(
            f'<br><span style="font-size:12px;color:#666">↳ '
            f'<a href="{url}" style="color:#666"><b>{source.upper()}:</b> '
            f'{title}</a></span>'
        )
    parts.append("</li>")
    return "".join(parts)


def _render_email_dump(dump_by_cat, items):
    """DUMP section: mechanical listing, grouped by category, score-descending."""
    parts = [
        "<div style='color:#888;font-size:11px;text-transform:uppercase;"
        "letter-spacing:.05em;margin-bottom:6px'>All stories</div>"
    ]
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
        for cluster in cat_clusters:
            parts.append(_render_email_dump_item(cluster, items))
        parts.append("</ul>")
    return "\n".join(parts)


def _render_email_sector(sector_key, sector_clusters, picks_for_sector, items):
    breakdown = _sector_breakdown(sector_clusters, picks_for_sector)
    spotlight_by_cat, dump_by_cat = breakdown["spotlight"], breakdown["dump"]

    parts = [
        f"<h3 style='border-bottom:1px solid #ddd;padding-bottom:4px;"
        f"margin-top:28px;color:#f59e0b'>{SECTORS[sector_key]} "
        f"<small style='color:#888;font-weight:400'>"
        f"({len(sector_clusters)} stories)</small></h3>"
    ]
    if spotlight_by_cat:
        parts.append(_render_email_spotlight(spotlight_by_cat, items))
    if spotlight_by_cat and dump_by_cat:
        parts.append("<hr style='border:none;border-top:1px dashed #ccc;margin:14px 0'>")
    if dump_by_cat:
        parts.append(_render_email_dump(dump_by_cat, items))
    return "\n".join(parts)


def _render_email_macro(macro):
    month_label = datetime.now().strftime("%B %Y")
    paragraph_style = "color:#aaa;font-size:13px;border-left:2px solid #444;padding-left:12px"
    body = macro.replace("\n\n", f"</p><p style='{paragraph_style}'>")
    return (
        f"<h3 style='color:#888;font-size:13px;margin-top:24px'>"
        f"Macro Trends: {month_label}</h3>"
        f"<p style='{paragraph_style}'>{body}</p>"
    )


def render_email(summary, cluster_index, top, items, macro=None):
    date_str = datetime.now().strftime("%b %d, %Y")
    picks = parse_spotlight(summary, cluster_index)

    parts = [f"<h2>Weekly Media Industry News: {date_str}</h2>"]
    for sector_key, sector_clusters in sectors_present(top):
        parts.append(_render_email_sector(
            sector_key, sector_clusters, picks.get(sector_key, {}), items
        ))
    if macro:
        parts.append(_render_email_macro(macro))
    return "\n".join(parts)


def send_email(html):
    msg = MIMEText(html, "html")
    msg["Subject"] = f"Weekly Media Industry News: {datetime.now().strftime('%b %d')}"
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = os.environ["DIGEST_TO"]
    with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], 465) as s:
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        s.send_message(msg)


# ---- Static site ----

def _static_spotlight_item_html(cluster, summary_text, items):
    links = _cluster_distinct_links(cluster, items, cap=CLUSTER_LINKS_CAP)
    if not links:
        return ""
    parts = []
    primary_url, primary_src, primary_title = links[0]
    parts.append(
        f'<li class="spot-item">'
        f'<div class="spot-summary">{summary_text}</div>'
        f'<div class="spot-links">'
        f'<a href="{primary_url}" target="_blank">'
        f'<span class="src">{primary_src.upper()}</span> '
        f'{primary_title}</a>'
    )
    for url, source, title in links[1:]:
        parts.append(
            f'<div class="also">↳ <a href="{url}" target="_blank">'
            f'<span class="src">{source.upper()}</span> {title}</a></div>'
        )
    parts.append('</div></li>')
    return "".join(parts)


def _static_spotlight_html(spotlight_by_cat, items):
    parts = ['<div class="spotlight">']
    for cat_key, _ in CATEGORIES:
        if cat_key not in spotlight_by_cat:
            continue
        parts.append(f'<h3 class="cat-block">{CATEGORY_NAMES[cat_key]}</h3>')
        parts.append('<ul class="spotlight-list">')
        for cluster, summary_text in spotlight_by_cat[cat_key]:
            parts.append(_static_spotlight_item_html(cluster, summary_text, items))
        parts.append('</ul>')
    parts.append('</div>')
    return "".join(parts)


def _static_dump_item_html(cluster, items):
    links = _cluster_distinct_links(cluster, items, cap=SECTION_DUMP_LINKS_CAP)
    if not links:
        return ""
    parts = []
    primary_url, primary_src, primary_title = links[0]
    parts.append(
        f'<li>'
        f'<a href="{primary_url}" target="_blank">'
        f'<span class="src">{primary_src.upper()}</span> '
        f'{primary_title}</a>'
    )
    for url, source, title in links[1:]:
        parts.append(
            f'<div class="also">↳ <a href="{url}" target="_blank">'
            f'<span class="src">{source.upper()}</span> {title}</a></div>'
        )
    parts.append('</li>')
    return "".join(parts)


def _static_dump_html(dump_by_cat, items):
    parts = ['<div class="dump">', '<h4 class="dump-header">All stories</h4>']
    for cat_key, _ in CATEGORIES:
        if cat_key not in dump_by_cat:
            continue
        cat_clusters = dump_by_cat[cat_key]
        parts.append(
            f'<h5 class="dump-cat">{CATEGORY_NAMES[cat_key]} '
            f'<span class="sn">{len(cat_clusters)}</span></h5>'
        )
        parts.append('<ul class="dump-list">')
        for cluster in cat_clusters:
            parts.append(_static_dump_item_html(cluster, items))
        parts.append('</ul>')
    parts.append('</div>')
    return "".join(parts)


def _static_sector_block(sector_key, sector_clusters, picks_for_sector, items):
    breakdown = _sector_breakdown(sector_clusters, picks_for_sector)
    spotlight_by_cat, dump_by_cat = breakdown["spotlight"], breakdown["dump"]

    block = [
        f'<section class="sec">'
        f'<h2 class="sector">{SECTORS[sector_key]} '
        f'<span class="sn">{len(sector_clusters)} stories</span></h2>'
    ]
    if spotlight_by_cat:
        block.append(_static_spotlight_html(spotlight_by_cat, items))
    if spotlight_by_cat and dump_by_cat:
        block.append('<hr class="sep">')
    if dump_by_cat:
        block.append(_static_dump_html(dump_by_cat, items))
    block.append('</section>')
    return "".join(block)


def _static_macro_section(macro):
    if not macro:
        return ""
    month_label = datetime.now().strftime("%B %Y")
    macro_html = "".join(
        f"<p>{paragraph.strip()}</p>" for paragraph in macro.split("\n\n") if paragraph.strip()
    )
    return (
        f'<section class="macro">'
        f'<h2>Macro Trends: {month_label}</h2>'
        f'<div class="macro-body">{macro_html}</div>'
        f'</section>'
    )


def _static_archive_section(archive_links):
    if not archive_links:
        return ""
    links_html = "".join(
        f'<li><a href="{url}">{wk}</a></li>' for wk, url in archive_links
    )
    return f'<nav class="archive"><h2>Archive</h2><ul>{links_html}</ul></nav>'


def _static_page_shell(page):
    """Wrap the rendered sections in the full HTML document.

    Args:
        page: {"date_str", "week", "story_count", "sector_html", "macro_section",
        "archive_section"}.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weekly Media Industry News: {page["date_str"]}</title>
<style>
{STATIC_PAGE_CSS}
</style>
</head>
<body>
<header>
  <h1>Weekly Media Industry News</h1>
  <div class="wk">Week {page["week"]} &nbsp;·&nbsp; {page["date_str"]} &nbsp;·&nbsp; {page["story_count"]} stories</div>
</header>
{page["sector_html"]}
{page["macro_section"]}
{page["archive_section"]}
<footer>Weekly Media Industry News &nbsp;·&nbsp; <a href="coverage.html">Coverage Analysis</a> &nbsp;·&nbsp; Generated {page["date_str"]}</footer>
</body>
</html>"""


def render_static(payload, macro=None, archive_links=None):
    """Render the full static digest page.

    Args:
        payload: {"summary", "cluster_index", "top", "items", "week"}, the
        same shape `_publish_everything` carries after the run.
        macro: prior-weeks macro-trend prose, or None.
        archive_links: [(week, url), ...] for the archive nav, or None.
    """
    summary = payload["summary"]
    cluster_index = payload["cluster_index"]
    top = payload["top"]
    items = payload["items"]
    date_str = datetime.now().strftime("%B %d, %Y")
    picks = parse_spotlight(summary, cluster_index)

    sector_blocks_html = []
    for sector_key, sector_clusters in sectors_present(top):
        sector_blocks_html.append(_static_sector_block(
            sector_key, sector_clusters, picks.get(sector_key, {}), items
        ))

    return _static_page_shell({
        "date_str": date_str,
        "week": payload["week"],
        "story_count": len(top),
        "sector_html": "".join(sector_blocks_html),
        "macro_section": _static_macro_section(macro),
        "archive_section": _static_archive_section(archive_links),
    })


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


# ---- Persistence ----

def load_week_items(conn):
    """Return every item in the rolling 7-day window, including ones an earlier
    digest already used. `used_in_digest` is recorded by save_digest for
    reference and deliberately does not gate retrieval, so a persistent topic
    can surface across two digests."""
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
    ids = []
    for cluster in top:
        for i in cluster["idxs"]:
            ids.append(items[i]["id"])
    conn.executemany("UPDATE items SET used_in_digest=1 WHERE id=?", [(i,) for i in ids])


# ---- Orchestration ----

def _build_corpus(rows):
    """Enrich this week's rows, cluster them, and embed them."""
    print(f"digest [2/12] enriching {len(rows)} items")
    items = enrich(rows)

    print("digest [3/12] building TF-IDF + clustering")
    fitted = build_tfidf(items)
    clusters = cluster_average_linkage(fitted["matrix"])

    print(f"digest [4/12] embedding {len(items)} items")
    texts = []
    for item in items:
        body = (item["body"] or "")[:_EMBED_BODY_CHARS]
        texts.append(f"{item['title']} {body}")
    embeddings = _embed(texts)
    if embeddings is None:
        print("embeddings: unavailable, using lexical-only fallback")
    else:
        print(f"embeddings: {embeddings.shape[0]} items × {embeddings.shape[1]} dims")

    return {
        "items": items,
        "matrix": fitted["matrix"],
        "vectorizer": fitted["vectorizer"],
        "clusters": clusters,
        "embeddings": embeddings,
    }


def _count_signal_entities(items):
    counts = defaultdict(int)
    for item in items:
        for entity, label in item["entities"].items():
            if _is_signal_entity(entity, label):
                counts[entity] += 1
    return counts


def _load_discourse_context(conn):
    """Load the IDF, aspects, coverage debt, topic bank, and tuned weights.

    Weight tuning runs here, before scoring, so this week uses the latest blend.
    It no-ops until cluster_signals has accumulated enough weeks.

    Returns:
        one dictionary of everything the scorer and the filters read.
    """
    print("digest [5/12] loading discourse context")
    idf_result = load_entity_idf(conn)
    entity_idf, max_idf = idf_result["idf"], idf_result["max_idf"]
    aspects = get_profile_aspects(conn)
    try:
        tune_weights(conn)
    except Exception as error:
        print(f"tune_weights failed, continuing with existing weights: {error}")
    history_weeks = conn.execute(
        "SELECT COUNT(DISTINCT week) FROM cluster_signals"
    ).fetchone()[0] or 0
    return {
        "entity_idf": entity_idf,
        "max_idf": max_idf,
        "aspects": aspects,
        "exclusion_aspects": get_exclusion_aspects(conn),
        "debt": compute_coverage_debt(conn, aspects),
        "bank": load_topic_bank(conn),
        "weights": load_weights(conn),
        "history_weeks": history_weeks,
    }


def _rank_clusters(corpus, context, run_state):
    """Score the clusters, drop the off-profile ones, then select and dedupe.

    Args:
        corpus: the output of `_build_corpus`.
        context: the output of `_load_discourse_context`.
        run_state: {"velocities": ..., "week": ...} for this run.

    Returns:
        {"top": the selected clusters, "aspect_coverage": coverage for the ledger}
    """
    clusters = corpus["clusters"]
    print(f"digest [6/12] scoring {len(clusters)} clusters")
    ranking = score_clusters({
        "clusters": clusters,
        "items": corpus["items"],
        "X": corpus["matrix"],
        "vec": corpus["vectorizer"],
        "velocities": run_state["velocities"],
        "bank": context["bank"],
        "aspects": context["aspects"],
        "debt": context["debt"],
        "entity_idf": context["entity_idf"],
        "max_idf": context["max_idf"],
        "weights": context["weights"],
        "week": run_state["week"],
        "exclusion_aspects": context["exclusion_aspects"],
        "embeddings": corpus["embeddings"],
    })

    history_weeks = context["history_weeks"]
    scored = ranking["scored"]
    kept = _filter_off_profile(scored, history_weeks)
    print(f"off-profile filter: {len(scored)} -> {len(kept)} clusters "
          f"(history_weeks={history_weeks})")

    print("digest [7/12] MMR select + dedup")
    # Dedup runs before the persistence writes so the topic bank and
    # cluster_signals reflect the same set the reader is shown.
    top = _dedupe_top(mmr_dynamic_select(kept), corpus["items"])
    return {"top": top, "aspect_coverage": ranking["aspect_coverage"]}


def _render_coverage(conn):
    from evaluate import report as eval_report, render_html as eval_render
    try:
        return eval_render(eval_report(conn, weeks=COVERAGE_REPORT_WEEKS))
    except Exception as error:
        print(f"coverage render failed: {error}")
        return None


def _send_monthly_metrics(week):
    """Send the self-learning metrics email on the first Sunday of the month."""
    try:
        from metrics_email import is_first_sunday_of_month, send_metrics_email
        if not is_first_sunday_of_month():
            return
        with sqlite3.connect(LOCAL_DB) as metrics_conn:
            send_metrics_email(metrics_conn, week)
        print(f"monthly metrics email sent for {week}")
    except Exception as error:
        print(f"metrics email failed (non-fatal): {error}")


def _publish_everything(conn, payload):
    """Push the database, publish the static site, and send the digest email.

    Everything read from `conn` is read before the connection closes, so the
    coverage report sees the state that was just committed.

    Args:
        conn: the open database connection, closed before publishing.
        payload: {"summary", "cluster_index", "top", "items", "week"}.
    """
    from evaluate import publish_coverage

    macro = load_macro_context(conn)
    archive_links = get_archive_links(conn)
    coverage_html = _render_coverage(conn)
    conn.close()

    print("digest [11/12] pushing db + publishing static site")
    push_db()
    week = payload["week"]
    publish_static(render_static(payload, macro, archive_links), week)
    if coverage_html:
        publish_coverage(coverage_html, week)

    print("digest [12/12] sending email")
    send_email(render_email(payload["summary"], payload["cluster_index"],
                            payload["top"], payload["items"], macro))
    _send_monthly_metrics(week)


def run():
    with pipeline_lock():
        print("digest: starting")
        for number, name in enumerate(STEP_PLAN, start=1):
            print(f"  · {number}/{len(STEP_PLAN)}  {name}")
        print("digest [1/12] pulling db")
        conn = pull_db()
        rows = load_week_items(conn)
        if len(rows) < MIN_ITEMS_FOR_DIGEST:
            print(f"too few items: {len(rows)}")
            return

        corpus = _build_corpus(rows)
        items = corpus["items"]
        week = datetime.now(timezone.utc).strftime("%Y-W%V")
        entity_counts = _count_signal_entities(items)
        velocities = compute_velocities(entity_counts, load_entity_history(conn))

        context = _load_discourse_context(conn)
        ranking = _rank_clusters(corpus, context, {
            "velocities": velocities,
            "week": week,
        })
        top = ranking["top"]

        longitudinal_context = build_longitudinal_context(
            conn, week, entity_counts, top, context["bank"],
        )

        print("digest [8/12] updating topic bank")
        update_topic_bank(conn, top, corpus["vectorizer"], week)
        save_cluster_signals(conn, week, top)
        save_coverage_ledger(conn, week, ranking["aspect_coverage"])

        history_depth = "full" if context["history_weeks"] >= FULL_HISTORY_WEEKS else "warmup"
        print(f"digest [9/12] LLM summarise ({len(top)} clusters, history={history_depth})")
        summarised = summarise(
            top, items, load_macro_context(conn), history_depth=history_depth,
            longitudinal_context=longitudinal_context,
        )
        summary, cluster_index = summarised["summary"], summarised["cluster_index"]

        print("digest [10/12] saving digest state")
        save_digest(conn, top, items, summary, corpus["vectorizer"])
        save_entity_history(conn, dict(entity_counts), week)
        macro_new = generate_macro_trends(conn)
        conn.commit()

        _publish_everything(conn, {
            "summary": summary,
            "cluster_index": cluster_index,
            "top": top,
            "items": items,
            "week": week,
        })

        print(
            f"digest done: {len(top)} clusters from {len(rows)} items "
            f"({len(corpus['clusters'])} total), {len(entity_counts)} entities tracked"
            + (f", macro updated ({week})" if macro_new else "")
        )


if __name__ == "__main__":
    run()
