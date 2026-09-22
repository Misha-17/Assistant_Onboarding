"""CPU passage retrieval with exact source spans and an optional pinned E5 lane.

Retrieval scores nominate text to inspect; they are never evidence of truth,
answerability or complete corpus coverage. No answer labels or remote services
are consulted. The caller supplies its current authorization scope on each use.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import unicodedata

import numpy as np

from .screen_budget import conservative_tokens


SCHEMA = "sisu.hybrid_passages.v1"
CHUNK_TOKENS = 384
CHUNK_OVERLAP = 64
RRF_K = 60
LANE_LIMIT = 128
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def _digest(value):
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


class RetrievalIntegrityError(RuntimeError):
    """A cached or supplied source no longer has its declared identity."""


class CpuE5Encoder:
    """The official mean-pooling E5 recipe, explicitly CPU and local-only."""

    def __init__(self, model_path):
        started = time.perf_counter()
        self.path = Path(model_path).resolve()
        manifest_path = self.path / "manifest.json"
        raw = manifest_path.read_bytes()
        try:
            manifest = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise RetrievalIntegrityError("Invalid encoder manifest") from exc
        if (manifest.get("schema") != "sisu.pinned_encoder.v1" or
                manifest.get("repository") != "intfloat/multilingual-e5-small" or
                manifest.get("revision") != E5_REVISION):
            raise RetrievalIntegrityError("Unrecognized pinned E5 manifest")
        seen = set()
        for item in manifest["files"]:
            name = item["path"]
            target = (self.path / name).resolve()
            if (name in seen or not target.is_relative_to(self.path) or
                    target.stat().st_size != item["bytes"] or
                    _file_hash(target) != item["sha256"]):
                raise RetrievalIntegrityError("Encoder artifact identity mismatch")
            seen.add(name)
        required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
        if not required <= seen:
            raise RetrievalIntegrityError("Encoder manifest omits a required artifact")
        # Transformers can consume optional tokenizer assets merely because they
        # exist beside config.json. Reject unpinned loader inputs before loading;
        # ordinary human-readable provenance does not influence tokenization.
        for target in self.path.rglob("*"):
            if not target.is_file():
                continue
            name = target.relative_to(self.path).as_posix()
            harmless = (name == "manifest.json" or target.suffix.casefold() == ".md" or
                        target.name.casefold() in {"license", "license.txt", "notice", "notice.txt"} or
                        (target.name.startswith("provenance_") and target.suffix == ".json"))
            if name not in seen and not harmless:
                raise RetrievalIntegrityError("Unmanifested encoder loader artifact: " + name)
        # No tokenizer/model code is fetched or executed from the model directory.
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False, use_fast=True)
        self.model = AutoModel.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32).to("cpu").eval()
        if not self.tokenizer.is_fast:
            raise RuntimeError("Exact passage offsets require a fast tokenizer")
        self.identity = _digest(_canonical({"manifest": _digest(raw), "pooling": "masked_mean_l2",
            "dtype": "float32", "device": "cpu", "prefixes": ["query: ", "passage: "],
            "implementation": _digest(inspect.getsource(CpuE5Encoder)),
            "libraries": {name: importlib.metadata.version(name)
                          for name in ("torch", "transformers", "tokenizers")}}))
        self.dimension = int(self.model.config.hidden_size)
        if self.dimension != 384:
            raise RetrievalIntegrityError("Unexpected E5 embedding dimension")
        self.load_s = time.perf_counter() - started

    def offsets(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                              truncation=False)["offset_mapping"]

    def document_input(self, text, title, heading, headers):
        metadata = " | ".join(x for x in (title, heading, *headers) if x)
        # Metadata affects retrieval only and is never inserted into quote text.
        meta_tokens = self.tokenizer(metadata, add_special_tokens=False)["input_ids"][:64]
        metadata = self.tokenizer.decode(meta_tokens, skip_special_tokens=True)
        return metadata + "\n" + text if metadata else text

    def encode(self, texts, *, query=False):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        result = []
        prefix = "query: " if query else "passage: "
        with self.torch.inference_mode():
            for start in range(0, len(texts), 16):
                batch = self.tokenizer([prefix + x for x in texts[start:start + 16]],
                    padding=True, truncation=True, max_length=512, return_tensors="pt")
                output = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).bool()
                pooled = output.masked_fill(~mask, 0).sum(1) / mask.sum(1).clamp_min(1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                result.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(result)


class HybridIndex:
    """Reusable index bound to source snapshots and a pinned local encoder.

    ``encoder`` is an explicit test/integration dependency; production callers
    normally use model_path. Its identity must bind its tokenization and weights.
    Cache integrity failures raise; unavailable optional encoder dependencies
    produce a visible lexical-only fallback. No background work is launched.
    """

    def __init__(self, config, store, model_path=None, *, encoder=None):
        self.config, self.store = config, store
        self.model_path = Path(model_path or os.environ.get("SISU_READER_ENCODER_PATH") or
                               Path(__file__).resolve().parents[1] / "models/multilingual-e5-small")
        self.cache_dir = Path(config.workspace_dir) / "hybrid_passages_v1"
        self._encoder = encoder
        self._encoder_attempted = encoder is not None
        self._fallback = None
        self._snapshot_id = None
        self._lock = threading.RLock()
        self._query_cache = {}
        self.passages = []
        self._vectors = None
        self._blocks = {}
        self._documents = {}
        self._postings = {}
        self._lengths = []
        self.last_search_stats = {}
        self.last_index_stats = {}

    def _get_encoder(self):
        if not self._encoder_attempted:
            self._encoder_attempted = True
            try:
                self._encoder = CpuE5Encoder(self.model_path)
            except RetrievalIntegrityError:
                self._encoder_attempted = False
                raise
            except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
                self._fallback = "Dense encoder unavailable: " + type(exc).__name__
        return self._encoder

    def _current_snapshot(self):
        snapshot = self.store.snapshot()
        # CorpusStore caches its snapshot and may retain an old SQLite handle
        # after another process atomically replaces the database. Check a fresh
        # read-only connection, not that cached handle, before/after retrieval.
        database = getattr(self.config, "db_path", None)
        if database is not None:
            path = Path(database).resolve()
            try:
                with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)) as connection:
                    active = connection.execute("SELECT m.value,s.manifest_sha256 FROM metadata m LEFT JOIN corpus_snapshots s ON s.snapshot_id=m.value WHERE m.key='active_snapshot_id'").fetchone()
            except sqlite3.Error as exc:
                raise RetrievalIntegrityError("Cannot verify active corpus database generation") from exc
            if active != (snapshot.snapshot_id, snapshot.manifest_sha256):
                raise RetrievalIntegrityError("Active corpus database changed; reload the corpus store and retrieval index")
        return snapshot

    def _spans(self, text, encoder):
        if not text.strip():
            return []
        if encoder is None:
            # A declared lexical fallback, not a claim about E5 token counts.
            offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
            size, overlap = 160, 32
        else:
            offsets = [(int(a), int(b)) for a, b in encoder.offsets(text) if b > a]
            size, overlap = CHUNK_TOKENS, CHUNK_OVERLAP
        if not offsets:
            return [(0, len(text))]
        spans = []
        for start in range(0, len(offsets), size - overlap):
            end = min(len(offsets), start + size)
            left = 0 if start == 0 else offsets[start][0]
            right = len(text) if end == len(offsets) else offsets[end][0]
            if left < 0 or right > len(text) or left >= right:
                raise RetrievalIntegrityError("Tokenizer produced invalid source offsets")
            spans.append((left, right))
            if end == len(offsets):
                break
        return spans

    def ensure(self):
        """Synchronously build/reuse content vectors and atomically publish state."""
        started = time.perf_counter()
        with self._lock:
            snapshot = self._current_snapshot()
            if self._snapshot_id == snapshot.snapshot_id:
                elapsed = time.perf_counter() - started
                return {**self.last_index_stats, "reused_in_memory": True,
                        "original_build_s": self.last_index_stats["index_s"],
                        "index_s": elapsed, "ensure_s": elapsed,
                        "encoder_load_s": 0.0, "embedding_s": 0.0,
                        "unique_vectors_encoded": 0}
            encoder_was_loaded = self._encoder is not None
            encoder = self._get_encoder()
            documents = tuple(self.store.documents())
            docmap, blocks, passages, encoded = {}, {}, [], []
            for document in documents:
                docid = document.document_revision_id
                if docid in docmap:
                    raise RetrievalIntegrityError("Duplicate document revision")
                docmap[docid] = document
                headings = {s.section_id: s.section_path for s in self.store.sections_for_document(docid)}
                for block in self.store.blocks_for_document(docid):
                    if (block.document_revision_id != docid or block.block_id in blocks or
                            _digest(block.text) != block.text_sha256):
                        raise RetrievalIntegrityError("Source block identity mismatch")
                    blocks[block.block_id] = block
                    for left, right in self._spans(block.text, encoder):
                        text = block.text[left:right]
                        heading = headings.get(block.section_id, "")
                        identity = _digest(_canonical([SCHEMA, block.block_id, block.text_sha256, left, right]))
                        row = {"passage_id": "passage_" + identity, "block_id": block.block_id,
                            "document_revision_id": docid, "document_title": document.title,
                            "source_path": document.source_path, "source_sha256": document.source_sha256,
                            "locator": block.locator, "section_id": block.section_id,
                            "section_path": heading, "headers": list(block.headers), "kind": block.kind,
                            "extraction_flags": list(block.extraction_flags), "text": text,
                            "char_start": left, "char_end": right, "block_text_sha256": block.text_sha256,
                            "text_sha256": _digest(text), "canonical_char_start": block.canonical_char_start + left,
                            "canonical_char_end": block.canonical_char_start + right,
                            "previous_block_id": block.previous_block_id, "next_block_id": block.next_block_id,
                            "snapshot_id": snapshot.snapshot_id}
                        passages.append(row)
                        encoded.append(encoder.document_input(text, document.title, heading, block.headers)
                                       if encoder else text)
            source_identity = _digest(_canonical({"snapshot": snapshot.snapshot_id,
                "manifest": snapshot.manifest_sha256, "passages": [
                    [p["passage_id"], p["source_sha256"], p["document_title"], p["section_path"], p["headers"]]
                    for p in passages]}))
            model_identity = encoder.identity if encoder else "lexical-only"
            identity = _digest(_canonical({"schema": SCHEMA, "source": source_identity,
                "model": model_identity, "chunk_tokens": CHUNK_TOKENS, "overlap": CHUNK_OVERLAP,
                "lexical_fallback_chunk_words": 160, "rrf_k": RRF_K}))
            vectors, cache_hits, encoded_count, encode_s = None, 0, 0, 0.0
            if encoder and encoded:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with closing(sqlite3.connect(self.cache_dir / "vectors.sqlite3", timeout=60)) as cache:
                    cache.execute("CREATE TABLE IF NOT EXISTS vectors (model TEXT, text_hash TEXT, dimension INTEGER, value BLOB, value_hash TEXT, PRIMARY KEY(model,text_hash))")
                    vectors = np.empty((len(encoded), encoder.dimension), dtype=np.float32)
                    missing = defaultdict(list)
                    for index, text in enumerate(encoded):
                        key = _digest(text)
                        saved = cache.execute("SELECT dimension,value,value_hash FROM vectors WHERE model=? AND text_hash=?", (model_identity, key)).fetchone()
                        if saved is None:
                            missing[key].append(index)
                        else:
                            dimension, value, value_hash = saved
                            if dimension != encoder.dimension or len(value) != dimension * 4 or _digest(value) != value_hash:
                                raise RetrievalIntegrityError("Cached embedding identity mismatch")
                            vector = np.frombuffer(value, dtype=np.float32)
                            self._validate_vectors(vector[None, :], 1, encoder.dimension)
                            vectors[index] = vector
                            cache_hits += 1
                    missing_keys = list(missing)
                    for start in range(0, len(missing_keys), 32):
                        keys = missing_keys[start:start + 32]
                        clock = time.perf_counter()
                        batch = np.asarray(encoder.encode([encoded[missing[k][0]] for k in keys]), dtype=np.float32)
                        encode_s += time.perf_counter() - clock
                        self._validate_vectors(batch, len(keys), encoder.dimension)
                        for key, vector in zip(keys, batch):
                            for index in missing[key]:
                                vectors[index] = vector
                            value = vector.tobytes()
                            cache.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?)",
                                (model_identity, key, encoder.dimension, value, _digest(value)))
                        cache.commit()
                        encoded_count += len(keys)
            postings = defaultdict(list)
            lengths = []
            for index, row in enumerate(passages):
                terms = Counter(_words(" ".join((row["document_title"], row["section_path"], *row["headers"], row["text"]))))
                lengths.append(sum(terms.values()))
                for word, count in terms.items():
                    postings[word].append((index, count))
            if self._current_snapshot().snapshot_id != snapshot.snapshot_id:
                raise RetrievalIntegrityError("Corpus snapshot changed while indexing")
            # Queries never see a partly built generation.
            self.passages, self._blocks, self._documents = passages, blocks, docmap
            self._vectors, self._postings, self._lengths = vectors, dict(postings), lengths
            self._snapshot_id = snapshot.snapshot_id
            self._query_cache.clear()
            self.last_index_stats = {"schema": SCHEMA, "index_identity": identity,
                "source_identity": source_identity, "snapshot_id": snapshot.snapshot_id,
                "model_identity": model_identity, "dense_available": encoder is not None,
                "dense_fallback_reason": self._fallback, "device": "cpu", "cpu_threads": 4,
                "documents": len(documents), "blocks": len(blocks), "passages": len(passages),
                "vectors_reused": cache_hits, "unique_vectors_encoded": encoded_count,
                "encoder_load_s": 0.0 if encoder_was_loaded else getattr(encoder, "load_s", 0.0),
                "encoder_initial_load_s": getattr(encoder, "load_s", 0.0), "embedding_s": encode_s,
                "vector_bytes": 0 if vectors is None else vectors.nbytes,
                "index_s": time.perf_counter() - started, "reused_in_memory": False,
                "persistent_cache": "SQLite content vectors; passage metadata rebuilt from immutable source blocks"}
            return dict(self.last_index_stats)

    @staticmethod
    def _validate_vectors(vectors, rows, dimension):
        if (vectors.shape != (rows, dimension) or not np.isfinite(vectors).all() or
                not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-3)):
            raise RetrievalIntegrityError("Invalid/non-normalized embedding matrix")

    def _allowed(self, values):
        if values is None or isinstance(values, (str, bytes)):
            raise TypeError("An explicit sequence of allowed document IDs is required")
        return frozenset(str(value) for value in values) & self._documents.keys()

    @staticmethod
    def _queries(values):
        if isinstance(values, (str, bytes)):
            raise TypeError("Queries must be a sequence of strings")
        if len(values) > 16:
            raise ValueError("At most 16 query variants per search")
        if any(not isinstance(x, str) or len(x) > 4096 for x in values):
            raise ValueError("Queries must be strings of at most 4096 characters")
        return tuple(dict.fromkeys(x.strip() for x in values if x.strip()))

    def search(self, queries, allowed_document_ids, limit=32):
        started = time.perf_counter()
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 512:
            raise ValueError("limit must be between 1 and 512")
        queries = self._queries(queries)
        with self._lock:
            self.ensure()
            allowed = self._allowed(allowed_document_ids)
            indices = [i for i, p in enumerate(self.passages) if p["document_revision_id"] in allowed]
            eligible = set(indices)
            scores, evidence = defaultdict(float), defaultdict(dict)
            query_encode_s = 0.0
            if indices and queries:
                average_length = sum(self._lengths[i] for i in indices) / len(indices) or 1
                for query_index, query in enumerate(queries):
                    lexical = defaultdict(float)
                    for term in set(_words(query)):
                        occurrences = [(i, tf) for i, tf in self._postings.get(term, ()) if i in eligible]
                        df = len(occurrences)
                        if not df:
                            continue
                        idf = math.log(1 + (len(indices) - df + .5) / (df + .5))
                        for index, tf in occurrences:
                            lexical[index] += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * self._lengths[index] / average_length))
                    ranked = sorted(lexical, key=lambda i: (-lexical[i], i))[:LANE_LIMIT]
                    for rank, index in enumerate(ranked, 1):
                        scores[index] += 1 / (RRF_K + rank)
                        evidence[index].setdefault(query_index, {"query": query}).update(
                            lexical_rank=rank, lexical_score=lexical[index])
                    if self._vectors is not None:
                        if query not in self._query_cache:
                            clock = time.perf_counter()
                            vector = np.asarray(self._encoder.encode([query], query=True), dtype=np.float32)
                            query_encode_s += time.perf_counter() - clock
                            self._validate_vectors(vector, 1, self._encoder.dimension)
                            if len(self._query_cache) >= 128:
                                self._query_cache.pop(next(iter(self._query_cache)))
                            self._query_cache[query] = vector[0]
                        # Bound temporary advanced-indexing copies when the
                        # authorization scope covers a large passage matrix.
                        similarities = np.empty(len(indices), dtype=np.float32)
                        for start in range(0, len(indices), 8192):
                            subset = indices[start:start + 8192]
                            similarities[start:start + len(subset)] = self._vectors[subset] @ self._query_cache[query]
                        order = np.argsort(-similarities, kind="stable")[:LANE_LIMIT]
                        for rank, position in enumerate(order, 1):
                            index = indices[int(position)]
                            scores[index] += 1 / (RRF_K + rank)
                            evidence[index].setdefault(query_index, {"query": query}).update(
                                dense_rank=rank, dense_score=float(similarities[position]))
            selected = sorted(scores, key=lambda i: (-scores[i], i))[:limit]
            if self._current_snapshot().snapshot_id != self._snapshot_id:
                raise RetrievalIntegrityError("Corpus snapshot changed during retrieval")
            self.last_search_stats = {"snapshot_id": self._snapshot_id,
                "index_identity": self.last_index_stats["index_identity"], "queries": len(queries),
                "authorized_documents": len(allowed), "authorized_passages": len(indices),
                "returned": len(selected), "dense_available": self._vectors is not None,
                "dense_fallback_reason": self._fallback, "query_embedding_s": query_encode_s,
                "search_s": time.perf_counter() - started, "rrf_k": RRF_K,
                "scores_are_answerability_probabilities": False}
            return [{**self.passages[index], "score": scores[index], "rank": rank,
                "retrieval_evidence": list(evidence[index].values())} for rank, index in enumerate(selected, 1)]

    def select(self, query_groups, allowed_document_ids, token_budget=10000):
        """Facet-fair evidence selection; neighboring spans remain separate rows."""
        if (not isinstance(query_groups, dict) or len(query_groups) > 16 or
                any(not isinstance(k, str) or len(k) > 512 for k in query_groups)):
            raise ValueError("Provide at most 16 named query groups")
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or not 0 <= token_budget <= 100000:
            raise ValueError("token_budget must be between 0 and 100000")
        with self._lock:
            self.ensure()
            selection_snapshot = self._snapshot_id
            allowed = self._allowed(allowed_document_ids)
            groups, searches = {}, []
            for facet, queries in query_groups.items():
                groups[facet] = self.search(queries, allowed, limit=32)
                if self._snapshot_id != selection_snapshot:
                    raise RetrievalIntegrityError("Corpus snapshot changed between facet searches")
                searches.append({"facet": facet, **self.last_search_stats})
            chosen, seen, used, doc_counts = [], set(), 0, Counter()

            def admit(hit, facet, reason, *, ceiling=None):
                nonlocal used
                if hit["passage_id"] in seen:
                    return False
                for prior in chosen:
                    if prior["block_id"] == hit["block_id"]:
                        intersection = max(0, min(prior["char_end"], hit["char_end"]) - max(prior["char_start"], hit["char_start"]))
                        if intersection >= .65 * min(len(prior["text"]), len(hit["text"])):
                            return False
                cost = 32 + conservative_tokens(hit["text"]) + conservative_tokens(
                    _canonical([hit["document_title"], hit["locator"], hit["headers"]]))
                if used + cost > (token_budget if ceiling is None else ceiling):
                    return False
                block = self._blocks[hit["block_id"]]
                if (hit["document_revision_id"] not in allowed or
                        block.text[hit["char_start"]:hit["char_end"]] != hit["text"] or
                        _digest(block.text) != hit["block_text_sha256"]):
                    raise RetrievalIntegrityError("Selected passage/source binding changed")
                chosen.append({**hit, "id": "P" + str(len(chosen) + 1), "facet": facet,
                    "selection_reason": reason, "estimated_tokens": cost})
                seen.add(hit["passage_id"]);used += cost;doc_counts[hit["document_revision_id"]] += 1
                return True

            # One useful location per facet first, then softly favor source diversity.
            for facet, hits in groups.items():
                for hit in hits:
                    if admit(hit, facet, "facet_first_pass"):
                        break
            candidates = [(facet, hit) for facet, hits in groups.items() for hit in hits]
            unselected = []
            while candidates:
                candidates.sort(key=lambda pair: (-(pair[1]["score"] / (1 + .12 * doc_counts[pair[1]["document_revision_id"]])), pair[1]["rank"]))
                facet, hit = candidates.pop(0)
                if not admit(hit, facet, "ranked_facet_and_document_diversity",
                             ceiling=max(used, int(token_budget * .8))):
                    unselected.append((facet, hit))
            # Existing structure supplies nearby qualifiers/headers, without
            # claiming their semantic necessity or splicing them into a quote.
            primary = list(chosen)
            by_block = defaultdict(list)
            for p in self.passages:
                if p["document_revision_id"] in allowed:
                    by_block[p["block_id"]].append(p)
            for hit in primary:
                same_block = by_block[hit["block_id"]]
                position = next(i for i, p in enumerate(same_block) if p["passage_id"] == hit["passage_id"])
                for near in (position - 1, position + 1):
                    if 0 <= near < len(same_block):
                        admit({**same_block[near], "score": 0.0, "rank": None, "retrieval_evidence": []},
                              hit["facet"], "adjacent_passage_context")
                for neighbor_id in (hit["previous_block_id"], hit["next_block_id"]):
                    neighbors = by_block.get(neighbor_id, ())
                    edge = neighbors[-1:] if neighbor_id == hit["previous_block_id"] else neighbors[:1]
                    for neighbor in edge:
                        if neighbor["section_id"] == hit["section_id"] or neighbor["kind"] == "table_header":
                            admit({**neighbor, "score": 0.0, "rank": None, "retrieval_evidence": []},
                                  hit["facet"], "adjacent_source_context")
            for facet, hit in unselected:
                admit(hit, facet, "remaining_budget_ranked_passage")
            if self._current_snapshot().snapshot_id != selection_snapshot:
                raise RetrievalIntegrityError("Corpus snapshot changed during evidence selection")
            return {"evidence": chosen, "stats": {"snapshot_id": self._snapshot_id,
                "index_identity": self.last_index_stats["index_identity"], "dense_available": self._vectors is not None,
                "dense_fallback_reason": self._fallback, "token_budget": token_budget,
                "estimated_tokens_used": used, "selected_passages": len(chosen),
                "selected_documents": len(doc_counts), "selected_facets": sorted({p["facet"] for p in chosen}),
                "queries": searches, "whole_corpus_coverage": False,
                "coverage_note": "Selection is bounded navigation; it does not establish semantic completeness or answerability."}}
