"""Source identity, authorization, cache and selection checks; no real model."""
import hashlib
import importlib.util
from pathlib import Path
import re
import sqlite3
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest

_package = ModuleType("_isolated_v10_retrieval_tests")
_package.__path__ = [str(Path(__file__).resolve().parents[1] / "sisu_reader")]
sys.modules[_package.__name__] = _package
_spec = importlib.util.spec_from_file_location(_package.__name__ + ".hybrid_retrieval",
    Path(_package.__path__[0]) / "hybrid_retrieval.py")
hr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = hr
_spec.loader.exec_module(hr)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Encoder:
    identity = "synthetic-encoder-v1"
    dimension = 4
    load_s = 0.0

    def __init__(self):
        self.calls = []

    def offsets(self, text):
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def document_input(self, text, title, heading, headers):
        return "\n".join((title, heading, *headers, text))

    def encode(self, texts, *, query=False):
        self.calls.append((query, tuple(texts)))
        vectors = []
        for text in texts:
            text = text.casefold()
            v = np.array([1.0 + sum(t in text for t in ("rain", "precipitation")),
                1.0 + sum(t in text for t in ("finance", "budget")),
                1.0 + sum(t in text for t in ("permit", "approval")), 0.2], dtype=np.float32)
            vectors.append(v / np.linalg.norm(v))
        return np.asarray(vectors)


class Store:
    def __init__(self, contents):
        self.snapshot_id = "snapshot_v1"
        self.docs, self.blocks = [], {}
        for docid, title, texts in contents:
            joined = "\n".join(texts)
            self.docs.append(SimpleNamespace(document_revision_id=docid, title=title,
                source_path="/fixture/" + docid, source_sha256=digest(joined)))
            self.blocks[docid] = []
            for i, text in enumerate(texts):
                self.blocks[docid].append(SimpleNamespace(block_id=f"{docid}_{i}",
                    document_revision_id=docid, section_id=docid + "_section", ordinal=i,
                    kind="paragraph", locator=f"paragraph {i+1}", text=text,
                    text_sha256=digest(text), canonical_char_start=1000*i,
                    canonical_char_end=1000*i+len(text), headers=("Applies to this section",),
                    extraction_flags=(), previous_block_id=f"{docid}_{i-1}" if i else None,
                    next_block_id=f"{docid}_{i+1}" if i+1<len(texts) else None))

    def snapshot(self):
        return SimpleNamespace(snapshot_id=self.snapshot_id, manifest_sha256=self.snapshot_id)

    def documents(self):
        return tuple(self.docs)

    def sections_for_document(self, docid):
        return (SimpleNamespace(section_id=docid+"_section", section_path="Operational limits"),)

    def blocks_for_document(self, docid):
        return tuple(self.blocks[docid])


def index(tmp_path, contents=None, encoder=None):
    store = Store(contents or [("a", "Weather", ["The local rain gauge records precipitation."]),
                              ("b", "Finance", ["The annual budget requires approval."])])
    encoder = Encoder() if encoder is None else encoder
    return hr.HybridIndex(SimpleNamespace(workspace_dir=tmp_path), store, encoder=encoder), store, encoder


def test_dense_synonym_retrieval_and_exact_source_binding(tmp_path):
    ix, store, _ = index(tmp_path)
    hits = ix.search(["precipitation"], ["a", "b"])
    assert hits[0]["document_revision_id"] == "a"
    assert hits[0]["retrieval_evidence"][0]["dense_rank"] == 1
    for row in hits:
        block = next(b for b in store.blocks[row["document_revision_id"]] if b.block_id == row["block_id"])
        assert row["text"] == block.text[row["char_start"]:row["char_end"]]
        assert row["text_sha256"] == digest(row["text"])
        assert row["block_text_sha256"] == block.text_sha256
        assert row["section_path"] not in row["text"]


def test_filter_applies_before_topk_and_lexical_corpus_statistics(tmp_path):
    ix, _, _ = index(tmp_path)
    only = ix.search(["annual budget requires approval"], ["a"], limit=1)
    assert len(only) == 1 and only[0]["document_revision_id"] == "a"
    assert only[0]["retrieval_evidence"][0]["dense_rank"] == 1
    assert ix.last_search_stats["authorized_documents"] == 1
    assert ix.last_search_stats["authorized_passages"] == 1
    assert ix.search(["finance"], [], limit=1) == []
    with pytest.raises(TypeError):
        ix.search(["budget"], None)
    with pytest.raises(TypeError):
        ix.search(["budget"], "b")


def test_content_vectors_survive_restart_and_new_snapshot_without_reencoding(tmp_path):
    ix, store, encoder = index(tmp_path)
    first = ix.ensure()
    assert first["unique_vectors_encoded"] == 2
    ix2 = hr.HybridIndex(SimpleNamespace(workspace_dir=tmp_path), store, encoder=Encoder())
    second = ix2.ensure()
    assert second["unique_vectors_encoded"] == 0 and second["vectors_reused"] == 2
    assert not ix2._encoder.calls
    store.snapshot_id = "snapshot_v2"
    third = ix2.ensure()
    assert third["snapshot_id"] != first["snapshot_id"]
    assert third["index_identity"] != first["index_identity"]
    assert third["unique_vectors_encoded"] == 0


def test_changed_source_and_encoder_identity_invalidate_only_matching_cache(tmp_path):
    ix, store, encoder = index(tmp_path)
    ix.ensure()
    block = store.blocks["a"][0]
    block.text = "Rain readings are revised to 27 millimeters."
    block.text_sha256 = digest(block.text)
    store.snapshot_id = "snapshot_v2"
    refreshed = ix.ensure()
    assert refreshed["unique_vectors_encoded"] == 1 and refreshed["vectors_reused"] == 1
    assert "27 millimeters" in ix.search(["rain"], ["a"])[0]["text"]
    other = Encoder();other.identity = "synthetic-v2-other-weights"
    ix2 = hr.HybridIndex(SimpleNamespace(workspace_dir=tmp_path), store, encoder=other)
    assert ix2.ensure()["unique_vectors_encoded"] == 2


def test_corrupted_vector_cache_is_rejected_not_silently_lexical(tmp_path):
    ix, store, _ = index(tmp_path);ix.ensure()
    with sqlite3.connect(tmp_path/"hybrid_passages_v1/vectors.sqlite3") as c:
        c.execute("UPDATE vectors SET value=?", (b"invalid",))
    fresh = hr.HybridIndex(SimpleNamespace(workspace_dir=tmp_path), store, encoder=Encoder())
    with pytest.raises(hr.RetrievalIntegrityError, match="Cached embedding"):
        fresh.ensure()


def test_forged_source_hash_and_duplicate_ids_rejected(tmp_path):
    ix, store, _ = index(tmp_path)
    store.blocks["a"][0].text = "Undeclared changed source."
    with pytest.raises(hr.RetrievalIntegrityError, match="Source block"):
        ix.ensure()
    ix2, s2, _ = index(tmp_path/"second")
    s2.blocks["a"].append(s2.blocks["a"][0])
    with pytest.raises(hr.RetrievalIntegrityError, match="Source block"):
        ix2.ensure()


def test_source_change_during_encoder_work_never_publishes_generation(tmp_path):
    ix, store, encoder = index(tmp_path)
    original = encoder.encode
    def mutate(texts, **kwargs):
        store.snapshot_id = "snapshot_during_build"
        return original(texts, **kwargs)
    encoder.encode = mutate
    with pytest.raises(hr.RetrievalIntegrityError, match="changed while indexing"):
        ix.ensure()
    assert ix._snapshot_id is None and ix.passages == []


def test_long_unicode_passages_are_contiguous_exact_spans(tmp_path):
    text = "  " + " ".join(f"järvi{i} 日本語" for i in range(500)) + "  "
    ix, store, _ = index(tmp_path, [("a", "Long", [text])]);ix.ensure()
    assert len(ix.passages) > 1
    assert ix.passages[0]["char_start"] == 0 and ix.passages[-1]["char_end"] == len(text)
    for p in ix.passages:
        assert p["text"] == text[p["char_start"]:p["char_end"]]
        assert len(re.findall(r"\S+", p["text"])) <= hr.CHUNK_TOKENS
    for left, right in zip(ix.passages, ix.passages[1:]):
        assert right["char_start"] <= left["char_end"]


def test_query_operator_strings_are_data_and_fallback_is_explicit(tmp_path):
    store = Store([("a", "Quoted", ['Åland says: budget OR "rain"; DROP TABLE vectors;'])])
    ix = hr.HybridIndex(SimpleNamespace(workspace_dir=tmp_path), store, model_path=tmp_path/"absent")
    stats = ix.ensure()
    assert stats["dense_available"] is False and "FileNotFoundError" in stats["dense_fallback_reason"]
    hits = ix.search(['ÅLAND OR "rain"; DROP TABLE vectors;'], ["a"])
    assert hits[0]["block_id"] == "a_0"
    assert ix.last_search_stats["dense_available"] is False


def test_selection_facet_coverage_budget_and_metadata_separation(tmp_path):
    ix, store, _ = index(tmp_path)
    selection = ix.select({"weather": ["rain precipitation"], "money": ["annual finance budget"]}, ["a", "b"], 1000)
    assert {p["facet"] for p in selection["evidence"]} == {"weather", "money"}
    assert {p["document_revision_id"] for p in selection["evidence"]} == {"a", "b"}
    assert selection["stats"]["estimated_tokens_used"] <= 1000
    assert selection["stats"]["whole_corpus_coverage"] is False
    for i,p in enumerate(selection["evidence"],1):
        assert p["id"] == f"P{i}" and p["text"] in store.blocks[p["document_revision_id"]][0].text
        assert "Operational limits" not in p["text"]
    assert ix.select({"tiny": ["rain"]}, ["a"], 0)["evidence"] == []


def test_empty_or_invalid_queries_never_claim_answerability(tmp_path):
    ix, _, _ = index(tmp_path)
    assert ix.search([], ["a"]) == []
    assert ix.last_search_stats["scores_are_answerability_probabilities"] is False
    with pytest.raises(ValueError):ix.search(["x"*4097], ["a"])
    with pytest.raises(TypeError):ix.search("rain", ["a"])
    with pytest.raises(ValueError):ix.select({"a": ["rain"]}, ["a"], -1)


def test_encoder_manifest_cannot_escape_model_directory(tmp_path):
    import json
    other=tmp_path/"secret";other.write_text("not a model")
    directory=tmp_path/"model";directory.mkdir()
    (directory/"manifest.json").write_text(json.dumps({"schema":"sisu.pinned_encoder.v1",
        "repository":"intfloat/multilingual-e5-small","revision":hr.E5_REVISION,
        "files":[{"path":"../secret","bytes":11,"sha256":digest("not a model")}]}))
    with pytest.raises(hr.RetrievalIntegrityError):hr.CpuE5Encoder(directory)


def test_no_mixed_snapshots_between_facet_queries(tmp_path):
    ix, store, _ = index(tmp_path)
    search = ix.search
    calls = []
    def change_between(queries, allowed_document_ids, limit=32):
        result = search(queries, allowed_document_ids, limit=limit)
        calls.append(queries)
        if len(calls) == 1:
            store.snapshot_id = "snapshot_changed_between_facets"
        return result
    ix.search = change_between
    with pytest.raises(hr.RetrievalIntegrityError, match="between facet"):
        ix.select({"weather": ["rain"], "money": ["budget"]}, ["a", "b"])


def test_external_database_generation_change_is_detected_despite_cached_store(tmp_path):
    ix, store, _ = index(tmp_path)
    database=tmp_path/"corpus.sqlite3"
    ix.config.db_path=database
    with sqlite3.connect(database) as c:
        c.execute("CREATE TABLE metadata(key TEXT,value TEXT)")
        c.execute("CREATE TABLE corpus_snapshots(snapshot_id TEXT,manifest_sha256 TEXT)")
        c.execute("INSERT INTO metadata VALUES ('active_snapshot_id','snapshot_v1')")
        c.execute("INSERT INTO corpus_snapshots VALUES ('snapshot_v1','snapshot_v1')")
    ix.ensure()
    with sqlite3.connect(database) as c:
        c.execute("UPDATE metadata SET value='external_rebuild' WHERE key='active_snapshot_id'")
        c.execute("INSERT INTO corpus_snapshots VALUES ('external_rebuild','new_manifest')")
    assert store.snapshot().snapshot_id == "snapshot_v1"
    with pytest.raises(hr.RetrievalIntegrityError, match="database changed"):
        ix.search(["rain"],["a"])


def test_missing_or_changed_corpus_database_does_not_publish_stale_index(tmp_path):
    ix, _, _ = index(tmp_path);ix.ensure();ix.config.db_path=tmp_path/"removed.sqlite3"
    with pytest.raises(hr.RetrievalIntegrityError, match="Cannot verify"):
        ix.select({"rain":["rain"]},["a"])
    assert not ix.config.db_path.exists()


def test_model_integrity_failure_is_not_downgraded_on_next_request(tmp_path,monkeypatch):
    _,store,_=index(tmp_path)
    ix=hr.HybridIndex(SimpleNamespace(workspace_dir=tmp_path),store)
    def reject(path):raise hr.RetrievalIntegrityError("bad pinned weights")
    monkeypatch.setattr(hr,"CpuE5Encoder",reject)
    for _ in range(2):
        with pytest.raises(hr.RetrievalIntegrityError,match="bad pinned weights"):
            ix.ensure()


def test_encoder_rejects_an_unapproved_but_well_formed_revision(tmp_path):
    import json
    (tmp_path/"manifest.json").write_text(json.dumps({"schema":"sisu.pinned_encoder.v1",
        "repository":"intfloat/multilingual-e5-small","revision":"a"*40,"files":[]}))
    with pytest.raises(hr.RetrievalIntegrityError,match="Unrecognized"):
        hr.CpuE5Encoder(tmp_path)


@pytest.mark.parametrize("name",["added_tokens.json","special_tokens_map.json","chat_template.jinja"])
def test_encoder_rejects_unpinned_optional_loader_assets_before_import(tmp_path,name):
    import json
    files=[]
    for required in ["config.json","model.safetensors","tokenizer.json","tokenizer_config.json"]:
        payload=b"fake fixture only"
        (tmp_path/required).write_bytes(payload)
        files.append({"path":required,"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest()})
    (tmp_path/"manifest.json").write_text(json.dumps({"schema":"sisu.pinned_encoder.v1",
        "repository":"intfloat/multilingual-e5-small","revision":hr.E5_REVISION,"files":files}))
    (tmp_path/name).write_text("unmanifested tokenizer behavior")
    with pytest.raises(hr.RetrievalIntegrityError,match="Unmanifested encoder loader artifact"):
        hr.CpuE5Encoder(tmp_path)
