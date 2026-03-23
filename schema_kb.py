from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st

from phase2_types import SchemaChunk


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------------------------------------------------------
# Lazy imports
# -----------------------------------------------------------------------------
def _import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise RuntimeError(
            "faiss is not installed. Install faiss-cpu (recommended) or faiss-gpu."
        ) from e


@st.cache_resource(show_spinner=False)
def _get_embedder(embed_model_name: str = DEFAULT_EMBED_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(embed_model_name)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
def _kb_paths(kb_dir: str, namespace: str, create: bool = False) -> Dict[str, Path]:
    base = Path(kb_dir).expanduser().resolve() / namespace
    if create:
        base.mkdir(parents=True, exist_ok=True)

    return {
        "base": base,
        "index": base / "schema.index",
        "meta": base / "schema_meta.jsonl",
        "learn_index": base / "learned.index",
        "learn_meta": base / "learned_meta.jsonl",
    }


# -----------------------------------------------------------------------------
# Chunking + parsing
# -----------------------------------------------------------------------------
def _split_schema_into_chunks(schema_text: str) -> List[str]:
    """
    Chunk schema into semantically meaningful blocks.
    Priority:
    1) Split per CREATE TABLE if DDL
    2) Else split per blank-line section
    """
    txt = (schema_text or "").strip()
    if not txt:
        return []

    txt = txt.replace("\r\n", "\n")

    if re.search(r"\bcreate\s+table\b", txt, re.IGNORECASE):
        parts = re.split(r"(?=\bcreate\s+table\b)", txt, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]

    parts = re.split(r"\n\s*\n", txt)
    return [p.strip() for p in parts if p.strip()]


def _extract_table_names(chunk: str) -> List[str]:
    tables: List[str] = []

    for m in re.finditer(r"\bTable\s+([A-Za-z0-9_\.]+)", chunk, re.IGNORECASE):
        tables.append(m.group(1).split(".")[-1])

    for m in re.finditer(
        r"\bcreate\s+table\s+(?:if\s+not\s+exists\s+)?([A-Za-z0-9_\"`\.]+)",
        chunk,
        re.IGNORECASE,
    ):
        name = m.group(1).strip('"`')
        tables.append(name.split(".")[-1])

    out: List[str] = []
    seen = set()
    for t in tables:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------
def _embed_texts(texts: List[str], embed_model_name: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    model = _get_embedder(embed_model_name)
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    return vecs.astype("float32")


# -----------------------------------------------------------------------------
# Cached index/meta loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_schema_index_cached(index_path_str: str):
    faiss = _import_faiss()
    return faiss.read_index(index_path_str)


@st.cache_data(show_spinner=False)
def _load_schema_meta_cached(meta_path_str: str) -> List[SchemaChunk]:
    meta_path = Path(meta_path_str)
    metas: List[SchemaChunk] = []

    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        metas.append(SchemaChunk(**obj))

    return metas


@st.cache_resource(show_spinner=False)
def _load_learned_index_cached(index_path_str: str):
    faiss = _import_faiss()
    return faiss.read_index(index_path_str)


@st.cache_data(show_spinner=False)
def _load_learned_meta_cached(meta_path_str: str) -> List[dict]:
    meta_path = Path(meta_path_str)
    return [
        json.loads(line)
        for line in meta_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# -----------------------------------------------------------------------------
# Schema KB build
# -----------------------------------------------------------------------------
def build_schema_kb(
    schema_text: str,
    kb_dir: str = "./kb",
    namespace: str = "default",
    source: str = "schema:uploaded",
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> Dict[str, Any]:
    """
    Create/overwrite schema KB for this namespace.
    """
    faiss = _import_faiss()
    paths = _kb_paths(kb_dir, namespace, create=True)

    chunks = _split_schema_into_chunks(schema_text)
    if not chunks:
        raise ValueError("Schema text is empty; cannot build KB.")

    vecs = _embed_texts(chunks, embed_model_name=embed_model_name)
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(paths["index"]))

    with paths["meta"].open("w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            meta = SchemaChunk(
                id=f"schema_{i}",
                text=ch,
                source=source,
                table_names=_extract_table_names(ch),
            )
            f.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")

    # clear caches because KB changed
    _load_schema_index_cached.clear()
    _load_schema_meta_cached.clear()

    return {
        "namespace": namespace,
        "chunks": len(chunks),
        "dim": int(dim),
        "kb_dir": str(paths["base"]),
        "index": str(paths["index"]),
        "meta": str(paths["meta"]),
    }


# -----------------------------------------------------------------------------
# Schema retrieval
# -----------------------------------------------------------------------------
def retrieve_schema_chunks(
    question: str,
    kb_dir: str = "./kb",
    namespace: str = "default",
    top_k: int = 8,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> List[SchemaChunk]:
    """
    Retrieve top-k schema chunks relevant to the question.
    """
    if not question.strip():
        return []

    paths = _kb_paths(kb_dir, namespace, create=False)

    if not paths["index"].exists() or not paths["meta"].exists():
        raise FileNotFoundError(
            f"Schema KB not built for namespace='{namespace}'. "
            f"Missing {paths['index']} or {paths['meta']}"
        )

    index = _load_schema_index_cached(str(paths["index"]))
    metas = _load_schema_meta_cached(str(paths["meta"]))

    qv = _embed_texts([question], embed_model_name=embed_model_name)
    scores, idxs = index.search(qv, int(top_k))

    out: List[SchemaChunk] = []
    for ix in idxs[0].tolist():
        if 0 <= ix < len(metas):
            out.append(metas[ix])

    return out


# -----------------------------------------------------------------------------
# Learning store
# -----------------------------------------------------------------------------
def append_learned_example(
    nl_question: str,
    sql_text: str,
    kb_dir: str = "./kb",
    namespace: str = "default",
    source: str = "learned:success",
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> Dict[str, Any]:
    """
    Append a successful NL→SQL pair to a learned example store.
    """
    faiss = _import_faiss()
    paths = _kb_paths(kb_dir, namespace, create=True)

    vec = _embed_texts([nl_question], embed_model_name=embed_model_name)
    dim = vec.shape[1]

    if paths["learn_index"].exists():
        index = faiss.read_index(str(paths["learn_index"]))
        if index.d != dim:
            raise ValueError(
                f"Embedding dim mismatch. Existing={index.d}, new={dim}"
            )
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(vec)
    faiss.write_index(index, str(paths["learn_index"]))

    meta_rec = {
        "id": f"learned_{index.ntotal - 1}",
        "text": f"NL: {nl_question}\nSQL: {sql_text}",
        "source": source,
        "table_names": [],
    }

    with paths["learn_meta"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta_rec, ensure_ascii=False) + "\n")

    # clear learned caches because store changed
    _load_learned_index_cached.clear()
    _load_learned_meta_cached.clear()

    return {
        "learned_total": int(index.ntotal),
        "learn_meta": str(paths["learn_meta"]),
    }


def retrieve_learned_fewshots(
    question: str,
    kb_dir: str = "./kb",
    namespace: str = "default",
    top_k: int = 3,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> List[str]:
    """
    Retrieve learned few-shot examples relevant to the current question.
    """
    if not question.strip():
        return []

    paths = _kb_paths(kb_dir, namespace, create=False)

    if not paths["learn_index"].exists() or not paths["learn_meta"].exists():
        return []

    index = _load_learned_index_cached(str(paths["learn_index"]))
    metas = _load_learned_meta_cached(str(paths["learn_meta"]))

    qv = _embed_texts([question], embed_model_name=embed_model_name)
    scores, idxs = index.search(qv, int(top_k))

    out: List[str] = []
    for ix in idxs[0].tolist():
        if 0 <= ix < len(metas):
            out.append(metas[ix]["text"])

    return out