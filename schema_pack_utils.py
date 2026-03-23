from __future__ import annotations

from typing import List

from phase2_types import SchemaChunk, SchemaPack


def build_schema_pack(chunks: List[SchemaChunk]) -> SchemaPack:
    text = "\n\n---\n\n".join([c.text.strip() for c in chunks if c.text.strip()])
    allowed_tables: List[str] = []
    seen = set()
    for c in chunks:
        for t in (c.table_names or []):
            if t not in seen:
                allowed_tables.append(t)
                seen.add(t)
    return SchemaPack(chunks=chunks, text=text, allowed_tables=allowed_tables)
