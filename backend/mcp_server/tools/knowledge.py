from __future__ import annotations

import math
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models.knowledge import KnowledgeChunk, KnowledgeDoc
from app.agents.services import agent_service, embedding, knowledge_service
from mcp_server.auth import CallContext
from mcp_server.registry import tool
from mcp_server.tools._helpers import resolve_workspace_id

@tool(
    name="search_knowledge_base",
    description=(
        "Semantic search across the calling agent's uploaded documents "
        "(PDF, MD, TXT). Returns the top matching text chunks with their "
        "similarity score. Use this when the user asks about something the "
        "agent has been given knowledge about (tax laws, accounting rules, "
        "personal notes, etc.)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "top_n": {"type": "integer", "minimum": 0, "maximum": 25},
            "similarity_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    tags=["read", "knowledge"],
)
async def search_knowledge_base(
    *,
    session: AsyncSession,
    ctx: CallContext,
    query: str,
    top_n: int | None = None,
    similarity_threshold: float | None = None,
) -> dict[str, Any]:
    if not ctx.agent_id:
        return {"error": "agent_id required (this tool only works inside an agent conversation)"}
    ws_id = await resolve_workspace_id(session, ctx)
    agent = await agent_service.get_agent(session, ctx.agent_id, ws_id)
    if agent is None:
        return {"error": "agent not found"}
    if top_n is not None and (type(top_n) is not int or not 0 <= top_n <= 25):
        return {"error": "invalid top_n"}
    if similarity_threshold is not None and (
        not isinstance(similarity_threshold, (int, float))
        or not math.isfinite(similarity_threshold)
        or not 0 <= similarity_threshold <= 1
    ):
        return {"error": "invalid similarity_threshold"}
    selected_n = top_n if top_n is not None else max(0, min(int(agent.top_n), 25))
    selected_threshold = (
        float(similarity_threshold) if similarity_threshold is not None
        else float(agent.similarity_threshold)
    )
    if selected_n == 0:
        return {"items": [], "total": 0, "warning": "retrieval_disabled"}
    vecs, model_label = await embedding.embed_texts([query])
    if not vecs:
        return {"items": [], "total": 0, "warning": "embedding failed"}
    hits = await knowledge_service.similarity_search(
        session, agent_id=ctx.agent_id, query_embedding=vecs[0],
        embedding_model=model_label, top_n=selected_n,
        similarity_threshold=selected_threshold,
    )
    if not hits:
        same_model = (await session.execute(
            select(KnowledgeChunk.id)
            .join(KnowledgeDoc, KnowledgeDoc.id == KnowledgeChunk.doc_id)
            .where(
                KnowledgeChunk.agent_id == ctx.agent_id,
                KnowledgeDoc.agent_id == ctx.agent_id,
                KnowledgeDoc.status == "ready",
                KnowledgeChunk.embedding.is_not(None),
                KnowledgeChunk.embedding_model == model_label,
            ).limit(1)
        )).first()
        if same_model:
            return {"items": [], "total": 0}
        wrong_model = (await session.execute(
            select(KnowledgeChunk.id)
            .join(KnowledgeDoc, KnowledgeDoc.id == KnowledgeChunk.doc_id)
            .where(
                KnowledgeChunk.agent_id == ctx.agent_id,
                KnowledgeDoc.agent_id == ctx.agent_id,
                KnowledgeDoc.status == "ready",
                KnowledgeChunk.embedding.is_not(None),
                KnowledgeChunk.embedding_model.is_distinct_from(model_label),
            ).limit(1)
        )).first()
        if wrong_model:
            return {"items": [], "total": 0, "warning": "reingest_required"}
    return {"items": hits, "total": len(hits)}
