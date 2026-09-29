"""Workspace-scoped knowledge retrieval on isolated PostgreSQL schemas."""
import uuid

import pytest
from sqlalchemy import select

import mcp_server.tools  # noqa: F401
from app.agents.models.agent import Agent
from app.agents.models.knowledge import KnowledgeChunk, KnowledgeDoc
from app.agents.services import embedding
from app.models.user import User
from app.models.workspace import Workspace, WorkspaceMember
from mcp_server.auth import CallContext
from mcp_server.registry import REGISTRY


@pytest.fixture
async def knowledge_context(postgres_sessions, monkeypatch):
    async with postgres_sessions() as session:
        user = User(
            id=uuid.uuid4(), email=f"rag-{uuid.uuid4()}@example.test",
            hashed_password="synthetic", is_active=True,
        )
        ws = Workspace(
            id=uuid.uuid4(), name="Synthetic knowledge", kind="personal",
            created_by_user_id=user.id, default_currency="BRL", locale="pt-BR",
        )
        session.add_all([user, ws])
        await session.flush()
        agent = Agent(
            id=uuid.uuid4(), user_id=user.id, workspace_id=ws.id,
            name="Knowledge", top_n=2, similarity_threshold=0.4,
        )
        session.add_all([
            agent, WorkspaceMember(
                id=uuid.uuid4(), user_id=user.id, workspace_id=ws.id, role="owner",
            ),
        ])
        await session.commit()

        async def synthetic_embedding(texts):
            return [[0.1] * 1536 for _ in texts], "synthetic:current"

        monkeypatch.setattr(embedding, "embed_texts", synthetic_embedding)
        yield session, CallContext(
            user_id=user.id, workspace_id=ws.id, agent_id=agent.id,
            conversation_id=uuid.uuid4(),
        ), agent


async def _document(session, ctx, *, status, label, title):
    doc = KnowledgeDoc(
        id=uuid.uuid4(), agent_id=ctx.agent_id, user_id=ctx.user_id,
        title=title, source=f"synthetic/{title}.md", mime="text/markdown",
        status=status, size_bytes=100, chunk_count=1,
    )
    session.add(doc)
    await session.flush()
    session.add(KnowledgeChunk(
        id=uuid.uuid4(), doc_id=doc.id, agent_id=ctx.agent_id,
        ordinal=0, content=f"Evidence in {title}",
        embedding=[0.1] * 1536, embedding_model=label,
    ))
    await session.commit()
    return doc


async def test_search_only_ready_current_model_with_citation(knowledge_context):
    session, ctx, agent = knowledge_context
    await _document(session, ctx, status="pending", label="synthetic:current", title="Pending")
    await _document(session, ctx, status="failed", label="synthetic:current", title="Failed")
    await _document(session, ctx, status="ready", label="synthetic:old", title="Stale")
    await _document(session, ctx, status="ready", label="synthetic:current", title="Ready")
    result = await REGISTRY["search_knowledge_base"].handler(session=session, ctx=ctx, query="evidence")
    assert result["total"] == 1
    assert result["items"][0]["title"] == "Ready"
    assert result["items"][0]["source"] == "synthetic/Ready.md"
    assert result["items"][0]["content"] == "Evidence in Ready"
    agent.top_n = 0
    await session.commit()
    assert (await REGISTRY["search_knowledge_base"].handler(
        session=session, ctx=ctx, query="evidence",
    )) == {"items": [], "total": 0, "warning": "retrieval_disabled"}
    overridden = await REGISTRY["search_knowledge_base"].handler(
        session=session, ctx=ctx, query="evidence", top_n=1,
    )
    assert overridden["total"] == 1


async def test_wrong_model_warns_but_threshold_miss_does_not(knowledge_context):
    session, ctx, agent = knowledge_context
    doc = await _document(session, ctx, status="ready", label="synthetic:old", title="Old")
    tool = REGISTRY["search_knowledge_base"].handler
    assert (await tool(session=session, ctx=ctx, query="evidence"))["warning"] == "reingest_required"
    chunk = (await session.execute(
        select(KnowledgeChunk).where(KnowledgeChunk.doc_id == doc.id)
    )).scalar_one()
    chunk.embedding_model = "synthetic:current"
    chunk.embedding = [1.0] + [0.0] * 1535
    await session.commit()
    missed = await tool(session=session, ctx=ctx, query="evidence", similarity_threshold=0.9)
    assert missed == {"items": [], "total": 0}


async def test_foreign_agent_cannot_search_other_workspace(knowledge_context):
    session, ctx, agent = knowledge_context
    foreign = CallContext(
        user_id=ctx.user_id, workspace_id=uuid.uuid4(), agent_id=agent.id,
        conversation_id=uuid.uuid4(),
    )
    result = await REGISTRY["search_knowledge_base"].handler(
        session=session, ctx=foreign, query="evidence",
    )
    assert result == {"error": "agent not found"}
