"""Usage totals and ownership are scoped to a conversation and its human."""
import uuid

import pytest

from app.agents.services import usage_service


pytestmark = pytest.mark.asyncio


async def test_conversation_usage_sums_chat_compaction_and_unknown_cost(
    client, session, auth_headers, test_user, test_agent, test_conversation,
):
    endpoint = f"/api/agents/conversations/{test_conversation.id}/usage"
    empty = await client.get(endpoint, headers=auth_headers)
    assert empty.status_code == 200
    assert empty.json() == {
        "input_tokens": 0, "output_tokens": 0, "chat_calls": 0,
        "compaction_calls": 0, "avg_latency_ms": None,
        "estimated_cost_usd": 0.0, "compacted_through_ordinal": 0,
    }
    test_conversation.summary = "Prefers concise replies"
    test_conversation.summary_through_ordinal = 10
    await session.commit()
    for kind, provider, model, input_tokens, output_tokens, latency in [
        ("chat", "openai", "gpt-4o-mini", 1000, 100, 100),
        ("compaction", "openai", "gpt-4o-mini", 500, 50, 300),
    ]:
        await usage_service.record_usage(
            session, user_id=test_user.id, agent_id=test_agent.id,
            conversation_id=test_conversation.id, message_id=None,
            provider=provider, model=model, kind=kind,
            input_tokens=input_tokens, output_tokens=output_tokens, latency_ms=latency,
        )
    known = await client.get(endpoint, headers=auth_headers)
    assert known.status_code == 200
    assert known.json() == {
        "input_tokens": 1500, "output_tokens": 150,
        "chat_calls": 1, "compaction_calls": 1,
        "avg_latency_ms": 200.0, "estimated_cost_usd": 0.000315,
        "compacted_through_ordinal": 10,
    }
    await usage_service.record_usage(
        session, user_id=test_user.id, agent_id=test_agent.id,
        conversation_id=test_conversation.id, message_id=None,
        provider="ollama", model="synthetic-local", kind="chat",
        input_tokens=20, output_tokens=4, latency_ms=None,
    )
    unknown = (await client.get(endpoint, headers=auth_headers)).json()
    assert unknown["input_tokens"] == 1520
    assert unknown["chat_calls"] == 2
    assert unknown["estimated_cost_usd"] is None
    assert unknown["avg_latency_ms"] == 200.0


async def test_usage_hidden_from_other_workspace_member_and_unknown_workspace(
    client, session, auth_headers, viewer_auth_headers,
    test_user, test_agent, test_conversation,
):
    endpoint = f"/api/agents/conversations/{test_conversation.id}/usage"
    assert (await client.get(endpoint, headers=auth_headers)).status_code == 200
    assert (await client.get(endpoint, headers=viewer_auth_headers)).status_code == 404
    foreign_header = dict(auth_headers, **{"X-Workspace-Id": str(uuid.uuid4())})
    assert (await client.get(endpoint, headers=foreign_header)).status_code == 404
    assert (await client.get(
        f"/api/agents/conversations/{uuid.uuid4()}/usage", headers=auth_headers,
    )).status_code == 404


async def test_usage_excludes_other_user_rows_even_if_conversation_id_matches(
    client, session, auth_headers, test_user, test_agent, test_conversation,
):
    # A stale/badly associated usage row must never contaminate the owner's bill.
    from app.models.user import User

    other = User(
        id=uuid.uuid4(), email=f"other-usage-{uuid.uuid4()}@example.test",
        hashed_password="synthetic", is_active=True,
    )
    session.add(other)
    await session.commit()
    await usage_service.record_usage(
        session, user_id=other.id, agent_id=test_agent.id,
        conversation_id=test_conversation.id, message_id=None,
        provider="openai", model="gpt-4o-mini", kind="chat",
        input_tokens=1000000, output_tokens=10,
    )
    body = (await client.get(
        f"/api/agents/conversations/{test_conversation.id}/usage", headers=auth_headers,
    )).json()
    assert body["input_tokens"] == 0
    assert body["chat_calls"] == 0
