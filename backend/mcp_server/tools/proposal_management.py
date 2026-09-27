from __future__ import annotations

import uuid
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.agents.services import proposal_service
from mcp_server.auth import CallContext
from mcp_server.registry import tool
from mcp_server.tools._helpers import resolve_workspace_id

@tool("list_proposals", description="List persistent proposals in the current workspace.", parameters={"type":"object","properties":{"status":{"type":"string"}},"additionalProperties":False})
async def list_proposals(*, session: AsyncSession, ctx: CallContext, status: str | None = None) -> list[dict[str, Any]]:
    ws = await resolve_workspace_id(session, ctx)
    return [proposal_service.serialize(p) for p in await proposal_service.list_proposals(session, ctx.user_id, ws, status=status)]

@tool("get_proposal", description="Get a persistent proposal by ID.", parameters={"type":"object","properties":{"proposal_id":{"type":"string","format":"uuid"}},"required":["proposal_id"],"additionalProperties":False})
async def get_proposal(*, session: AsyncSession, ctx: CallContext, proposal_id: str) -> dict[str, Any]:
    ws = await resolve_workspace_id(session, ctx)
    return proposal_service.serialize(await proposal_service.get(session, ctx.user_id, ws, uuid.UUID(proposal_id)))

@tool("approve_proposal", description="Approve and apply a pending proposal after explicit human approval.", parameters={"type":"object","properties":{"proposal_id":{"type":"string","format":"uuid"},"expected_hash":{"type":"string"}},"required":["proposal_id","expected_hash"],"additionalProperties":False}, external_only=True)
async def approve_proposal(*, session: AsyncSession, ctx: CallContext, proposal_id: str, expected_hash: str) -> dict[str, Any]:
    ws = await resolve_workspace_id(session, ctx)
    return await proposal_service.approve_and_apply(session, ctx.user_id, ws, uuid.UUID(proposal_id), expected_hash)

@tool("reject_proposal", description="Reject a pending proposal after explicit human decision.", parameters={"type":"object","properties":{"proposal_id":{"type":"string","format":"uuid"},"expected_hash":{"type":"string"}},"required":["proposal_id","expected_hash"],"additionalProperties":False}, external_only=True)
async def reject_proposal(*, session: AsyncSession, ctx: CallContext, proposal_id: str, expected_hash: str) -> dict[str, Any]:
    ws = await resolve_workspace_id(session, ctx)
    row = await proposal_service.reject(session, ctx.user_id, ws, uuid.UUID(proposal_id), expected_hash)
    return proposal_service.serialize(row)
