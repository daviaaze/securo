from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import current_active_user
from app.core.database import get_async_session
from app.core.workspace_context import current_workspace, WorkspaceContext
from app.models.user import User
from app.agents.services import proposal_service

router = APIRouter(prefix="/api/agents/proposals", tags=["agent-proposals"])


class Decision(BaseModel):
    expected_hash: str


@router.get("")
async def list_agent_proposals(status: str | None = None, ctx: WorkspaceContext = Depends(current_workspace), session: AsyncSession = Depends(get_async_session)) -> list[dict[str, Any]]:
    rows = await proposal_service.list_proposals(session, ctx.user.id, ctx.workspace.id, status=status)
    return [proposal_service.serialize(row) for row in rows]


@router.get("/{proposal_id}")
async def get_agent_proposal(proposal_id: uuid.UUID, ctx: WorkspaceContext = Depends(current_workspace), session: AsyncSession = Depends(get_async_session)) -> dict[str, Any]:
    row = await proposal_service.get(session, ctx.user.id, ctx.workspace.id, proposal_id)
    return proposal_service.serialize(row)


@router.post("/{proposal_id}/apply")
async def apply_agent_proposal(proposal_id: uuid.UUID, body: Decision, ctx: WorkspaceContext = Depends(current_workspace), session: AsyncSession = Depends(get_async_session)) -> dict[str, Any]:
    return await proposal_service.approve_and_apply(session, ctx.user.id, ctx.workspace.id, proposal_id, body.expected_hash)


@router.post("/{proposal_id}/reject")
async def reject_agent_proposal(proposal_id: uuid.UUID, body: Decision, ctx: WorkspaceContext = Depends(current_workspace), session: AsyncSession = Depends(get_async_session)) -> dict[str, Any]:
    row = await proposal_service.reject(session, ctx.user.id, ctx.workspace.id, proposal_id, body.expected_hash)
    return proposal_service.serialize(row)
