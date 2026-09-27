"""Tool registry for the MCP server."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession
from mcp_server.auth import CallContext

ToolHandler = Callable[..., Awaitable[Any]]

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    is_proposal: bool = False
    tags: list[str] = field(default_factory=list)
    external_only: bool = False

REGISTRY: dict[str, ToolSpec] = {}

def tool(name: str, *, description: str, parameters: dict[str, Any], is_proposal: bool = False, tags: list[str] | None = None, external_only: bool = False) -> Callable[[ToolHandler], ToolHandler]:
    def deco(fn: ToolHandler) -> ToolHandler:
        if name in REGISTRY:
            raise RuntimeError(f"duplicate tool registration: {name}")
        schema = dict(parameters)
        if is_proposal:
            schema["properties"] = {k: v for k, v in parameters.get("properties", {}).items() if k != "apply"}
        REGISTRY[name] = ToolSpec(name, description, schema, fn, is_proposal, list(tags or []), external_only)
        return fn
    return deco

def list_tools(ctx: CallContext | None = None) -> list[dict[str, Any]]:
    return [
        {"name": s.name, "description": s.description, "inputSchema": s.parameters,
         "_securo": {"is_proposal": s.is_proposal, "tags": s.tags}}
        for s in REGISTRY.values() if not s.external_only or (ctx is not None and ctx.external)
    ]

async def call_tool(session: AsyncSession, ctx: CallContext, name: str, arguments: dict[str, Any] | None) -> Any:
    spec = REGISTRY.get(name)
    if spec is None or (spec.external_only and not ctx.external):
        raise KeyError(f"unknown tool: {name}")
    from app.core.app_clock import use_timezone
    from mcp_server.tools._helpers import resolve_workspace_id
    try:
        workspace_id = await resolve_workspace_id(session, ctx)
    except ValueError:
        workspace_id = None
    call_args = dict(arguments or {})
    if spec.is_proposal and "apply" in call_args:
        raise ValueError("Proposal tools are preview-only; use approve_proposal after explicit approval")
    async with use_timezone(session, workspace_id):
        result = await spec.handler(session=session, ctx=ctx, **call_args)
        if spec.is_proposal and isinstance(result, dict) and not result.get("error") and not result.get("proposal_id"):
            if workspace_id is None:
                raise ValueError("No workspace available for proposal")
            from app.agents.services.proposal_service import create
            preview = {k: v for k, v in result.items() if k not in {"apply_endpoint", "proposal_id", "expected_hash", "expires_at", "status"}}
            stored = await create(session, user_id=ctx.user_id, workspace_id=workspace_id,
                kind=str(result.get("kind", spec.name.removeprefix("propose_"))),
                arguments=call_args, preview=preview, conversation_id=ctx.conversation_id, agent_id=ctx.agent_id)
            await session.commit()
            return {**result, "proposal_id": str(stored.id), "expected_hash": stored.contract_hash,
                    "expires_at": stored.expires_at.isoformat(), "status": stored.status}
        return result
