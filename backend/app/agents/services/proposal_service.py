from __future__ import annotations
import hashlib, json, uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.agents.models import AgentProposal
from app.models.workspace import Workspace, WorkspaceMember
from app.core.app_clock import app_today
from app.schemas.budget import BudgetCreate
from app.schemas.category import CategoryCreate
from app.schemas.goal import GoalCreate
from app.schemas.recurring_transaction import RecurringTransactionCreate, RecurringTransactionUpdate
from app.schemas.rule import RuleCreate, RuleUpdate
from app.schemas.transaction import TransactionCreate
from app.services import budget_service, category_service, goal_service, recurring_transaction_service
from app.services import rule_service, transaction_service
_WRITE_ROLES = {"owner", "editor", "manager"}

def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Proposal not found")

async def _authorize(session: AsyncSession, user_id: uuid.UUID, workspace_id: uuid.UUID, write: bool) -> None:
    row = await session.execute(select(Workspace.id, WorkspaceMember.role).join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id).where(Workspace.id == workspace_id, WorkspaceMember.user_id == user_id, Workspace.is_archived.is_(False)))
    member = row.first()
    if member is None:
        raise _not_found()
    if write and member.role not in _WRITE_ROLES:
        raise HTTPException(status_code=403, detail="Read-only role")

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _expired(value: datetime) -> bool:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value <= _now()

def _hash(kind: str, arguments: dict[str, Any], preview: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({"kind": kind, "arguments": arguments, "preview": preview}, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

async def create(session: AsyncSession, *, user_id: uuid.UUID, workspace_id: uuid.UUID, kind: str, arguments: dict[str, Any], preview: dict[str, Any], target_snapshot: dict[str, Any] | None = None, conversation_id: uuid.UUID | None = None, agent_id: uuid.UUID | None = None) -> AgentProposal:
    await _authorize(session, user_id, workspace_id, True)
    row = AgentProposal(workspace_id=workspace_id, user_id=user_id, kind=kind, arguments=arguments, preview=preview, target_snapshot=target_snapshot, contract_hash=_hash(kind, arguments, preview), expires_at=datetime.now(timezone.utc)+timedelta(hours=24), conversation_id=conversation_id, agent_id=agent_id)
    session.add(row)
    await session.flush()
    return row

async def get(session: AsyncSession, user_id: uuid.UUID, workspace_id: uuid.UUID, proposal_id: uuid.UUID, *, write: bool = False) -> AgentProposal:
    await _authorize(session, user_id, workspace_id, write)
    row = await session.scalar(select(AgentProposal).where(AgentProposal.id == proposal_id, AgentProposal.user_id == user_id, AgentProposal.workspace_id == workspace_id))
    if row is None:
        raise _not_found()
    return row

async def list_proposals(session: AsyncSession, user_id: uuid.UUID, workspace_id: uuid.UUID, *, status: str | None = None) -> list[AgentProposal]:
    await _authorize(session, user_id, workspace_id, False)
    query = select(AgentProposal).where(AgentProposal.user_id == user_id, AgentProposal.workspace_id == workspace_id)
    if status:
        query = query.where(AgentProposal.status == status)
    return list((await session.scalars(query.order_by(AgentProposal.created_at.desc()))).all())

def serialize(row: AgentProposal) -> dict[str, Any]:
    return {"proposal_id": str(row.id), "workspace_id": str(row.workspace_id), "kind": row.kind, "preview": row.preview, "status": row.status, "expected_hash": row.contract_hash, "expires_at": row.expires_at.isoformat(), "created_at": row.created_at.isoformat() if row.created_at else None, "conversation_id": str(row.conversation_id) if row.conversation_id else None, "agent_id": str(row.agent_id) if row.agent_id else None, "result": row.result}
async def _verify_preview_target(session: AsyncSession, row: AgentProposal) -> None:
    args, preview, ws = row.arguments, row.preview, row.workspace_id
    if row.kind == "categorize":
        from app.models.category import Category
        from app.models.transaction import Transaction
        category_id = uuid.UUID(args["category_id"])
        if await session.scalar(select(Category.id).where(
            Category.id == category_id, Category.workspace_id == ws
        )) is None:
            raise HTTPException(status_code=409, detail="Proposal category changed or was removed")
        expected = {uuid.UUID(t["id"]): t for t in preview.get("affected", [])}
        current = {
            item.id: item
            for item in (await session.scalars(select(Transaction).where(
                Transaction.id.in_(list(expected)), Transaction.workspace_id == ws
            ))).all()
        }
        if len(current) != len(expected) or any(
            (
                (str(current[tid].category_id) if current[tid].category_id else None)
                != target.get("current_category_id")
                or current[tid].description != target.get("description")
                or float(current[tid].amount) != target.get("amount")
                or current[tid].currency != target.get("currency")
            )
            for tid, target in expected.items()
        ):
            raise HTTPException(status_code=409, detail="Proposal target changed; create a new proposal")
    elif row.kind in ("update_recurring_transaction", "cancel_recurring_transaction"):
        from app.models.recurring_transaction import RecurringTransaction
        target = preview.get("target", {})
        item = await session.scalar(select(RecurringTransaction).where(
            RecurringTransaction.id == uuid.UUID(args["recurring_id"]),
            RecurringTransaction.workspace_id == ws,
        ))
        if item is None or any(
            (str(getattr(item, key)) if key == "id" else
             float(getattr(item, key)) if key == "amount" else
             bool(getattr(item, key)) if key == "is_active" else getattr(item, key))
            != value
            for key, value in target.items()
            if key in {"description", "amount", "currency", "frequency", "weekend_adjustment", "day_of_month", "is_active"}
        ):
            raise HTTPException(status_code=409, detail="Proposal target changed; create a new proposal")
    elif row.kind in ("update_rule", "delete_rule"):
        from app.schemas.rule import RuleRead
        item = await rule_service.get_rule(session, uuid.UUID(args["rule_id"]), ws)
        target = preview.get("target")
        if item is None or target is None:
            raise HTTPException(status_code=409, detail="Proposal target changed; create a new proposal")
        current = RuleRead.model_validate(item).model_dump(mode="json", exclude={"user_id"})
        if current != target:
            raise HTTPException(status_code=409, detail="Proposal target changed; create a new proposal")

async def _apply_stored_arguments(
    session: AsyncSession, row: AgentProposal
) -> dict[str, Any]:
    """Execute only the persisted proposal arguments; never accept a client payload."""
    args = dict(row.arguments)
    ws, user, kind = row.workspace_id, row.user_id, row.kind
    if kind == "categorize":
        count = await transaction_service.bulk_update_category(
            session, ws, [uuid.UUID(v) for v in args["transaction_ids"]],
            uuid.UUID(args["category_id"]), commit=False,
        )
        return {"updated": count}
    if kind == "create_category":
        from app.models.category_group import CategoryGroup
        args["icon"] = args.get("icon") or "circle-help"
        args["color"] = args.get("color") or "#6B7280"
        group_id = args.get("group_id")
        if group_id and await session.scalar(select(CategoryGroup.id).where(
            CategoryGroup.id == uuid.UUID(group_id), CategoryGroup.workspace_id == ws
        )) is None:
            raise HTTPException(status_code=409, detail="Proposal category group changed or was removed")
        obj = await category_service.create_category(
            session, ws, user, CategoryCreate.model_validate(args), commit=False
        )
    elif kind == "create_budget":
        from app.models.category import Category
        category_id = uuid.UUID(args["category_id"])
        if await session.scalar(select(Category.id).where(
            Category.id == category_id, Category.workspace_id == ws
        )) is None:
            raise HTTPException(status_code=409, detail="Proposal category changed or was removed")
        raw = {k: v for k, v in args.items() if k != "currency"}
        obj = await budget_service.create_budget(
            session, ws, user, BudgetCreate.model_validate(raw), commit=False
        )
    elif kind == "create_transaction":
        raw = dict(args)
        raw.pop("group_id", None)
        raw["date"] = raw.get("date") or row.preview.get("proposed", {}).get("date") or app_today()
        split_args = raw.pop("splits", None)
        if split_args is not None:
            from app.schemas.transaction_split import TransactionSplitsInput
            raw["splits"] = TransactionSplitsInput.model_validate({
                "share_type": split_args["share_type"],
                "splits": [
                    {"group_member_id": m.get("group_member_id", m.get("member_id")),
                     "share_amount": m.get("share_amount"), "share_pct": m.get("share_pct"),
                     "notes": m.get("notes")}
                    for m in split_args.get("splits", split_args.get("members", []))
                ],
            })
        obj = await transaction_service.create_transaction(
            session, ws, user, TransactionCreate.model_validate(raw), commit=False
        )
    elif kind == "create_recurring_transaction":
        proposed = row.preview.get("proposed", {})
        for key in ("start_date", "end_date", "currency"):
            if args.get(key) is None:
                args[key] = proposed.get(key)
        if args.get("start_date") is None:
            args["start_date"] = app_today()
        if args.get("currency") is None:
            args.pop("currency", None)
        obj = await recurring_transaction_service.create_recurring_transaction(
            session, ws, user, RecurringTransactionCreate.model_validate(args), commit=False
        )
    elif kind == "update_recurring_transaction":
        rid = uuid.UUID(args["recurring_id"])
        changes = {k: v for k, v in args.items() if k != "recurring_id"}
        obj = await recurring_transaction_service.update_recurring_transaction(
            session, rid, ws, RecurringTransactionUpdate.model_validate(changes), commit=False
        )
        if obj is None:
            raise HTTPException(status_code=409, detail="Proposal target no longer exists")
    elif kind == "cancel_recurring_transaction":
        rid = uuid.UUID(args["recurring_id"])
        if args.get("mode", "deactivate") == "delete":
            result = await recurring_transaction_service.delete_recurring_transaction(
                session, rid, ws, commit=False
            )
        else:
            obj = await recurring_transaction_service.update_recurring_transaction(
                session, rid, ws, RecurringTransactionUpdate(is_active=False), commit=False
            )
            result = obj is not None
        if not result:
            raise HTTPException(status_code=409, detail="Proposal target no longer exists")
        return {"success": True}
    elif kind == "create_goal":
        goal_args = dict(args)
        proposed = row.preview.get("proposed", {})
        for key in ("currency", "icon", "color"):
            if goal_args.get(key) is None:
                goal_args[key] = proposed.get(key)
        if "deadline" in goal_args:
            goal_args["target_date"] = goal_args.pop("deadline")
        if "initial_amount" in goal_args:
            goal_args["current_amount"] = goal_args.pop("initial_amount")
        obj = await goal_service.create_goal(
            session, ws, user, GoalCreate.model_validate(goal_args), commit=False
        )
    elif kind == "create_rule":
        draft = RuleCreate.model_validate(args)
        should_apply = draft.is_active and draft.apply_to_existing
        rule = await rule_service.create_rule(
            session, ws, user,
            draft.model_copy(update={"apply_to_existing": False}),
            commit=False,
        )
        if should_apply:
            await rule_service.apply_single_rule(
                session, ws, rule,
                overwrite_existing_categories=draft.overwrite_existing_categories,
                commit=False,
            )
        obj = rule
    elif kind == "update_rule":
        rid = uuid.UUID(args["rule_id"])
        changes = {k: v for k, v in args.items() if k != "rule_id"}
        draft = RuleUpdate.model_validate(changes)
        changed = draft.model_dump(exclude_unset=True)
        should_apply = (
            draft.apply_to_existing
            if draft.apply_to_existing is not None
            else bool({"conditions", "conditions_op", "actions"}.intersection(changed))
        )
        rule = await rule_service.update_rule(
            session, rid, ws,
            draft.model_copy(update={"apply_to_existing": False}),
            commit=False,
        )
        if rule is None:
            raise HTTPException(status_code=409, detail="Proposal target no longer exists")
        if rule.is_active and should_apply:
            await rule_service.apply_single_rule(
                session, ws, rule,
                overwrite_existing_categories=draft.overwrite_existing_categories,
                commit=False,
            )
        obj = rule
    elif kind == "delete_rule":
        if not await rule_service.delete_rule(
            session, uuid.UUID(args["rule_id"]), ws, commit=False
        ):
            raise HTTPException(status_code=409, detail="Proposal target no longer exists")
        return {"success": True}
    elif kind == "create_payee_rule":
        pattern = str(args["match_pattern"])
        data = RuleCreate.model_validate({
            "name": f"Payee: {pattern}",
            "conditions": [{"field": "description", "op": "contains", "value": pattern}],
            "actions": [{"op": "set_category", "value": str(args["category_id"])}],
            "apply_to_existing": False,
        })
        obj = await rule_service.create_rule(session, ws, user, data, commit=False)
    else:
        raise HTTPException(status_code=409, detail="Unsupported proposal kind")
    return {"id": str(getattr(obj, "id", "")), "kind": kind}

async def approve_and_apply(
    session: AsyncSession,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    proposal_id: uuid.UUID,
    expected_hash: str,
) -> dict[str, Any]:
    had_transaction = session.in_transaction()
    context = session.begin_nested() if had_transaction else session.begin()
    applied_now = False
    async with context:
        row = await session.scalar(
            select(AgentProposal).where(AgentProposal.id == proposal_id).with_for_update()
        )
        if row is None or row.user_id != user_id or row.workspace_id != workspace_id:
            raise _not_found()
        await _authorize(session, user_id, workspace_id, True)
        if row.status == "applied" and row.contract_hash == expected_hash:
            result = serialize(row)
        else:
            if (
                row.status != "pending"
                or _expired(row.expires_at)
                or row.contract_hash != expected_hash
                or _hash(row.kind, row.arguments, row.preview) != row.contract_hash
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Proposal is stale, expired, or already decided; create a new proposal",
                )
            await _verify_preview_target(session, row)
            row.result = await _apply_stored_arguments(session, row)
            applied_now = True
            row.status = "applied"
            row.decided_at = datetime.now(timezone.utc)
            row.decided_by = user_id
            await session.flush()
            result = serialize(row)
    if had_transaction and applied_now:
        await session.commit()
    return result
async def reject(
    session: AsyncSession,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    proposal_id: uuid.UUID,
    expected_hash: str,
) -> AgentProposal:
    had_transaction = session.in_transaction()
    context = session.begin_nested() if had_transaction else session.begin()
    async with context:
        row = await session.scalar(
            select(AgentProposal).where(AgentProposal.id == proposal_id).with_for_update()
        )
        if row is None or row.user_id != user_id or row.workspace_id != workspace_id:
            raise _not_found()
        await _authorize(session, user_id, workspace_id, True)
        if (
            row.status != "pending"
            or _expired(row.expires_at)
            or row.contract_hash != expected_hash
            or _hash(row.kind, row.arguments, row.preview) != row.contract_hash
        ):
            raise HTTPException(status_code=409, detail="Proposal is stale, expired, or already decided")
        row.status = "rejected"
        row.decided_at = datetime.now(timezone.utc)
        row.decided_by = user_id
        await session.flush()
    if had_transaction:
        await session.commit()
    return row
