from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import pytest
from sqlalchemy import select, update

from app.agents.models import AgentProposal
from app.agents.services import proposal_service
from app.models.category import Category
from app.models.user import User
from app.models.workspace import Workspace, WorkspaceMember


async def _user_workspace(session, email: str, role: str = "owner"):
    user = User(
        id=uuid.uuid4(),
        email=email,
        hashed_password=bcrypt.hashpw(b"testpass123", bcrypt.gensalt(rounds=4)).decode(),
        is_active=True,
        is_verified=True,
        preferences={"timezone": "UTC", "currency_display": "USD"},
    )
    workspace = Workspace(
        id=uuid.uuid4(),
        name=f"Workspace {email}",
        kind="personal",
        created_by_user_id=user.id,
        default_currency="USD",
    )
    session.add_all([user, workspace])
    await session.flush()
    session.add(WorkspaceMember(
        id=uuid.uuid4(), workspace_id=workspace.id, user_id=user.id, role=role
    ))
    await session.commit()
    return user, workspace


async def _category_preview(session, user, workspace, name="Approved"):
    arguments = {"name": name, "icon": "circle-help", "color": "#6B7280"}
    preview = {"kind": "create_category", "proposed": arguments}
    return await proposal_service.create(
        session,
        user_id=user.id,
        workspace_id=workspace.id,
        kind="create_category",
        arguments=arguments,
        preview=preview,
    )


@pytest.mark.asyncio
async def test_postgres_proposal_apply_is_persistent_and_idempotent(postgres_sessions):
    async with postgres_sessions() as session:
        user, workspace = await _user_workspace(session, "proposal-apply@example.com")
        workspace_id = workspace.id
        proposal = await _category_preview(session, user, workspace)
        proposal_id = proposal.id
        contract_hash = proposal.contract_hash

        result = await proposal_service.approve_and_apply(
            session, user.id, workspace.id, proposal_id, contract_hash
        )
        assert result["status"] == "applied"
        assert result["result"]["kind"] == "create_category"

        categories = (await session.scalars(select(Category).where(
            Category.workspace_id == workspace_id,
            Category.name == "Approved",
        ))).all()
        assert len(categories) == 1

        repeated = await proposal_service.approve_and_apply(
            session, user.id, workspace.id, proposal_id, contract_hash
        )
        assert repeated["result"] == result["result"]
        assert len((await session.scalars(select(Category).where(
            Category.workspace_id == workspace_id,
            Category.name == "Approved",
        ))).all()) == 1


@pytest.mark.asyncio
async def test_postgres_proposal_reject_hash_and_expiry_are_fail_closed(postgres_sessions):
    async with postgres_sessions() as session:
        user, workspace = await _user_workspace(session, "proposal-reject@example.com")
        workspace_id = workspace.id
        proposal = await _category_preview(session, user, workspace, "Rejected")

        with pytest.raises(Exception) as wrong_hash:
            await proposal_service.reject(
                session, user.id, workspace.id, proposal.id, "wrong-hash"
            )
        assert getattr(wrong_hash.value, "status_code", None) == 409
        assert await session.scalar(select(Category.id).where(
            Category.workspace_id == workspace_id, Category.name == "Rejected"
        )) is None

        proposal.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        await session.commit()
        with pytest.raises(Exception) as expired:
            await proposal_service.approve_and_apply(
                session, user.id, workspace.id, proposal.id, proposal.contract_hash
            )
        assert getattr(expired.value, "status_code", None) == 409
        assert await session.scalar(select(Category.id).where(
            Category.workspace_id == workspace_id, Category.name == "Rejected"
        )) is None

@pytest.mark.asyncio
async def test_postgres_proposal_rechecks_revoked_role_and_workspace_scope(postgres_sessions):
    async with postgres_sessions() as session:
        user, workspace = await _user_workspace(session, "proposal-revoke@example.com")
        workspace_id = workspace.id
        proposal = await _category_preview(session, user, workspace, "Revoked")
        await session.execute(update(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace.id,
            WorkspaceMember.user_id == user.id,
        ).values(role="viewer"))
        await session.commit()

        with pytest.raises(Exception) as denied:
            await proposal_service.approve_and_apply(
                session, user.id, workspace.id, proposal.id, proposal.contract_hash
            )
        assert getattr(denied.value, "status_code", None) == 403
        assert await session.scalar(select(Category.id).where(
            Category.workspace_id == workspace_id, Category.name == "Revoked"
        )) is None

@pytest.mark.asyncio
async def test_postgres_proposal_rolls_back_domain_change_on_executor_failure(
    postgres_sessions, monkeypatch
):
    async with postgres_sessions() as session:
        user, workspace = await _user_workspace(session, "proposal-rollback@example.com")
        workspace_id = workspace.id
        proposal = await _category_preview(session, user, workspace, "Rolled Back")

        async def fail_after_mutation(inner_session, row):
            await proposal_service.category_service.create_category(
                inner_session,
                workspace.id,
                user.id,
                proposal_service.CategoryCreate.model_validate(row.arguments),
                commit=False,
            )
            raise RuntimeError("simulated executor failure")

        monkeypatch.setattr(proposal_service, "_apply_stored_arguments", fail_after_mutation)
        with pytest.raises(RuntimeError):
            await proposal_service.approve_and_apply(
                session, user.id, workspace.id, proposal.id, proposal.contract_hash
            )

        assert await session.scalar(select(Category.id).where(
            Category.workspace_id == workspace_id, Category.name == "Rolled Back"
        )) is None
        stored = await session.get(AgentProposal, proposal.id)
        assert stored.status == "pending"
