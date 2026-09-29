from __future__ import annotations

import uuid
from typing import Any, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models.conversation import Conversation, Message


async def list_conversations(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    agent_id: Optional[uuid.UUID] = None,
    limit: int = 50,
) -> list[Conversation]:
    q = select(Conversation).where(Conversation.workspace_id == workspace_id)
    if agent_id:
        q = q.where(Conversation.agent_id == agent_id)
    q = q.order_by(Conversation.updated_at.desc()).limit(limit)
    return list((await session.execute(q)).scalars().all())


async def get_conversation(
    session: AsyncSession, conversation_id: uuid.UUID, workspace_id: uuid.UUID
) -> Optional[Conversation]:
    return (await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.workspace_id == workspace_id,
        )
    )).scalar_one_or_none()


async def create_conversation(
    session: AsyncSession,
    *,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    agent_id: uuid.UUID,
    channel: str = "web",
    title: Optional[str] = None,
) -> Conversation:
    conv = Conversation(
        workspace_id=workspace_id,
        user_id=user_id,
        agent_id=agent_id,
        channel=channel,
        title=title,
    )
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv


async def delete_conversation(
    session: AsyncSession, conversation_id: uuid.UUID, workspace_id: uuid.UUID
) -> bool:
    conv = await get_conversation(session, conversation_id, workspace_id)
    if conv is None:
        return False
    await session.delete(conv)
    await session.commit()
    return True


async def list_messages(
    session: AsyncSession, conversation_id: uuid.UUID, limit: int = 200
) -> list[Message]:
    # Newest `limit` messages, returned in ascending ordinal order. Ordering
    # ascending BEFORE applying the limit pinned the window to the start of
    # the conversation: once a thread outgrew the executor's replay budget,
    # the model never received the newest user turn and kept re-answering
    # the last question inside the stale window.
    q = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.ordinal.desc())
        .limit(limit)
    )
    rows = list((await session.execute(q)).scalars().all())
    rows.reverse()
    return rows


async def uncheckpointed_messages(
    session: AsyncSession, conversation_id: uuid.UUID, cursor: int = 0,
) -> list[Message]:
    """Read all raw rows after the last committed summary checkpoint."""
    return list((await session.execute(
        select(Message).where(
            Message.conversation_id == conversation_id, Message.ordinal > cursor,
        ).order_by(Message.ordinal)
    )).scalars().all())


def complete_turns(rows: list[Message]) -> tuple[list[list[Message]], list[Message], bool]:
    """Split complete user→assistant turns without severing parallel tool calls.

    The last, newly persisted user row is returned separately. Any interrupted
    historical exchange is omitted as a unit, never replayed as orphaned tools.
    """
    turns: list[list[Message]] = []
    pending: list[Message] = []
    outstanding: set[str] = set()
    incomplete = False
    for row in rows:
        if row.role == "user":
            if pending:
                incomplete = True
            pending = [row]
            outstanding.clear()
        elif not pending:
            incomplete = True
        elif row.role == "assistant":
            pending.append(row)
            if row.tool_calls:
                ids = {call.get("id") for call in row.tool_calls}
                if outstanding or None in ids or len(ids) != len(row.tool_calls):
                    incomplete = True
                    pending = []
                else:
                    outstanding = ids
            elif not outstanding:
                turns.append(pending)
                pending = []
        elif row.role == "tool":
            call_id = (row.tool_result or {}).get("tool_call_id")
            if call_id not in outstanding:
                incomplete = True
                pending = []
                outstanding.clear()
            else:
                pending.append(row)
                outstanding.remove(call_id)
        else:
            incomplete = True
            pending = []
            outstanding.clear()
    current = pending if len(pending) == 1 and pending[0].role == "user" else []
    if pending and not current:
        incomplete = True
    return turns, current, incomplete


async def checkpoint_summary(
    session: AsyncSession, conversation_id: uuid.UUID,
    previous_cursor: Optional[int], through_ordinal: int, summary: str,
) -> bool:
    """Compare-and-set the summary cursor; concurrent runs cannot overwrite it."""
    prior = (Conversation.summary_through_ordinal.is_(None) if previous_cursor is None
             else Conversation.summary_through_ordinal == previous_cursor)
    result = await session.execute(
        update(Conversation).where(
            Conversation.id == conversation_id, prior,
            (Conversation.summary_through_ordinal.is_(None) if previous_cursor is None
             else Conversation.summary_through_ordinal < through_ordinal),
        ).values(summary=summary, summary_through_ordinal=through_ordinal)
    )
    await session.commit()
    return result.rowcount == 1


async def append_message(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    role: str,
    content: Optional[str] = None,
    tool_calls: Optional[list[Any]] = None,
    tool_result: Optional[dict[str, Any]] = None,
    citations: Optional[list[Any]] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> Message:
    next_ord = (await session.execute(
        select(func.coalesce(func.max(Message.ordinal), 0) + 1).where(
            Message.conversation_id == conversation_id
        )
    )).scalar_one()
    msg = Message(
        conversation_id=conversation_id,
        ordinal=int(next_ord),
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_result=tool_result,
        citations=citations,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    return msg


async def update_title_if_empty(
    session: AsyncSession, conversation_id: uuid.UUID, candidate: str
) -> None:
    conv = (await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )).scalar_one_or_none()
    if conv is None or conv.title:
        return
    conv.title = (candidate or "").strip()[:200] or None
    await session.commit()


async def update_title(
    session: AsyncSession, conversation_id: uuid.UUID, workspace_id: uuid.UUID, title: str
) -> Optional[Conversation]:
    """Always overwrites the title (used by the rename UI and by the
    LLM-generated title endpoint). Returns the updated row, or None if
    not found / not in this workspace."""
    conv = await get_conversation(session, conversation_id, workspace_id)
    if conv is None:
        return None
    conv.title = (title or "").strip()[:200] or None
    await session.commit()
    await session.refresh(conv)
    return conv
