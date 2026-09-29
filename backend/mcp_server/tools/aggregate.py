from __future__ import annotations

import re
from typing import Any

from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.account import Account
from app.models.category import Category
from app.models.transaction import Transaction
from app.models.user import User
from app.services import admin_service
from app.services._query_filters import counts_as_pnl, is_not_ignored, reporting_date_col
from mcp_server.auth import CallContext
from mcp_server.registry import tool
from mcp_server.tools._helpers import num, parse_date, parse_uuid_list, resolve_workspace_id


@tool(
    name="aggregate",
    description=(
        "Aggregate transactions without writing custom SQL. Choose a metric "
        "(sum|count|avg) and a group_by (category|month|account|payee|day). "
        "Returns {bucket,value,count} rows in primary currency unless a native "
        "currency filter is set. Missing FX returns an error, never a mixed sum. "
        "Gross transaction basis (not split-adjusted). Defaults to posted rows; "
        "status='pending' or 'all' includes other statuses."
    ),
    parameters={
        "type": "object",
        "properties": {
            "metric": {"type": "string", "enum": ["sum", "count", "avg"], "default": "sum"},
            "group_by": {"type": "string", "enum": ["category", "month", "account", "payee", "day"], "default": "category"},
            "from_date": {"type": "string", "format": "date"},
            "to_date": {"type": "string", "format": "date"},
            "account_ids": {"type": "array", "items": {"type": "string", "format": "uuid"}},
            "category_ids": {"type": "array", "items": {"type": "string", "format": "uuid"}},
            "payee_id": {"type": "string", "format": "uuid", "description": "Restrict to a single payee — useful for 'how much did I spend at X?'"},
            "currency": {"type": "string", "description": "Restrict to one native currency (BRL, USD, EUR, ...)"},
            "tx_type": {"type": "string", "enum": ["expense", "income"], "description": "Filter to expenses or income only"},
            "description_contains": {"type": "string", "description": "Case-insensitive substring match against the transaction description — use this to scope to a merchant/keyword like 'uber', 'spotify', 'amigos do bem'."},
            "status": {"type": "string", "enum": ["posted", "pending", "all"], "default": "posted", "description": "Default 'posted' = only money that already moved. Use 'pending' for scheduled/recurring not yet settled, or 'all' to include both."},
            "exclude_transfers": {"type": "boolean", "default": True},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 25, "description": "Max bucket rows in the result. Capped at 50."},
        },
        "additionalProperties": False,
    },
    tags=["read", "aggregate"],
)
async def aggregate(
    *,
    session: AsyncSession,
    ctx: CallContext,
    metric: str = "sum",
    group_by: str = "category",
    from_date: str | None = None,
    to_date: str | None = None,
    account_ids: list[str] | None = None,
    category_ids: list[str] | None = None,
    payee_id: str | None = None,
    currency: str | None = None,
    tx_type: str | None = None,
    description_contains: str | None = None,
    status: str = "posted",
    exclude_transfers: bool = True,
    limit: int = 25,
) -> dict[str, Any]:
    # Hard cap regardless of LLM input — keeps payload small enough for
    # token-tight providers.
    limit = max(1, min(int(limit), 50))
    ws_id = await resolve_workspace_id(session, ctx)
    user = await session.get(User, ctx.user_id)
    if user is None or not re.fullmatch(r"[A-Za-z]{3}", user.primary_currency or ""):
        return {"error": "primary_currency_unavailable"}
    primary_currency = user.primary_currency.upper()
    mode = await admin_service.get_credit_card_accounting_mode(session)
    reporting_date = reporting_date_col(mode)
    amount_col = (
        Transaction.amount if currency else
        case(
            (Transaction.currency == primary_currency, Transaction.amount),
            else_=Transaction.amount_primary,
        )
    )

    # Group expression + label expression.
    if group_by == "category":
        bucket_id = Transaction.category_id
        label_q = select(Category.id, Category.name).where(Category.workspace_id == ws_id)
    elif group_by == "account":
        bucket_id = Transaction.account_id
        label_q = None
    elif group_by == "payee":
        bucket_id = Transaction.payee_id
        label_q = None
    elif group_by == "month":
        bucket_id = func.to_char(reporting_date, "YYYY-MM")
        label_q = None
    elif group_by == "day":
        bucket_id = func.to_char(reporting_date, "YYYY-MM-DD")
        label_q = None
    else:
        return {"error": f"unknown group_by: {group_by}"}

    if metric == "sum":
        value_expr = func.sum(amount_col)
    elif metric == "avg":
        value_expr = func.avg(amount_col)
    else:  # count
        value_expr = func.count(Transaction.id)

    filters = [
        Transaction.workspace_id == ws_id,
        Account.workspace_id == ws_id,
        Account.is_closed.is_(False),
        Transaction.source != "opening_balance",
        counts_as_pnl() if exclude_transfers else and_(
            is_not_ignored(),
            Transaction.exclude_from_pnl.is_(False),
            ~and_(Transaction.source == "settlement", Transaction.type == "debit"),
        ),
    ]
    fd = parse_date(from_date)
    td = parse_date(to_date)
    if fd:
        filters.append(reporting_date >= fd)
    if td:
        filters.append(reporting_date <= td)

    accs = parse_uuid_list(account_ids)
    if accs:
        filters.append(Transaction.account_id.in_(accs))
    cats = parse_uuid_list(category_ids)
    if cats:
        filters.append(Transaction.category_id.in_(cats))
    if payee_id:
        # parse_uuid_list returns a list — single payee_id wrapped works too.
        pids = parse_uuid_list([payee_id])
        if pids:
            filters.append(Transaction.payee_id == pids[0])
    if currency:
        filters.append(Transaction.currency == currency.upper())
    if description_contains:
        filters.append(Transaction.description.ilike(f"%{description_contains}%"))
    if status == "posted":
        filters.append(Transaction.status == "posted")
    elif status == "pending":
        filters.append(Transaction.status == "pending")
    if tx_type == "expense":
        filters.append(Transaction.type == "debit")
    elif tx_type == "income":
        filters.append(Transaction.type == "credit")

    base = select(Transaction.id).join(Account, Account.id == Transaction.account_id).where(*filters)
    if metric != "count" and not currency:
        missing_fx = (await session.execute(
            base.where(
                Transaction.currency != primary_currency,
                Transaction.amount_primary.is_(None),
            ).limit(1)
        )).first()
        if missing_fx:
            return {"error": "missing_fx", "currency": primary_currency}

    q = (
        select(bucket_id.label("bucket"), value_expr.label("value"),
               func.count(Transaction.id).label("count"))
        .join(Account, Account.id == Transaction.account_id)
        .where(*filters)
        .group_by(bucket_id).order_by(value_expr.desc().nulls_last()).limit(limit)
    )

    rows = (await session.execute(q)).mappings().all()
    label_map: dict[Any, str] = {}
    if label_q is not None:
        for cid, cname in (await session.execute(label_q)).all():
            label_map[cid] = cname

    items = [
        {
            "bucket": str(r["bucket"]) if r["bucket"] is not None else None,
            "label": label_map.get(r["bucket"]),
            "value": num(r["value"]),
            "count": int(r.get("count", 0)),
        }
        for r in rows
    ]
    return {
        "metric": metric,
        "group_by": group_by,
        "from_date": fd.isoformat() if fd else None,
        "to_date": td.isoformat() if td else None,
        "items": items,
        "currency": None if metric == "count" else (currency.upper() if currency else primary_currency),
        "basis": "gross_transactions",
        "transfers_included": not exclude_transfers,
        "total": len(items),
    }
