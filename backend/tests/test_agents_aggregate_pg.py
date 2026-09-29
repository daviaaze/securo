"""Real PostgreSQL finance-tool semantics with isolated pgvector schemas."""
import uuid
from datetime import date
from decimal import Decimal

import pytest

import mcp_server.tools  # noqa: F401
from app.models.category import Category
from app.models.credit_card_bill import CreditCardBill
from app.models.account import Account
from app.models.transaction import Transaction
from app.models.user import User
from app.models.workspace import Workspace, WorkspaceMember
from mcp_server.auth import CallContext
from mcp_server.registry import REGISTRY


@pytest.fixture
async def finance_context(postgres_sessions):
    async with postgres_sessions() as session:
        user = User(
            id=uuid.uuid4(), email=f"finance-{uuid.uuid4()}@example.test",
            hashed_password="synthetic", is_active=True,
            preferences={"currency_display": "BRL"},
        )
        workspace = Workspace(
            id=uuid.uuid4(), name="Synthetic", kind="personal",
            created_by_user_id=user.id, default_currency="BRL", locale="pt-BR",
        )
        session.add_all([user, workspace])
        await session.flush()
        session.add(WorkspaceMember(
            id=uuid.uuid4(), user_id=user.id, workspace_id=workspace.id, role="owner",
        ))
        account = Account(
            id=uuid.uuid4(), user_id=user.id, workspace_id=workspace.id,
            name="Synthetic BRL", type="checking", currency="BRL", balance=Decimal("0"),
        )
        session.add(account)
        await session.commit()
        yield session, CallContext(
            user_id=user.id, workspace_id=workspace.id, conversation_id=uuid.uuid4(),
        ), account

async def _expense(session, ctx, account, *, amount, currency, primary, **fields):
    transaction = Transaction(
        id=uuid.uuid4(), user_id=ctx.user_id, workspace_id=ctx.workspace_id,
        account_id=account.id, description=f"Synthetic {currency}",
        amount=Decimal(amount), amount_primary=Decimal(primary) if primary is not None else None,
        currency=currency, date=date(2026, 2, 20), type="debit",
        source="manual", status="posted",
    )
    for key, value in fields.items():
        setattr(transaction, key, value)
    session.add(transaction)
    await session.commit()
    return transaction


async def test_aggregate_rejects_missing_fx_without_partial_sum(finance_context):
    session, ctx, account = finance_context
    await _expense(session, ctx, account, amount="100", currency="BRL", primary=None)
    await _expense(session, ctx, account, amount="10", currency="USD", primary=None)
    result = await REGISTRY["aggregate"].handler(
        session=session, ctx=ctx, metric="sum", group_by="month", tx_type="expense",
    )
    assert result == {"error": "missing_fx", "currency": "BRL"}


async def test_aggregate_converts_foreign_amount_without_mixing_currencies(finance_context):
    session, ctx, account = finance_context
    await _expense(session, ctx, account, amount="100", currency="BRL", primary=None)
    await _expense(session, ctx, account, amount="10", currency="USD", primary="50")
    result = await REGISTRY["aggregate"].handler(
        session=session, ctx=ctx, metric="sum", group_by="month", tx_type="expense",
    )
    assert result["currency"] == "BRL"
    assert result["basis"] == "gross_transactions"
    assert result["items"] == [{
        "bucket": "2026-02", "label": None, "value": 150.0, "count": 2,
    }]


async def test_missing_fx_in_limited_out_of_view_bucket_blocks_all_totals(finance_context):
    session, ctx, account = finance_context
    await _expense(session, ctx, account, amount="100", currency="BRL", primary=None)
    await _expense(
        session, ctx, account, amount="10", currency="USD", primary=None,
        date=date(2026, 3, 20),
    )
    total = await REGISTRY["aggregate"].handler(
        session=session, ctx=ctx, group_by="month", tx_type="expense", limit=1,
    )
    count = await REGISTRY["aggregate"].handler(
        session=session, ctx=ctx, metric="count", group_by="month", tx_type="expense",
    )
    assert total == {"error": "missing_fx", "currency": "BRL"}
    assert count["currency"] is None
    assert sum(item["value"] for item in count["items"]) == 2


async def test_aggregate_filters_and_explicit_transfer_inclusion(finance_context):
    session, ctx, account = finance_context
    transfer = Category(
        id=uuid.uuid4(), user_id=ctx.user_id, workspace_id=ctx.workspace_id,
        name="Movement", treat_as_transfer=True,
    )
    ignored = Category(
        id=uuid.uuid4(), user_id=ctx.user_id, workspace_id=ctx.workspace_id,
        name="Ignored", is_ignored=True,
    )
    closed = Account(
        id=uuid.uuid4(), user_id=ctx.user_id, workspace_id=ctx.workspace_id,
        name="Closed", type="checking", currency="BRL",
        balance=Decimal("0"), is_closed=True,
    )
    session.add_all([transfer, ignored, closed])
    await session.commit()
    for amount, fields in [
        ("100", {}),
        ("10", {"transfer_pair_id": uuid.uuid4()}),
        ("20", {"category_id": transfer.id}),
        ("30", {"is_ignored": True}),
        ("40", {"category_id": ignored.id}),
        ("50", {"exclude_from_pnl": True}),
        ("60", {"source": "settlement"}),
        ("70", {"source": "opening_balance"}),
    ]:
        await _expense(session, ctx, account, amount=amount, currency="BRL", primary=None, **fields)
    await _expense(session, ctx, closed, amount="80", currency="BRL", primary=None)
    default = await REGISTRY["aggregate"].handler(
        session=session, ctx=ctx, group_by="month", tx_type="expense",
    )
    included = await REGISTRY["aggregate"].handler(
        session=session, ctx=ctx, group_by="month", tx_type="expense",
        exclude_transfers=False,
    )
    assert default["items"][0]["value"] == 100.0
    assert default["transfers_included"] is False
    assert included["items"][0]["value"] == 130.0
    assert included["transfers_included"] is True


async def test_aggregate_uses_bill_override_and_native_currency(finance_context):
    session, ctx, account = finance_context
    await _expense(
        session, ctx, account, amount="25", currency="BRL", primary=None,
        effective_bill_date=date(2026, 3, 15),
    )
    await _expense(session, ctx, account, amount="10", currency="USD", primary="50")
    march = await REGISTRY["aggregate"].handler(
        session=session, ctx=ctx, metric="sum", group_by="month",
        from_date="2026-03-01", to_date="2026-03-31", tx_type="expense",
    )
    usd = await REGISTRY["aggregate"].handler(
        session=session, ctx=ctx, metric="sum", group_by="month",
        currency="USD", tx_type="expense",
    )
    assert march["items"] == [{"bucket": "2026-03", "label": None, "value": 25.0, "count": 1}]
    assert usd["items"][0]["value"] == 10.0
    assert usd["currency"] == "USD"


async def test_card_bills_status_order_and_workspace_isolation(finance_context):
    session, ctx, account = finance_context
    card = Account(
        id=uuid.uuid4(), user_id=ctx.user_id, workspace_id=ctx.workspace_id,
        name="Card", type="credit_card", currency="BRL", balance=Decimal("0"),
        credit_limit=Decimal("1000"), statement_close_day=10, payment_due_day=20,
    )
    session.add(card)
    await session.commit()
    bills_tool = REGISTRY["list_credit_card_bills"].handler
    assert (await bills_tool(session=session, ctx=ctx, account_id=str(card.id)))["status"] == "no_provider_bills"
    assert (await bills_tool(session=session, ctx=ctx, account_id=str(account.id)))["status"] == "not_credit_card"
    for day in (10, 20, 30):
        session.add(CreditCardBill(
            id=uuid.uuid4(), user_id=ctx.user_id, workspace_id=ctx.workspace_id,
            account_id=card.id, external_id=f"bill-{day}",
            due_date=date(2026, 4, day), total_amount=Decimal(str(day)),
            currency="BRL", minimum_payment=Decimal("5"),
        ))
    foreign_ws = Workspace(
        id=uuid.uuid4(), name="Foreign", kind="personal",
        created_by_user_id=ctx.user_id, default_currency="BRL", locale="pt-BR",
    )
    session.add(foreign_ws)
    await session.flush()
    foreign_card = Account(
        id=uuid.uuid4(), user_id=ctx.user_id, workspace_id=foreign_ws.id,
        name="Other card", type="credit_card", currency="BRL", balance=Decimal("0"),
    )
    session.add(foreign_card)
    await session.commit()
    got = await bills_tool(session=session, ctx=ctx, account_id=str(card.id), limit=2)
    assert got["status"] == "available"
    assert [row["due_date"] for row in got["items"]] == ["2026-04-30", "2026-04-20"]
    assert [row["total_amount"] for row in got["items"]] == [30.0, 20.0]
    assert all(row["minimum_payment"] == 5.0 for row in got["items"])
    assert await bills_tool(session=session, ctx=ctx, account_id=str(foreign_card.id)) == {
        "error": "account not found",
    }


async def test_transaction_installment_metadata_and_nulls(finance_context):
    session, ctx, account = finance_context
    series = uuid.uuid4()
    transaction = await _expense(
        session, ctx, account, amount="25", currency="BRL", primary=None,
        installment_number=2, total_installments=4, installment_total_amount=Decimal("100"),
        installment_purchase_date=date(2026, 1, 15), installment_series_id=series,
        effective_bill_date=date(2026, 3, 15),
    )
    tool = REGISTRY["list_transactions"].handler
    result = await tool(session=session, ctx=ctx, account_ids=[str(account.id)], limit=10)
    row = next(item for item in result["items"] if item["id"] == str(transaction.id))
    assert (row["installment_number"], row["total_installments"], row["installment_total_amount"]) == (2, 4, 100.0)
    assert row["installment_purchase_date"] == "2026-01-15"
    assert row["installment_series_id"] == str(series)
    assert row["effective_bill_date"] == "2026-03-15"
    assert row["bill_id"] is None
    plain = await _expense(session, ctx, account, amount="2", currency="BRL", primary=None)
    result = await tool(session=session, ctx=ctx, account_ids=[str(account.id)], limit=10)
    plain_row = next(item for item in result["items"] if item["id"] == str(plain.id))
    assert all(plain_row[key] is None for key in (
        "installment_number", "total_installments", "installment_total_amount",
        "installment_purchase_date", "installment_series_id", "bill_id", "effective_bill_date",
    ))
