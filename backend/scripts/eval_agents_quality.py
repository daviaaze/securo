"""Run synthetic, read-only Securo agent cases against a loopback Ollama model.

Prints case IDs, pass/fail, latency, and score only. No private conversations,
prompts, generated text, tool payloads or runtime identifiers leave this process.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def _loopback_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise argparse.ArgumentTypeError("Ollama URL must be loopback HTTP")
    if parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("Ollama URL must contain only loopback host and port")
    return value.rstrip("/")


def _numbers(text: str) -> set[Decimal]:
    values: set[Decimal] = set()
    for raw in re.findall(r"(?<![\w])\d[\d.,]*(?![\w])", text):
        digits = raw.rstrip(".,")
        if not digits:
            continue
        if "," in digits and "." in digits:
            normalized = digits.replace(".", "").replace(",", ".") if digits.rfind(",") > digits.rfind(".") else digits.replace(",", "")
        elif "," in digits:
            normalized = digits.replace(",", ".")
        elif digits.count(".") > 1:
            normalized = digits.replace(".", "")
        else:
            normalized = digits
        try:
            values.add(Decimal(normalized))
        except InvalidOperation:
            pass
    return values


async def _evaluate(url: str, model: str, cases: list[dict]) -> int:
    import sqlalchemy.types
    import pgvector.sqlalchemy as pgv
    from sqlalchemy.dialects.postgresql import UUID
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    # Same SQLite compatibility used by the test suite, before models import Vector.
    class VectorJSON(sqlalchemy.types.JSON):
        cache_ok = True

        def __init__(self, dim=None, *args, **kwargs):
            super().__init__(*args, **kwargs)

    pgv.Vector = VectorJSON

    @compiles(UUID, "sqlite")
    def _uuid_as_text(type_, compiler, **kwargs):
        return "CHAR(32)"

    from app.main import app as _registered_app  # noqa: F401 — register Base models
    from app.core.database import Base
    from app.agents.models.agent import Agent
    from app.agents.models.conversation import Conversation
    from app.agents.mcp.client import MCPRegistry, ToolHandle
    from app.agents.providers.ollama import OllamaProvider
    from app.agents.runtime.executor import AgentExecutor
    from app.models.user import User
    from app.models.workspace import Workspace, WorkspaceMember

    class ReadOnlyFixture(MCPRegistry):
        def __init__(self, responses: dict):
            self.responses = responses
            self.calls: list[str] = []

        async def discover(self, **kwargs):
            return [ToolHandle(
                server="securo", name=name, description=f"Read synthetic {name} data",
                parameters={"type": "object", "properties": {}, "additionalProperties": True},
            ) for name in self.responses]

        async def call(self, *, wire_name, arguments, **kwargs):
            name = wire_name.removeprefix("securo__")
            self.calls.append(name)
            return {"ok": "error" not in self.responses[name], "data": self.responses[name], "text": ""}

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    provider = OllamaProvider(base_url=url, model=model)
    passed = 0
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        for case in cases:
            started = time.monotonic()
            good = False
            try:
                async with factory() as session:
                    user = User(
                        id=uuid.uuid4(), email=f"synthetic-{uuid.uuid4()}@example.test",
                        hashed_password="synthetic", is_active=True,
                        preferences={"currency_display": "BRL"},
                    )
                    ws = Workspace(
                        id=uuid.uuid4(), name="Synthetic", kind="personal",
                        created_by_user_id=user.id, default_currency="BRL", locale="pt-BR",
                    )
                    session.add_all([user, ws])
                    await session.flush()
                    agent = Agent(
                        id=uuid.uuid4(), user_id=user.id, workspace_id=ws.id,
                        name="Synthetic quality evaluator", provider="ollama", model=model,
                        system_prompt="Use only available tools for financial facts. State currency explicitly.",
                        auto_context=False, temperature=0,
                    )
                    session.add_all([
                        WorkspaceMember(id=uuid.uuid4(), user_id=user.id, workspace_id=ws.id, role="owner"),
                        agent,
                    ])
                    await session.flush()
                    conv = Conversation(
                        id=uuid.uuid4(), user_id=user.id, agent_id=agent.id,
                        workspace_id=ws.id, channel="web",
                    )
                    session.add(conv)
                    await session.commit()
                    fixture = ReadOnlyFixture(case["tool_responses"])
                    with patch("app.agents.runtime.executor._provider_and_model_for", return_value=(provider, model)):
                        events = [event async for event in AgentExecutor(mcp=fixture).run(
                            session=session, agent=agent, user_id=user.id, workspace_id=ws.id,
                            conversation_id=conv.id, user_message=case["prompt"],
                        )]
                    answer = "".join(event.text or "" for event in events if event.type == "text_delta")
                    failed = any(event.type == "error" for event in events)
                    expected_error = case["expect_error"]
                    # A missing-FX tool error is an expected refusal, not necessarily
                    # an executor error event; require explicit refusal in answer.
                    refusal = bool(re.search(r"(missing|unavailable|cannot|can't|não|sem|impossível|rate|câmbio)", answer, re.I))
                    tools_ok = set(case["expected_tools"]).issubset(fixture.calls)
                    amount_ok = case["expected_decimal"] is None or Decimal(case["expected_decimal"]) in _numbers(answer)
                    currency_ok = case["expected_currency"] is None or case["expected_currency"] in answer.upper()
                    good = tools_ok and (failed or refusal if expected_error else not failed and amount_ok and currency_ok)
            except Exception:
                # Never print exception text: provider errors can contain payloads.
                good = False
            passed += int(good)
            print(f"{case['id']} {'PASS' if good else 'FAIL'} {time.monotonic() - started:.2f}s", flush=True)
    finally:
        await engine.dispose()
    print(f"score {passed}/{len(cases)}")
    return 0 if passed == len(cases) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", required=True, type=_loopback_url)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    os.environ.setdefault("AGENTS_ENABLED", "true")
    os.environ.setdefault("AGENTS_MCP_JWT_SECRET", "synthetic-eval-secret-not-production")
    os.environ.setdefault("CREDENTIALS_DIRECTORY", "")
    cases = [json.loads(line) for line in Path(__file__).with_name("agents_quality_cases.jsonl").read_text().splitlines() if line]
    return asyncio.run(_evaluate(args.ollama_url, args.model, cases))


if __name__ == "__main__":
    raise SystemExit(main())
