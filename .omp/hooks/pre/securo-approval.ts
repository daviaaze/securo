import type { HookAPI } from "@oh-my-pi/pi-coding-agent/extensibility/hooks";

const APPROVAL_TOOLS: Record<string, true> = {
  mcp__securo_approve_proposal: true,
  mcp__securo_reject_proposal: true,
};
const MCP_CONFIG = `${process.env.HOME ?? ""}/.omp/agent/mcp.json`;
type JsonRpcResponse = { result?: unknown; error?: { message?: string } };
type Proposal = {
  proposal_id?: unknown;
  kind?: unknown;
  preview?: unknown;
  status?: unknown;
  expected_hash?: unknown;
  expires_at?: unknown;
  result?: unknown;
};

async function configuredMcpCall(toolName: string, args: Record<string, unknown>): Promise<unknown> {
  const config = JSON.parse(await Bun.file(MCP_CONFIG).text()) as {
    mcpServers?: Record<string, { url?: string; headers?: Record<string, string> }>;
  };
  const server = config.mcpServers?.securo;
  if (typeof server?.url !== "string") throw new Error("Securo MCP connection is not configured");

  const headers: Record<string, string> = {
    Accept: "application/json, text/event-stream",
    "Content-Type": "application/json",
    ...(server.headers ?? {}),
  };
  const initialize = await fetch(server.url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "securo-omp-approval", version: "1" },
      },
    }),
  });
  if (!initialize.ok) throw new Error(`Securo MCP unavailable (${initialize.status})`);
  const initializeBody = await initialize.text();
  const initData = initializeBody.split("\n").find((line) => line.startsWith("data:"));
  const initResult = JSON.parse(initData ? initData.slice(5).trim() : initializeBody) as JsonRpcResponse;
  if (initResult.error || !initResult.result) throw new Error("Securo MCP initialization failed");
  headers["MCP-Protocol-Version"] = "2024-11-05";
  const sessionId = initialize.headers.get("mcp-session-id");
  if (sessionId) headers["MCP-Session-Id"] = sessionId;
  await fetch(server.url, {
    method: "POST",
    headers,
    body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
  });

  const response = await fetch(server.url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/call",
      params: { name: toolName, arguments: args },
    }),
  });
  if (!response.ok) throw new Error(`Securo MCP call failed (${response.status})`);
  const body = await response.text();
  const dataLine = body.split("\n").find((line) => line.startsWith("data:"));
  const result = JSON.parse(dataLine ? dataLine.slice(5).trim() : body) as JsonRpcResponse;
  if (result.error) throw new Error("Securo MCP rejected proposal lookup");
  const payload = result.result;
  if (payload && typeof payload === "object" && Array.isArray((payload as { content?: unknown }).content)) {
    for (const item of (payload as { content: Array<{ text?: unknown }> }).content) {
      if (typeof item.text !== "string") continue;
      try {
        return JSON.parse(item.text);
      } catch {
        // Continue looking for structured proposal data.
      }
    }
  }
  return payload;
}

export default function securoApprovalHook(pi: HookAPI): void {
  pi.on("tool_call", async (event, ctx) => {
    if (APPROVAL_TOOLS[event.toolName] !== true) return;
    if (!ctx.hasUI) return { block: true, reason: "Securo proposal approval requires an interactive UI" };

    const proposalId = event.input.proposal_id;
    const expectedHash = event.input.expected_hash;
    if (typeof proposalId !== "string" || typeof expectedHash !== "string") {
      return { block: true, reason: "Securo approval requires proposal_id and expected_hash" };
    }

    let proposal: Proposal;
    try {
      proposal = (await configuredMcpCall("get_proposal", { proposal_id: proposalId })) as Proposal;
    } catch {
      return { block: true, reason: "Unable to fetch the Securo proposal; approval blocked" };
    }
    if (proposal.status !== "pending" || proposal.expected_hash !== expectedHash) {
      return { block: true, reason: "Securo proposal is no longer pending or its hash changed" };
    }

    const action = event.toolName === "mcp__securo_reject_proposal" ? "Reject" : "Approve and apply";
    const preview = JSON.stringify({
      proposal_id: proposal.proposal_id,
      kind: proposal.kind,
      workspace: proposal.preview && typeof proposal.preview === "object"
        ? (proposal.preview as Record<string, unknown>).workspace
        : undefined,
      status: proposal.status,
      expires_at: proposal.expires_at,
      preview: proposal.preview,
      result: proposal.result,
    }, null, 2);
    const confirmed = await ctx.ui.confirm(
      `Securo: ${action} proposal ${proposalId}`,
      `Review the persisted proposal before continuing:\n${preview}\n\nConfirm only if this exact workspace, target, and impact are intended.`,
    );
    if (!confirmed) return { block: true, reason: "Securo proposal action cancelled" };
  });
}
