import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { AlertCircle, Check, ChevronDown, ChevronRight, Loader2, Sparkles } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { agents } from '@/lib/api'
import type { ProposalKind, ProposalData } from '@/lib/agent-proposals'

interface Props { data: ProposalData }

export function ProposalCard({ data }: Props) {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const [showJson, setShowJson] = useState(false)
  const id = data.proposal_id
  const state = useQuery({
    queryKey: ['agent-proposal', id], queryFn: () => agents.proposals.get(id!), enabled: !!id,
    initialData: id && data.status ? { proposal_id: id, kind: data.kind || '', preview: data.preview || data, status: data.status, expected_hash: data.expected_hash || '', expires_at: data.expires_at || '', result: data.result || null } : undefined,
    refetchOnWindowFocus: true,
  })
  const proposal = state.data
  const status = proposal?.status ?? data.status
  const decision = { expected_hash: proposal?.expected_hash || '' }
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ['agent-proposal', id] })
    qc.invalidateQueries({ queryKey: ['agent-proposals'] })
    qc.invalidateQueries({ queryKey: ['agent-conv-messages'] })
  }
  const apply = useMutation({
    mutationFn: () => agents.proposals.apply(id!, decision),
    onSuccess: (result) => { qc.setQueryData(['agent-proposal', id], result); refresh(); toast.success(t('agents.proposal.applied')); ['transactions','accounts','categories','category-groups','recurring-transactions','budgets','rules','goals','dashboard'].forEach((key) => qc.invalidateQueries({ queryKey: [key] })) },
    onError: (err: unknown) => { const e = err as { response?: { status?: number; data?: { detail?: string } } }; toast.error(e.response?.data?.detail || t('agents.proposal.applyFailed')); if (e.response?.status === 409) refresh() },
  })
  const reject = useMutation({
    mutationFn: () => agents.proposals.reject(id!, decision),
    onSuccess: (result) => { qc.setQueryData(['agent-proposal', id], result); refresh() },
    onError: () => { toast.error(t('agents.proposal.applyFailed')); refresh() },
  })
  if (data.error) return <div className="rounded-md border border-rose-300/50 bg-rose-50/40 dark:bg-rose-950/15 px-3 py-2 text-sm flex items-center gap-2 text-rose-700 dark:text-rose-200"><AlertCircle className="h-4 w-4 shrink-0" /><span className="truncate">{data.error}</span></div>
  if (!data.kind) return null
  const view = (proposal?.preview || data.preview || data) as ProposalData
  const applied = status === 'applied'
  const actionable = !!id && !!proposal?.expected_hash && status === 'pending' && !state.isError
  return <div className={cn('rounded-md border text-sm overflow-hidden', applied ? 'border-emerald-300/40 bg-emerald-50/30 dark:bg-emerald-950/15' : 'border-amber-300/50 bg-amber-50/40 dark:bg-amber-950/15')}>
    <div className="px-3 py-2.5 flex items-start gap-3">
      <div className="shrink-0 mt-0.5">{applied ? <Check className="h-4 w-4 text-emerald-600" /> : <Sparkles className="h-4 w-4 text-amber-600" />}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap"><span className="font-semibold">{kindTitle(data.kind, t)}</span><span className={cn('text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded', applied ? 'bg-emerald-200/70 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200' : 'bg-amber-200/70 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200')}>{applied ? t('agents.proposal.applied') : status === 'rejected' ? t('agents.proposal.rejected', 'Rejected') : status === 'expired' ? t('agents.proposal.expired', 'Expired') : t('agents.proposal.label')}</span></div>
        <div className="mt-1 text-muted-foreground text-[13px] leading-snug">{renderSummary(data.kind, view, t)}</div>
        <SplitPreview proposed={view.proposed} />
        {applied && proposal?.result && <div className="mt-1 text-xs text-emerald-700 dark:text-emerald-300">{safeStringify(proposal.result)}</div>}
        {!id && <div className="mt-1 text-xs text-muted-foreground">{t('agents.proposal.legacyReadOnly', 'Legacy preview — generate a new proposal to approve.')}</div>}
        {id && state.isError && <div className="mt-1 text-xs text-rose-600">{t('agents.proposal.applyFailed')}</div>}
      </div>
      <div className="shrink-0 flex items-center gap-1.5">{applied ? <span className="text-xs text-emerald-700 dark:text-emerald-300 inline-flex items-center gap-1"><Check className="h-3.5 w-3.5" />{t('agents.proposal.applied')}</span> : actionable ? <><Button size="sm" onClick={() => apply.mutate()} disabled={apply.isPending || reject.isPending}>{apply.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" /> : <Check className="h-3.5 w-3.5 mr-1.5" />}{t('agents.proposal.apply')}</Button><Button size="sm" variant="outline" onClick={() => reject.mutate()} disabled={apply.isPending || reject.isPending}>{reject.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : t('agents.proposal.reject', 'Reject')}</Button></> : null}</div>
    </div>
    <button type="button" onClick={() => setShowJson((v) => !v)} className="w-full text-[11px] uppercase tracking-wider text-muted-foreground hover:text-foreground border-t px-3 py-1.5 flex items-center gap-1.5">{showJson ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}{t('agents.proposal.showRaw')}</button>
    {showJson && <pre className="border-t px-3 py-2 text-[11px] leading-snug font-mono bg-background/60 overflow-x-auto">{safeStringify(view)}</pre>}
  </div>
}

function kindTitle(kind: ProposalKind, t: ReturnType<typeof useTranslation>['t']): string {
  if (kind === 'create_rule') return t('rules.newRule')
  if (kind === 'update_rule') return t('rules.editRule')
  if (kind === 'delete_rule') return t('rules.confirmDeleteTitle')
  return t(`agents.proposal.kind.${kind}`)
}

function renderSummary(kind: ProposalKind, d: ProposalData, t: ReturnType<typeof useTranslation>['t']): string {
  const p = d.proposed || {}; const tgt = d.target || {}
  const fmt = (amount: unknown, currency: unknown) => { if (amount == null) return '?'; const n = typeof amount === 'number' ? amount : Number(amount) || 0; const c = typeof currency === 'string' ? currency : 'BRL'; try { return new Intl.NumberFormat(undefined, { style: 'currency', currency: c }).format(n) } catch { return `${c} ${n.toFixed(2)}` } }
  switch (kind) {
    case 'categorize': return t('agents.proposal.summary.categorize', { count: d.affected_count ?? d.affected?.length ?? 0, category: d.target_category?.name ?? '?' })
    case 'create_category': return t('agents.proposal.summary.createCategory', { name: String(p.name ?? '?') }) + (d.name_collision ? ` — ${t('agents.proposal.collision', { name: d.name_collision.name })}` : '')
    case 'create_budget': return t('agents.proposal.summary.createBudget', { category: String(p.category_name ?? p.category_id ?? '?'), amount: fmt(p.amount, p.currency), month: String(p.month ?? '?') })
    case 'create_payee_rule': return t('agents.proposal.summary.createPayeeRule', { pattern: String(p.match_pattern ?? '?'), category: String(p.category_name ?? p.category_id ?? '?') })
    case 'create_rule': return `“${String(p.name ?? '?')}”`
    case 'update_rule': return `${String(tgt.name ?? '?')} → ${Object.keys(d.changes || {}).join(', ')}`
    case 'delete_rule': return `“${String(tgt.name ?? '?')}”`
    case 'create_transaction': return t('agents.proposal.summary.createTransaction', { description: String(p.description ?? '?'), amount: fmt(p.amount, p.currency), type: String(p.type ?? '?'), date: String(p.date ?? ''), account: String(p.account_name ?? '?') })
    case 'create_recurring_transaction': return t('agents.proposal.summary.createRecurring', { description: String(p.description ?? '?'), amount: fmt(p.amount, p.currency), frequency: String(p.frequency ?? '?'), day: p.day_of_month ? `${p.day_of_month}` : '?', account: String(p.account_name ?? '?') })
    case 'update_recurring_transaction': return t('agents.proposal.summary.updateRecurring', { description: String(tgt.description ?? '?'), changes: Object.entries(d.changes || {}).map(([k,v]) => `${k}: ${String(tgt[k] ?? '∅')} → ${String(v ?? '∅')}`).join(', ') || '—' })
    case 'cancel_recurring_transaction': return t(d.mode === 'delete' ? 'agents.proposal.summary.deleteRecurring' : 'agents.proposal.summary.deactivateRecurring', { description: String(tgt.description ?? '?') })
    case 'create_goal': return t('agents.proposal.summary.createGoal', { name: String(p.name ?? '?'), target: fmt(p.target_amount, p.currency), deadline: p.deadline ? String(p.deadline) : '—' })
    default: return ''
  }
}
function safeStringify(o: unknown): string { try { return JSON.stringify(o, null, 2) } catch { return String(o) } }
function SplitPreview({ proposed }: { proposed?: Record<string, unknown> }) {
  const p = proposed || {}; const splits = p.splits as { share_type?: string; items?: Array<Record<string, unknown>> } | undefined
  if (!splits || !Array.isArray(splits.items) || !splits.items.length) return null
  const currency = typeof p.currency === 'string' ? p.currency : 'BRL'
  return <div className="mt-1.5 text-[12px] border rounded px-2 py-1.5 bg-background/40"><div className="text-muted-foreground mb-1">{p.group_name ? `${p.group_name} · ` : ''}{splits.share_type || 'equal'}</div><ul className="space-y-0.5">{splits.items.map((s, i) => <li key={i} className="flex justify-between gap-3"><span>{String(s.member_name ?? s.member_id ?? '?')}</span><span className="tabular-nums">{String(s.share_amount ?? '?')} {currency}{s.share_pct != null ? ` (${Number(s.share_pct)}%)` : ''}</span></li>)}</ul></div>
}
