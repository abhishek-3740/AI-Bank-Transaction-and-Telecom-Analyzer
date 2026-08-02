import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getCustomer, getStrReport } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import RiskBadge from '../components/RiskBadge'
import RulesBadge from '../components/RulesBadge'
import { formatCurrency, formatDate, formatDateTime, formatRatio, formatRisk, getBandClass, getBandFromRisk, parseReason } from '../utils/formatters'
import './CustomerDetail.css'

function CustomerReportModal({ report, onClose }) {
  if (!report) return null
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <section className="report-modal" role="dialog" aria-modal="true" aria-labelledby="report-modal-title">
        <header className="report-modal-header">
          <div><span className="section-kicker">System-generated dossier</span><h2 id="report-modal-title">Suspicious Transaction Report</h2><p>{report.report_id}</p></div>
          <button className="icon-button" onClick={onClose} aria-label="Close report">×</button>
        </header>
        <div className="report-modal-body">
          <div className="report-meta-grid"><div><span>Generated</span><strong>{formatDateTime(report.generated_at)}</strong></div><div><span>Reporting officer</span><strong>{report.reporting_officer || 'System'}</strong></div><div><span>Customer</span><strong>{report.customer_name}</strong></div><div><span>Customer ID</span><strong className="mono">{report.customer_id}</strong></div></div>
          <div className="report-summary-grid"><div><span>Suspicious transactions</span><strong>{report.total_suspicious_transactions}</strong></div><div><span>Total amount</span><strong>{formatCurrency(report.total_suspicious_amount)}</strong></div><div><span>Date range</span><strong>{formatDate(report.date_range_from)} — {formatDate(report.date_range_to)}</strong></div><div><span>Primary band</span><RiskBadge band={report.primary_risk_band} /></div></div>
          {report.scenario_types_detected?.length > 0 && <div className="report-section"><span className="section-kicker">Scenario types detected</span><div className="scenario-tags">{report.scenario_types_detected.map((scenario) => <span key={scenario} className="scenario-tag">{scenario.replace(/_/g, ' ')}</span>)}</div></div>}
          <div className="report-section"><span className="section-kicker">Investigator narrative</span><blockquote>{report.narrative}</blockquote></div>
          <div className="report-section"><span className="section-kicker">Flagged transactions</span><div className="table-scroll report-table-scroll"><table className="data-table"><thead><tr><th>Date</th><th>Transaction ID</th><th>Amount</th><th>Risk</th><th>Reasons</th><th>Rules</th></tr></thead><tbody>{(report.transactions || []).map((transaction) => <tr key={transaction.Transaction_ID}><td>{formatDate(transaction.Date)}</td><td className="mono primary-cell">{transaction.Transaction_ID}</td><td className="numeric">{formatCurrency(transaction.Transaction_Amount)}</td><td><RiskBadge band={transaction.risk_band} score={transaction.risk_score} showScore /></td><td><span className="report-reason-cell">{(transaction.reasons || []).join(' · ') || '—'}</span></td><td><RulesBadge rules={transaction.rules_fired} limit={2} /></td></tr>)}</tbody></table></div></div>
          {(report.graph_suspicion_score !== null && report.graph_suspicion_score !== undefined) && <div className="report-section"><span className="section-kicker">Graph context</span><div className="graph-context-grid"><div><span>Suspicion score</span><strong>{Number(report.graph_suspicion_score).toFixed(4)}</strong></div><div><span>In / out ratio</span><strong>{formatRatio(report.graph_in_out_ratio)}</strong></div><div><span>Mule flag</span><strong className={report.graph_mule_flag ? 'danger-text' : 'success-text'}>{report.graph_mule_flag ? 'FLAGGED' : 'CLEAR'}</strong></div></div></div>}
        </div>
      </section>
    </div>
  )
}

function RiskReason({ reason }) {
  const parsed = parseReason(reason)
  const contributionClass = parsed.contribution >= 0 ? 'positive-contribution' : 'negative-contribution'
  return <span className="risk-reason" title={reason}><span>{parsed.feature}</span>{parsed.contribution !== null && <strong className={contributionClass}>({parsed.contribution > 0 ? '+' : ''}{parsed.contribution.toFixed(2)})</strong>}</span>
}

export default function CustomerDetail() {
  const { customerId } = useParams()
  const navigate = useNavigate()
  const [customer, setCustomer] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showAll, setShowAll] = useState(false)
  const [report, setReport] = useState(null)
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState('')

  useEffect(() => {
    let mounted = true
    setLoading(true)
    setError('')
    getCustomer(customerId)
      .then((data) => { if (mounted) setCustomer(data) })
      .catch((requestError) => { if (mounted) setError(requestError.message) })
      .finally(() => { if (mounted) setLoading(false) })
    return () => { mounted = false }
  }, [customerId])

  const riskBands = useMemo(() => {
    const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 }
    ;(customer?.top_transactions || []).forEach((transaction) => { const band = transaction.risk_band || getBandFromRisk(transaction.risk_score); counts[band] = (counts[band] || 0) + 1 })
    return Object.entries(counts).map(([band, count]) => ({ band, count }))
  }, [customer])
  const transactions = showAll ? (customer?.top_transactions || []) : (customer?.top_transactions || []).slice(0, 5)

  const handleGenerateReport = () => {
    setReportLoading(true)
    setReportError('')
    getStrReport(customerId)
      .then((data) => setReport(data))
      .catch((requestError) => setReportError(requestError.message))
      .finally(() => setReportLoading(false))
  }

  if (loading) return <div className="page customer-page"><div className="page-loading"><LoadingSpinner size="lg" /><span>Loading customer investigation…</span></div></div>
  if (error || !customer) return <div className="page customer-page"><button className="button button-ghost back-button" onClick={() => navigate('/alerts')}>← Back to alerts</button><div className="error-banner" role="alert"><span aria-hidden="true">!</span><div><strong>Customer investigation unavailable</strong><br />{error || 'Customer not found.'}</div></div></div>

  const columns = [
    { key: 'Date', label: 'Date', render: (row) => formatDate(row.Date) },
    { key: 'Transaction_Mode', label: 'Mode', render: (row) => row.Transaction_Mode || 'BANK' },
    { key: 'Transaction_Amount', label: 'Amount', align: 'right', render: (row) => formatCurrency(row.Transaction_Amount) },
    { key: 'risk_score', label: 'Risk Score', align: 'right', render: (row) => <RiskBadge band={row.risk_band} score={row.risk_score} showScore /> },
    { key: 'reason_1', label: 'Reason 1', render: (row) => <RiskReason reason={row.reason_1} /> },
    { key: 'rules_fired', label: 'Rules', render: (row) => <RulesBadge rules={row.rules_fired} limit={2} /> },
  ]

  return (
    <div className="page customer-page">
      <div className="page-header customer-page-header"><div><button className="button button-ghost back-button" onClick={() => navigate('/alerts')}>← Back to alert queue</button><span className="section-kicker">Customer investigation / linked evidence</span><h2 className="page-heading">{customer.customer_name}</h2><p className="customer-id">Customer ID <span>{customer.customer_id}</span></p></div><button className="button button-danger generate-report-button" onClick={handleGenerateReport} disabled={reportLoading}><span aria-hidden="true">▣</span>{reportLoading ? 'Generating report…' : 'Generate STR Report'}</button></div>
      {reportError && <div className="error-banner" role="alert"><span aria-hidden="true">!</span><div><strong>STR report unavailable</strong><br />{reportError}</div></div>}

      <section className="panel customer-header-card"><div className="customer-header-intro"><span className="section-kicker">Risk subject</span><h3>{customer.customer_name}</h3><p>{customer.customer_id}</p></div><div className="customer-header-stats"><div><span>Total transactions</span><strong>{Number(customer.total_transactions || 0).toLocaleString('en-IN')}</strong></div><div><span>Alerts</span><strong className="danger-text">{Number(customer.alert_count || 0).toLocaleString('en-IN')}</strong></div><div><span>Max risk score</span><strong className={`band-${getBandClass(customer.dominant_risk_band)}`}>{formatRisk(customer.max_risk_score)}</strong></div><div><span>Primary band</span><RiskBadge band={customer.dominant_risk_band} /></div></div></section>

      <section className="customer-investigation-grid">
        <article className="panel risk-profile-panel"><div className="panel-header"><div><span className="section-kicker">Signal profile</span><h3 className="section-heading">Risk profile</h3></div></div><div className="risk-score-display"><span>Maximum observed risk</span><strong className={`band-${getBandClass(customer.dominant_risk_band)}`}>{formatRisk(customer.max_risk_score)}</strong><RiskBadge band={customer.dominant_risk_band} /></div><div className="customer-rule-section"><span className="section-kicker">Rules fired</span><div className="customer-rules"><RulesBadge rules={customer.rules_fired_summary} /></div></div><div className="mini-chart-header"><span className="section-kicker">Top-transaction band mix</span><span className="text-muted">{(customer.top_transactions || []).length} observed</span></div><div className="mini-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={riskBands} margin={{ top: 6, right: 10, left: -22, bottom: 0 }}><CartesianGrid stroke="var(--border)" vertical={false} /><XAxis dataKey="band" tick={{ fill: 'var(--text-muted)', fontSize: 9 }} tickLine={false} axisLine={false} /><YAxis allowDecimals={false} tick={{ fill: 'var(--text-muted)', fontSize: 9 }} tickLine={false} axisLine={false} /><Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text-primary)' }} /><Bar dataKey="count" radius={[3, 3, 0, 0]}>{riskBands.map((entry) => <Cell key={entry.band} fill={{ CRITICAL: 'var(--critical)', HIGH: 'var(--high)', MEDIUM: 'var(--medium)', LOW: 'var(--low)' }[entry.band]} />)}</Bar></BarChart></ResponsiveContainer></div></article>

        <article className="panel customer-transactions-panel"><div className="panel-header"><div><span className="section-kicker">Highest-risk observations</span><h3 className="section-heading">Top transactions</h3></div>{(customer.top_transactions || []).length > 5 && <button className="button button-ghost button-small" onClick={() => setShowAll((current) => !current)}>{showAll ? 'Show top 5' : 'Show all'}</button>}</div><div className="customer-table-wrap"><div className="table-scroll"><table className="data-table"><thead><tr>{columns.map((column) => <th key={column.key} className={column.align === 'right' ? 'align-right' : ''}>{column.label}</th>)}</tr></thead><tbody>{transactions.map((row) => <tr key={row.Transaction_ID}>{columns.map((column) => <td key={column.key} className={column.align === 'right' ? 'numeric' : ''}>{column.render ? column.render(row) : row[column.key] || '—'}</td>)}</tr>)}</tbody></table></div></div>{transactions.length === 0 && <div className="empty-state"><span className="empty-state-icon">⌕</span><h3 className="empty-state-title">No transactions returned</h3></div>}</article>
      </section>

      {report && <CustomerReportModal report={report} onClose={() => setReport(null)} />}
    </div>
  )
}
