import { useEffect, useMemo, useState } from 'react'
import { getReportsSummary, getStrBatch, getStrReport } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PipelineEmpty from '../components/PipelineEmpty'
import RiskBadge from '../components/RiskBadge'
import RulesBadge from '../components/RulesBadge'
import { downloadJson, formatCurrency, formatDate, formatDateTime, formatRatio } from '../utils/formatters'
import './STRReports.css'

export default function STRReports() {
  const [summary, setSummary] = useState(null)
  const [customers, setCustomers] = useState([])
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState('')
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [reportLoading, setReportLoading] = useState(false)
  const [error, setError] = useState('')
  const [pipelineEmpty, setPipelineEmpty] = useState(false)
  const [reportError, setReportError] = useState('')

  useEffect(() => {
    let mounted = true
    setLoading(true)
    Promise.all([getReportsSummary(), getStrBatch({ top_n: 20 })])
      .then(([summaryData, batchData]) => {
        if (!mounted) return
        setSummary(summaryData)
        setCustomers(batchData || [])
      })
      .catch((requestError) => {
        if (!mounted) return
        setPipelineEmpty(Boolean(requestError.isPipelineNotReady))
        setError(requestError.isPipelineNotReady ? '' : requestError.message)
      })
      .finally(() => { if (mounted) setLoading(false) })
    return () => { mounted = false }
  }, [])

  const filteredCustomers = useMemo(() => {
    const query = search.trim().toLowerCase()
    if (!query) return customers
    return customers.filter((customer) => `${customer.customer_name} ${customer.customer_id}`.toLowerCase().includes(query))
  }, [customers, search])

  const selectCustomer = (customerId) => {
    setSelectedId(customerId)
    setReportLoading(true)
    setReport(null)
    setReportError('')
    getStrReport(customerId)
      .then((data) => setReport(data))
      .catch((requestError) => setReportError(requestError.message))
      .finally(() => setReportLoading(false))
  }

  if (loading) return <div className="page reports-page"><div className="page-loading"><LoadingSpinner size="lg" /><span>Loading STR portfolio…</span></div></div>

  return <div className="page reports-page">
    <div className="page-header"><div><span className="section-kicker">Compliance workspace / generated dossiers</span><h2 className="page-heading">STR reports</h2><p className="page-subtitle">Generate, review, and export suspicious transaction reports for the highest-priority customers.</p></div><span className="report-generated-note">Last indexed {formatDateTime(summary?.generated_at)}</span></div>
    {error && <div className="error-banner" role="alert"><span aria-hidden="true">!</span><div><strong>Report portfolio unavailable</strong><br />{error}</div></div>}
    {pipelineEmpty && <PipelineEmpty what="transactions to report on" />}
    <section className="reports-summary-grid"><div className="report-kpi"><span>Total alerts</span><strong>{Number(summary?.total_alerts || 0).toLocaleString('en-IN')}</strong><small>Threshold ≥ {summary?.min_risk_threshold ?? 70}</small></div><div className="report-kpi"><span>Unique alerted customers</span><strong>{Number(summary?.unique_alerted_customers || 0).toLocaleString('en-IN')}</strong><small>Customers in current portfolio</small></div><div className="report-kpi report-kpi-wide"><span>Total suspicious amount</span><strong>{formatCurrency(summary?.total_suspicious_amount_inr)}</strong><small>Aggregated flagged amount</small></div><div className="report-kpi"><span>Min risk threshold</span><strong>{Number(summary?.min_risk_threshold ?? 70).toFixed(0)}</strong><small>Score gate</small></div></section>
    <section className="reports-master-detail">
      <aside className="panel report-customer-panel"><div className="panel-header"><div><span className="section-kicker">Customer queue</span><h3 className="section-heading">Priority subjects</h3></div><span className="panel-meta">{filteredCustomers.length} shown</span></div><div className="report-search"><input className="form-control" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search name or ID…" aria-label="Search report customers" /></div><div className="report-customer-list">{filteredCustomers.length ? filteredCustomers.map((customer) => <button key={customer.customer_id} className={`report-customer-card${selectedId === customer.customer_id ? ' active' : ''}`} onClick={() => selectCustomer(customer.customer_id)}><span className="report-customer-main"><strong>{customer.customer_name}</strong><small>{customer.customer_id}</small></span><span className="report-customer-meta"><strong>{customer.alert_count} alerts</strong><RiskBadge band={Number(customer.max_risk) >= 90 ? 'CRITICAL' : Number(customer.max_risk) >= 70 ? 'HIGH' : 'MEDIUM'} score={customer.max_risk} showScore /></span></button>) : <div className="empty-state report-list-empty"><span className="empty-state-icon">⌕</span><h3 className="empty-state-title">No customers found</h3><p>Search by customer name or ID.</p></div>}</div></aside>
      <article className="panel report-viewer-panel">{reportLoading ? <div className="report-loading"><LoadingSpinner size="lg" /><span>Generating report for {selectedId}…</span></div> : report ? <ReportViewer report={report} /> : reportError ? <div className="empty-state"><span className="empty-state-icon">!</span><h3 className="empty-state-title">Unable to generate report</h3><p>{reportError}</p></div> : <div className="empty-state report-placeholder"><span className="report-document-icon" aria-hidden="true">▤</span><h3 className="empty-state-title">Select a customer to view their STR report</h3><p>Choose a subject from the priority queue to generate a full suspicious transaction dossier.</p></div>}</article>
    </section>
  </div>
}

function ReportViewer({ report }) {
  return <div className="report-viewer"><header className="report-viewer-header"><div><span className="section-kicker">Forensic compliance document</span><h3>Suspicious Transaction Report</h3><p>Report ID <span>{report.report_id}</span></p></div><button className="button button-primary button-small" onClick={() => downloadJson(`${report.customer_id}_str_report.json`, report)}>Export as JSON <span aria-hidden="true">↓</span></button></header><div className="report-viewer-scroll"><div className="report-document-meta"><div><span>Generated</span><strong>{formatDateTime(report.generated_at)}</strong></div><div><span>Reporting officer</span><strong>{report.reporting_officer}</strong></div><div><span>Customer</span><strong>{report.customer_name}</strong></div><div><span>Customer ID</span><strong className="mono">{report.customer_id}</strong></div></div><section className="report-document-section"><span className="section-kicker">Summary</span><div className="report-document-summary"><div><span>Total suspicious transactions</span><strong>{report.total_suspicious_transactions}</strong></div><div><span>Total amount</span><strong>{formatCurrency(report.total_suspicious_amount)}</strong></div><div><span>Date range</span><strong>{formatDate(report.date_range_from)} — {formatDate(report.date_range_to)}</strong></div><div><span>Primary risk band</span><RiskBadge band={report.primary_risk_band} /></div></div></section><section className="report-document-section"><span className="section-kicker">Detected fraud scenarios</span><div className="scenario-tags">{report.scenario_types_detected?.length ? report.scenario_types_detected.map((scenario) => <span key={scenario} className="scenario-tag">{scenario.replace(/_/g, ' ')}</span>) : <span className="text-muted">No linked scenario types.</span>}</div></section><section className="report-document-section"><span className="section-kicker">Investigator narrative</span><blockquote className="investigator-quote">{report.narrative}</blockquote></section><section className="report-document-section"><div className="report-section-title"><span className="section-kicker">Flagged transactions</span><span className="panel-meta">{report.transactions?.length || 0} records</span></div><div className="table-scroll"><table className="data-table report-document-table"><thead><tr><th>Date</th><th>Mode</th><th>Amount</th><th>Risk Score</th><th>Reasons</th><th>Rules</th></tr></thead><tbody>{(report.transactions || []).map((transaction) => <tr key={transaction.Transaction_ID}><td>{formatDate(transaction.Date)}</td><td>{transaction.Transaction_Mode || 'BANK'}</td><td className="numeric">{formatCurrency(transaction.Transaction_Amount)}</td><td><RiskBadge band={transaction.risk_band} score={transaction.risk_score} showScore /></td><td><span className="report-reasons-inline" title={(transaction.reasons || []).join(' | ')}>{(transaction.reasons || []).join(' | ') || '—'}</span></td><td><RulesBadge rules={transaction.rules_fired} limit={2} /></td></tr>)}</tbody></table></div></section>{report.graph_suspicion_score !== null && report.graph_suspicion_score !== undefined && <section className="report-document-section"><span className="section-kicker">Graph context</span><div className="graph-context-grid"><div><span>Suspicion score</span><strong>{Number(report.graph_suspicion_score).toFixed(4)}</strong></div><div><span>In / out ratio</span><strong>{formatRatio(report.graph_in_out_ratio)}</strong></div><div><span>Mule flag</span><strong className={report.graph_mule_flag ? 'danger-text' : 'success-text'}>{report.graph_mule_flag ? 'FLAGGED' : 'CLEAR'}</strong></div></div></section>}</div></div>
}
