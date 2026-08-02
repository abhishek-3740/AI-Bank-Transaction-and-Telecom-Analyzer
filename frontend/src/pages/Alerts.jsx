import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getAlerts } from '../api/client'
import AlertsTable from '../components/AlertsTable'
import { PAGE_SIZES, RISK_BANDS, SPLITS, formatCurrency, formatDate, splitReasons } from '../utils/formatters'
import RiskBadge from '../components/RiskBadge'
import RulesBadge from '../components/RulesBadge'
import './Alerts.css'

export default function Alerts() {
  const navigate = useNavigate()
  const [filters, setFilters] = useState({ minRisk: 70, band: '', split: '', rule: '', pageSize: 50 })
  const [appliedFilters, setAppliedFilters] = useState(filters)
  const [page, setPage] = useState(1)
  const [payload, setPayload] = useState({ total: 0, page: 1, page_size: 50, results: [] })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [sortDirection, setSortDirection] = useState('desc')

  const loadAlerts = useCallback(() => {
    let mounted = true
    setLoading(true)
    setError('')
    getAlerts({
      min_risk: Number(appliedFilters.minRisk),
      band: appliedFilters.band || undefined,
      split: appliedFilters.split || undefined,
      rule: appliedFilters.rule || undefined,
      page,
      page_size: Number(appliedFilters.pageSize),
    })
      .then((data) => {
        if (!mounted) return
        const results = appliedFilters.rule
          ? (data.results || []).filter((row) => String(row.rules_fired || '').toLowerCase().includes(appliedFilters.rule.toLowerCase()))
          : data.results || []
        setPayload({ ...data, results, total: appliedFilters.rule ? results.length : Number(data.total || 0) })
      })
      .catch((requestError) => {
        if (mounted) setError(requestError.message)
      })
      .finally(() => {
        if (mounted) setLoading(false)
      })
    return () => { mounted = false }
  }, [appliedFilters, page])

  useEffect(() => loadAlerts(), [loadAlerts])

  const sortedResults = useMemo(() => {
    const results = [...(payload.results || [])]
    return results.sort((left, right) => {
      const direction = sortDirection === 'asc' ? 1 : -1
      return (Number(left.risk_score || 0) - Number(right.risk_score || 0)) * direction
    })
  }, [payload.results, sortDirection])

  const totalPages = Math.max(1, Math.ceil(Number(payload.total || 0) / Number(appliedFilters.pageSize)))
  const startResult = payload.total ? (page - 1) * Number(appliedFilters.pageSize) + 1 : 0
  const endResult = payload.total ? Math.min(page * Number(appliedFilters.pageSize), payload.total) : 0

  const updateFilter = (key, value) => setFilters((current) => ({ ...current, [key]: value }))
  const applyFilters = (event) => {
    event.preventDefault()
    const next = { ...filters, minRisk: Math.min(100, Math.max(0, Number(filters.minRisk) || 0)) }
    setFilters(next)
    setAppliedFilters(next)
    setPage(1)
  }
  const resetFilters = () => {
    const next = { minRisk: 70, band: '', split: '', rule: '', pageSize: 50 }
    setFilters(next)
    setAppliedFilters(next)
    setPage(1)
  }

  const columns = [
    { key: 'row-number', label: '#', render: (_, index) => (page - 1) * Number(appliedFilters.pageSize) + index + 1 },
    { key: 'Transaction_ID', label: 'Transaction ID', className: 'mono primary-cell' },
    { key: 'Date', label: 'Date', render: (row) => formatDate(row.Date) },
    { key: 'Sender_Customer_Name', label: 'Customer Name', className: 'primary-cell' },
    { key: 'Transaction_Amount', label: 'Amount (₹)', align: 'right', render: (row) => formatCurrency(row.Transaction_Amount) },
    { key: 'risk_score', label: <button className="table-sort-button" onClick={(event) => { event.stopPropagation(); setSortDirection((current) => current === 'desc' ? 'asc' : 'desc') }}>Risk Score <span aria-hidden="true">{sortDirection === 'desc' ? '↓' : '↑'}</span></button>, align: 'right', render: (row) => <RiskBadge band={row.risk_band} score={row.risk_score} showScore /> },
    { key: 'risk_band', label: 'Band', render: (row) => <RiskBadge band={row.risk_band} /> },
    { key: 'reason_1', label: 'Reason 1', render: (row) => <span className="table-reason" title={row.reason_1}>{splitReasons(row)[0] || '—'}</span> },
    { key: 'rules_fired', label: 'Rules', render: (row) => <RulesBadge rules={row.rules_fired} limit={2} /> },
    { key: 'is_suspicious_gt', label: 'GT Label', align: 'center', render: (row) => <span className={Number(row.is_suspicious_gt) ? 'gt-positive' : 'gt-negative'}>{Number(row.is_suspicious_gt) ? '✓' : '✗'}</span> },
  ]

  return (
    <div className="page alerts-page">
      <div className="page-header">
        <div>
          <span className="section-kicker">Forensic queue / server filtered</span>
          <h2 className="page-heading">Alert queue</h2>
          <p className="page-subtitle">Review scored transactions, compare model signals, and open a customer investigation.</p>
        </div>
      </div>

      {error && <div className="error-banner" role="alert"><span aria-hidden="true">!</span><div><strong>Unable to load alerts</strong><br />{error}</div></div>}

      <form className="panel alerts-filter-panel" onSubmit={applyFilters}>
        <div className="alerts-filter-grid">
          <label className="filter-field filter-range-field"><span className="form-label">Min risk <strong>{filters.minRisk}</strong></span><input type="range" min="0" max="100" value={filters.minRisk} onChange={(event) => updateFilter('minRisk', event.target.value)} /></label>
          <label className="filter-field"><span className="form-label">Risk band</span><select className="form-control" value={filters.band} onChange={(event) => updateFilter('band', event.target.value)}><option value="">All bands</option>{RISK_BANDS.map((band) => <option key={band} value={band}>{band}</option>)}</select></label>
          <label className="filter-field"><span className="form-label">Split</span><select className="form-control" value={filters.split} onChange={(event) => updateFilter('split', event.target.value)}><option value="">All splits</option>{SPLITS.map((split) => <option key={split} value={split}>{split}</option>)}</select></label>
          <label className="filter-field"><span className="form-label">Rule</span><input className="form-control" value={filters.rule} onChange={(event) => updateFilter('rule', event.target.value)} placeholder="Filter by rule…" /></label>
          <div className="alerts-filter-actions"><button className="button button-primary" type="submit">Apply filters</button><button className="button button-ghost" type="button" onClick={resetFilters}>Reset</button></div>
        </div>
      </form>

      <section className="panel alerts-results-panel">
        <div className="alerts-results-toolbar"><div><span className="section-kicker">Investigation records</span><p className="results-summary">Showing <strong>{startResult}–{endResult}</strong> of <strong>{Number(payload.total || 0).toLocaleString('en-IN')}</strong> results</p></div><label className="page-size-field"><span>Rows</span><select className="form-control" value={filters.pageSize} onChange={(event) => { const value = Number(event.target.value); updateFilter('pageSize', value); setAppliedFilters((current) => ({ ...current, pageSize: value })); setPage(1) }}>{PAGE_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}</select></label></div>
        <AlertsTable data={sortedResults} loading={loading} columns={columns} onRowClick={(row) => navigate(`/customer/${row.Sender_Customer_ID}`)} emptyMessage="No alerts match the current filters" />
        <div className="pagination-bar"><button className="button button-ghost pagination-button" disabled={page <= 1 || loading} onClick={() => setPage((current) => Math.max(1, current - 1))}>← Prev</button><span>Page <strong>{page}</strong> of <strong>{totalPages}</strong></span><button className="button button-ghost pagination-button" disabled={page >= totalPages || loading} onClick={() => setPage((current) => Math.min(totalPages, current + 1))}>Next →</button></div>
      </section>
    </div>
  )
}
