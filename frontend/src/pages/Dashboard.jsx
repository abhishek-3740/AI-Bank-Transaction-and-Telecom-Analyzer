import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Cell, Legend, Pie, PieChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getAlerts, getScoringStats } from '../api/client'
import AlertsTable from '../components/AlertsTable'
import LoadingSpinner from '../components/LoadingSpinner'
import StatCard from '../components/StatCard'
import { formatCurrency, formatPercentage } from '../utils/formatters'
import './Dashboard.css'

const bandColors = {
  CRITICAL: '#f85149',
  HIGH: '#f0883e',
  MEDIUM: '#d29922',
  LOW: '#3fb950',
}

const ruleOrder = ['ODD_HOUR', 'HIGH_AMOUNT_ANOMALY', 'RAPID_SUCCESSION', 'NEW_BENEFICIARY_FLAG', 'TELECOM_BURST']

function numberFormat(value) {
  return Number(value || 0).toLocaleString('en-IN')
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [stats, setStats] = useState(null)
  const [criticalAlerts, setCriticalAlerts] = useState([])
  const [statsLoading, setStatsLoading] = useState(true)
  const [alertsLoading, setAlertsLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let mounted = true
    setStatsLoading(true)
    getScoringStats()
      .then((data) => {
        if (mounted) setStats(data)
      })
      .catch((requestError) => {
        if (mounted) setError(requestError.message)
      })
      .finally(() => {
        if (mounted) setStatsLoading(false)
      })
    return () => { mounted = false }
  }, [])

  useEffect(() => {
    let mounted = true
    setAlertsLoading(true)
    getAlerts({ min_risk: 90, band: 'CRITICAL', page: 1, page_size: 10 })
      .then((data) => {
        if (mounted) setCriticalAlerts(data?.results || [])
      })
      .catch((requestError) => {
        if (mounted) setError((current) => current || requestError.message)
      })
      .finally(() => {
        if (mounted) setAlertsLoading(false)
      })
    return () => { mounted = false }
  }, [])

  const bandData = useMemo(() => (
    Object.entries(stats?.band_distribution || bandColors).map(([name, value]) => ({
      name,
      value: Number(value || 0),
      color: bandColors[name] || '#8b949e',
    }))
  ), [stats])

  const ruleData = useMemo(() => ruleOrder.map((name) => ({
    name: name.replace(/_FLAG$/, '').replace(/_/g, ' '),
    count: Number(stats?.rule_fire_counts?.[name] || 0),
  })), [stats])

  const alertColumns = [
    { key: 'Transaction_ID', label: 'Transaction ID', className: 'mono primary-cell' },
    { key: 'Sender_Customer_Name', label: 'Customer', render: (row) => <span className="dashboard-customer">{row.Sender_Customer_Name}<small>{row.Sender_Customer_ID}</small></span> },
    { key: 'Transaction_Amount', label: 'Amount', align: 'right', render: (row) => formatCurrency(row.Transaction_Amount) },
    { key: 'risk_score', label: 'Risk Score', align: 'right', render: (row) => <span className="dashboard-risk-score">{Number(row.risk_score || 0).toFixed(1)}</span> },
    { key: 'reason_1', label: 'Reason 1', render: (row) => <span className="table-reason" title={row.reason_1}>{row.reason_1 || '—'}</span> },
    { key: 'rules_fired', label: 'Rules', render: (row) => <span className="dashboard-rule-count">{row.rules_fired ? row.rules_fired.split('|').filter(Boolean).length : 0} fired</span> },
    { key: 'is_suspicious_gt', label: 'GT', align: 'center', render: (row) => <span className={Number(row.is_suspicious_gt) ? 'gt-positive' : 'gt-negative'}>{Number(row.is_suspicious_gt) ? '✓' : '✗'}</span> },
  ]

  return (
    <div className="page dashboard-page">
      <div className="page-header dashboard-header">
        <div>
          <span className="section-kicker">Command center / live intelligence</span>
          <h2 className="page-heading">Investigation overview</h2>
          <p className="page-subtitle">A live view of model signals, rule activity, and the highest-priority transaction alerts.</p>
        </div>
        <div className="dashboard-refresh-note"><span className="refresh-mark">↻</span> Live API data</div>
      </div>

      {error && <div className="error-banner" role="alert"><span aria-hidden="true">!</span><div><strong>Data connection issue</strong><br />{error}</div></div>}

      <section className="dashboard-stat-grid" aria-label="Key metrics">
        {statsLoading ? (
          Array.from({ length: 5 }).map((_, index) => <div className="stat-card stat-card-skeleton" key={index}><span className="skeleton skeleton-line" /><span className="skeleton skeleton-value" /></div>)
        ) : (
          <>
            <StatCard title="Total Transactions" value={numberFormat(stats?.total_transactions)} subtitle="Scored observations" icon="↗" />
            <StatCard title="Total Alerts" value={numberFormat(stats?.total_alerts)} subtitle="Risk score ≥ 70" color="var(--accent-red)" icon="!" />
            <StatCard title="Alert Rate" value={`${Number(stats?.alert_rate_pct || 0).toFixed(2)}%`} subtitle="Across all transactions" color="var(--accent-yellow)" icon="%" />
            <StatCard title="Model Precision" value={formatPercentage(stats?.test_precision)} subtitle="Test split" color="var(--accent-green)" icon="✓" />
            <StatCard title="Model Recall" value={formatPercentage(stats?.test_recall)} subtitle="Test split" color="var(--accent-blue)" icon="◎" />
          </>
        )}
      </section>

      <section className="dashboard-chart-grid" aria-label="Risk intelligence charts">
        <article className="panel chart-panel">
          <div className="panel-header"><div><span className="section-kicker">Severity distribution</span><h3 className="section-heading">Alerts by Risk Band</h3></div><span className="panel-meta">{numberFormat(stats?.total_alerts)} alerts</span></div>
          <div className="chart-wrap chart-pie-wrap">
            {statsLoading ? <div className="chart-loading"><LoadingSpinner /><span>Loading distribution…</span></div> : <ResponsiveContainer width="100%" height="100%"><PieChart>
              <Pie data={bandData} dataKey="value" nameKey="name" cx="50%" cy="45%" innerRadius="52%" outerRadius="77%" paddingAngle={3} stroke="var(--bg-card)" strokeWidth={3}>
                {bandData.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
              </Pie>
              <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text-primary)' }} formatter={(value) => numberFormat(value)} />
              <Legend verticalAlign="bottom" iconType="circle" formatter={(value) => <span className="chart-legend-text">{value}</span>} />
            </PieChart></ResponsiveContainer>}
          </div>
        </article>

        <article className="panel chart-panel">
          <div className="panel-header"><div><span className="section-kicker">Independent signals</span><h3 className="section-heading">Rule Engine Fires</h3></div><span className="panel-meta">5 rules</span></div>
          <div className="chart-wrap chart-bar-wrap">
            {statsLoading ? <div className="chart-loading"><LoadingSpinner /><span>Loading rules…</span></div> : <ResponsiveContainer width="100%" height="100%"><BarChart data={ruleData} layout="vertical" margin={{ top: 4, right: 24, left: 4, bottom: 4 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" stroke="var(--text-muted)" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="name" width={148} stroke="var(--text-muted)" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} tickLine={false} axisLine={false} />
              <Tooltip cursor={{ fill: 'rgba(88,166,255,0.06)' }} contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text-primary)' }} formatter={(value) => numberFormat(value)} />
              <Bar dataKey="count" fill="var(--accent-blue)" radius={[0, 4, 4, 0]} barSize={18} />
            </BarChart></ResponsiveContainer>}
          </div>
        </article>
      </section>

      <section className="panel dashboard-alerts-panel">
        <div className="panel-header"><div><span className="section-kicker">Immediate attention</span><h3 className="section-heading">Recent Critical Alerts (Top 10)</h3></div><button className="button button-ghost button-small" onClick={() => navigate('/alerts')}>Open alert queue <span aria-hidden="true">→</span></button></div>
        <AlertsTable data={criticalAlerts} loading={alertsLoading} columns={alertColumns} onRowClick={(row) => navigate(`/customer/${row.Sender_Customer_ID}`)} emptyMessage="No critical alerts returned" compact />
      </section>
    </div>
  )
}
