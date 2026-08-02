import { useCallback, useEffect, useMemo, useState } from 'react'
import { ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, CartesianGrid } from 'recharts'
import { getGraphNode, getGraphNodes, getGraphSummary } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import PipelineEmpty from '../components/PipelineEmpty'
import RiskBadge from '../components/RiskBadge'
import StatCard from '../components/StatCard'
import { formatCurrency, formatRatio, formatRisk } from '../utils/formatters'
import './GraphAnalytics.css'

function GraphTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const node = payload[0]?.payload
  if (!node) return null
  return <div className="graph-tooltip"><strong>{node.node_id}</strong><span>In-degree: {node.in_degree}</span><span>Out-degree: {node.out_degree}</span><span>Suspicion: {Number(node.suspicion_score || 0).toFixed(4)}</span><span>{node.is_mule_account ? 'Mule account' : 'Regular node'}</span></div>
}

function EdgeTable({ title, edges = [] }) {
  return <div className="edge-table-section"><div className="edge-table-title"><span>{title}</span><strong>{edges.length}</strong></div>{edges.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Transaction ID</th><th>{title === 'Incoming transactions' ? 'Source' : 'Destination'}</th><th>Amount</th><th>Risk</th></tr></thead><tbody>{edges.slice(0, 20).map((edge) => <tr key={`${edge.Transaction_ID}-${edge.src}-${edge.dst}`}><td className="mono primary-cell">{edge.Transaction_ID}</td><td className="mono">{title === 'Incoming transactions' ? edge.src : edge.dst}</td><td className="numeric">{formatCurrency(edge.Transaction_Amount)}</td><td><RiskBadge band={edge.risk_band} score={edge.risk_score} showScore /></td></tr>)}</tbody></table></div> : <div className="edge-empty">No linked transactions.</div>}</div>
}

function NodeDrawer({ detail, loading, onClose }) {
  return <div className="graph-drawer-shell" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}><aside className="graph-drawer" role="dialog" aria-modal="true" aria-labelledby="node-drawer-title"><header className="drawer-header"><div><span className="section-kicker">Selected graph node</span><h2 id="node-drawer-title">{detail?.node?.node_id || 'Loading node'}</h2></div><button className="icon-button" onClick={onClose} aria-label="Close node detail">×</button></header>{loading ? <div className="drawer-loading"><LoadingSpinner /><span>Loading node evidence…</span></div> : detail?.node ? <div className="drawer-body"><div className="drawer-summary-grid"><div><span>Suspicion</span><strong>{Number(detail.node.suspicion_score || 0).toFixed(4)}</strong></div><div><span>Alerts</span><strong>{detail.node.alert_count}</strong></div><div><span>In / out</span><strong>{formatRatio(detail.node.in_out_ratio)}</strong></div><div><span>Mule flag</span><strong className={detail.node.is_mule_account ? 'danger-text' : 'success-text'}>{detail.node.is_mule_account ? 'YES' : 'NO'}</strong></div></div><div className="drawer-stats-line"><span>Received <strong>{formatCurrency(detail.node.total_received)}</strong></span><span>Sent <strong>{formatCurrency(detail.node.total_sent)}</strong></span><span>PageRank <strong>{Number(detail.node.pagerank || 0).toFixed(5)}</strong></span></div><EdgeTable title="Incoming transactions" edges={detail.incoming_edges} /><EdgeTable title="Outgoing transactions" edges={detail.outgoing_edges} /></div> : <div className="empty-state"><span className="empty-state-icon">⌕</span><h3 className="empty-state-title">Node unavailable</h3></div>}</aside></div>
}

export default function GraphAnalytics() {
  const [summary, setSummary] = useState(null)
  const [nodes, setNodes] = useState([])
  const [loading, setLoading] = useState(true)
  const [nodesLoading, setNodesLoading] = useState(false)
  const [error, setError] = useState('')
  const [pipelineEmpty, setPipelineEmpty] = useState(false)
  const [mulesOnly, setMulesOnly] = useState(false)
  const [selectedNode, setSelectedNode] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)

  useEffect(() => {
    let mounted = true
    setLoading(true)
    setError('')
    getGraphSummary(20)
      .then((data) => {
        if (!mounted) return
        setSummary(data)
        setNodes(data?.top_suspicious_nodes || [])
      })
      .catch((requestError) => {
        if (!mounted) return
        setPipelineEmpty(Boolean(requestError.isPipelineNotReady))
        setError(requestError.isPipelineNotReady ? '' : requestError.message)
      })
      .finally(() => { if (mounted) setLoading(false) })
    return () => { mounted = false }
  }, [])

  const reloadNodes = useCallback((onlyMules) => {
    if (!summary) return
    if (!onlyMules) {
      setNodes(summary.top_suspicious_nodes || [])
      return
    }
    let mounted = true
    setNodesLoading(true)
    getGraphNodes({ sort_by: 'suspicion_score', mule_only: true, page: 1, page_size: 100 })
      .then((data) => { if (mounted) setNodes(data || []) })
      .catch((requestError) => { if (mounted) setError(requestError.isPipelineNotReady ? '' : requestError.message) })
      .finally(() => { if (mounted) setNodesLoading(false) })
    return () => { mounted = false }
  }, [summary])

  const scatterData = useMemo(() => nodes.map((node) => ({ ...node, ratio_plot: Math.min(Number(node.in_out_ratio || 0), 1000) })), [nodes])
  const regularNodes = scatterData.filter((node) => !Number(node.is_mule_account))
  const muleNodes = scatterData.filter((node) => Number(node.is_mule_account))

  const openNode = (node) => {
    setSelectedNode({ node })
    setDetailLoading(true)
    getGraphNode(node.node_id)
      .then((data) => setSelectedNode(data))
      .catch((requestError) => setError(requestError.message))
      .finally(() => setDetailLoading(false))
  }

  const toggleMules = (event) => {
    const next = event.target.checked
    setMulesOnly(next)
    reloadNodes(next)
  }

  if (loading) return <div className="page graph-page"><div className="page-loading"><LoadingSpinner size="lg" /><span>Loading graph intelligence…</span></div></div>

  return <div className="page graph-page">
    <div className="page-header"><div><span className="section-kicker">Network intelligence / suspicious entities</span><h2 className="page-heading">Graph analytics</h2><p className="page-subtitle">Trace high-risk transfer nodes, mule-account signals, and the transaction edges behind them.</p></div><label className="mule-toggle"><input type="checkbox" checked={mulesOnly} onChange={toggleMules} /><span className="toggle-track" /><span>Show mules only</span></label></div>
    {error && <div className="error-banner" role="alert"><span aria-hidden="true">!</span><div><strong>Graph data issue</strong><br />{error}</div></div>}
    {pipelineEmpty && <PipelineEmpty what="graph" />}
    <section className="graph-stat-grid"><StatCard title="Total Nodes" value={Number(summary?.total_nodes || 0).toLocaleString('en-IN')} subtitle="Entities in transfer graph" icon="◈" /><StatCard title="Total Edges" value={Number(summary?.total_edges || 0).toLocaleString('en-IN')} subtitle="Directed transactions" color="var(--accent-blue)" icon="↗" /><StatCard title="Mule Accounts Detected" value={Number(summary?.known_mule_nodes || 0).toLocaleString('en-IN')} subtitle="Known or suspected" color="var(--accent-red)" icon="!" /></section>
    <section className="panel graph-chart-panel"><div className="panel-header"><div><span className="section-kicker">Top suspicious nodes</span><h3 className="section-heading">Node Risk Scatter — Size = Alert Count</h3></div><div className="graph-legend"><span><i className="legend-dot legend-mule" /> Mule Account</span><span><i className="legend-dot legend-regular" /> Regular Node</span></div></div><div className="graph-chart-wrap">{nodesLoading ? <div className="chart-loading"><LoadingSpinner /><span>Filtering node set…</span></div> : <ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 18, right: 24, bottom: 22, left: 4 }}><CartesianGrid stroke="var(--border)" strokeDasharray="3 3" /><XAxis type="number" dataKey="ratio_plot" name="In/out ratio" domain={[0, 1000]} tick={{ fill: 'var(--text-muted)', fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={(value) => value === 1000 ? '1k+' : value} label={{ value: 'In / out ratio (capped)', position: 'insideBottom', fill: 'var(--text-muted)', fontSize: 10, offset: -12 }} /><YAxis type="number" dataKey="suspicion_score" name="Suspicion score" domain={[0, 1]} tick={{ fill: 'var(--text-muted)', fontSize: 10 }} tickLine={false} axisLine={false} /><ZAxis type="number" dataKey="alert_count" range={[40, 400]} name="Alert count" /><Tooltip cursor={{ strokeDasharray: '3 3', stroke: 'var(--accent-blue)' }} content={<GraphTooltip />} /><Scatter name="Regular Nodes" data={regularNodes} fill="var(--accent-blue)" shape={(props) => <circle cx={props.cx} cy={props.cy} r={Math.max(4, Math.min(20, 4 + Math.sqrt(Number(props.payload?.alert_count || 0)) * 1.8))} fill="var(--accent-blue)" fillOpacity="0.7" stroke="var(--accent-blue)" strokeWidth="1" />} /><Scatter name="Mule Accounts" data={muleNodes} fill="var(--accent-red)" shape={(props) => <circle cx={props.cx} cy={props.cy} r={Math.max(5, Math.min(20, 5 + Math.sqrt(Number(props.payload?.alert_count || 0)) * 1.8))} fill="var(--accent-red)" fillOpacity="0.8" stroke="#ffb4af" strokeWidth="1" />} /></ScatterChart></ResponsiveContainer>}</div></section>
    <section className="panel graph-table-panel"><div className="panel-header"><div><span className="section-kicker">Prioritized entities</span><h3 className="section-heading">Top Suspicious Nodes</h3></div><span className="panel-meta">{nodes.length} nodes shown</span></div>{nodes.length ? <div className="table-scroll"><table className="data-table graph-node-table"><thead><tr><th>Rank</th><th>Node ID</th><th>In-degree</th><th>Out-degree</th><th>In/Out Ratio</th><th>Suspicion Score</th><th>Alerts</th><th>PageRank</th><th>Mule?</th></tr></thead><tbody>{nodes.map((node, index) => <tr key={node.node_id} className="clickable-row" onClick={() => openNode(node)} onKeyDown={(event) => { if (event.key === 'Enter') openNode(node) }} tabIndex="0"><td className="mono">{index + 1}</td><td className="mono primary-cell">{node.node_id}</td><td className="numeric">{node.in_degree}</td><td className="numeric">{node.out_degree}</td><td className="numeric">{formatRatio(node.in_out_ratio)}</td><td className="numeric graph-suspicion">{Number(node.suspicion_score || 0).toFixed(4)}</td><td className="numeric">{node.alert_count}</td><td className="numeric">{Number(node.pagerank || 0).toFixed(5)}</td><td>{Number(node.is_mule_account) ? <span className="mule-badge">⚠ MULE</span> : <span className="text-muted">—</span>}</td></tr>)}</tbody></table></div> : <div className="empty-state"><span className="empty-state-icon">◈</span><h3 className="empty-state-title">No graph nodes returned</h3></div>}</section>
    {selectedNode && <NodeDrawer detail={selectedNode} loading={detailLoading} onClose={() => setSelectedNode(null)} />}
  </div>
}
