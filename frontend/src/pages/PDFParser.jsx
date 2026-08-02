import { useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { parsePdf } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import RiskBadge from '../components/RiskBadge'
import { downloadCsv, formatCurrency, formatFileSize, getBandFromRisk } from '../utils/formatters'
import './PDFParser.css'

function findColumn(columns, aliases) {
  return columns.find((column) => aliases.some((alias) => column.toLowerCase().replace(/[^a-z0-9]/g, '').includes(alias)))
}

function displayParsedValue(column, value, row) {
  if (value === null || value === undefined || value === '') return <span className="text-muted">—</span>
  if (column === 'Transaction_Amount' || column.toLowerCase().includes('amount')) return formatCurrency(value)
  if (column === 'risk_score') return <RiskBadge band={row.risk_band || getBandFromRisk(value)} score={value} showScore />
  if (column === 'risk_band') return <RiskBadge band={value} />
  if (column === 'Transaction_Mode' || column.toLowerCase().includes('mode')) return <span className="mode-badge">{String(value)}</span>
  return String(value)
}

/**
 * What happened to the parsed rows after parsing.
 *
 * Parsing alone never changed the dashboard — the rows have to be scored and
 * indexed. Reporting that explicitly (including what had to be inferred from an
 * incomplete statement) is the difference between a working upload and a silent
 * dead end.
 */
function IngestSummary({ ingest }) {
  if (!ingest) return null

  if (!ingest.dashboard_updated) {
    return <div className="ingest-summary ingest-summary-warning" role="status">
      <span className="ingest-icon" aria-hidden="true">!</span>
      <div>
        <strong>Parsed, but not added to the dashboard</strong>
        <span>{ingest.reason || 'The scoring pipeline could not run on this file.'}</span>
      </div>
    </div>
  }

  const caveats = [
    ingest.rows_dropped_incomplete > 0 && `${ingest.rows_dropped_incomplete} non-transaction row(s) skipped (balances, totals)`,
    ingest.timestamps_imputed > 0 && `${ingest.timestamps_imputed} row(s) had no clock time — time-of-day and velocity signals are approximate`,
    ingest.receivers_derived_from_narration > 0 && `${ingest.receivers_derived_from_narration} beneficiary account(s) inferred from the narration text`,
  ].filter(Boolean)

  return <div className="ingest-summary" role="status">
    <span className="ingest-icon" aria-hidden="true">✓</span>
    <div>
      <strong>Scored and indexed — {ingest.scored_transactions} transaction(s), {ingest.alerts} alert(s) at risk ≥ 70</strong>
      <span>Graph rebuilt with {ingest.graph_nodes} node(s) and {ingest.graph_edges} edge(s).{ingest.cdr_context ? ' CDR context applied.' : ' No CDR context uploaded — telecom signals are unavailable.'}</span>
      {caveats.length > 0 && <ul className="ingest-caveats">{caveats.map((caveat) => <li key={caveat}>{caveat}</li>)}</ul>}
      <div className="ingest-actions"><Link className="button button-primary button-small" to="/">Open dashboard <span aria-hidden="true">→</span></Link><Link className="button button-ghost button-small" to="/alerts">View alert queue</Link></div>
    </div>
  </div>
}

export default function PDFParser() {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [parsing, setParsing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const selectFile = (nextFile) => {
    if (!nextFile) return
    setFile(nextFile)
    setResult(null)
    setError('')
  }
  const handleDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    selectFile(event.dataTransfer.files?.[0])
  }
  const handleParse = () => {
    if (!file) return
    setParsing(true)
    setError('')
    const formData = new FormData()
    formData.append('file', file)
    parsePdf(formData)
      .then((data) => setResult(data))
      .catch((requestError) => setError(requestError.message))
      .finally(() => setParsing(false))
  }
  const downloadParsedCsv = () => {
    if (!result?.columns?.length) return
    const baseName = file?.name?.replace(/\.pdf$/i, '') || 'tri-netra'
    downloadCsv(`${baseName}_parsed.csv`, result.columns, result.data || [])
  }

  const columns = result?.columns || []
  const rows = result?.data || []
  const firstRow = rows[0] || {}
  const metadata = [
    ['Bank Name', findColumn(columns, ['senderbankname', 'bankname', 'bank'])],
    ['Account Number', findColumn(columns, ['senderaccountnumber', 'accountnumber'])],
    ['IFSC', findColumn(columns, ['senderifsc', 'ifsc'])],
    ['Customer ID', findColumn(columns, ['sendercustomerid', 'customerid'])],
  ].map(([label, column]) => ({ label, column, value: column ? firstRow[column] : null })).filter((item) => item.value !== null && item.value !== undefined && item.value !== '')

  return <div className="page pdf-page">
    <div className="page-header"><div><span className="section-kicker">Data ingestion / source evidence</span><h2 className="page-heading">PDF parser</h2><p className="page-subtitle">Ingest bank statements, call detail records, and IPDR extracts into the canonical investigation schema.</p></div></div>
    {error && <div className="error-banner" role="alert"><span aria-hidden="true">!</span><div><strong>PDF parsing failed</strong><br />{error}</div></div>}
    <section className="pdf-workspace">
      <article className={`panel pdf-upload-panel${error ? ' has-error' : ''}`}>
        <div className="panel-header"><div><span className="section-kicker">Source document</span><h3 className="section-heading">Upload evidence</h3></div><span className="file-type-label">PDF only</span></div>
        <div className={`upload-zone${dragging ? ' dragging' : ''}`} onDragEnter={(event) => { event.preventDefault(); setDragging(true) }} onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragging(false)} onDrop={handleDrop} onClick={() => inputRef.current?.click()} role="button" tabIndex="0" onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click() }}>
          <input ref={inputRef} type="file" accept="application/pdf,.pdf" hidden onChange={(event) => selectFile(event.target.files?.[0])} />
          <span className="upload-icon" aria-hidden="true">⇧</span>
          <strong>Drag & drop a Bank Statement, CDR, or IPDR PDF</strong>
          <span>Supported formats: Bank (Axis, HDFC, SBI, ICICI), CDR, IPDR</span>
          <button type="button" className="button button-ghost browse-button" onClick={(event) => { event.stopPropagation(); inputRef.current?.click() }}>Browse files</button>
        </div>
        {file && <div className="selected-file"><span className="selected-file-icon" aria-hidden="true">▤</span><div><strong>{file.name}</strong><span>{formatFileSize(file.size)}</span></div><button className="icon-button" aria-label="Remove selected file" onClick={() => { setFile(null); setResult(null); setError('') }}>×</button></div>}
        <button className="button button-primary parse-button" disabled={!file || parsing} onClick={handleParse}>{parsing ? <><LoadingSpinner size="sm" /> Parsing PDF…</> : <>Parse PDF <span aria-hidden="true">→</span></>}</button>
        <div className="extraction-info"><span className="section-kicker">What gets extracted</span><div className="extraction-list"><div><strong>Bank</strong><span>Account number, IFSC, bank name, transactions, mode, amount, and date.</span></div><div><strong>CDR</strong><span>Call records with party numbers, duration, IMEI, and cell ID.</span></div><div><strong>IPDR</strong><span>Session records with IP addresses, ports, IMSI, and device context.</span></div></div></div>
      </article>
      <article className="panel pdf-results-panel"><div className="panel-header"><div><span className="section-kicker">Canonical output</span><h3 className="section-heading">Parsed results</h3></div>{result && <button className="button button-ghost button-small" onClick={downloadParsedCsv}>Download as CSV <span aria-hidden="true">↓</span></button>}</div>{!result ? <div className="empty-state pdf-results-empty"><span className="report-document-icon" aria-hidden="true">▤</span><h3 className="empty-state-title">Upload a PDF to see parsed results</h3><p>Parsed columns, metadata, and records will appear here.</p></div> : <div className="pdf-results-content"><div className="parse-success"><span className="success-check">✓</span><div><strong>Successfully parsed {result.rows} rows</strong><span>Dataset type: {result.dataset_type || 'auto'}</span></div></div><IngestSummary ingest={result.ingest} />{metadata.length > 0 && <div className="pdf-metadata-grid">{metadata.map((item) => <div key={item.label}><span>{item.label}</span><strong>{String(item.value)}</strong></div>)}</div>}<div className="parsed-table-heading"><span className="section-kicker">Extracted rows</span><span className="panel-meta">Showing {Math.min(rows.length, 50)} of {rows.length}</span></div><div className="table-scroll parsed-data-table"><table className="data-table"><thead><tr>{columns.map((column) => <th key={column}>{column.replace(/_/g, ' ')}</th>)}</tr></thead><tbody>{rows.slice(0, 50).map((row, index) => <tr key={`row-${index}`}>{columns.map((column) => <td key={column} className={column === 'Transaction_Amount' || column === 'risk_score' ? 'numeric' : ''}>{displayParsedValue(column, row[column], row)}</td>)}</tr>)}</tbody></table></div>{rows.length > 50 && <p className="table-footnote">Only the first 50 rows are displayed. Download the CSV for the complete dataset.</p>}</div>}</article>
    </section>
  </div>
}
