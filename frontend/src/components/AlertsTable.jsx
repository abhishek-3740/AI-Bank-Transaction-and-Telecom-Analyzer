import LoadingSpinner from './LoadingSpinner'
import RiskBadge from './RiskBadge'
import RulesBadge from './RulesBadge'
import { formatCurrency, formatDate, formatRisk, splitReasons } from '../utils/formatters'
import './AlertsTable.css'

const defaultColumns = [
  { key: 'Transaction_ID', label: 'Transaction ID', className: 'mono primary-cell' },
  { key: 'Date', label: 'Date', render: (row) => formatDate(row.Date) },
  { key: 'Sender_Customer_Name', label: 'Customer', className: 'primary-cell' },
  { key: 'Transaction_Amount', label: 'Amount', align: 'right', render: (row) => formatCurrency(row.Transaction_Amount) },
  { key: 'risk_score', label: 'Risk Score', align: 'right', render: (row) => <RiskBadge band={row.risk_band} score={row.risk_score} showScore /> },
  { key: 'reason_1', label: 'Reason 1', render: (row) => <span className="table-reason" title={row.reason_1}>{splitReasons(row)[0] || '—'}</span> },
  { key: 'rules_fired', label: 'Rules', render: (row) => <RulesBadge rules={row.rules_fired} limit={2} /> },
  { key: 'is_suspicious_gt', label: 'GT', align: 'center', render: (row) => <span className={Number(row.is_suspicious_gt) ? 'gt-positive' : 'gt-negative'}>{Number(row.is_suspicious_gt) ? '✓' : '✗'}</span> },
]

export default function AlertsTable({ data = [], loading = false, onRowClick, columns = defaultColumns, emptyMessage = 'No alerts match the current filters', compact = false }) {
  return (
    <div className={`alerts-table ${compact ? 'alerts-table-compact' : ''}`}>
      {loading ? (
        <div className="table-loading">
          <LoadingSpinner size="md" label="Loading table" />
          <span>Loading investigation records…</span>
        </div>
      ) : data.length === 0 ? (
        <div className="empty-state alerts-table-empty">
          <span className="empty-state-icon" aria-hidden="true">⌕</span>
          <h3 className="empty-state-title">{emptyMessage}</h3>
          <p>Try widening the filters or check the API connection.</p>
        </div>
      ) : (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                {columns.map((column) => (
                  <th key={column.key} className={column.align === 'right' ? 'align-right' : column.align === 'center' ? 'align-center' : ''}>
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, index) => (
                <tr
                  key={row.Transaction_ID || row.node_id || index}
                  className={onRowClick ? 'clickable-row' : ''}
                  onClick={() => onRowClick?.(row)}
                  onKeyDown={(event) => {
                    if (onRowClick && (event.key === 'Enter' || event.key === ' ')) {
                      event.preventDefault()
                      onRowClick(row)
                    }
                  }}
                  tabIndex={onRowClick ? 0 : undefined}
                >
                  {columns.map((column) => (
                    <td key={column.key} className={`${column.className || ''} ${column.align === 'right' ? 'numeric' : column.align === 'center' ? 'align-center' : ''}`}>
                      {column.render ? column.render(row, index) : row[column.key] ?? '—'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
