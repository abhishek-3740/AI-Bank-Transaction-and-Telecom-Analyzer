export const RISK_BANDS = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
export const SPLITS = ['train', 'val', 'test']
export const PAGE_SIZES = [25, 50, 100]

export function formatCurrency(value) {
  const amount = Number(value)
  if (!Number.isFinite(amount)) return '—'
  return `₹${amount.toLocaleString('en-IN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`
}

export function formatDate(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return date.toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  })
}

export function formatDateTime(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value)
  return date.toLocaleString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatRisk(value) {
  const score = Number(value)
  return Number.isFinite(score) ? score.toFixed(1) : '—'
}

export function formatPercentage(value, fractionDigits = 1) {
  const probability = Number(value)
  return Number.isFinite(probability)
    ? `${(probability * 100).toFixed(fractionDigits)}%`
    : '—'
}

export function formatRatio(value) {
  const ratio = Number(value)
  if (!Number.isFinite(ratio)) return '—'
  return ratio > 9999 ? `>${Math.round(ratio).toLocaleString('en-IN')}` : ratio.toFixed(2)
}

export function getBandClass(band = '') {
  return String(band).toLowerCase().replace(/\s+/g, '-') || 'low'
}

export function getBandFromRisk(value) {
  const score = Number(value)
  if (score >= 90) return 'CRITICAL'
  if (score >= 70) return 'HIGH'
  if (score >= 40) return 'MEDIUM'
  return 'LOW'
}

export function splitRules(rules) {
  if (Array.isArray(rules)) return rules.filter(Boolean)
  return String(rules || '').split('|').map((rule) => rule.trim()).filter(Boolean)
}

export function splitReasons(rowOrReasons) {
  if (Array.isArray(rowOrReasons)) return rowOrReasons.filter(Boolean)
  if (rowOrReasons && typeof rowOrReasons === 'object') {
    return [rowOrReasons.reason_1, rowOrReasons.reason_2, rowOrReasons.reason_3].filter(Boolean)
  }
  return String(rowOrReasons || '').split('|').map((reason) => reason.trim()).filter(Boolean)
}

export function parseReason(reason) {
  const value = String(reason || '').trim()
  const match = value.match(/^(.*?)(?:\s+\(([+-]?\d+(?:\.\d+)?)\))$/)
  if (!match) return { feature: value || 'Unknown signal', contribution: null }
  return {
    feature: match[1].replace(/_/g, ' '),
    contribution: Number(match[2]),
  }
}

export function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

export function downloadJson(filename, value) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

export function downloadCsv(filename, columns, rows) {
  const escapeCell = (value) => {
    const text = value === null || value === undefined ? '' : String(value)
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
  }
  const csv = [
    columns.map(escapeCell).join(','),
    ...rows.map((row) => columns.map((column) => escapeCell(row[column])).join(',')),
  ].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}
