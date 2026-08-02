import axios from 'axios'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 30000,
})

function stripUndefined(params = {}) {
  return Object.fromEntries(
    Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== ''),
  )
}

function getErrorMessage(error) {
  const detail = error?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((item) => item?.msg || String(item)).join(', ')
  return error?.message || 'The request could not be completed.'
}

async function request(promise) {
  try {
    const response = await promise
    return response.data
  } catch (error) {
    const wrapped = new Error(getErrorMessage(error))
    wrapped.status = error?.response?.status
    wrapped.cause = error
    throw wrapped
  }
}

export function getScoringStats() {
  return request(client.get('/v1/scoring/stats'))
}

export function getAlerts(params = {}) {
  return request(client.get('/v1/scoring/alerts', { params: stripUndefined(params) }))
}

export function getTransactions(params = {}) {
  return request(client.get('/v1/scoring/transactions', { params: stripUndefined(params) }))
}

export function getCustomer(customerId) {
  return request(client.get(`/v1/scoring/customer/${encodeURIComponent(customerId)}`))
}

export function scoreTransaction(transactionObj) {
  return request(client.post('/v1/scoring/score', { transaction: transactionObj }))
}

export function getGraphSummary(topN = 10) {
  return request(client.get('/v1/graph/summary', { params: { top_n: topN } }))
}

export function getGraphNodes(params = {}) {
  return request(client.get('/v1/graph/nodes', { params: stripUndefined(params) }))
}

export function getGraphNode(nodeId) {
  return request(client.get(`/v1/graph/node/${encodeURIComponent(nodeId)}`))
}

export function getGraphMules() {
  return request(client.get('/v1/graph/mules'))
}

export function getGraphEdges(params = {}) {
  return request(client.get('/v1/graph/edges', { params: stripUndefined(params) }))
}

export function getStrBatch(params = {}) {
  return request(client.get('/v1/reports/str/batch', { params: stripUndefined(params) }))
}

export function getStrReport(customerId, params = {}) {
  return request(client.get(`/v1/reports/str/${encodeURIComponent(customerId)}`, { params: stripUndefined(params) }))
}

export function getReportsSummary(minRisk) {
  return request(client.get('/v1/reports/summary', { params: stripUndefined({ min_risk: minRisk }) }))
}

export function parsePdf(formData) {
  return request(client.post('/v1/pdf/parse', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }))
}
