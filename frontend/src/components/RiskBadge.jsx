import { getBandClass, formatRisk } from '../utils/formatters'
import './RiskBadge.css'

export default function RiskBadge({ band, score, showScore = false }) {
  const label = band || 'LOW'
  return (
    <span className={`risk-badge risk-${getBandClass(label)}`}>
      <span>{label}</span>
      {showScore && score !== undefined && <strong>{formatRisk(score)}</strong>}
    </span>
  )
}
