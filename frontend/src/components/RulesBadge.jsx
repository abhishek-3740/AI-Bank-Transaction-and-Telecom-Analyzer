import { splitRules } from '../utils/formatters'
import './RulesBadge.css'

const ruleClass = {
  ODD_HOUR: 'rule-blue',
  HIGH_AMOUNT_ANOMALY: 'rule-orange',
  RAPID_SUCCESSION: 'rule-red',
  NEW_BENEFICIARY_FLAG: 'rule-yellow',
  TELECOM_BURST: 'rule-purple',
}

export default function RulesBadge({ rules, limit }) {
  const ruleList = splitRules(rules)
  const visibleRules = limit ? ruleList.slice(0, limit) : ruleList
  const hiddenCount = Math.max(ruleList.length - visibleRules.length, 0)
  if (!ruleList.length) return <span className="rules-empty">—</span>

  return (
    <span className="rules-badge-list" title={ruleList.join(' · ')}>
      {visibleRules.map((rule) => (
        <span key={rule} className={`rule-badge ${ruleClass[rule] || 'rule-default'}`}>
          {rule.replace(/_FLAG$/, '').replace(/_/g, ' ')}
        </span>
      ))}
      {hiddenCount > 0 && <span className="rule-more">+{hiddenCount}</span>}
    </span>
  )
}
