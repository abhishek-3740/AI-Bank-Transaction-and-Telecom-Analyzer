import './StatCard.css'

export default function StatCard({ title, value, subtitle, color = 'var(--accent-blue)', icon }) {
  return (
    <article className="stat-card" style={{ '--stat-accent': color }}>
      <div className="stat-card-topline">
        <span className="stat-card-label">{title}</span>
        {icon && <span className="stat-card-icon" aria-hidden="true">{icon}</span>}
      </div>
      <strong className="stat-card-value">{value}</strong>
      {subtitle && <span className="stat-card-subtitle">{subtitle}</span>}
    </article>
  )
}
