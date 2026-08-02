import { NavLink } from 'react-router-dom'
import './Sidebar.css'

const navigation = [
  { to: '/', label: 'Dashboard', icon: '⌂' },
  { to: '/alerts', label: 'Alerts', icon: '!' },
  { to: '/graph', label: 'Graph Analytics', icon: '◈' },
  { to: '/reports', label: 'STR Reports', icon: '▤' },
  { to: '/pdf', label: 'PDF Parser', icon: '□' },
]

export default function Sidebar() {
  return (
    <aside className="sidebar" aria-label="Primary navigation">
      <div className="sidebar-brand">
        <div className="brand-mark" aria-hidden="true">TN</div>
        <div className="brand-copy">
          <strong>TRI-NETRA</strong>
          <span>Investigation Console</span>
        </div>
      </div>

      <div className="sidebar-section-label">Workspaces</div>
      <nav className="sidebar-nav">
        {navigation.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}
          >
            <span className="sidebar-icon" aria-hidden="true">{item.icon}</span>
            <span className="sidebar-link-label">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span className="sidebar-footer-dot" aria-hidden="true" />
        <div>
          <strong>Secure workspace</strong>
          <span>Local API connection</span>
        </div>
      </div>
    </aside>
  )
}
