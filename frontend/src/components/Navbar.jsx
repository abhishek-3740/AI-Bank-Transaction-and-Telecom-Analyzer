import { useLocation } from 'react-router-dom'
import './Navbar.css'

export default function Navbar({ title }) {
  const location = useLocation()
  const pathLabel = location.pathname === '/' ? 'Overview' : location.pathname.split('/')[1]?.replace(/-/g, ' ')

  return (
    <header className="navbar">
      <div className="navbar-context">
        <span className="navbar-eyebrow">TRI-NETRA / {pathLabel || 'workspace'}</span>
        <h1>{title}</h1>
      </div>
      <div className="navbar-status" aria-label="System Active">
        <span className="status-dot" aria-hidden="true" />
        <span>System Active</span>
      </div>
    </header>
  )
}
