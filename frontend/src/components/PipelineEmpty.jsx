import { Link } from 'react-router-dom'
import './PipelineEmpty.css'

/**
 * Banner for the "no scored data yet" state (API 503).
 *
 * Every page used to swallow that response and render an empty table, which
 * reads as "nothing suspicious found" rather than "nothing has been scored".
 * The distinction matters in an investigation tool, so say it out loud.
 */
export default function PipelineEmpty({ what = 'data' }) {
  return (
    <div className="empty-state-banner" role="status">
      <span className="empty-state-icon" aria-hidden="true">📂</span>
      <div>
        <strong>No scored {what} yet</strong>
        <p>
          Upload a bank statement PDF on the <Link to="/pdf">PDF Parser</Link> page — it is
          scored and indexed automatically. To load the full demo dataset instead, run{' '}
          <code>python scripts/score.py</code> and <code>python scripts/graph_analytics.py</code>{' '}
          from the <code>backend/</code> directory.
        </p>
      </div>
    </div>
  )
}
