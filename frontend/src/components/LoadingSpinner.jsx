import './LoadingSpinner.css'

export default function LoadingSpinner({ size = 'md', label = 'Loading' }) {
  return (
    <span className={`loading-spinner loading-spinner-${size}`} role="status" aria-label={label} />
  )
}
