import { BrowserRouter, Navigate, Outlet, Route, Routes, useLocation } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Alerts from './pages/Alerts'
import CustomerDetail from './pages/CustomerDetail'
import GraphAnalytics from './pages/GraphAnalytics'
import STRReports from './pages/STRReports'
import PDFParser from './pages/PDFParser'

const pageTitles = {
  '/': 'Investigation Overview',
  '/alerts': 'Alert Queue',
  '/graph': 'Graph Analytics',
  '/reports': 'STR Reports',
  '/pdf': 'PDF Parser',
}

function Layout() {
  const location = useLocation()
  const title = location.pathname.startsWith('/customer/')
    ? 'Customer Investigation'
    : pageTitles[location.pathname] || 'Investigation Overview'

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="app-main">
        <Navbar title={title} />
        <main className="main-content">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/customer/:customerId" element={<CustomerDetail />} />
          <Route path="/graph" element={<GraphAnalytics />} />
          <Route path="/reports" element={<STRReports />} />
          <Route path="/pdf" element={<PDFParser />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
