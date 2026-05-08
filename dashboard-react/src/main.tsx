import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App.tsx'
import { queryClient } from './api/queryClient'
import { ErrorBoundary } from './components/ui/ErrorBoundary'
import { registerServiceWorker } from './api/pwa'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </QueryClientProvider>
  </StrictMode>,
)

// Service worker registration — non-blocking. In dev mode the SW would
// cache stale Vite-served assets and break HMR, so skip it there.
if (import.meta.env.PROD) {
  registerServiceWorker();
}
