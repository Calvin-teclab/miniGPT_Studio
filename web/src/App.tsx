import { lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'

const PipelinePage = lazy(() => import('./pages/PipelinePage'))
const ConfigPage = lazy(() => import('./pages/ConfigPage'))
const TrainingPage = lazy(() => import('./pages/TrainingPage'))
const EvalPage = lazy(() => import('./pages/EvalPage'))
const ChatPage = lazy(() => import('./pages/ChatPage'))
const MusicPage = lazy(() => import('./pages/MusicPage'))

function PageLoading() {
  return (
    <div className="p-8 text-sm text-text-muted">
      页面加载中...
    </div>
  )
}

export default function App() {
  return (
    <Suspense fallback={<PageLoading />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<PipelinePage />} />
          <Route path="config" element={<ConfigPage />} />
          <Route path="training" element={<TrainingPage />} />
          <Route path="eval" element={<EvalPage />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="music" element={<MusicPage />} />
        </Route>
      </Routes>
    </Suspense>
  )
}
