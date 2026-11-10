import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import {
  Activity,
  AlertTriangle,
  Camera,
  Check,
  CheckCircle2,
  CircleStop,
  FileVideo,
  Gauge,
  Play,
  Radio,
  RefreshCw,
  Shield,
  Sliders,
  Trash2,
  TrendingUp,
  UploadCloud,
  Users,
  Wifi,
  WifiOff,
  Zap,
} from 'lucide-react'

const ENV_API_BASE = ((import.meta as any).env?.VITE_API_BASE_URL || '').trim().replace(/\/+$/, '')

const resolveInitialApiUrl = () => {
  try {
    const saved = localStorage.getItem('crowdguard_api_url')
    if (saved) return saved.trim().replace(/\/+$/, '')
  } catch {
    // ignore storage errors
  }
  if (ENV_API_BASE) return ENV_API_BASE
  if (
    typeof window !== 'undefined' &&
    (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ) {
    return 'http://127.0.0.1:8000'
  }
  return ''
}

const resolveInitialToken = () => {
  try {
    return sessionStorage.getItem('crowdguard_api_token') || ''
  } catch {
    return ''
  }
}

const wsBaseUrl = (base: string) => {
  if (base.startsWith('https://')) return base.replace('https://', 'wss://')
  if (base.startsWith('http://')) return base.replace('http://', 'ws://')
  return base
}

type Zone = {
  zone_id: string
  row: number
  col: number
  people_detected: number
  effective_people?: number
  density_p_m2?: number
  fruin_los?: string
  relative_density: number
  motion_magnitude: number
  motion_level: string
  level: string
}

type PanicInfo = {
  panic_score: number
  panic_level: string
  panic_type: string
  crowd_pressure: number
  kinetic_energy?: number
  kinetic_z_score?: number
  signals?: Record<string, number>
  reasons: string[]
}

type Risk = {
  score: number
  level: string
  predicted_score_30s?: number
  predicted_level_30s?: string
  prediction_trend?: string
  estimated_seconds_to_critical?: number | null
  risk_slope_per_sec?: number
  reasons: string[]
  model_type?: string
  probability?: number | null
}

type FlowDynamics = {
  mean_mag: number
  max_mag?: number
  flow_entropy: number
  counter_flow: number
  positive_divergence: number
  negative_divergence: number
  velocity_variance?: number
}

type TrackingDynamics = {
  active_count: number
  mean_norm_speed: number
  direction_coherence: number
  fallen_count: number
  fallen_track_ids?: number[]
}

type SpatialSummary = {
  overall_density_p_m2: number
  peak_zone_density_p_m2: number
  overall_los: string
  peak_los: string
  monitored_area_m2: number
}

type FrameRow = {
  frame: number
  time_seconds: number
  people_detected: number
  motion_magnitude: number
  flow_dynamics?: FlowDynamics
  tracking_dynamics?: TrackingDynamics
  spatial_summary?: SpatialSummary
  panic?: PanicInfo
  risk: Risk
  zone_metrics: Zone[]
  boxes: number[][]
  detector: string
}

type Annotated = {
  time_seconds: number
  image_base64: string
  panic_type?: string
  risk_score?: number
}

type Analysis = {
  analysis_id: string
  annotated_video_url?: string
  filename: string
  frames_analyzed: number
  detector: string
  warning?: string | null
  calibration?: {
    monitored_area_m2: number
    row_weights: number[]
  }
  summary: {
    max_people_detected: number
    average_people_detected: number
    peak_density_p_m2?: number
    peak_los?: string
    max_heuristic_risk: number
    max_predicted_risk_30s?: number
    max_panic_score?: number
    dominant_panic_type?: string
    peak_crowd_pressure?: number
    peak_flow_entropy?: number
    total_fallen_alerts?: number
    peak_time_seconds: number
    peak_level: string
    hotspot_zone: string
  }
  timeline: FrameRow[]
  annotated_frames: Annotated[]
  safety_notice: string
}

type LiveMetrics = {
  timestamp?: string
  people_detected?: number
  motion_magnitude?: number
  flow_dynamics?: FlowDynamics
  tracking_dynamics?: TrackingDynamics
  spatial_summary?: SpatialSummary
  panic?: PanicInfo
  risk?: Risk
  zone_metrics?: Zone[]
  detector?: string
}

type CameraRecord = {
  camera_id: string
  name: string
  venue_zone: string
  masked_url: string
  status: string
  error?: string | null
  started_at?: string | null
  fps?: number
  calibration?: {
    monitored_area_m2: number
    row_weights: number[]
  }
  metrics?: LiveMetrics | null
}

type Incident = {
  id: string
  created_at: string
  camera_id?: string | null
  source: string
  category?: string
  level: string
  score: number
  panic_score?: number
  predicted_score?: number
  status: string
  message: string
  details?: Record<string, any>
}

const levelClass = (s?: string) =>
  s === 'CRITICAL' || s === 'HIGH'
    ? 'high'
    : s === 'ELEVATED' || s === 'WARNING' || s === 'WATCH'
      ? 'elevated'
      : 'normal'

const timeLabel = (s: number) =>
  `${Math.floor(s / 60)
    .toString()
    .padStart(2, '0')}:${Math.floor(s % 60)
    .toString()
    .padStart(2, '0')}`

export default function App() {
  const [navSection, setNavSection] = useState<'overview' | 'cameras' | 'incidents'>('overview')
  const [apiUrl, setApiUrl] = useState<string>(() => resolveInitialApiUrl())
  const [apiToken, setApiToken] = useState<string>(() => resolveInitialToken())
  const [showServerConfig, setShowServerConfig] = useState<boolean>(false)
  const [draftApiUrl, setDraftApiUrl] = useState<string>(() => resolveInitialApiUrl())
  const [draftToken, setDraftToken] = useState<string>(() => resolveInitialToken())

  const [health, setHealth] = useState<any>(null)
  const [file, setFile] = useState<File | null>(null)
  const [areaM2, setAreaM2] = useState<number>(36)
  const [rowWeights, setRowWeights] = useState<string>('1.35,1.0,0.85')
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<number>(0)
  const [error, setError] = useState('')
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [recentAnalyses, setRecentAnalyses] = useState<any[]>([])

  const [incidents, setIncidents] = useState<Incident[]>([])
  const [incidentFilter, setIncidentFilter] = useState<string>('ALL')

  const [cameras, setCameras] = useState<CameraRecord[]>([])
  const [selectedCamId, setSelectedCamId] = useState<string>('cam-primary')
  const [camName, setCamName] = useState('Main Concourse Gate A')
  const [camZone, setCamZone] = useState('North Entrance')
  const [camUrl, setCamUrl] = useState('demo://crowd-sim')
  const [camAreaM2, setCamAreaM2] = useState<number>(36)
  const [liveImage, setLiveImage] = useState('')
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics | null>(null)
  const [liveFps, setLiveFps] = useState<number>(0)
  const [wsConnected, setWsConnected] = useState(false)
  const [rtspBusy, setRtspBusy] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)

  async function apiFetch(path: string, init: RequestInit = {}) {
    if (!apiUrl) {
      throw new Error('Backend server URL is not configured.')
    }
    const headers = new Headers(init.headers || {})
    if (apiToken) {
      headers.set('X-API-Key', apiToken)
    }
    return fetch(`${apiUrl}${path}`, { ...init, headers })
  }

  function saveServerConfig(e?: React.FormEvent) {
    if (e) e.preventDefault()
    const cleanUrl = draftApiUrl.trim().replace(/\/+$/, '')
    const cleanTok = draftToken.trim()
    setApiUrl(cleanUrl)
    setApiToken(cleanTok)
    try {
      if (cleanUrl) {
        localStorage.setItem('crowdguard_api_url', cleanUrl)
      } else {
        localStorage.removeItem('crowdguard_api_url')
      }
      if (cleanTok) {
        sessionStorage.setItem('crowdguard_api_token', cleanTok)
      } else {
        sessionStorage.removeItem('crowdguard_api_token')
      }
    } catch {
      // ignore storage errors
    }
    setShowServerConfig(false)
    setError('')
  }

  async function refreshCore() {
    if (!apiUrl) {
      setHealth(null)
      return
    }
    try {
      const h = await apiFetch('/health')
      if (h.ok) {
        setHealth(await h.json())
      } else {
        setHealth(null)
      }

      const inc = await apiFetch('/api/v1/incidents')
      if (inc.ok) {
        setIncidents((await inc.json()).items || [])
      }

      const cams = await apiFetch('/api/v1/cameras')
      if (cams.ok) {
        const list: CameraRecord[] = (await cams.json()).items || []
        setCameras(list)
        if (list.length > 0 && !list.some((c) => c.camera_id === selectedCamId)) {
          setSelectedCamId(list[0].camera_id)
        }
      }

      const hist = await apiFetch('/api/v1/analyses?limit=10')
      if (hist.ok) {
        setRecentAnalyses((await hist.json()).items || [])
      }
    } catch {
      setHealth(null)
    }
  }

  useEffect(() => {
    void refreshCore()
    const id = setInterval(() => {
      void refreshCore()
    }, 4000)
    return () => clearInterval(id)
  }, [apiUrl, apiToken])

  const selectedCamera = useMemo(
    () => cameras.find((c) => c.camera_id === selectedCamId) || null,
    [cameras, selectedCamId],
  )

  // Real-time WebSocket stream subscription for the selected camera
  useEffect(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setWsConnected(false)

    if (!apiUrl || !selectedCamera || selectedCamera.status === 'stopped') {
      setLiveImage('')
      setLiveMetrics(null)
      return
    }

    let closedByEffect = false
    const tokenQuery = apiToken ? `?token=${encodeURIComponent(apiToken)}` : ''
    const wsUrl = `${wsBaseUrl(apiUrl)}/ws/cameras/${encodeURIComponent(selectedCamera.camera_id)}${tokenQuery}`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      if (!closedByEffect) setWsConnected(true)
    }
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        if (data.image_base64) {
          setLiveImage(`data:image/jpeg;base64,${data.image_base64}`)
        }
        if (data.metrics) {
          setLiveMetrics(data.metrics)
        }
        if (typeof data.fps === 'number') {
          setLiveFps(data.fps)
        }
      } catch {
        // Ignore malformed WS packets
      }
    }
    ws.onerror = () => {
      if (!closedByEffect) setWsConnected(false)
    }
    ws.onclose = () => {
      if (!closedByEffect) setWsConnected(false)
    }

    return () => {
      closedByEffect = true
      ws.close()
    }
  }, [apiUrl, apiToken, selectedCamera?.camera_id, selectedCamera?.status])

  async function analyzeAsync() {
    if (!file) {
      setError('Choose a video file first.')
      return
    }
    setBusy(true)
    setProgress(0.03)
    setError('')
    setAnalysis(null)

    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('monitored_area_m2', String(areaM2))
      fd.append('row_weights', rowWeights)

      const startRes = await apiFetch('/api/v1/analyze/async', {
        method: 'POST',
        body: fd,
      })
      const job = await startRes.json()
      if (!startRes.ok) {
        throw new Error(job.detail || 'Could not queue video analysis.')
      }

      const aid = job.analysis_id
      while (true) {
        await new Promise((r) => setTimeout(r, 750))
        const pollRes = await apiFetch(`/api/v1/analysis/${aid}`)
        const state = await pollRes.json()
        if (!pollRes.ok) {
          throw new Error(state.detail || 'Failed checking analysis status.')
        }
        if (typeof state.progress === 'number') {
          setProgress(state.progress)
        }
        if (state.status === 'COMPLETED' && state.result) {
          setAnalysis(state.result)
          setProgress(1)
          await refreshCore()
          break
        }
        if (state.status === 'FAILED') {
          throw new Error(state.error || 'Video analysis failed.')
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analysis failed')
    } finally {
      setBusy(false)
    }
  }

  async function loadPreviousAnalysis(aid: string) {
    if (!aid) return
    setError('')
    try {
      const r = await apiFetch(`/api/v1/analysis/${aid}`)
      const d = await r.json()
      if (r.ok && d.result) {
        setAnalysis(d.result)
        setNavSection('overview')
      }
    } catch {
      setError('Could not load saved analysis.')
    }
  }

  async function startCamera(presetUrl?: string) {
    const targetUrl = (presetUrl ?? camUrl).trim()
    if (!targetUrl) {
      setError('Enter an RTSP/HTTP URL, webcam:0, or select a demo stream.')
      return
    }
    setRtspBusy(true)
    setError('')
    try {
      const cid = `cam-${camName.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 14) || 'venue'}`
      const r = await apiFetch('/api/v1/cameras/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_id: cid,
          name: camName.trim() || 'Venue Camera',
          venue_zone: camZone.trim() || 'Main Concourse',
          url: targetUrl,
          calibration: {
            monitored_area_m2: camAreaM2,
            row_weights: rowWeights.split(',').map((v) => parseFloat(v.trim()) || 1.0),
          },
        }),
      })
      const d = await r.json()
      if (!r.ok) {
        throw new Error(d.detail || 'Could not start camera stream.')
      }
      setSelectedCamId(d.camera_id)
      await refreshCore()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start camera')
    } finally {
      setRtspBusy(false)
    }
  }

  async function stopCamera(cid: string) {
    try {
      await apiFetch(`/api/v1/cameras/${encodeURIComponent(cid)}/stop`, { method: 'POST' })
      await refreshCore()
    } catch {
      setError('Unable to stop camera stream.')
    }
  }

  async function removeCamera(cid: string) {
    try {
      await apiFetch(`/api/v1/cameras/${encodeURIComponent(cid)}`, { method: 'DELETE' })
      await refreshCore()
    } catch {
      setError('Unable to remove camera.')
    }
  }

  async function updateCamArea(cid: string, area: number) {
    try {
      await apiFetch(`/api/v1/cameras/${encodeURIComponent(cid)}/calibration`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          monitored_area_m2: area,
          row_weights: [1.35, 1.0, 0.85],
        }),
      })
      await refreshCore()
    } catch {
      setError('Failed updating perspective calibration.')
    }
  }

  async function updateIncidentStatus(id: string, status: string) {
    try {
      await apiFetch(`/api/v1/incidents/${encodeURIComponent(id)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      })
      await refreshCore()
    } catch {
      setError('Could not update incident status.')
    }
  }

  async function clearAllIncidents() {
    try {
      await apiFetch('/api/v1/incidents', { method: 'DELETE' })
      await refreshCore()
    } catch {
      setError('Could not clear incidents.')
    }
  }

  const peakFrame = useMemo(() => {
    if (!analysis?.timeline?.length) return undefined
    return analysis.timeline.reduce((a, b) => (b.risk.score > a.risk.score ? b : a))
  }, [analysis])

  const zones = peakFrame?.zone_metrics || []
  const liveZones = liveMetrics?.zone_metrics || []
  const openIncidentCount = incidents.filter((x) => x.status === 'OPEN').length
  const filteredIncidents =
    incidentFilter === 'ALL' ? incidents : incidents.filter((x) => x.status === incidentFilter)

  const activeRiskScore = liveMetrics?.risk?.score ?? analysis?.summary.max_heuristic_risk
  const activePredScore =
    liveMetrics?.risk?.predicted_score_30s ?? analysis?.summary.max_predicted_risk_30s
  const activePanicScore = liveMetrics?.panic?.panic_score ?? analysis?.summary.max_panic_score

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandmark">
            <Shield size={23} />
          </div>
          <div>
            <b>
              CrowdGuard <span>AI</span>
            </b>
            <small>Behavior, Panic &amp; Forecast</small>
          </div>
        </div>

        <div className="section-label">OPERATIONS</div>

        <button
          className={`nav ${navSection === 'overview' ? 'active' : ''}`}
          onClick={() => setNavSection('overview')}
        >
          <Activity size={17} /> Video Intelligence
        </button>

        <button
          className={`nav ${navSection === 'cameras' ? 'active' : ''}`}
          onClick={() => setNavSection('cameras')}
        >
          <Camera size={17} /> Live Venue Feeds
          <i>{cameras.filter((c) => c.status === 'running').length}</i>
        </button>

        <button
          className={`nav ${navSection === 'incidents' ? 'active' : ''}`}
          onClick={() => setNavSection('incidents')}
        >
          <AlertTriangle size={17} /> Incident Queue
          <i>{openIncidentCount}</i>
        </button>

        <div className="side-note">
          <Shield size={17} />
          <b>Physics + ML Decision Support</b>
          <p>
            Combines YOLOv8 tracking, optical-flow entropy &amp; divergence, Helbing crowd pressure,
            and +30s temporal forecasting. Verify physically before intervention.
          </p>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <div className="eyebrow">
              PUBLIC VENUE COMMAND / REAL-TIME CROWD DYNAMICS
            </div>
            <h1>
              {navSection === 'overview'
                ? 'Recorded footage & behavior analysis'
                : navSection === 'cameras'
                  ? 'Live multi-camera venue monitoring'
                  : 'Persistent incident review & triage'}
            </h1>
            <p>
              Kalman-Hungarian tracking · Vector flow entropy &amp; radial dispersion · Crush
              stagnation · Fruin LOS density · +30s temporal forecasting
            </p>
          </div>

          <div className={`connection ${health ? 'online' : 'offline'}`}>
            <span className="dot" />
            {health
              ? `API v${health.version || '1.0'} · ${health.risk_model_loaded ? 'ML Model' : 'Physics Engine'}`
              : apiUrl
                ? 'API disconnected'
                : 'Backend not configured'}
            <button
              className="ghost"
              onClick={() => void refreshCore()}
              aria-label="Refresh API status"
              title="Refresh API status"
            >
              <RefreshCw size={15} />
            </button>
            <button
              className="ghost"
              onClick={() => {
                setDraftApiUrl(apiUrl)
                setDraftToken(apiToken)
                setShowServerConfig((prev) => !prev)
              }}
              aria-label="Configure backend URL"
              title="Configure backend server URL"
            >
              <Sliders size={15} />
            </button>
          </div>
        </header>

        {(showServerConfig || !apiUrl) && (
          <form className="panel server-config" onSubmit={saveServerConfig}>
            <div className="panel-title">
              <div>
                <div className="eyebrow">CLOUD / VERCEL BACKEND CONNECTION</div>
                <h2>Connect FastAPI Vision &amp; ML Engine</h2>
              </div>
              {showServerConfig && apiUrl && (
                <button
                  type="button"
                  className="chip-btn"
                  onClick={() => setShowServerConfig(false)}
                >
                  Close
                </button>
              )}
            </div>
            <p className="muted" style={{ marginTop: 0 }}>
              Vercel hosts the React dashboard. Point this UI to your hosted Python backend URL
              (Render, Railway, Fly.io, Cloud VM, or HTTPS tunnel) or set{' '}
              <code>VITE_API_BASE_URL</code> in Vercel Project Environment Variables.
            </p>
            <div className="server-fields">
              <input
                type="url"
                value={draftApiUrl}
                onChange={(e) => setDraftApiUrl(e.target.value)}
                placeholder="https://crowdguard-ai-backend.onrender.com"
              />
              <input
                type="password"
                value={draftToken}
                onChange={(e) => setDraftToken(e.target.value)}
                placeholder="Optional API Auth Token (if AUTH_ENABLED=true)"
              />
              <button type="submit" className="primary">
                Save &amp; Connect
              </button>
            </div>
          </form>
        )}

        <section className="stats">
          <Stat
            icon={<Users />}
            label="People / Peak Density"
            value={
              liveMetrics
                ? `${liveMetrics.people_detected ?? 0} · ${liveMetrics.spatial_summary?.peak_zone_density_p_m2?.toFixed(2) ?? '0.00'}/m²`
                : analysis
                  ? `${analysis.summary.max_people_detected} · ${(analysis.summary.peak_density_p_m2 ?? 0).toFixed(2)}/m²`
                  : '—'
            }
            foot={
              liveMetrics?.spatial_summary
                ? `Fruin LOS ${liveMetrics.spatial_summary.peak_los} (Live)`
                : analysis?.summary.peak_los
                  ? `Fruin LOS ${analysis.summary.peak_los} (Peak frame)`
                  : 'Perspective-calibrated density'
            }
          />

          <Stat
            icon={<Zap />}
            label="Panic / Anomaly Index"
            value={typeof activePanicScore === 'number' ? `${activePanicScore}/100` : '—'}
            foot={
              liveMetrics?.panic
                ? `${liveMetrics.panic.panic_type} (${liveMetrics.panic.panic_level})`
                : analysis?.summary.dominant_panic_type
                  ? `Peak mode: ${analysis.summary.dominant_panic_type}`
                  : 'Dispersion, chaos & crush stagnation'
            }
          />

          <Stat
            icon={<TrendingUp />}
            label="Risk & +30s Forecast"
            value={
              typeof activeRiskScore === 'number'
                ? `${activeRiskScore} → ${activePredScore ?? activeRiskScore}`
                : '—'
            }
            foot={
              liveMetrics?.risk?.estimated_seconds_to_critical != null
                ? `ETCT: ${liveMetrics.risk.estimated_seconds_to_critical}s (${liveMetrics.risk.prediction_trend})`
                : liveMetrics?.risk?.prediction_trend ||
                  (analysis
                    ? `Peak trend: ${peakFrame?.risk?.prediction_trend || 'ANALYZED'}`
                    : 'Sliding-window horizon projection')
            }
          />

          <Stat
            icon={<AlertTriangle />}
            label="Open Venue Alerts"
            value={String(openIncidentCount)}
            foot={`${incidents.length} total logged in SQLite`}
          />
        </section>

        {error && <div className="error">{error}</div>}

        {/* --- VIDEO INTELLIGENCE VIEW --- */}
        {navSection === 'overview' && (
          <>
            <section className="panel">
              <div className="panel-title">
                <div>
                  <div className="eyebrow">OFFLINE &amp; FORENSIC ANALYSIS</div>
                  <h2>Upload venue video with perspective calibration</h2>
                </div>
                {recentAnalyses.length > 0 && (
                  <select
                    className="history-select"
                    value={analysis?.analysis_id || ''}
                    onChange={(e) => void loadPreviousAnalysis(e.target.value)}
                  >
                    <option value="" disabled>
                      Load previous analysis ({recentAnalyses.length})
                    </option>
                    {recentAnalyses.map((item) => (
                      <option key={item.analysis_id} value={item.analysis_id}>
                        {item.filename} ({item.status})
                      </option>
                    ))}
                  </select>
                )}
              </div>

              <div className="upload-layout">
                <div>
                  <label className="dropzone">
                    <input
                      type="file"
                      accept="video/*,.mp4,.mov,.avi,.mkv,.webm"
                      onChange={(e) => {
                        setFile(e.target.files?.[0] || null)
                        setError('')
                      }}
                    />
                    <UploadCloud size={27} />
                    <b>{file ? file.name : 'Choose MP4, MOV, AVI, MKV, or WEBM'}</b>
                    <span>
                      {file
                        ? `${(file.size / 1024 / 1024).toFixed(1)} MB selected`
                        : 'Non-blocking async processing with live progress'}
                    </span>
                  </label>

                  <div className="cal-bar">
                    <label>
                      <Sliders size={13} /> Monitored Floor Area (m²):
                      <input
                        type="number"
                        min={6}
                        max={1000}
                        value={areaM2}
                        onChange={(e) => setAreaM2(Number(e.target.value) || 36)}
                      />
                    </label>
                    <label>
                      Row Weights (Top,Mid,Bot):
                      <input
                        type="text"
                        value={rowWeights}
                        onChange={(e) => setRowWeights(e.target.value)}
                      />
                    </label>
                  </div>

                  <div className="actions">
                    <button
                      className="primary"
                      disabled={!health || busy || !file}
                      onClick={() => void analyzeAsync()}
                    >
                      {busy ? (
                        <>
                          <RefreshCw className="spin" size={16} />
                          Analyzing ({Math.round(progress * 100)}%)…
                        </>
                      ) : (
                        <>
                          <FileVideo size={16} />
                          Analyze video
                        </>
                      )}
                    </button>

                    {analysis && (
                      <button
                        className="secondary"
                        onClick={() => {
                          setAnalysis(null)
                          setFile(null)
                          setError('')
                        }}
                      >
                        Clear result
                      </button>
                    )}
                  </div>

                  {busy && (
                    <div className="progress-track">
                      <div
                        className="progress-fill"
                        style={{ width: `${Math.max(4, Math.round(progress * 100))}%` }}
                      />
                    </div>
                  )}
                </div>

                <div className="video-info">
                  <div className="eyebrow">BEHAVIOR &amp; PHYSICS DIAGNOSTICS</div>
                  <h3>{analysis ? analysis.filename : 'Ready for spatio-temporal analysis'}</h3>
                  {analysis ? (
                    <div className="diag-list">
                      <div>
                        <span>Detector &amp; Tracker</span>
                        <b>{analysis.detector}</b>
                      </div>
                      <div>
                        <span>Peak Density &amp; LOS</span>
                        <b>
                          {(analysis.summary.peak_density_p_m2 ?? 0).toFixed(2)} p/m² (LOS{' '}
                          {analysis.summary.peak_los || 'A'})
                        </b>
                      </div>
                      <div>
                        <span>Dominant Panic Signature</span>
                        <b>
                          {analysis.summary.dominant_panic_type || 'NORMAL_FLOW'} (
                          {analysis.summary.max_panic_score ?? 0}/100)
                        </b>
                      </div>
                      <div>
                        <span>Helbing Crowd Pressure</span>
                        <b>{(analysis.summary.peak_crowd_pressure ?? 0).toFixed(2)}</b>
                      </div>
                      <div>
                        <span>8-Bin Shannon Flow Entropy</span>
                        <b>{(analysis.summary.peak_flow_entropy ?? 0).toFixed(2)}</b>
                      </div>
                      <div>
                        <span>Fallen / Posture Collapse Alerts</span>
                        <b>{analysis.summary.total_fallen_alerts ?? 0} frames</b>
                      </div>
                    </div>
                  ) : (
                    <p>
                      Computes multi-object trajectories, full optical-flow vector divergence &amp;
                      entropy, Helbing crowd pressure, crush deadlock stagnation, and +30s risk
                      trajectory forecasting.
                    </p>
                  )}
                  {analysis?.warning && <div className="inline-warning">{analysis.warning}</div>}
                </div>
              </div>
            </section>

            {analysis && (
              <>
                <section className="panel">
                  <div className="panel-title">
                    <div>
                      <div className="eyebrow">ANNOTATED PLAYBACK (TRACKS + FLOW VECTORS + HUD)</div>
                      <h2>Processed venue video</h2>
                    </div>
                    {analysis.annotated_video_url && (
                      <a
                        className="download-link"
                        href={`${apiUrl}${analysis.annotated_video_url}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open / download video
                      </a>
                    )}
                  </div>

                  {analysis.annotated_video_url ? (
                    <video
                      key={analysis.analysis_id}
                      className="annotated-player"
                      controls
                      preload="metadata"
                      src={`${apiUrl}${analysis.annotated_video_url}`}
                    />
                  ) : (
                    <div className="empty">Annotated video unavailable.</div>
                  )}
                </section>

                <section className="panel">
                  <div className="panel-title">
                    <div>
                      <div className="eyebrow">TEMPORAL FORECASTING &amp; PANIC TIMELINE</div>
                      <h2>Current Risk vs +30s Forecast &amp; Panic Score</h2>
                    </div>
                    <span className="chip">
                      Peak @ {timeLabel(analysis.summary.peak_time_seconds)} · Zone{' '}
                      {analysis.summary.hotspot_zone}
                    </span>
                  </div>

                  <div className="timeline-legend">
                    <span>
                      <i className="swatch risk" /> Current Risk
                    </span>
                    <span>
                      <i className="swatch pred" /> +30s Forecast
                    </span>
                    <span>
                      <i className="swatch panic" /> Panic Index
                    </span>
                  </div>

                  <div className="timeline">
                    {analysis.timeline
                      .filter(
                        (_, i) =>
                          i % Math.max(1, Math.floor(analysis.timeline.length / 28)) === 0,
                      )
                      .slice(0, 28)
                      .map((r, i) => (
                        <div
                          className="timeline-bar"
                          title={`${timeLabel(r.time_seconds)} · Risk ${r.risk.score} | +30s ${r.risk.predicted_score_30s ?? r.risk.score} | Panic ${r.panic?.panic_score ?? 0}`}
                          key={`${r.frame}-${i}`}
                        >
                          <div className="bar-group">
                            <div
                              className={`bar ${levelClass(r.risk.level)}`}
                              style={{ height: `${Math.max(4, r.risk.score)}%` }}
                            />
                            <div
                              className="bar pred-bar"
                              style={{
                                height: `${Math.max(
                                  4,
                                  r.risk.predicted_score_30s ?? r.risk.score,
                                )}%`,
                              }}
                            />
                            <div
                              className="bar panic-bar"
                              style={{ height: `${Math.max(4, r.panic?.panic_score ?? 0)}%` }}
                            />
                          </div>
                          <small>{i % 4 === 0 ? timeLabel(r.time_seconds) : ''}</small>
                        </div>
                      ))}
                  </div>
                </section>

                <section className="panel">
                  <div className="panel-title">
                    <div>
                      <div className="eyebrow">CALIBRATED SPATIAL GRID</div>
                      <h2>3 × 3 Density (p/m²) &amp; Fruin LOS (Peak Frame)</h2>
                    </div>
                    <span className="chip">
                      Area: {analysis.calibration?.monitored_area_m2 ?? areaM2} m²
                    </span>
                  </div>

                  <ZoneGrid zones={zones} />

                  <div className="zone-table">
                    {zones.map((z) => (
                      <div className="zone-row" key={z.zone_id}>
                        <b>{z.zone_id}</b>
                        <span>
                          {z.people_detected} people ({(z.density_p_m2 ?? 0).toFixed(2)}/m²)
                        </span>
                        <span>
                          LOS {z.fruin_los || 'A'} · Flow {z.motion_magnitude.toFixed(2)}
                        </span>
                        <span className={`pill ${levelClass(z.level)}`}>{z.level}</span>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="panel">
                  <div className="panel-title">
                    <div>
                      <div className="eyebrow">SAMPLED ANNOTATED KEYFRAMES</div>
                      <h2>Frame-by-frame inspection</h2>
                    </div>
                    <span className="chip">{analysis.annotated_frames?.length || 0} frames</span>
                  </div>

                  <div className="frame-grid">
                    {(analysis.annotated_frames || []).map((f, i) => (
                      <div className="frame-card" key={`${f.time_seconds}-${i}`}>
                        <img
                          src={`data:image/jpeg;base64,${f.image_base64}`}
                          alt={`Frame ${i + 1}`}
                        />
                        <div>
                          <b>
                            {timeLabel(f.time_seconds)} · {f.panic_type || 'NORMAL'}
                          </b>
                          <span>Risk {f.risk_score ?? 0}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              </>
            )}
          </>
        )}

        {/* --- LIVE MULTI-CAMERA VENUE VIEW --- */}
        {navSection === 'cameras' && (
          <>
            <section className="panel">
              <div className="panel-title">
                <div>
                  <div className="eyebrow">MULTI-CAMERA INGESTION &amp; WEBSOCKET STREAMING</div>
                  <h2>Connect RTSP, webcam, or simulated venue stream</h2>
                </div>
                <span className={`chip ${wsConnected ? 'ws-on' : ''}`}>
                  {wsConnected ? `WS Live · ${liveFps.toFixed(1)} FPS` : 'WebSocket Standby'}
                </span>
              </div>

              <div className="rtsp-layout">
                <div className="rtsp-form">
                  <div className="two-col">
                    <div>
                      <label>Camera Name</label>
                      <input value={camName} onChange={(e) => setCamName(e.target.value)} />
                    </div>
                    <div>
                      <label>Venue Zone</label>
                      <input value={camZone} onChange={(e) => setCamZone(e.target.value)} />
                    </div>
                  </div>

                  <label>Stream URL (RTSP / HTTP / webcam:0 / demo://*)</label>
                  <input
                    value={camUrl}
                    onChange={(e) => setCamUrl(e.target.value)}
                    placeholder="rtsp://user:pass@192.168.1.50:554/stream1 or demo://crowd-sim"
                  />

                  <div className="preset-row">
                    <span>Quick Test Scenarios:</span>
                    <button
                      type="button"
                      className="chip-btn"
                      onClick={() => {
                        setCamUrl('demo://crowd-sim')
                        void startCamera('demo://crowd-sim')
                      }}
                    >
                      <Play size={12} /> Full Cycle Sim
                    </button>
                    <button
                      type="button"
                      className="chip-btn"
                      onClick={() => {
                        setCamUrl('demo://panic-sim')
                        void startCamera('demo://panic-sim')
                      }}
                    >
                      <Zap size={12} /> Panic Dispersion Sim
                    </button>
                    <button
                      type="button"
                      className="chip-btn"
                      onClick={() => {
                        setCamUrl('demo://crush-sim')
                        void startCamera('demo://crush-sim')
                      }}
                    >
                      <Users size={12} /> Crush Stagnation Sim
                    </button>
                  </div>

                  <div className="cal-bar">
                    <label>
                      <Sliders size={13} /> Monitored Area (m²):
                      <input
                        type="number"
                        min={6}
                        max={1000}
                        value={camAreaM2}
                        onChange={(e) => setCamAreaM2(Number(e.target.value) || 36)}
                      />
                    </label>
                  </div>

                  <div className="actions">
                    <button
                      className="primary"
                      disabled={!health || rtspBusy}
                      onClick={() => void startCamera()}
                    >
                      <Radio size={16} />
                      {rtspBusy ? 'Connecting…' : 'Start / Update Feed'}
                    </button>
                    {selectedCamera && selectedCamera.status !== 'stopped' && (
                      <button
                        className="secondary"
                        onClick={() => void stopCamera(selectedCamera.camera_id)}
                      >
                        <CircleStop size={16} /> Stop Active
                      </button>
                    )}
                  </div>

                  {cameras.length > 0 && (
                    <div className="cam-list">
                      {cameras.map((cam) => (
                        <div
                          key={cam.camera_id}
                          className={`cam-card ${cam.camera_id === selectedCamId ? 'selected' : ''}`}
                          onClick={() => setSelectedCamId(cam.camera_id)}
                        >
                          <div>
                            <b>{cam.name}</b>
                            <small>
                              {cam.venue_zone} · {cam.status.toUpperCase()} ({cam.fps ?? 0} FPS)
                            </small>
                          </div>
                          <div className="cam-actions">
                            <button
                              type="button"
                              title="Adjust +10 m2"
                              onClick={(e) => {
                                e.stopPropagation()
                                const cur = cam.calibration?.monitored_area_m2 ?? 36
                                void updateCamArea(cam.camera_id, cur + 10)
                              }}
                            >
                              {cam.calibration?.monitored_area_m2 ?? 36}m²
                            </button>
                            <button
                              type="button"
                              title="Remove camera"
                              onClick={(e) => {
                                e.stopPropagation()
                                void removeCamera(cam.camera_id)
                              }}
                            >
                              <Trash2 size={14} />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="live-preview">
                  {liveImage ? (
                    <img src={liveImage} alt="Live WebSocket annotated frame" />
                  ) : (
                    <div className="preview-empty">
                      <WifiOff size={30} />
                      <b>
                        {selectedCamera
                          ? `Status: ${selectedCamera.status.toUpperCase()}`
                          : 'No active camera'}
                      </b>
                      <span>
                        {selectedCamera?.error ||
                          'Connect an RTSP camera or click a Quick Test Scenario.'}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </section>

            {liveMetrics && (
              <section className="panel">
                <div className="panel-title">
                  <div>
                    <div className="eyebrow">REAL-TIME CROWD PHYSICS &amp; FORECAST TELEMETRY</div>
                    <h2>{selectedCamera?.name || 'Live Stream'} — Zone &amp; Panic Analysis</h2>
                  </div>
                  <span className="live-tag">
                    <span className="dot" /> LIVE WEBSOCKET
                  </span>
                </div>

                <div className="live-telemetry-grid">
                  <div className="tele-pill">
                    <span>Behavior Mode</span>
                    <b>{liveMetrics.panic?.panic_type || 'NORMAL_FLOW'}</b>
                  </div>
                  <div className="tele-pill">
                    <span>Crowd Pressure</span>
                    <b>{(liveMetrics.panic?.crowd_pressure ?? 0).toFixed(2)}</b>
                  </div>
                  <div className="tele-pill">
                    <span>Flow Entropy / Counter-Flow</span>
                    <b>
                      {(liveMetrics.flow_dynamics?.flow_entropy ?? 0).toFixed(2)} /{' '}
                      {(liveMetrics.flow_dynamics?.counter_flow ?? 0).toFixed(2)}
                    </b>
                  </div>
                  <div className="tele-pill">
                    <span>+30s Forecast &amp; Trend</span>
                    <b>
                      {liveMetrics.risk?.predicted_score_30s ?? 0}/100 (
                      {liveMetrics.risk?.prediction_trend || 'STABLE'})
                    </b>
                  </div>
                </div>

                <ZoneGrid zones={liveZones} />
              </section>
            )}
          </>
        )}

        {/* --- INCIDENTS QUEUE VIEW --- */}
        {navSection === 'incidents' && (
          <section className="panel">
            <div className="panel-title">
              <div>
                <div className="eyebrow">PERSISTENT SQLITE AUDIT LOG</div>
                <h2>Venue safety incidents &amp; alerts</h2>
              </div>
              <div className="actions" style={{ marginTop: 0 }}>
                {['ALL', 'OPEN', 'ACKNOWLEDGED', 'RESOLVED'].map((st) => (
                  <button
                    key={st}
                    className={`chip-btn ${incidentFilter === st ? 'active-chip' : ''}`}
                    onClick={() => setIncidentFilter(st)}
                  >
                    {st}
                  </button>
                ))}
                {incidents.length > 0 && (
                  <button className="secondary" onClick={() => void clearAllIncidents()}>
                    <Trash2 size={14} /> Clear all
                  </button>
                )}
              </div>
            </div>

            {filteredIncidents.length ? (
              filteredIncidents.map((x) => (
                <div className="incident" key={x.id}>
                  <AlertTriangle size={18} />
                  <div>
                    <b>
                      [{x.status}] {x.level} · {x.category || 'CROWD_RISK'} · {x.source}
                    </b>
                    <p>{x.message}</p>
                    <small>
                      {x.created_at ? new Date(x.created_at).toLocaleString() : ''} · Risk{' '}
                      {x.score}/100 · Panic {x.panic_score ?? 0}/100 · +30s Pred{' '}
                      {x.predicted_score ?? x.score}/100
                    </small>
                  </div>
                  <div className="inc-actions">
                    {x.status === 'OPEN' && (
                      <button
                        className="chip-btn"
                        onClick={() => void updateIncidentStatus(x.id, 'ACKNOWLEDGED')}
                      >
                        Acknowledge
                      </button>
                    )}
                    {x.status !== 'RESOLVED' && (
                      <button
                        className="chip-btn"
                        onClick={() => void updateIncidentStatus(x.id, 'RESOLVED')}
                      >
                        <Check size={12} /> Resolve
                      </button>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="empty">
                <CheckCircle2 size={24} />
                No incidents matching '{incidentFilter}'.
              </div>
            )}
          </section>
        )}

        <footer>
          <span>
            <Shield size={14} /> CrowdGuard AI · Real-Time Behavior, Panic &amp; Forecast Engine
          </span>
          <span>
            <Wifi size={14} /> {apiUrl || 'No backend URL configured'}
          </span>
        </footer>
      </main>
    </div>
  )
}

function Stat({
  icon,
  label,
  value,
  foot,
}: {
  icon: ReactNode
  label: string
  value: string
  foot: string
}) {
  return (
    <div className="stat">
      <div className="stat-head">
        <span>{icon}</span>
        {label}
      </div>
      <strong>{value}</strong>
      <small>{foot}</small>
    </div>
  )
}

function ZoneGrid({ zones }: { zones: Zone[] }) {
  const byId = new Map(zones.map((z) => [z.zone_id, z]))
  return (
    <div className="heat-grid">
      {Array.from({ length: 9 }, (_, i) => {
        const id = `R${Math.floor(i / 3) + 1}C${(i % 3) + 1}`
        const z = byId.get(id)
        return (
          <div key={id} className={`heat-cell ${levelClass(z?.level || 'NORMAL')}`}>
            <small>
              {id} · {z?.fruin_los ? `LOS ${z.fruin_los}` : 'LOS —'}
            </small>
            <strong>{z?.people_detected ?? '—'}</strong>
            <span>{z ? `${(z.density_p_m2 ?? 0).toFixed(2)} people/m²` : 'No data'}</span>
            <div className="cell-bottom">
              {z ? `${z.motion_level} flow (${z.motion_magnitude.toFixed(1)})` : 'Standby'}
            </div>
          </div>
        )
      })}
    </div>
  )
}
