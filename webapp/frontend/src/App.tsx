import React, { useState, useEffect, useRef } from "react"
import { HashRouter as Router, Routes, Route, Link, useLocation } from "react-router-dom"
import {
  Search as SearchIcon,
  Database,
  RefreshCw,
  Sliders,
  AlertCircle,
  Layers,
  Image as ImageIcon,
  Video,
  Target,
  BookOpen,
  UploadCloud,
  Film,
  MessageCircle,
  History,
  X,
  BarChart3,
} from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { ResultCard, type ResultHit } from "@/components/ResultCard"
import { CandidateInspectionDialog } from "@/components/CandidateInspectionDialog"
import { BrowseVideoDialog } from "@/components/BrowseVideoDialog"
import { VBSAuditWorkspace } from "@/components/VBSAuditWorkspace"
import { AuditHistoryView } from "@/components/AuditHistoryView"
import { RAGBenchmarkWorkspace } from "@/components/RAGBenchmarkWorkspace"
import { VQAWorkspace } from "@/components/VQAWorkspace"
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""
// -------------------------------------------------------------
// NAVIGATION COMPONENT
// -------------------------------------------------------------
function Navbar() {
  const location = useLocation()
  
  const navItems = [
    { path: "/", label: "Live Search", icon: SearchIcon },
    { path: "/benchmark", label: "RAG Benchmark", icon: BarChart3 },
    { path: "/audit", label: "System Audit", icon: Layers },
    { path: "/history", label: "Audit History", icon: History },
    { path: "/database", label: "Vector Store", icon: Database }
  ]
  return (
    <nav className="vbs-nav sticky top-0 z-40">
      <div className="w-full px-4 sm:px-6 lg:px-8 py-2.5 flex items-center justify-between gap-4">
        <Link to="/" className="flex items-center gap-3 min-w-0 group">
          <div className="vbs-mark">
            <Video className="h-5 w-5" />
            <span className="vbs-mark-dot" />
          </div>
          <div className="min-w-0 text-left">
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-base sm:text-lg font-extrabold tracking-tight text-slate-900 m-0">
                AEGIS <span className="text-xs font-semibold text-slate-400 font-normal">| TGLTW-RMIT</span>
              </h1>
              <span className="vbs-badge">VBS 2027</span>
            </div>
            <p className="hidden sm:block text-[11px] text-slate-500 font-semibold tracking-wide uppercase mt-0.5">
              Adaptive Evidence-Grounded Interactive Search
            </p>
          </div>
        </Link>

        <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path
            const Icon = item.icon
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`vbs-nav-link ${isActive ? "vbs-nav-link-active" : ""}`}
              >
                <Icon className="h-4 w-4 shrink-0" />
                <span className="hidden sm:inline">{item.label}</span>
              </Link>
            )
          })}
        </div>

        <div className="hidden xl:flex items-center gap-2 text-[10px] uppercase tracking-[0.16em] font-bold text-slate-500 shrink-0">
          <span className="vbs-live-dot" /> Live system
        </div>
      </div>
    </nav>
  )
}

// -------------------------------------------------------------
// VIEW 1: SEARCH PAGE & BATCH QUERIES
// -------------------------------------------------------------
type TaskMode = "kis-t" | "kis-c" | "kis-v" | "avs" | "vqa"

function SearchView() {
  // Single Query states
  const [query, setQuery] = useState("")
  const [taskMode, setTaskMode] = useState<TaskMode>("kis-t")
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [, setExpandedIndex] = useState<number | null>(null)
  const [inspectingHit, setInspectingHit] = useState<{ hit: ResultHit; rank: number } | null>(null)
  // KIS-C clarification question (Phase L) - set when /api/search's Type 1
  // flow detects an ambiguous result set and asks a narrowing question.
  const [clarification, setClarification] = useState<string | null>(null)
  const [clarificationAnswer, setClarificationAnswer] = useState("")
  const [kisCMessages, setKisCMessages] = useState<Array<{ role: "operator" | "system"; text: string }>>([])
  // Video player modal state
  const [selectedVideo, setSelectedVideo] = useState<{name: string, time: number, frameIdx?: number | null} | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)

  // KIS-V clip upload
  const videoUploadRef = useRef<HTMLInputElement>(null)
  const [kisVideoFile, setKisVideoFile] = useState<File | null>(null)
  const [videoDropActive, setVideoDropActive] = useState(false)

  // VBS interactive-session states (Phase C backend / Phase D UI of the
  // batch-to-interactive plan): temporal query mode, video-browse dialog,
  // and DRES login/current-task so "Nộp câu trả lời" has a task_id to
  // submit against.
  const [browsingVideo, setBrowsingVideo] = useState<string | null>(null)
  const [dresLoggedIn, setDresLoggedIn] = useState(false)
  const [currentTask, setCurrentTask] = useState<any>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)

  // Backend /api/search lanes: 1 = KIS-T *and* KIS-C (all conversational
  // machinery - clarification boost, negative feedback filter, ambiguity
  // question - is gated behind request.type == 1 in main.py), 2 = VQA,
  // 3 = TRAKE temporal, 4 = AVS. Visual (kis-v) tasks short-circuit to
  // /api/search-by-video, so the numeric type is unused there; sending 5
  // would be rejected by the backend (valid: 1-4).
  const queryType = taskMode === "vqa" ? 2 : taskMode === "avs" ? 4 : 1
  const isVisualTask = taskMode === "kis-v"
  const isConversationalTask = taskMode === "kis-c"
  const isAvsTask = taskMode === "avs"

  const handleTaskModeChange = (mode: TaskMode) => {
    setTaskMode(mode)
    setResults([])
    setError(null)
    setClarification(null)
    setClarificationAnswer("")
    setKisCMessages([])
    if (mode !== "kis-v") setKisVideoFile(null)
  }
  const acceptKisVideo = (file: File | undefined) => {
    if (!file) return
    const isVideo = file.type.startsWith("video/") || /\.(mp4|mov|webm|mkv|avi)$/i.test(file.name)
    const isImage = file.type.startsWith("image/") || /\.(jpg|jpeg|png|webp)$/i.test(file.name)
    if (!isVideo && !isImage) {
      setError("KIS-V yêu cầu một file video (MP4, MOV...) hoặc ảnh khung hình (JPG, PNG...).")
      return
    }
    setError(null)
    setKisVideoFile(file)
    setResults([])
  }

  // -----------------------------------------------------------
  // SINGLE QUERY INFERENCE
  // -----------------------------------------------------------
  const handleSearchByVideo = async () => {
    if (!kisVideoFile) return
    setLoading(true)
    setError(null)
    setResults([])
    setExpandedIndex(null)
    try {
      const isImage = kisVideoFile.type.startsWith("image/") || /\.(jpg|jpeg|png|webp)$/i.test(kisVideoFile.name)
      const endpoint = isImage ? "/api/search-by-image" : "/api/search-by-video"
      const formData = new FormData()
      formData.append("file", kisVideoFile)
      const response = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Visual search failed")
      }
      const data = await response.json()
      setResults(data.results || [])
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Visual search failed"
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (isVisualTask) {
      await handleSearchByVideo()
      return
    }
    if (!query.trim()) return
    if (isConversationalTask && clarification && !clarificationAnswer.trim()) return

    const requestQuery = isConversationalTask && clarificationAnswer.trim()
      ? `${query}\nAdditional detail from operator: ${clarificationAnswer.trim()}`
      : query

    // Auto-detect sequential temporal chain if query has " then " or " -> "
    const temporalSteps = query.split(/\s+then\s+|->/i).map((s) => s.trim()).filter(Boolean)
    const isTemporal = temporalSteps.length > 1

    setLoading(true)
    setError(null)
    setResults([])
    setExpandedIndex(null)
    setClarification(null)

    try {
      const endpoint = isTemporal ? "/api/temporal-search" : "/api/search"
      const body = isTemporal
        ? { queries: temporalSteps, exact: true, verify: true }
        : {
            type: queryType,
            query: requestQuery,
            exact: true,
            verify: true,
            ...(isConversationalTask && clarificationAnswer.trim()
              ? { clarification_answer: clarificationAnswer.trim() }
              : {}),
          }
      const response = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Search request failed")
      }

      const data = await response.json()
      if (isConversationalTask) {
        setKisCMessages((messages) => [
          ...messages,
          { role: "operator", text: requestQuery },
          ...(data.clarification ? [{ role: "system" as const, text: data.clarification }] : []),
        ])
        setClarificationAnswer("")
      }

      interface TemporalMatch {
        video_name: string
        frames: number[]
        score: number
        payloads?: Array<{ caption?: string; scene_narrative?: string; timestamp?: number }>
      }

      const normalized = isTemporal
        ? ((data.results || []) as TemporalMatch[]).map((m) => ({
            id: `${m.video_name}:${(m.frames || []).join("-")}`,
            score: m.score,
            payload: {
              source_file: m.video_name,
              timestamp: m.payloads?.[0]?.timestamp ?? 0,
              frame_idx: m.frames?.[0] ?? null,
              caption: (m.payloads || [])
                .map((p, i) => p?.caption || `(step ${i + 1})`)
                .join(" -> "),
              scene_narrative: (m.payloads || [])
                .map((p) => p?.scene_narrative)
                .filter(Boolean)
                .join(" -> "),
              modality: "temporal_sequence",
            },
          }))
        : (data.results || [])
      setResults(normalized)
      if (!isTemporal && data.clarification) setClarification(data.clarification)
    } catch (err) {
      const msg = err instanceof Error ? err.message : "An unexpected error occurred."
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  // -----------------------------------------------------------
  // VBS INTERACTIVE-SESSION ACTIONS (feedback, query-by-example,
  // in-video search, DRES) - Phase D of the batch-to-interactive plan.
  // Each re-runs a search and swaps `results`, same as handleSearch above.
  // -----------------------------------------------------------
  const runSessionSearch = async (endpoint: string, body: any) => {
    setLoading(true)
    setError(null)
    setExpandedIndex(null)
    try {
      const response = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Request failed")
      }
      const data = await response.json()
      setResults(data.results || [])
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.")
    } finally {
      setLoading(false)
    }
  }

  const handleFeedback = (id: string, positive: boolean) => {
    runSessionSearch("/api/feedback", positive ? { positive_ids: [id] } : { negative_ids: [id] })
  }

  const handleUseAsQuery = (id: string) => {
    runSessionSearch("/api/query-by-example", { point_id: id })
  }

  const handleInVideoSearch = async (videoName: string) => {
    if (!query.trim()) return
    await runSessionSearch("/api/in-video-search", { query, video_name: videoName })
  }

  const handleBrowseVideo = (videoName: string) => setBrowsingVideo(videoName)

  const handleDresLogin = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/dres/login`, { method: "POST" })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "DRES login failed")
      }
      setDresLoggedIn(true)
      const taskResponse = await fetch(`${BACKEND_URL}/api/dres/current-task`)
      if (taskResponse.ok) setCurrentTask(await taskResponse.json())
    } catch (err: any) {
      setActionMessage(err.message || "DRES login failed")
    }
  }

  const handleSubmitToDres = async (hit: ResultHit, force = false) => {
    if (!currentTask?.task_id) {
      setActionMessage("Chưa có task DRES hiện tại - đăng nhập DRES trước.")
      return
    }
    try {
      // Payload shape is unverified against a live DRES schema (see
      // dres_client.py's docstring) - this is a best-effort KIS-style
      // {mediaItemName, timestamp} guess, adjust once the real schema
      // for VBS 2027's DRES instance is confirmed.
      const response = await fetch(`${BACKEND_URL}/api/dres/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task_id: currentTask.task_id,
          payload: { mediaItemName: hit.payload.source_file, timestamp: hit.payload.timestamp },
          video_name: hit.payload.source_file,
          force,
        }),
      })
      const data = await response.json()
      if (response.ok) {
        setActionMessage(`Đã nộp: ${JSON.stringify(data)}`)
        return
      }
      // AVS duplicate-video guard (Phase H): 409 is a soft warning, not a
      // hard block - let the operator confirm and force-resubmit if they
      // have a real reason to.
      if (response.status === 409 && data.detail?.warning) {
        if (window.confirm(`${data.detail.warning}\n\nNộp lại?`)) {
          await handleSubmitToDres(hit, true)
        }
        return
      }
      setActionMessage(data.detail?.warning || data.detail || "Nộp thất bại")
    } catch (err: any) {
      setActionMessage(err.message || "Nộp thất bại")
    }
  }

  useEffect(() => {
    if (selectedVideo && videoRef.current) {
      videoRef.current.currentTime = selectedVideo.time
    }
  }, [selectedVideo])
  return (
    <main className="w-full px-3 sm:px-5 lg:px-6 py-4 relative text-left">
      {/* Unified Command Cockpit Deck */}
      <div className="bg-white rounded-xl border border-slate-200/90 shadow-xs mb-4 overflow-hidden relative">
        {loading && <div className="scan-line" />}

        {/* Top Control Strip: Task Mode Selector & DRES Status */}
        <div className="px-3.5 py-2 bg-slate-50/80 border-b border-slate-200/80 flex flex-wrap items-center justify-between gap-2.5">
          {/* Left: Task Mode Selection Pills */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[10px] font-black uppercase tracking-wider text-slate-400 mr-1 hidden sm:inline">
              Task Mode:
            </span>
            {(
              [
                { id: "kis-t", label: "KIS-T", desc: "Text", icon: Target },
                { id: "kis-c", label: "KIS-C", desc: "Chat", icon: MessageCircle },
                { id: "vqa", label: "VQA", desc: "Question", icon: BookOpen },
                { id: "avs", label: "AVS", desc: "Ad-hoc", icon: Layers },
                { id: "kis-v", label: "KIS-V", desc: "Visual Clip", icon: ImageIcon },
              ] as const
            ).map((m) => {
              const active = taskMode === m.id
              const Icon = m.icon
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => handleTaskModeChange(m.id as TaskMode)}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold transition-all ${
                    active
                      ? "bg-indigo-600 text-white shadow-xs"
                      : "bg-white border border-slate-200/90 text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  <span>{m.label}</span>
                  <span className={`text-[10px] font-normal ${active ? "text-indigo-200" : "text-slate-400"}`}>
                    {m.desc}
                  </span>
                </button>
              )
            })}
          </div>
          {/* Right Controls: DRES Status & Action Notification */}
          <div className="flex items-center gap-2 flex-wrap">
            {actionMessage && (
              <span className="text-slate-500 text-xs italic truncate max-w-xs" title={actionMessage}>
                {actionMessage}
              </span>
            )}
            <div className="inline-flex items-center gap-2 px-2.5 py-1 bg-white border border-slate-200 rounded-lg text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  dresLoggedIn ? "bg-emerald-500 animate-pulse" : "bg-slate-300"
                }`}
              />
              <span className="font-semibold text-slate-700">
                {dresLoggedIn
                  ? currentTask?.task_id || "DRES Connected"
                  : "DRES Offline"}
              </span>
              <button
                type="button"
                onClick={handleDresLogin}
                className="text-[10px] font-bold text-indigo-600 hover:underline ml-1"
              >
                {dresLoggedIn ? "Refresh" : "Connect"}
              </button>
            </div>
          </div>
        </div>

        {/* Command Search Bar */}
        <div className="p-3 sm:p-4">
          <form onSubmit={handleSearch} className="flex flex-col md:flex-row gap-2">
            {!isVisualTask && (
              <div className="flex-1 text-left">
                <div className="relative">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={
                      queryType === 1
                        ? isConversationalTask
                          ? "Start with a rough memory; the system can ask for more detail..."
                          : isAvsTask
                          ? "Describe a visual concept (e.g. cars in front of trees...)"
                          : "Describe the target shot (e.g. a motorbike riding through rain...)"
                        : queryType === 2
                        ? "Ask about the video (e.g. what is the license plate of the red car?)"
                        : "Describe the sequence (e.g. a motorbike passes, then a red car...)"
                    }
                    className="w-full bg-slate-50/50 hover:bg-white focus:bg-white border border-slate-200 rounded-lg pl-9 pr-4 py-2 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all font-medium"
                  />
                  <SearchIcon className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
                </div>
              </div>
            )}

              {isVisualTask && (
                <div className="w-full md:flex-1 min-w-0 text-left">
                  <input
                    ref={videoUploadRef}
                    type="file"
                    accept="video/*,image/*"
                    className="hidden"
                    onChange={(event) => acceptKisVideo(event.target.files?.[0])}
                  />
                  <div
                    className={`vbs-video-dropzone ${videoDropActive ? "vbs-video-dropzone-active" : ""} ${kisVideoFile ? "vbs-video-dropzone-ready" : ""}`}
                    role="button"
                    tabIndex={0}
                    onClick={() => videoUploadRef.current?.click()}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") videoUploadRef.current?.click()
                    }}
                    onDragEnter={(event) => {
                      event.preventDefault()
                      setVideoDropActive(true)
                    }}
                    onDragOver={(event) => event.preventDefault()}
                    onDragLeave={() => setVideoDropActive(false)}
                    onDrop={(event) => {
                      event.preventDefault()
                      setVideoDropActive(false)
                      acceptKisVideo(event.dataTransfer.files?.[0])
                    }}
                  >
                    <div className="vbs-video-drop-icon">
                      {kisVideoFile ? <Film className="h-4 w-4" /> : <UploadCloud className="h-4 w-4" />}
                    </div>
                    <div className="min-w-0 flex flex-wrap items-baseline gap-x-2 gap-y-0">
                      <p className="text-xs font-extrabold text-slate-800 truncate m-0">
                        {kisVideoFile ? kisVideoFile.name : "Drop KIS-V video clip or keyframe image here"}
                      </p>
                      <span className="text-[11px] text-slate-500 font-medium whitespace-nowrap">
                        {kisVideoFile ? `${(kisVideoFile.size / (1024 * 1024)).toFixed(1)} MB` : "or browse (MP4, JPG, PNG)"}
                      </span>
                    </div>
                    {kisVideoFile && (
                      <button
                        type="button"
                        className="vbs-clear-file"
                        aria-label="Remove selected clip"
                        onClick={(event) => {
                          event.stopPropagation()
                          setKisVideoFile(null)
                        }}
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                </div>
              )}

              <button
                type="submit"
                disabled={loading || (isVisualTask && !kisVideoFile)}
                className="inline-flex items-center justify-center gap-1.5 px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs rounded-lg transition-all shadow-sm shadow-indigo-600/20 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
              >
                {loading ? (
                  <>
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    <span>Searching...</span>
                  </>
                ) : (
                  <>
                    <SearchIcon className="h-3.5 w-3.5" />
                    <span>Search</span>
                  </>
                )}
              </button>
            </form>
          </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-3.5 py-2 rounded-lg mb-4 flex items-center gap-2.5 text-xs font-semibold text-left">
          <AlertCircle className="h-4 w-4 text-red-500 shrink-0" />
          <div>{error}</div>
        </div>
      )}

          {/* KIS-C Conversation / Clarification Box */}
          {isConversationalTask && (kisCMessages.length > 0 || clarification) && (
            <div className="vbs-conversation-card mb-4 text-left p-3.5 rounded-xl border border-indigo-100 bg-white shadow-2xs">
              <div className="flex items-center gap-2 mb-2.5">
                <div className="vbs-conversation-icon"><MessageCircle className="h-3.5 w-3.5" /></div>
                <div>
                  <p className="text-[11px] font-extrabold text-slate-800 uppercase tracking-wider m-0">KIS-C Clarification Dialogue</p>
                  <p className="text-[11px] text-slate-500 m-0">Clarify vague memory attributes to resolve ambiguous candidate pools.</p>
                </div>
              </div>
              <div className="space-y-1.5 mb-3 max-h-36 overflow-y-auto">
                {kisCMessages.map((message, index) => (
                  <div key={`${message.role}-${index}`} className={`vbs-chat-bubble ${message.role === "operator" ? "vbs-chat-bubble-operator" : "vbs-chat-bubble-system"}`}>
                    <span className="vbs-chat-label">{message.role === "operator" ? "You" : "System"}</span>
                    <span className="text-xs">{message.text}</span>
                  </div>
                ))}
              </div>
              {clarification && (
                <div className="vbs-clarification-form pt-2 border-t border-slate-100">
                  <div className="flex gap-2">
                    <input
                      id="clarification-answer"
                      type="text"
                      value={clarificationAnswer}
                      onChange={(event) => setClarificationAnswer(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && clarificationAnswer.trim()) handleSearch()
                      }}
                      placeholder="Answer system question to refine candidate ranking..."
                      className="flex-1 text-xs bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 focus:bg-white focus:outline-none focus:border-indigo-500"
                    />
                    <button
                      type="button"
                      disabled={loading || !clarificationAnswer.trim()}
                      onClick={() => handleSearch()}
                      className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-xs font-bold rounded-lg shadow-2xs"
                    >
                      Refine
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Results Summary Bar */}
          {!loading && results.length > 0 && (
            <div className="mb-3 text-left flex justify-between items-center bg-white border border-slate-200/80 px-3.5 py-1.5 rounded-lg shadow-2xs">
              <p className="text-xs text-slate-600 font-semibold m-0 flex items-center gap-1.5">
                <span className="px-1.5 py-0.5 bg-indigo-50 text-indigo-700 font-bold rounded font-mono text-[11px]">
                  {results.length}
                </span>
                <span>candidate shots for: <strong className="text-slate-800 font-medium">"{query}"</strong></span>
              </p>
              <span className="text-[10px] text-slate-400 font-mono">
                Click any thumbnail to inspect high-res frame & evidence
              </span>
            </div>
          )}

          {/* Loading Skeletons */}
          {loading && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8 gap-2.5">
              {Array.from({ length: 16 }).map((_, n) => (
                <div key={n} className="bg-white rounded-lg border border-slate-200/80 overflow-hidden animate-pulse">
                  <div className="aspect-video bg-slate-200" />
                  <div className="p-2 space-y-1.5">
                    <div className="h-3 bg-slate-200 rounded w-3/4" />
                    <div className="h-2.5 bg-slate-100 rounded w-1/2" />
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Results Gallery / VQA Dedicated Workspace */}
          {!loading && results.length > 0 && (
            taskMode === "vqa" ? (
              <VQAWorkspace
                query={query}
                results={results}
                onPlay={(name, time, frameIdx) => setSelectedVideo({ name, time, frameIdx })}
                onInspect={(h, i) => setInspectingHit({ hit: h, rank: i })}
                onSubmitToDres={handleSubmitToDres}
                onFeedback={handleFeedback}
                onUseAsQuery={handleUseAsQuery}
                onInVideoSearch={handleInVideoSearch}
                onBrowseVideo={handleBrowseVideo}
              />
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8 gap-2.5">
                {results.map((hit, idx) => (
                  <ResultCard
                    key={hit.id || idx}
                    hit={hit}
                    idx={idx}
                    onInspect={(h, i) => setInspectingHit({ hit: h, rank: i + 1 })}
                    onPlay={(name, time, frameIdx) => setSelectedVideo({ name, time, frameIdx })}
                    onFeedback={handleFeedback}
                    onUseAsQuery={handleUseAsQuery}
                    onInVideoSearch={handleInVideoSearch}
                    onBrowseVideo={handleBrowseVideo}
                    onSubmitToDres={handleSubmitToDres}
                  />
                ))}
              </div>
            )
          )}

          {/* Empty States */}
          {!loading && results.length === 0 && query && (
            <Card className="tech-card py-12 text-center max-w-md mx-auto border-slate-200 bg-white">
              <CardContent className="space-y-3">
                <div className="bg-slate-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto border border-slate-200">
                  <SearchIcon className="h-5 w-5 text-slate-400" />
                </div>
                <h3 className="text-base font-bold text-slate-800">No candidate shots found</h3>
                <p className="text-slate-500 text-xs">
                  Try a more concrete description, a different task mode, or use visual search for KIS-V.
                </p>
              </CardContent>
            </Card>
          )}
      {/* High-Resolution Candidate Inspection Dialog */}
      <CandidateInspectionDialog
        hit={inspectingHit?.hit ?? null}
        rank={inspectingHit?.rank ?? null}
        open={Boolean(inspectingHit)}
        onClose={() => setInspectingHit(null)}
        onPlay={(name, time, frameIdx) => setSelectedVideo({ name, time, frameIdx })}
        onFeedback={handleFeedback}
        onUseAsQuery={handleUseAsQuery}
        onInVideoSearch={handleInVideoSearch}
        onBrowseVideo={handleBrowseVideo}
        onSubmitToDres={handleSubmitToDres}
      />

      {/* Video Modal Player */}
      {selectedVideo && (
        <Dialog open={!!selectedVideo} onOpenChange={(open) => !open && setSelectedVideo(null)}>
          <DialogContent className="border-slate-200 bg-white p-0 overflow-hidden shadow-2xl rounded-xl">
            <DialogHeader className="p-6 pb-2">
              <DialogTitle className="text-slate-800 truncate pr-6">{selectedVideo.name}</DialogTitle>
              <DialogDescription className="text-slate-500">
                Playing starting at timestamp: <span className="text-indigo-600 font-semibold">{selectedVideo.time.toFixed(2)}s</span>
                {selectedVideo.frameIdx != null && (
                  <> · frame <span className="text-indigo-600 font-semibold">{selectedVideo.frameIdx}</span></>
                )}
              </DialogDescription>
            </DialogHeader>

            <div className="aspect-video bg-black relative flex items-center justify-center w-full">
              <video
                ref={videoRef}
                controls
                autoPlay
                className="w-full h-full object-contain"
                src={`${BACKEND_URL}/api/media/video/${encodeURIComponent(selectedVideo.name)}`}
              />
            </div>

            <div className="p-6 bg-slate-50 border-t border-slate-100 flex justify-end">
              <button
                onClick={() => setSelectedVideo(null)}
                className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg text-sm font-semibold transition-colors"
              >
                Close Media Player
              </button>
            </div>
          </DialogContent>
        </Dialog>
      )}

      <BrowseVideoDialog
        videoName={browsingVideo}
        onClose={() => setBrowsingVideo(null)}
        onPlayFrame={(name, time, frameIdx) => {
          setBrowsingVideo(null)
          setSelectedVideo({ name, time, frameIdx })
        }}
      />
    </main>
  )
}

// -------------------------------------------------------------
// VIEW 3: DATABASE VIEW
// -------------------------------------------------------------
function DatabaseView() {
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  const fetchStats = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/status`)
      const data = await response.json()
      setStats(data.qdrant)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStats()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[500px]">
        <RefreshCw className="h-8 w-8 text-indigo-500 animate-spin" />
      </div>
    )
  }

  const isConnected = stats?.status === "connected"

  return (
    <div className="w-full px-4 sm:px-6 lg:px-8 py-6">
      <Card className="tech-card text-left bg-white/90">
        <CardHeader className="border-b border-slate-100 pb-6 flex flex-row items-center justify-between">
          <div>
            <CardTitle className="text-slate-800 text-lg font-bold">Vector Collection Statistics</CardTitle>
            <CardDescription className="text-xs font-medium">
              Detailed breakdown of the multimedia vectors available to the VBS search console.
            </CardDescription>
          </div>
          <Badge variant={isConnected ? "success" : "destructive"} className="text-xs uppercase font-bold px-3 py-1">
            {isConnected ? "Active" : "Offline"}
          </Badge>
        </CardHeader>
        <CardContent className="pt-6 space-y-6">
          {isConnected ? (
            <div className="space-y-6">
              {/* Summary Metrics Row */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="bg-slate-50 border border-slate-200/80 p-4 rounded-xl text-left">
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                    Total Ingested Records
                  </div>
                  <div className="text-2xl font-extrabold text-indigo-600 mt-1 font-mono">
                    {(stats.total_points || stats.visual_points || 0).toLocaleString()}
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">Across all Qdrant collections</div>
                </div>

                <div className="bg-slate-50 border border-slate-200/80 p-4 rounded-xl text-left">
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                    Qdrant Instance Status
                  </div>
                  <div className="text-2xl font-extrabold text-emerald-600 mt-1 flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
                    Online
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1 font-mono">{stats.host || "localhost"}:{stats.port || 6333}</div>
                </div>

                <div className="bg-slate-50 border border-slate-200/80 p-4 rounded-xl text-left">
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                    Indexed Collections
                  </div>
                  <div className="text-2xl font-extrabold text-slate-800 mt-1 font-mono">
                    {stats.collections?.length || 1}
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">Active HNSW vector graphs</div>
                </div>
              </div>

              {/* Collections Cards Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {(stats.collections && stats.collections.length > 0 ? stats.collections : [
                  {
                    name: "visual_keyframes_v1",
                    points: stats.visual_points || 66499,
                    indexed: 64689,
                    dim: 2048,
                    distance: "Cosine",
                    status: "green",
                  }
                ]).map((col: { name: string; points?: number; indexed?: number; dim?: number; distance?: string; status?: string }, idx: number) => (
                  <div
                    key={col.name || idx}
                    className="bg-white border border-slate-200/90 hover:border-indigo-400 p-5 rounded-xl text-left space-y-3 transition shadow-xs"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                        Collection #{idx + 1}
                      </span>
                      <span className="inline-flex items-center gap-1 text-[10px] font-bold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full capitalize">
                        {col.status || "Ready"}
                      </span>
                    </div>

                    <div>
                      <h3 className="text-base font-bold text-slate-900 font-mono truncate" title={col.name}>
                        {col.name}
                      </h3>
                      <div className="text-2xl font-black text-indigo-600 mt-1 font-mono">
                        {(col.points || 0).toLocaleString()} <span className="text-xs font-semibold text-slate-500">records</span>
                      </div>
                    </div>

                    <div className="pt-2.5 border-t border-slate-100 flex flex-col gap-1 text-xs text-slate-500">
                      <div className="flex justify-between">
                        <span>Embedding Dimension:</span>
                        <strong className="font-mono text-slate-700">{col.dim || 2048}d (WeMM-4B)</strong>
                      </div>
                      <div className="flex justify-between">
                        <span>Distance Metric:</span>
                        <strong className="font-mono text-slate-700">{col.distance || "Cosine"}</strong>
                      </div>
                      {col.indexed != null && (
                        <div className="flex justify-between">
                          <span>Indexed HNSW Vectors:</span>
                          <strong className="font-mono text-indigo-600">{(col.indexed || 0).toLocaleString()}</strong>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Schema description */}
              <div className="bg-indigo-50/40 border border-indigo-100/60 p-5 rounded-xl space-y-2 text-left">
                <h4 className="text-slate-900 font-bold text-sm flex items-center gap-2">
                  <Sliders className="h-4 w-4 text-indigo-600" />
                  Ingested Metadata Schema Properties
                </h4>
                <p className="text-xs text-slate-600 leading-relaxed font-medium m-0">
                  The <code className="text-indigo-600 bg-white border border-indigo-100 px-1.5 py-0.5 rounded font-mono font-bold">visual_keyframes_v1</code> collection indexes video shots using 2048-dimensional dense representations (Tencent WeMM-Embedding-4B with Matryoshka Representation Learning). Indexed payloads carry source video paths, native frame indices, timestamps, PP-OCRv6 normalized on-screen text, YOLOE-26 detected objects, Whisper speech transcripts, and dense scene narratives.
                </p>
              </div>
            </div>
          ) : (
            <div className="py-12 text-center text-slate-400 space-y-2 p-6 border border-dashed border-red-200 bg-red-50/20 rounded-xl">
              <AlertCircle className="h-10 w-10 text-red-500 mx-auto" />
              <p className="font-bold text-red-700">Database Offline</p>
              <p className="text-xs">Ensure Qdrant is hosted on port <code className="text-slate-650 bg-slate-100 px-1 rounded">6333</code>. Run the host script inside preprocessing.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// -------------------------------------------------------------
// MAIN APPLICATION ROOT
// -------------------------------------------------------------
function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col justify-between selection:bg-indigo-500/20 text-slate-800">
        <div>
          <Navbar />
          <Routes>
            <Route path="/" element={<SearchView />} />
            <Route path="/benchmark" element={<RAGBenchmarkWorkspace />} />
            <Route path="/audit" element={<VBSAuditWorkspace />} />
            <Route path="/history" element={<AuditHistoryView />} />
            <Route path="/database" element={<DatabaseView />} />
          </Routes>
        </div>
        <footer className="py-6 text-center text-xs text-slate-400 border-t border-slate-200 bg-white/60 mt-12">
          <p className="font-semibold">Video Browser Showdown 2027 &bull; The Gays Lead The World &bull; RMIT University Vietnam</p>
          <p className="mt-1 text-[11px] text-slate-400">Interactive video retrieval &bull; DRES-enabled research system</p>
        </footer>
      </div>
    </Router>
  )
}

export default App
