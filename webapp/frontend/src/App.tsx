import React, { useState, useEffect, useRef } from "react"
import { HashRouter as Router, Routes, Route, Link, useLocation } from "react-router-dom"
import {
  Search as SearchIcon,
  Database,
  Cpu,
  RefreshCw,
  Sliders,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Terminal,
  FileCode,
  FolderOpen,
  Layers,
  PlayCircle,
  HelpCircle,
  Image as ImageIcon,
  Video,
  Radio,
  Timer,
  Target,
  BookOpen,
  UploadCloud,
  Film,
  MessageCircle,
  X,
} from "lucide-react"

import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select"
import { ResultCard, type ResultHit } from "@/components/ResultCard"
import { BrowseVideoDialog } from "@/components/BrowseVideoDialog"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

// -------------------------------------------------------------
// NAVIGATION COMPONENT
// -------------------------------------------------------------
function Navbar() {
  const location = useLocation()
  
  const navItems = [
    { path: "/", label: "Live Search", icon: SearchIcon },
    { path: "/preprocess", label: "Indexing Lab", icon: Cpu },
    { path: "/database", label: "Vector Store", icon: Database }
  ]

  return (
    <nav className="vbs-nav sticky top-0 z-40">
      <div className="max-w-[1600px] mx-auto px-5 sm:px-8 py-3 flex items-center justify-between gap-6">
        <Link to="/" className="flex items-center gap-3 min-w-0 group">
          <div className="vbs-mark">
            <Video className="h-5 w-5" />
            <span className="vbs-mark-dot" />
          </div>
          <div className="min-w-0 text-left">
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-base sm:text-lg font-extrabold tracking-tight text-slate-900 m-0">
                The Gays Lead The World
              </h1>
              <span className="vbs-badge">VBS 2027</span>
            </div>
            <p className="hidden sm:block text-[11px] text-slate-500 font-semibold tracking-wide uppercase mt-0.5">
              Interactive video retrieval console
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
  const [activeTab, setActiveTab] = useState<"single" | "batch">("single")
  
  // Single Query states
  const [query, setQuery] = useState("")
  const [taskMode, setTaskMode] = useState<TaskMode>("kis-t")
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null)
  // KIS-C clarification question (Phase L) - set when /api/search's Type 1
  // flow detects an ambiguous result set and asks a narrowing question.
  const [clarification, setClarification] = useState<string | null>(null)
  const [clarificationAnswer, setClarificationAnswer] = useState("")
  const [kisCMessages, setKisCMessages] = useState<Array<{ role: "operator" | "system"; text: string }>>([])
  
  // Batch Query states
  const [batchFiles, setBatchFiles] = useState<any[]>([])
  const [batchRunning, setBatchRunning] = useState(false)
  const [batchLogs, setBatchLogs] = useState<string[]>([])
  const [batchResults, setBatchResults] = useState<any[]>([])
  const [batchError, setBatchError] = useState<string | null>(null)
  const batchLogContainerRef = useRef<HTMLDivElement>(null)
  
  // Video player modal state
  const [selectedVideo, setSelectedVideo] = useState<{name: string, time: number} | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)

  // KIS-V clip upload
  const videoUploadRef = useRef<HTMLInputElement>(null)
  const [kisVideoFile, setKisVideoFile] = useState<File | null>(null)
  const [videoDropActive, setVideoDropActive] = useState(false)

  // VBS interactive-session states (Phase C backend / Phase D UI of the
  // batch-to-interactive plan): temporal query mode, video-browse dialog,
  // and DRES login/current-task so "Nộp câu trả lời" has a task_id to
  // submit against.
  const [temporalMode, setTemporalMode] = useState(false)
  // Steps after the first ("query" itself is step 1) - generalizes the old
  // fixed 2-query temporal search into an N-step chain (Exquisitor-inspired
  // sequence-chain matching, see HybridSearcher.temporal_chain_match).
  const [extraQueries, setExtraQueries] = useState<string[]>([""])
  const addTemporalStep = () => setExtraQueries((prev) => (prev.length < 4 ? [...prev, ""] : prev))
  const removeTemporalStep = (idx: number) =>
    setExtraQueries((prev) => (prev.length > 1 ? prev.filter((_, i) => i !== idx) : prev))
  const updateTemporalStep = (idx: number, value: string) =>
    setExtraQueries((prev) => prev.map((v, i) => (i === idx ? value : v)))

  // Escalate-precision-on-demand (Phase I, U-Cker/PraK-inspired): both
  // default off so the server's config defaults (fast/no-verify) apply -
  // an operator ticks these on mid-task when stuck on a hard query.
  const [exactSearch, setExactSearch] = useState(false)
  const [verifyResults, setVerifyResults] = useState(false)
  const [browsingVideo, setBrowsingVideo] = useState<string | null>(null)
  const [dresLoggedIn, setDresLoggedIn] = useState(false)
  const [currentTask, setCurrentTask] = useState<any>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)

  const queryType = taskMode === "vqa" ? 2 : 1
  const isVisualTask = taskMode === "kis-v"
  const isConversationalTask = taskMode === "kis-c"
  const isAvsTask = taskMode === "avs"

  const clearConversationState = () => {
    setResults([])
    setError(null)
    setClarification(null)
    setClarificationAnswer("")
    setKisCMessages([])
    setExpandedIndex(null)
  }

  // One backend process serves a whole VBS session of consecutive tasks, and
  // the backend's CQR history/Rocchio vector are global to that process - so
  // clearing only React state would leave the NEXT task's first query being
  // rewritten against the previous task's turns. Always clear both sides.
  const handleResetSession = async (notify = true) => {
    clearConversationState()
    try {
      const response = await fetch(`${BACKEND_URL}/api/session/reset`, { method: "POST" })
      if (!response.ok) throw new Error("reset failed")
      const data = await response.json()
      if (notify) setActionMessage(`Đã reset phiên - xoá ${data.cleared_turns} lượt hội thoại.`)
    } catch {
      setError("Không reset được phiên trên backend - lịch sử CQR có thể còn sót lại từ task trước.")
    }
  }

  const handleTaskModeChange = (mode: TaskMode) => {
    setTaskMode(mode)
    setTemporalMode(false)
    if (mode !== "kis-v") setKisVideoFile(null)
    // Switching task mode is a task boundary in practice - reset the backend
    // session too. Not sufficient on its own (two consecutive KIS-C tasks
    // never fire this), which is why the explicit button exists as well.
    handleResetSession(false)
  }

  const acceptKisVideo = (file: File | undefined) => {
    if (!file) return
    const looksLikeVideo = file.type.startsWith("video/") || /\.(mp4|mov|webm|mkv|avi)$/i.test(file.name)
    if (!looksLikeVideo) {
      setError("KIS-V cần một file video (MP4, MOV, WebM hoặc định dạng video tương tự).")
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
      const formData = new FormData()
      formData.append("file", kisVideoFile)
      const response = await fetch(`${BACKEND_URL}/api/search-by-video`, {
        method: "POST",
        body: formData,
      })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Visual clip search failed")
      }
      const data = await response.json()
      setResults(data.results || [])
    } catch (err: any) {
      setError(err.message || "Visual clip search failed")
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
    if (temporalMode && extraQueries.some((q) => !q.trim())) return
    if (isConversationalTask && clarification && !clarificationAnswer.trim()) return

    // Single line on purpose: this string lands in the CQR prompt's
    // "Latest Query:" slot, and a newline split it into a phantom extra field
    // that matches none of the few-shot examples. Keep the wording in sync
    // with CQR_FEWSHOT_EXAMPLES' Example 4, which demonstrates exactly this
    // shape so the model pattern-matches the live slot.
    const requestQuery = isConversationalTask && clarificationAnswer.trim()
      ? `${query}. Additional detail from operator: ${clarificationAnswer.trim()}`
      : query

    setLoading(true)
    setError(null)
    setResults([])
    setExpandedIndex(null)
    setClarification(null)

    try {
      const endpoint = temporalMode ? "/api/temporal-search" : "/api/search"
      const body = temporalMode
        ? { queries: [query, ...extraQueries] }
        : {
            type: queryType,
            query: requestQuery,
            ...(exactSearch ? { exact: true } : {}),
            ...(verifyResults ? { verify: true } : {}),
            // KIS-C: sent separately from `query` (which still carries it
            // appended, for retrieval) so the backend can boost the exact
            // candidates the clarifying question was about.
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
      // /api/temporal-search returns {video_name, frames, payloads} per
      // match (one entry per chain step) instead of the usual
      // {id, score, payload} shape - normalize into the same ResultHit
      // shape ResultCard expects (using the first step's payload as the
      // displayed frame, joining every step's caption to show the whole
      // chain) rather than building a second rendering path just for this
      // one mode.
      const normalized = temporalMode
        ? (data.results || []).map((m: any) => ({
            id: `${m.video_name}:${(m.frames || []).join("-")}`,
            score: m.score,
            payload: {
              ...(m.payloads?.[0] || {}),
              caption: (m.payloads || [])
                .map((p: any, i: number) => p?.caption || `(sự kiện ${i + 1})`)
                .join(" → "),
            },
          }))
        : (data.results || [])
      setResults(normalized)
      if (!temporalMode && data.clarification) setClarification(data.clarification)
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.")
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

  // -----------------------------------------------------------
  // BATCH QUERIES INFERENCE
  // -----------------------------------------------------------
  const fetchBatchStatus = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/batch/status`)
      const data = await response.json()
      setBatchFiles(data.files || [])
      setBatchRunning(data.running)
    } catch (err) {
      console.error(err)
    }
  }

  const fetchBatchLogs = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/batch/logs`)
      const data = await response.json()
      setBatchLogs(data.logs || [])
      setBatchRunning(data.running)
    } catch (err) {
      console.error(err)
    }
  }

  const fetchBatchResults = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/batch/results`)
      if (response.ok) {
        const data = await response.json()
        setBatchResults(data || [])
      }
    } catch (err) {
      console.error("Batch results file not generated yet.")
    }
  }

  const triggerBatchQuery = async () => {
    setBatchRunning(true)
    setBatchError(null)
    setBatchResults([])
    try {
      const response = await fetch(`${BACKEND_URL}/api/batch/run`, { method: "POST" })
      const data = await response.json()
      console.log(data)
      fetchBatchLogs()
    } catch (err: any) {
      setBatchError(err.message || "Failed to trigger batch processing.")
      setBatchRunning(false)
    }
  }

  useEffect(() => {
    if (activeTab === "batch") {
      fetchBatchStatus()
      fetchBatchResults()
    }
  }, [activeTab])

  useEffect(() => {
    let interval: any
    if (batchRunning) {
      interval = setInterval(() => {
        fetchBatchLogs()
        fetchBatchStatus()
      }, 2000)
    } else {
      fetchBatchLogs()
      fetchBatchResults()
    }
    return () => clearInterval(interval)
  }, [batchRunning])

  useEffect(() => {
    if (batchLogContainerRef.current) {
      batchLogContainerRef.current.scrollTop = batchLogContainerRef.current.scrollHeight
    }
  }, [batchLogs])

  useEffect(() => {
    if (selectedVideo && videoRef.current) {
      videoRef.current.currentTime = selectedVideo.time
    }
  }, [selectedVideo])

  // Helper to parse batch result string to play in video modal
  const parseBatchResult = (resStr: string) => {
    if (!resStr || resStr === "N/A" || resStr.startsWith("Error")) return null
    
    // Check if result has video and timestamp format: "video1.mp4, 12.50"
    const parts = resStr.split(",").map(p => p.trim())
    if (parts.length >= 2) {
      const video = parts[0]
      const time = parseFloat(parts[1])
      if (video && !isNaN(time)) {
        return { video, time, answer: parts.slice(2).join(", ") }
      }
    }
    return null
  }

  return (
    <main className="max-w-[1440px] mx-auto px-5 sm:px-8 py-8 relative">
      <section className="vbs-hero mb-7">
        <div className="relative z-10 max-w-3xl text-left">
          <div className="vbs-eyebrow"><Radio className="h-3.5 w-3.5" /> VBS 2027 · live retrieval session</div>
          <h2 className="vbs-hero-title">Find the right shot<br className="hidden sm:block" /> before the clock runs out.</h2>
          <p className="vbs-hero-copy">
            Search, inspect, refine, and submit video moments with a human-in-the-loop workflow built for the Video Browser Showdown.
          </p>
        </div>
        <div className="vbs-hero-meta relative z-10">
          <div className="vbs-hero-meta-label"><Timer className="h-4 w-4" /> Competition mindset</div>
          <div className="flex flex-wrap gap-2 mt-3">
            <span className="vbs-task-chip"><Target className="h-3.5 w-3.5" /> KIS-T</span>
            <span className="vbs-task-chip"><Target className="h-3.5 w-3.5" /> KIS-C</span>
            <span className="vbs-task-chip"><ImageIcon className="h-3.5 w-3.5" /> KIS-V</span>
            <span className="vbs-task-chip"><BookOpen className="h-3.5 w-3.5" /> VQA</span>
            <span className="vbs-task-chip"><Layers className="h-3.5 w-3.5" /> AVS</span>
          </div>
          <p className="text-xs text-slate-500 font-medium mt-3 max-w-xs leading-relaxed">
            Every result is a candidate shot. Verify it visually before sending a DRES submission.
          </p>
        </div>
      </section>
      
      {/* Subnavigation Tabs */}
      <div className="vbs-tabs mb-6" role="tablist" aria-label="Search workspace">
        <button
          onClick={() => setActiveTab("single")}
          className={`vbs-tab ${activeTab === "single" ? "vbs-tab-active" : ""}`}
          role="tab"
          aria-selected={activeTab === "single"}
        >
          <SearchIcon className="h-4 w-4" />
          Live Search
        </button>
        <button
          onClick={() => setActiveTab("batch")}
          className={`vbs-tab ${activeTab === "batch" ? "vbs-tab-active" : ""}`}
          role="tab"
          aria-selected={activeTab === "batch"}
        >
          <Layers className="h-4 w-4" />
          Batch Evaluation
        </button>
      </div>

      {/* ------------------------------------------------------- */}
      {/* TAB A: SINGLE QUERY CONSOLE */}
      {/* ------------------------------------------------------- */}
      {activeTab === "single" && (
        <>
          {/* DRES panel: login + current task display so "Nộp câu trả lời"
              on each result card has a task_id to submit against. */}
          <Card className="vbs-session-card tech-card mb-4">
            <CardContent className="pt-4 pb-4 flex items-center justify-between gap-4 flex-wrap">
              <div className="flex items-center gap-3 text-left">
                <div className={`vbs-session-icon ${dresLoggedIn ? "vbs-session-icon-live" : ""}`}>
                  <Radio className="h-4 w-4" />
                </div>
                <div>
                  <span className="text-[10px] font-extrabold text-slate-500 uppercase tracking-[0.16em] block mb-0.5">DRES session</span>
                {currentTask ? (
                  <span className="text-sm font-bold text-slate-800">
                    Task loaded · {currentTask.task_id || currentTask.type || JSON.stringify(currentTask)}
                  </span>
                ) : (
                  <span className="text-sm text-slate-500 font-semibold">{dresLoggedIn ? "Connected · no current task" : "Not connected"}</span>
                )}
                </div>
              </div>
              <div className="flex items-center gap-3">
                {actionMessage && <span className="text-xs text-slate-500 max-w-xs truncate" title={actionMessage}>{actionMessage}</span>}
                <button
                  onClick={handleDresLogin}
                  className="vbs-dark-button"
                >
                  {dresLoggedIn ? "Refresh task" : "Connect DRES"}
                </button>
              </div>
            </CardContent>
          </Card>

          <Card className="tech-card border-indigo-100/60 mb-8 overflow-hidden relative bg-white">
            {loading && <div className="scan-line" />}
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 mb-4 text-left">
                <input
                  id="temporal-mode"
                  type="checkbox"
                  checked={temporalMode}
                  onChange={(e) => setTemporalMode(e.target.checked)}
                  className="h-4 w-4"
                />
                  <label htmlFor="temporal-mode" className="text-xs font-bold text-slate-600 uppercase tracking-wider">
                    Temporal chain · nhiều sự kiện nối tiếp
                </label>
              </div>

              {!temporalMode && (
                <div className="flex items-center gap-4 mb-4 text-left">
                  <div className="flex items-center gap-2">
                    <input
                      id="exact-search"
                      type="checkbox"
                      checked={exactSearch}
                      onChange={(e) => setExactSearch(e.target.checked)}
                      className="h-4 w-4"
                    />
                    <label htmlFor="exact-search" className="text-xs font-bold text-slate-600 uppercase tracking-wider">
                      High precision · chậm hơn
                    </label>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      id="verify-results"
                      type="checkbox"
                      checked={verifyResults}
                      onChange={(e) => setVerifyResults(e.target.checked)}
                      className="h-4 w-4"
                    />
                    <label htmlFor="verify-results" className="text-xs font-bold text-slate-600 uppercase tracking-wider">
                      Verify candidates · chậm hơn
                    </label>
                  </div>
                </div>
              )}
              <form onSubmit={handleSearch} className="flex flex-col md:flex-row gap-4">

                {/* Shadcn Select Component */}
                {!temporalMode && (
                <div className="w-full md:w-72 text-left">
                  <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
                    Task mode
                  </label>
                  <Select
                    value={taskMode}
                    onValueChange={(val) => handleTaskModeChange(val as TaskMode)}
                  >
                    <SelectTrigger className="w-full bg-white border border-slate-200 rounded-lg text-slate-800">
                      <SelectValue placeholder="Select method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="kis-t">KIS-T · Textual known-item</SelectItem>
                      <SelectItem value="kis-c">KIS-C · Conversational known-item</SelectItem>
                      <SelectItem value="kis-v">KIS-V · Visual known-item</SelectItem>
                      <SelectItem value="avs">AVS · Ad-hoc video search</SelectItem>
                      <SelectItem value="vqa">VQA · Visual question</SelectItem>
                    </SelectContent>
                  </Select>
                  <button
                    type="button"
                    onClick={() => handleResetSession()}
                    disabled={loading}
                    title="Xoá lịch sử hội thoại trên backend trước khi bắt đầu task mới"
                    className="mt-2 inline-flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-slate-800 disabled:opacity-50"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                    Reset phiên (task mới)
                  </button>
                </div>
                )}

                {!isVisualTask && <div className="flex-1 text-left">
                  <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
                    {temporalMode ? "Sự kiện thứ 1" : queryType === 2 ? "Question / answer prompt" : isConversationalTask ? "What do you remember?" : isAvsTask ? "AVS search description" : "Task description"}
                  </label>
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
                          ? "Ask about the video (e.g. how many people cross the road?)"
                          : "Describe the sequence (e.g. a motorbike passes, then a red car...)"
                      }
                      className="w-full bg-white border border-slate-200 rounded-lg pl-10 pr-4 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                    />
                    <SearchIcon className="absolute left-3.5 top-3.5 h-4 w-4 text-slate-400" />
                  </div>
                </div>}

                {isVisualTask && (
                  <div className="w-full md:flex-1 min-w-0 text-left">
                    <input
                      ref={videoUploadRef}
                      type="file"
                      accept="video/*"
                      className="hidden"
                      onChange={(event) => acceptKisVideo(event.target.files?.[0])}
                    />
                    <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
                      KIS-V clip
                    </label>
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
                        {kisVideoFile ? <Film className="h-5 w-5" /> : <UploadCloud className="h-5 w-5" />}
                      </div>
                      <div className="min-w-0 flex flex-wrap items-baseline gap-x-2 gap-y-0">
                        <p className="text-sm font-extrabold text-slate-800 truncate m-0">
                          {kisVideoFile ? kisVideoFile.name : "Drop the KIS-V clip here"}
                        </p>
                        <span className="text-xs text-slate-500 font-medium whitespace-nowrap">
                          {kisVideoFile ? `${(kisVideoFile.size / (1024 * 1024)).toFixed(1)} MB · ready to search` : "or click to browse · MP4, MOV, WebM"}
                        </span>
                      </div>
                      {kisVideoFile && (
                        <button
                          type="button"
                          className="vbs-clear-file"
                          aria-label="Remove selected KIS-V clip"
                          onClick={(event) => {
                            event.stopPropagation()
                            setKisVideoFile(null)
                          }}
                        >
                          <X className="h-4 w-4" />
                        </button>
                      )}
                    </div>
                  </div>
                )}

                {temporalMode && extraQueries.map((q, idx) => (
                  <div key={idx} className="flex-1 text-left">
                    <label className="flex items-center justify-between text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
                      <span>Sự kiện thứ {idx + 2} (xảy ra sau)</span>
                      {extraQueries.length > 1 && (
                        <button
                          type="button"
                          onClick={() => removeTemporalStep(idx)}
                          className="text-slate-400 hover:text-red-500 normal-case font-semibold"
                        >
                          Bớt bước
                        </button>
                      )}
                    </label>
                    <div className="relative">
                      <input
                        type="text"
                        value={q}
                        onChange={(e) => updateTemporalStep(idx, e.target.value)}
                        placeholder="e.g. ô tô màu đỏ đi qua"
                        className="w-full bg-white border border-slate-200 rounded-lg pl-10 pr-4 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                      />
                      <SearchIcon className="absolute left-3.5 top-3.5 h-4 w-4 text-slate-400" />
                    </div>
                  </div>
                ))}

                {temporalMode && extraQueries.length < 4 && (
                  <div className="flex items-end">
                    <button
                      type="button"
                      onClick={addTemporalStep}
                      className="bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 text-xs font-bold px-3 py-2.5 rounded-lg transition-colors h-[40px]"
                    >
                      + Thêm bước
                    </button>
                  </div>
                )}

                <div className="flex items-end">
                  <button
                    type="submit"
                    disabled={loading || (isVisualTask && !kisVideoFile)}
                    className="w-full md:w-auto bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm px-6 py-2.5 rounded-lg transition-all duration-300 shadow-md shadow-indigo-600/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 h-[40px] tech-glow-button"
                  >
                    {loading ? (
                      <>
                        <RefreshCw className="h-4 w-4 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <SearchIcon className="h-4 w-4" />
                        {isVisualTask ? "Search visual clip" : isAvsTask ? "Find diverse shots" : "Search candidates"}
                      </>
                    )}
                  </button>
                </div>

              </form>
            </CardContent>
          </Card>

          {/* Error Alert */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-8 flex items-center gap-3 text-left">
              <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
              <div className="text-sm font-semibold">{error}</div>
            </div>
          )}

          {isConversationalTask && (kisCMessages.length > 0 || clarification) && (
            <div className="vbs-conversation-card mb-8 text-left">
              <div className="flex items-center gap-2 mb-4">
                <div className="vbs-conversation-icon"><MessageCircle className="h-4 w-4" /></div>
                <div>
                  <p className="text-xs font-extrabold text-slate-800 uppercase tracking-[0.14em] m-0">KIS-C conversation</p>
                  <p className="text-xs text-slate-500 font-medium mt-0.5 m-0">Add one detail at a time when the candidate set is ambiguous.</p>
                </div>
              </div>
              <div className="space-y-2 mb-4">
                {kisCMessages.map((message, index) => (
                  <div key={`${message.role}-${index}`} className={`vbs-chat-bubble ${message.role === "operator" ? "vbs-chat-bubble-operator" : "vbs-chat-bubble-system"}`}>
                    <span className="vbs-chat-label">{message.role === "operator" ? "You" : "System"}</span>
                    <span>{message.text}</span>
                  </div>
                ))}
              </div>
              {clarification && (
                <div className="vbs-clarification-form">
                  <label htmlFor="clarification-answer" className="text-xs font-extrabold text-slate-700 uppercase tracking-wider">
                    Narrow the search
                  </label>
                  <div className="flex flex-col sm:flex-row gap-2 mt-2">
                    <input
                      id="clarification-answer"
                      type="text"
                      value={clarificationAnswer}
                      onChange={(event) => setClarificationAnswer(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && clarificationAnswer.trim()) handleSearch()
                      }}
                      placeholder="Add the missing detail..."
                      className="vbs-clarification-input"
                    />
                    <button
                      type="button"
                      disabled={loading || !clarificationAnswer.trim()}
                      onClick={() => handleSearch()}
                      className="vbs-clarification-button"
                    >
                      Refine search
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* KIS-C clarification fallback for non-conversational task modes. */}
          {clarification && !isConversationalTask && (
            <div className="bg-amber-50 border border-amber-200 text-amber-800 px-4 py-3 rounded-lg mb-8 flex items-center gap-3 text-left">
              <HelpCircle className="h-5 w-5 text-amber-500 flex-shrink-0" />
              <div className="text-sm font-semibold">{clarification}</div>
            </div>
          )}

          {/* Loading Grid */}
          {loading && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[1, 2, 3].map((n) => (
                <Card key={n} className="border-indigo-50/50 bg-white/40 animate-pulse h-96">
                  <div className="bg-slate-200 h-48 rounded-t-xl" />
                  <div className="p-6 space-y-3">
                    <div className="h-4 bg-slate-200 rounded w-2/3" />
                    <div className="h-4 bg-slate-200 rounded w-1/2" />
                    <div className="h-10 bg-slate-200 rounded pt-3" />
                  </div>
                </Card>
              ))}
            </div>
          )}

          {/* Results List */}
          {!loading && results.length > 0 && (
            <div className="mb-6 text-left flex justify-between items-center bg-white/60 border border-slate-200/60 px-4 py-2 rounded-lg">
              <p className="text-sm text-slate-500 font-semibold m-0">
                <span className="text-indigo-600 font-bold">{results.length}</span> candidate shots for: "{query}"
              </p>
              <Badge variant="outline" className="border-indigo-100 text-indigo-600 bg-indigo-50/50">
                FusedDenseSparse
              </Badge>
            </div>
          )}

          {!loading && results.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {results.map((hit, idx) => (
                <ResultCard
                  key={hit.id || idx}
                  hit={hit}
                  idx={idx}
                  isExpanded={expandedIndex === idx}
                  onToggleExpand={() => setExpandedIndex(expandedIndex === idx ? null : idx)}
                  onPlay={(name, time) => setSelectedVideo({ name, time })}
                  onFeedback={handleFeedback}
                  onUseAsQuery={handleUseAsQuery}
                  onInVideoSearch={handleInVideoSearch}
                  onBrowseVideo={handleBrowseVideo}
                  onSubmitToDres={handleSubmitToDres}
                />
              ))}
            </div>
          )}

          {/* Empty States */}
          {!loading && results.length === 0 && query && (
            <Card className="tech-card py-16 text-center max-w-lg mx-auto border-indigo-100/60 bg-white/75">
              <CardContent className="space-y-4">
                <div className="bg-slate-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto border border-slate-200">
                  <SearchIcon className="h-6 w-6 text-slate-400" />
                </div>
                <h3 className="text-lg font-bold text-slate-800">No candidate shots found</h3>
                <p className="text-slate-500 text-sm">
                  Try a more concrete description, a different task mode, or use visual search for KIS-V.
                </p>
              </CardContent>
            </Card>
          )}

          {!loading && results.length === 0 && !query && (
            <div className="text-center py-20 max-w-xl mx-auto space-y-6">
              <div className="inline-flex bg-indigo-600/10 p-4 rounded-full border border-indigo-500/20">
                <Sparkles className="h-10 w-10 text-indigo-600 animate-pulse" />
              </div>
              <h2 className="text-2xl font-extrabold text-slate-800 tracking-tight">
                Ready for the next VBS task
              </h2>
              <p className="text-slate-500 text-sm leading-relaxed font-semibold">
                Search visual frames, spoken speech, and sound environments. Refine the query, inspect the evidence, then submit only when the shot is convincing.
              </p>
            </div>
          )}
        </>
      )}

      {/* ------------------------------------------------------- */}
      {/* TAB B: BATCH QUERIES DASHBOARD */}
      {/* ------------------------------------------------------- */}
      {activeTab === "batch" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 text-left">
          
          {/* Query Files Panel */}
          <div className="lg:col-span-1 space-y-6">
            <Card className="tech-card h-full flex flex-col justify-between bg-white">
              <div>
                <CardHeader>
                  <CardTitle className="text-slate-800 flex items-center gap-2 text-base">
                    <FolderOpen className="h-5 w-5 text-indigo-500" />
                    Query Registry · batch evaluation
                  </CardTitle>
                  <CardDescription className="text-xs font-semibold">
                    Directory storing batch query files inside the <code className="text-indigo-600 font-mono font-bold">queries/</code> folder.
                  </CardDescription>
                </CardHeader>
                
                <CardContent className="space-y-4">
                  {batchFiles.length === 0 ? (
                    <div className="text-slate-400 italic text-center py-10 border border-dashed border-slate-200 rounded-lg">
                      No files found. Please ensure queries/queries.json template is present.
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {batchFiles.map((file, fIdx) => (
                        <div key={fIdx} className="flex justify-between items-center bg-slate-50 border border-slate-100 p-3 rounded-lg">
                          <div>
                            <span className="text-sm font-bold text-slate-800 block truncate">{file.name}</span>
                            <span className="text-xs text-slate-500 font-semibold">{file.size_kb} KB</span>
                          </div>
                          <Badge variant="secondary" className="bg-slate-200 text-slate-700 text-[10px] uppercase font-bold">
                            {file.name.split('.').pop()}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="bg-indigo-50/50 border border-indigo-100 p-4 rounded-xl space-y-2 mt-4">
                    <h5 className="text-xs font-bold text-indigo-950 uppercase flex items-center gap-1.5">
                      <Sliders className="h-3.5 w-3.5 text-indigo-650" />
                      Usage Instruction
                    </h5>
                    <p className="text-xs text-indigo-900 leading-relaxed font-semibold">
                      To run a custom batch evaluation: place your list of queries in <code className="text-indigo-600 bg-indigo-100/50 px-1 rounded">queries/queries.json</code>, then start the evaluation run.
                    </p>
                  </div>
                </CardContent>
              </div>

              <CardFooter className="pt-4 border-t border-slate-100 bg-slate-50/50">
                <button
                  onClick={triggerBatchQuery}
                  disabled={batchRunning || batchFiles.length === 0}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-2.5 rounded-lg text-sm shadow-md shadow-indigo-600/20 disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2 tech-glow-button"
                >
                  {batchRunning ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Running Inference
                    </>
                  ) : (
                    <>
                      <Cpu className="h-4 w-4" />
                      Run batch evaluation
                    </>
                  )}
                </button>
              </CardFooter>
            </Card>
          </div>

          {/* Running Console & Results */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Terminal Live Output logs */}
            <Card className="tech-card bg-white overflow-hidden relative">
              {batchRunning && <div className="scan-line" />}
              <CardHeader className="border-b border-slate-100 pb-4">
                <CardTitle className="text-slate-800 text-base flex items-center gap-2">
                  <Terminal className="h-5 w-5 text-indigo-500" />
                  Batch Inference Log console
                </CardTitle>
                <CardDescription className="text-xs font-semibold">
                  Streams live console logs of subprocess batch queries execution.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4">
                {batchError && (
                  <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm font-semibold text-red-700">
                    {batchError}
                  </div>
                )}
                <div 
                  ref={batchLogContainerRef}
                  className="bg-slate-900 border border-slate-950 rounded-lg p-4 font-mono text-xs text-slate-200 h-64 overflow-y-auto space-y-1 select-text shadow-inner"
                >
                  {batchLogs.length === 0 ? (
                    <div className="text-slate-500 italic py-20 text-center font-mono">
                      Log console idle. Start batch query to run queries list.
                    </div>
                  ) : (
                    batchLogs.map((log, lIdx) => (
                      <div 
                        key={lIdx} 
                        className={`py-0.5 leading-relaxed break-all ${
                          log.startsWith("ERROR") || log.includes("Error") 
                            ? "text-red-400 font-semibold" 
                            : log.startsWith("---") 
                            ? "text-indigo-400 font-bold" 
                            : "text-emerald-400"
                        }`}
                      >
                        {log}
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Results Grid Table */}
            <Card className="tech-card bg-white">
              <CardHeader className="border-b border-slate-100 pb-4">
                <CardTitle className="text-slate-800 text-base flex items-center justify-between">
                  <span>Batch Output Results</span>
                  {batchResults.length > 0 && (
                    <Badge className="bg-emerald-500 text-white border-none text-[10px] font-bold py-0.5 px-2">
                      Generated
                    </Badge>
                  )}
                </CardTitle>
                <CardDescription className="text-xs font-semibold">
                  List of resolved retrieval outputs matching queries list. Saved to <code className="text-indigo-650">queries/batch_results.json</code>.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4">
                {batchResults.length === 0 ? (
                  <div className="text-slate-400 italic text-center py-20 border border-dashed border-slate-100 rounded-lg text-sm">
                    No results compiled yet. Run batch processing above to generate.
                  </div>
                ) : (
                  <div className="overflow-x-auto border border-slate-200 rounded-lg">
                    <table className="w-full text-sm text-left text-slate-500">
                      <thead className="text-xs text-slate-700 uppercase bg-slate-50 border-b border-slate-200">
                        <tr>
                          <th className="px-4 py-3 font-extrabold w-12">Idx</th>
                          <th className="px-4 py-3 font-extrabold w-24">Type</th>
                          <th className="px-4 py-3 font-extrabold">Query String</th>
                          <th className="px-4 py-3 font-extrabold w-56">Result Output</th>
                          <th className="px-4 py-3 font-extrabold w-16 text-center">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchResults.map((res, rIdx) => {
                          const parsed = parseBatchResult(res.result)
                          return (
                            <tr key={rIdx} className="bg-white border-b border-slate-150 hover:bg-slate-50/50">
                              <td className="px-4 py-3.5 font-bold font-mono text-slate-800">{rIdx + 1}</td>
                              <td className="px-4 py-3.5 font-bold text-slate-650">
                                <Badge variant="outline" className="border-indigo-100 text-indigo-600 bg-indigo-50 font-bold text-[10px] py-0">
                                  Type {res.type}
                                </Badge>
                              </td>
                              <td className="px-4 py-3.5 font-semibold text-slate-700 leading-snug">{res.query}</td>
                              <td className="px-4 py-3.5 font-mono text-xs text-slate-900 font-semibold break-all leading-normal bg-slate-50/40">
                                {res.result}
                              </td>
                              <td className="px-4 py-3.5 text-center">
                                {parsed ? (
                                  <button
                                    onClick={() => setSelectedVideo({ name: parsed.video, time: parsed.time })}
                                    className="bg-indigo-50 hover:bg-indigo-100 text-indigo-650 p-1.5 rounded-lg border border-indigo-100 flex items-center justify-center transition-colors shadow-sm"
                                    title="Play Clip"
                                  >
                                    <PlayCircle className="h-4.5 w-4.5" />
                                  </button>
                                ) : (
                                  <span className="text-slate-350 text-xs font-semibold font-mono">N/A</span>
                                )}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Video Modal Player */}
      {selectedVideo && (
        <Dialog open={!!selectedVideo} onOpenChange={(open) => !open && setSelectedVideo(null)}>
          <DialogContent className="border-slate-200 bg-white p-0 overflow-hidden shadow-2xl rounded-xl">
            <DialogHeader className="p-6 pb-2">
              <DialogTitle className="text-slate-800 truncate pr-6">{selectedVideo.name}</DialogTitle>
              <DialogDescription className="text-slate-500">
                Playing starting at timestamp: <span className="text-indigo-600 font-semibold">{selectedVideo.time.toFixed(2)}s</span>
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
        onPlayFrame={(name, time) => {
          setBrowsingVideo(null)
          setSelectedVideo({ name, time })
        }}
      />
    </main>
  )
}

// -------------------------------------------------------------
// VIEW 2: PREPROCESS DASHBOARD
// -------------------------------------------------------------
function PreprocessView() {
  const [statusData, setStatusData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [logs, setLogs] = useState<string[]>([])
  const [preprocessorRunning, setPreprocessorRunning] = useState(false)
  const logContainerRef = useRef<HTMLDivElement>(null)

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/status`)
      const data = await response.json()
      setStatusData(data)
      setPreprocessorRunning(data.preprocessing_active)
    } catch (err) {
      console.error("Error fetching status:", err)
    } finally {
      setLoading(false)
    }
  }

  const fetchLogs = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/preprocess/logs`)
      const data = await response.json()
      setLogs(data.logs || [])
      setPreprocessorRunning(data.running)
    } catch (err) {
      console.error("Error fetching logs:", err)
    }
  }

  const triggerPreprocessing = async () => {
    try {
      setPreprocessorRunning(true)
      const response = await fetch(`${BACKEND_URL}/api/preprocess/run`, { method: "POST" })
      const data = await response.json()
      console.log("Triggered pipeline:", data)
      fetchLogs()
    } catch (err) {
      console.error("Error triggering pipeline:", err)
      setPreprocessorRunning(false)
    }
  }

  useEffect(() => {
    fetchStatus()
  }, [])

  useEffect(() => {
    let interval: any
    if (preprocessorRunning) {
      interval = setInterval(() => {
        fetchLogs()
        fetchStatus()
      }, 2000)
    } else {
      fetchLogs()
    }
    return () => clearInterval(interval)
  }, [preprocessorRunning])

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[500px]">
        <RefreshCw className="h-8 w-8 text-indigo-500 animate-spin" />
      </div>
    )
  }

  const datasetFiles = statusData?.dataset_files || []
  const qdrant = statusData?.qdrant || {}

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 relative">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <Card className="tech-card text-left bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-500 font-semibold">Qdrant connection</CardDescription>
            <CardTitle className="text-slate-800 flex items-center gap-2 text-lg font-bold mt-1">
              {qdrant.status === "connected" ? (
                <>
                  <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                  Connected
                </>
              ) : (
                <>
                  <AlertCircle className="h-5 w-5 text-red-500" />
                  Disconnected
                </>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-500 font-semibold">
            {qdrant.status === "connected" ? (
              <p>Running on: <code className="text-indigo-600 bg-indigo-50 border border-indigo-100/50 px-1.5 py-0.5 rounded font-mono">{qdrant.host}:{qdrant.port}</code></p>
            ) : (
              <p className="text-red-500">Could not resolve vector database.</p>
            )}
          </CardContent>
        </Card>

        <Card className="tech-card text-left bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-500 font-semibold">Indexed media</CardDescription>
            <CardTitle className="text-slate-800 text-lg font-bold mt-1 flex items-center gap-2">
              <FolderOpen className="h-5 w-5 text-indigo-500" />
              {datasetFiles.length} files detected
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-500 font-semibold">
            <p>Stored in root folder <code className="text-indigo-600 bg-indigo-50 border border-indigo-100/50 px-1.5 py-0.5 rounded font-mono">datasets/</code>.</p>
          </CardContent>
        </Card>

        <Card className="tech-card text-left bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription className="text-slate-500 font-semibold">Active configuration</CardDescription>
            <CardTitle className="text-slate-800 text-lg font-bold mt-1 flex items-center gap-2">
              <FileCode className="h-5 w-5 text-indigo-500" />
              VLM: {statusData?.vlm_option || "openai"}
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-500 font-semibold">
            <p>Object detector: <code className="text-indigo-600 bg-indigo-50 border border-indigo-100/50 px-1.5 py-0.5 rounded font-mono">{statusData?.detector_option || "local"}</code></p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 space-y-6">
          <Card className="tech-card text-left h-full flex flex-col justify-between bg-white/90">
            <div>
              <CardHeader>
                <CardTitle className="text-slate-800 flex items-center gap-2 text-base">
                  <FolderOpen className="h-5 w-5 text-indigo-500" />
                  Scanned Media files
                </CardTitle>
                <CardDescription className="font-medium text-xs">
                  Files scanned inside the <code className="text-indigo-600 font-mono">datasets/</code> directory.
                </CardDescription>
              </CardHeader>
              <CardContent className="overflow-y-auto max-h-96 pr-2">
                {datasetFiles.length === 0 ? (
                  <div className="py-12 text-center text-slate-400 text-sm border border-dashed border-slate-200 rounded-lg p-6">
                    <p className="font-semibold mb-1">No media files detected</p>
                    <p className="text-xs">Place your raw videos or images in the root folder <code className="text-slate-600">datasets/</code> to process them.</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {datasetFiles.map((file: any, fIdx: number) => (
                      <div key={fIdx} className="flex justify-between items-center bg-slate-50 border border-slate-100 rounded-lg p-3 hover:border-slate-350 transition-colors">
                        <div className="truncate pr-2">
                          <span className="text-sm text-slate-800 font-semibold block truncate" title={file.name}>
                            {file.name}
                          </span>
                          <span className="text-xs text-slate-500 capitalize font-medium">
                            {file.type} &bull; {file.size_mb} MB
                          </span>
                        </div>
                        <Badge variant="outline" className="border-slate-200 bg-white text-slate-500 font-mono text-[10px] shrink-0">
                          {file.name.substring(file.name.lastIndexOf('.'))}
                        </Badge>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </div>
            <CardFooter className="pt-4 border-t border-slate-100 bg-slate-50/50">
              <button
                onClick={fetchStatus}
                className="w-full bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 py-2 rounded-lg text-sm font-semibold transition-colors flex justify-center items-center gap-2 shadow-sm"
              >
                <RefreshCw className="h-4 w-4 text-slate-500" />
                Rescan Folder
              </button>
            </CardFooter>
          </Card>
        </div>

        <div className="lg:col-span-2 space-y-6">
          <Card className="tech-card text-left h-full flex flex-col justify-between bg-white/90 overflow-hidden relative">
            {preprocessorRunning && <div className="scan-line" />}
            <CardHeader className="flex flex-row items-center justify-between gap-4 border-b border-slate-100 pb-4">
              <div>
                <CardTitle className="text-slate-800 flex items-center gap-2 text-base">
                  <Terminal className="h-5 w-5 text-indigo-500" />
                  Indexing Pipeline Controls
                </CardTitle>
                <CardDescription className="text-xs font-medium">
                  Triggers keyframe selection, normalized Unicode OCR indexing, PhoWhisper transcribing, and CLAP ambient audio embeddings.
                </CardDescription>
              </div>
              <div>
                <button
                  onClick={triggerPreprocessing}
                  disabled={preprocessorRunning || datasetFiles.length === 0}
                  className="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm px-4 py-2 rounded-lg transition-colors shadow-md shadow-indigo-600/20 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 tech-glow-button"
                >
                  {preprocessorRunning ? (
                    <><RefreshCw className="h-4 w-4 animate-spin" />Indexing...</>
                  ) : (
                    <><Cpu className="h-4 w-4" />Start Pipeline</>
                  )}
                </button>
              </div>
            </CardHeader>
            <CardContent className="pt-6 flex-1 flex flex-col justify-between">
              {preprocessorRunning && (
                <div className="mb-4 space-y-2 bg-slate-50 border border-slate-100 p-4 rounded-lg">
                  <div className="flex justify-between text-xs font-bold">
                    <span className="text-indigo-600">Extracting descriptors & captions...</span>
                    <span className="text-slate-500">In Progress</span>
                  </div>
                  <Progress value={50} className="h-2" />
                  <p className="text-[10px] text-slate-400 font-semibold">Executing preprocessing python script in the background. Keep this browser open.</p>
                </div>
              )}

              <div className="bg-slate-900 border border-slate-950 rounded-lg p-4 font-mono text-xs text-slate-200 h-96 overflow-y-auto space-y-1 select-text shadow-inner">
                {logs.length === 0 ? (
                  <div className="text-slate-500 italic py-32 text-center font-mono">
                    Console idle. Click "Start Pipeline" to execute indexing.
                  </div>
                ) : (
                  logs.map((log, lIdx) => (
                    <div 
                      key={lIdx} 
                      className={`py-0.5 leading-relaxed break-all ${
                        log.startsWith("ERROR") || log.includes("Error") 
                          ? "text-red-400 font-semibold" 
                          : log.startsWith("---") 
                          ? "text-indigo-400 font-bold" 
                          : "text-emerald-400"
                      }`}
                    >
                      {log}
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
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
    <div className="max-w-4xl mx-auto px-6 py-8">
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
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-slate-50 border border-slate-100 p-5 rounded-xl text-left space-y-2 hover:border-indigo-200 transition-colors">
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                    Collection target
                  </div>
                  <div className="text-base font-bold text-slate-800 font-mono">
                    visual_index
                  </div>
                  <div className="text-2xl font-extrabold text-indigo-600">
                    {stats.visual_points} <span className="text-xs font-semibold text-slate-500">records ingested</span>
                  </div>
                  <div className="text-xs text-slate-400 pt-2 border-t border-slate-200/60 font-semibold">
                    Dim: 1536 (QwenVL8BEmbedder text-visual space)
                  </div>
                </div>

                <div className="bg-slate-50 border border-slate-100 p-5 rounded-xl text-left space-y-2 hover:border-indigo-200 transition-colors">
                  <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                    Collection target
                  </div>
                  <div className="text-base font-bold text-slate-800 font-mono">
                    audio_env_index
                  </div>
                  <div className="text-2xl font-extrabold text-indigo-600">
                    {stats.audio_points} <span className="text-xs font-semibold text-slate-500">records ingested</span>
                  </div>
                  <div className="text-xs text-slate-400 pt-2 border-t border-slate-200/60 font-semibold">
                    Dim: 512 (M2D-CLAP ambient sound space)
                  </div>
                </div>
              </div>

              <div className="bg-indigo-50/30 border border-indigo-100/40 p-5 rounded-xl space-y-3">
                <h4 className="text-slate-800 font-bold text-sm flex items-center gap-1.5">
                  <Sliders className="h-4 w-4 text-indigo-600" />
                  Ingested Metadata Schema Properties
                </h4>
                <p className="text-xs text-slate-500 leading-relaxed font-medium">
                  The <code className="text-indigo-600 bg-indigo-50 border border-indigo-100/50 px-1 rounded font-mono">visual_index</code> maps visual keyframes using dense representations. Indexed metadata payloads contain raw transcripts, NFC-normalized OCR strings, detected object tags (bounding box coords resolved via LocateAnything), and detailed keyframe caption narratives.
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
            <Route path="/preprocess" element={<PreprocessView />} />
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
