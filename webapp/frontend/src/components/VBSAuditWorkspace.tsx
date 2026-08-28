import React, { useEffect, useState, useMemo } from "react"
import {
  Activity,
  CheckCircle2,
  Download,
  Eye,
  Folder,
  Layers,
  Loader2,
  Play,
  RefreshCw,
  Sparkles,
  Zap,
  Film,
  History,
  Terminal,
  Cpu,
} from "lucide-react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { BrowseVideoDialog } from "./BrowseVideoDialog"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

export type VBSTaskType = 1 | 2 | 3 | 4 | 5

export const VBS_TASK_LABELS: Record<VBSTaskType, { label: string; desc: string; badgeClass: string }> = {
  1: { label: "KIS-T", desc: "Textual Known-Item Search", badgeClass: "bg-blue-50 text-blue-700 border-blue-200" },
  2: { label: "VQA", desc: "Video Question Answering", badgeClass: "bg-amber-50 text-amber-700 border-amber-200" },
  3: { label: "KIS-C", desc: "Conversational Search", badgeClass: "bg-purple-50 text-purple-700 border-purple-200" },
  4: { label: "AVS", desc: "Ad-hoc Video Search", badgeClass: "bg-indigo-50 text-indigo-700 border-indigo-200" },
  5: { label: "KIS-V", desc: "Visual Known-Item Search", badgeClass: "bg-teal-50 text-teal-700 border-teal-200" },
}

interface TraceHistoryItem {
  trace_id: string
  query: string
  type: number
  type_name: string
  timestamp: string
  total_latency_ms: number
  results_count: number
  summary?: Record<string, unknown>
}

interface CandidateItem {
  rank: number
  video_id: string
  frame_id: string | number
  answer?: string | null
  src_file?: string | null
  thumbnail_url?: string | null
  is_prior?: boolean
  score?: number
  row: string[]
}

interface TraceStageData {
  stage: string
  title: string
  subtitle: string
  status: "idle" | "running" | "completed" | "error"
  duration_ms?: number
  data?: Record<string, unknown>
}

export const VBSAuditWorkspace: React.FC = () => {
  // Navigation tabs: "trace" (Interactive step-by-step audit) vs "batch" (Manifest runner & Priors)
  const [activeTab, setActiveTab] = useState<"trace" | "batch">("trace")

  // --- Trace Lab States ---
  const [traceQuery, setTraceQuery] = useState<string>("a person riding a red motorcycle on a mountain road in the morning")
  const [traceType, setTraceType] = useState<VBSTaskType>(1)
  const [fastMode, setFastMode] = useState<boolean>(true)
  const [isTracing, setIsTracing] = useState<boolean>(false)
  const [currentTraceId, setCurrentTraceId] = useState<string | null>(null)
  const [traceHistory, setTraceHistory] = useState<TraceHistoryItem[]>([])
  const [activeStageIndex, setActiveStageIndex] = useState<number>(0)
  const [traceResults, setTraceResults] = useState<Record<string, unknown> | null>(null)
  const [traceCandidates, setTraceCandidates] = useState<CandidateItem[]>([])

  // --- Batch & Manifest States ---
  const [folders, setFolders] = useState<{ path: string; name: string; has_queries: boolean }[]>([])
  const [selectedFolder, setSelectedFolder] = useState<string>("queries")
  const [queries, setQueries] = useState<any[]>([])
  const [loadingQueries, setLoadingQueries] = useState<boolean>(false)
  const [selectedQueryId, setSelectedQueryId] = useState<string | null>(null)
  const [queryDetail, setQueryDetail] = useState<any>(null)

  // Video Inspection Modal
  const [browseDialog, setBrowseDialog] = useState<{
    open: boolean
    videoName: string
    initialTimestamp?: number
  }>({
    open: false,
    videoName: "",
    initialTimestamp: 0,
  })

  // 1. Fetch Trace History
  const fetchTraceHistory = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/diagnostics/history`)
      if (res.ok) {
        const data = await res.json()
        setTraceHistory(data.history || [])
      }
    } catch (err) {
      console.error("Failed to load trace history:", err)
    }
  }

  // 2. Fetch Folders
  const fetchFolders = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/vbs-audit/folders`)
      if (res.ok) {
        const data = await res.json()
        setFolders(data.folders || [])
      }
    } catch (err) {
      console.error("Failed to load folders:", err)
    }
  }

  // 3. Fetch Queries
  const fetchQueries = async (folder: string) => {
    setLoadingQueries(true)
    try {
      const res = await fetch(`${BACKEND_URL}/api/vbs-audit/queries?folder=${encodeURIComponent(folder)}`)
      if (res.ok) {
        const data = await res.json()
        const qList = data.queries || []
        setQueries(qList)
        if (qList.length > 0 && (!selectedQueryId || !qList.some((q: any) => q.id === selectedQueryId))) {
          setSelectedQueryId(qList[0].id)
        }
      }
    } catch (err) {
      console.error("Failed to load queries:", err)
    } finally {
      setLoadingQueries(false)
    }
  }

  // 4. Fetch Query Detail
  const fetchQueryDetail = async (folder: string, queryId: string) => {
    try {
      const res = await fetch(
        `${BACKEND_URL}/api/vbs-audit/query-detail?folder=${encodeURIComponent(folder)}&query_id=${encodeURIComponent(queryId)}`
      )
      if (res.ok) {
        const data = await res.json()
        setQueryDetail(data)
      }
    } catch (err) {
      console.error("Failed to load query detail:", err)
    }
  }

  useEffect(() => {
    fetchTraceHistory()
    fetchFolders()
  }, [])

  useEffect(() => {
    if (selectedFolder) {
      fetchQueries(selectedFolder)
    }
  }, [selectedFolder])

  useEffect(() => {
    if (selectedFolder && selectedQueryId) {
      fetchQueryDetail(selectedFolder, selectedQueryId)
    }
  }, [selectedFolder, selectedQueryId])

  // Execute Step-by-Step Trace Run
  const handleRunTrace = async () => {
    if (!traceQuery.trim()) return
    setIsTracing(true)
    setTraceResults(null)
    setTraceCandidates([])

    try {
      const res = await fetch(`${BACKEND_URL}/api/diagnostics/debug-run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: traceQuery,
          type: traceType,
          fast_submission: fastMode,
          top_k: 30,
          include_content: true,
          include_prompts: true,
        }),
      })

      if (res.ok) {
        const data = await res.json()
        setTraceResults(data)
        setCurrentTraceId(data.trace_id || null)

        // Parse candidates from result preview
        const rawResults = (data.result_preview || data.results || []) as Array<Record<string, unknown>>
        const candidates: CandidateItem[] = rawResults.map((r, idx) => {
          const payload = (r.payload || {}) as Record<string, unknown>
          const vId = String(payload.source_file || payload.video_id || r.video_name || "unknown").replace(".mp4", "")
          const fId = payload.frame_idx ?? r.frame_idx ?? 0
          return {
            rank: idx + 1,
            video_id: vId,
            frame_id: fId as number,
            answer: r.answer ? String(r.answer) : null,
            src_file: `${vId}.mp4`,
            thumbnail_url: `/api/media/${vId}.mp4`,
            score: typeof r.score === "number" ? r.score : undefined,
            row: [vId, String(fId)],
          }
        })
        setTraceCandidates(candidates)
        fetchTraceHistory()
      } else {
        const errData = await res.json().catch(() => ({}))
        alert(`Diagnostic run error: ${errData.detail || "Unknown error"}`)
      }
    } catch (err) {
      console.error("Diagnostic execution error:", err)
    } finally {
      setIsTracing(false)
    }
  }

  // Load a historic trace
  const handleLoadHistoryTrace = async (traceId: string) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/diagnostics/trace/${encodeURIComponent(traceId)}`)
      if (res.ok) {
        const data = await res.json()
        setTraceResults(data)
        setCurrentTraceId(traceId)
        if (data.query?.query_text) {
          setTraceQuery(String(data.query.query_text))
        }
        if (data.query?.query_type) {
          setTraceType(Number(data.query.query_type) as VBSTaskType)
        }

        const rawResults = (data.final_result?.results || data.results || []) as Array<Record<string, unknown>>
        const candidates: CandidateItem[] = rawResults.map((r, idx) => {
          const payload = (r.payload || {}) as Record<string, unknown>
          const vId = String(payload.source_file || payload.video_id || r.video_name || "unknown").replace(".mp4", "")
          const fId = payload.frame_idx ?? r.frame_idx ?? 0
          return {
            rank: idx + 1,
            video_id: vId,
            frame_id: fId as number,
            answer: r.answer ? String(r.answer) : null,
            src_file: `${vId}.mp4`,
            thumbnail_url: `/api/media/${vId}.mp4`,
            row: [vId, String(fId)],
          }
        })
        setTraceCandidates(candidates)
      }
    } catch (err) {
      console.error("Failed to load historic trace:", err)
    }
  }

  // Stages definition for stepped pipeline tracker
  const stages: TraceStageData[] = useMemo(() => {
    const timings = (traceResults?.timings_ms || {}) as Record<string, number>
    return [
      {
        stage: "01_query",
        title: "Stage 1: Query & HyDE",
        subtitle: "CQR expansion, anchor extraction & hypothetical visual prose",
        status: isTracing ? "running" : traceResults ? "completed" : "idle",
        duration_ms: timings.query_processing_ms,
        data: traceResults?.query as Record<string, unknown> | undefined,
      },
      {
        stage: "02_retrieval",
        title: "Stage 2: Multimodal Retrieval & Fusion",
        subtitle: "Dense Qwen3-VL + Sparse BM25 + SigLIP fused via 4-Way RRF",
        status: isTracing ? "running" : traceResults ? "completed" : "idle",
        duration_ms: timings.retrieval_ms,
        data: traceResults?.retrieval as Record<string, unknown> | undefined,
      },
      {
        stage: "03_diversity",
        title: "Stage 3: Scene Diversity & Ambiguity",
        subtitle: "Temporal coherence boost, KIS-C ambiguity gating & AVS diversity filter",
        status: isTracing ? "running" : traceResults ? "completed" : "idle",
        duration_ms: timings.context_construction_ms,
        data: traceResults?.context as Record<string, unknown> | undefined,
      },
      {
        stage: "04_vlm",
        title: "Stage 4: VLM Grounding & VQA",
        subtitle: "Keyframe resolution, visual verification & grounded answer extraction",
        status: isTracing ? "running" : traceResults ? "completed" : "idle",
        duration_ms: timings.reranking_ms,
        data: traceResults?.reranking as Record<string, unknown> | undefined,
      },
      {
        stage: "05_output",
        title: "Stage 5: Priors & Submission",
        subtitle: "Evidence-backed prior fusion, rank ordering & submission export",
        status: isTracing ? "running" : traceResults ? "completed" : "idle",
        duration_ms: traceResults ? 12 : undefined,
        data: traceResults?.final_result as Record<string, unknown> | undefined,
      },
    ]
  }, [traceResults, isTracing])

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans pb-16">
      {/* Top Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm">
        <div className="max-w-[1700px] mx-auto px-6 h-16 flex items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-700">
              <Layers className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-extrabold tracking-tight text-slate-900 m-0">
                VBS System Audit &amp; Benchmark Lab
              </h1>
              <p className="text-xs text-slate-500 font-medium mt-0.5">
                Interactive step-by-step pipeline diagnostics and ground-truth audit
              </p>
            </div>
          </div>

          {/* Mode Switcher Tabs */}
          <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200 gap-1">
            <button
              onClick={() => setActiveTab("trace")}
              className={`px-4 py-1.5 rounded-md text-xs font-bold transition flex items-center gap-2 ${
                activeTab === "trace"
                  ? "bg-white text-slate-900 shadow-sm border border-slate-200"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Activity className="h-3.5 w-3.5 text-emerald-600" />
              <span>Step-by-Step Trace Lab</span>
            </button>

            <button
              onClick={() => setActiveTab("batch")}
              className={`px-4 py-1.5 rounded-md text-xs font-bold transition flex items-center gap-2 ${
                activeTab === "batch"
                  ? "bg-white text-slate-900 shadow-sm border border-slate-200"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Folder className="h-3.5 w-3.5 text-indigo-600" />
              <span>Manifest &amp; Priors Workspace</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-[1700px] mx-auto px-6 pt-6">
        {activeTab === "trace" ? (
          /* TAB 1: STEP-BY-STEP TRACE LAB */
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Trace History (3 cols) */}
            <div className="lg:col-span-3 space-y-4">
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <History className="h-4 w-4 text-slate-500" />
                    <h2 className="text-xs font-extrabold uppercase tracking-wider text-slate-700 m-0">
                      Audit History
                    </h2>
                  </div>
                  <button
                    onClick={fetchTraceHistory}
                    className="p-1 hover:bg-slate-100 rounded text-slate-400 hover:text-slate-700 transition"
                    title="Refresh History"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                  </button>
                </div>

                <div className="divide-y divide-slate-100 max-h-[620px] overflow-y-auto space-y-1 pr-1">
                  {traceHistory.length === 0 ? (
                    <div className="py-8 text-center text-xs text-slate-400">
                      No audit traces recorded yet. Run a query to start.
                    </div>
                  ) : (
                    traceHistory.map((item) => {
                      const isSelected = item.trace_id === currentTraceId
                      return (
                        <div
                          key={item.trace_id}
                          onClick={() => handleLoadHistoryTrace(item.trace_id)}
                          className={`p-3 rounded-lg cursor-pointer transition text-left mt-1 border ${
                            isSelected
                              ? "bg-emerald-50/70 border-emerald-400 shadow-sm"
                              : "bg-white border-slate-200/70 hover:bg-slate-50"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-1 mb-1">
                            <span className="text-[10px] font-mono text-slate-400">
                              {item.timestamp.split(" ")[1] || item.timestamp}
                            </span>
                            <span className="px-1.5 py-0.2 bg-slate-100 border border-slate-200 rounded text-[9px] font-bold text-slate-700">
                              {item.type_name}
                            </span>
                          </div>
                          <p className="text-xs text-slate-800 font-semibold line-clamp-2 leading-snug">
                            {item.query}
                          </p>
                          <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 font-mono">
                            <span>{item.total_latency_ms} ms</span>
                            <span>{item.results_count} hits</span>
                          </div>
                        </div>
                      )
                    })
                  )}
                </div>
              </div>
            </div>

            {/* Right Main Column: Interactive Query Input & Stepped Pipeline (9 cols) */}
            <div className="lg:col-span-9 space-y-6">
              {/* 1. Query Dispatch Card */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-5">
                <div className="flex items-center justify-between gap-4 flex-wrap">
                  <div className="flex items-center gap-2">
                    <Terminal className="h-4 w-4 text-emerald-600" />
                    <h2 className="text-sm font-extrabold uppercase tracking-wide text-slate-900 m-0">
                      Query Execution &amp; Pipeline Inspector
                    </h2>
                  </div>

                  {/* Task Type Pills */}
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {([1, 2, 3, 4, 5] as VBSTaskType[]).map((t) => {
                      const info = VBS_TASK_LABELS[t]
                      const isActive = traceType === t
                      return (
                        <button
                          key={t}
                          onClick={() => setTraceType(t)}
                          className={`px-3 py-1 rounded-md text-xs font-bold transition border ${
                            isActive
                              ? "bg-slate-900 text-white border-slate-900 shadow-sm"
                              : "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100"
                          }`}
                        >
                          {info.label}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Query Input Box */}
                <div className="space-y-3">
                  <textarea
                    rows={2}
                    value={traceQuery}
                    onChange={(e) => setTraceQuery(e.target.value)}
                    placeholder="Enter VBS test query..."
                    className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3.5 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:bg-white focus:border-slate-400 focus:ring-1 focus:ring-slate-300 font-medium leading-relaxed"
                  />

                  <div className="flex items-center justify-between gap-4 flex-wrap">
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => setFastMode(!fastMode)}
                        className={`px-3 py-1 text-xs rounded border transition font-medium flex items-center gap-1.5 ${
                          fastMode
                            ? "bg-amber-50 border-amber-300 text-amber-800"
                            : "bg-slate-100 border-slate-200 text-slate-600 hover:text-slate-900"
                        }`}
                      >
                        <Zap className="h-3 w-3 text-amber-500" />
                        <span>{fastMode ? "Fast Mode (No VLM)" : "Full VLM Verification"}</span>
                      </button>

                      {currentTraceId && (
                        <span className="text-[11px] font-mono text-slate-400">
                          Trace ID: {currentTraceId}
                        </span>
                      )}
                    </div>

                    <button
                      onClick={handleRunTrace}
                      disabled={isTracing}
                      className="px-5 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-bold flex items-center gap-2 transition shadow-sm disabled:opacity-50"
                    >
                      {isTracing ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Play className="h-4 w-4 fill-white" />
                      )}
                      <span>Run Step-by-Step Audit</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* 2. Stepped Pipeline Progress & Intermediate Inspector */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                  <div className="flex items-center gap-2">
                    <Cpu className="h-4 w-4 text-indigo-600" />
                    <h3 className="text-sm font-extrabold text-slate-900 m-0 uppercase tracking-wide">
                      Multi-Stage Pipeline Execution
                    </h3>
                  </div>

                  {traceResults && (
                    <div className="flex items-center gap-2">
                      <span className="px-2.5 py-1 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-md text-xs font-bold flex items-center gap-1.5">
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                        Total: {String(traceResults.total_latency_ms || 0)} ms
                      </span>
                    </div>
                  )}
                </div>

                {/* 5-Step Stepper Cards */}
                <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
                  {stages.map((st, idx) => {
                    const isActive = activeStageIndex === idx
                    return (
                      <div
                        key={st.stage}
                        onClick={() => setActiveStageIndex(idx)}
                        className={`p-3.5 rounded-lg border cursor-pointer transition text-left space-y-2 ${
                          isActive
                            ? "bg-slate-900 text-white border-slate-900 shadow-md"
                            : st.status === "completed"
                            ? "bg-slate-50/80 border-slate-200 hover:bg-slate-100 text-slate-800"
                            : "bg-slate-50/40 border-slate-100 text-slate-400"
                        }`}
                      >
                        <div className="flex items-center justify-between text-[10px] font-mono font-bold">
                          <span>STAGE 0{idx + 1}</span>
                          {st.duration_ms !== undefined && (
                            <span className={isActive ? "text-emerald-400" : "text-emerald-700 font-semibold"}>
                              {Math.round(st.duration_ms)} ms
                            </span>
                          )}
                        </div>

                        <h4 className="text-xs font-bold leading-snug line-clamp-1">
                          {st.title.split(": ")[1]}
                        </h4>

                        <p className={`text-[10px] line-clamp-2 leading-relaxed ${isActive ? "text-slate-300" : "text-slate-500"}`}>
                          {st.subtitle}
                        </p>
                      </div>
                    )
                  })}
                </div>

                {/* Active Stage Data Inspector Box */}
                <div className="p-5 bg-slate-50 rounded-xl border border-slate-200 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-extrabold uppercase tracking-wider text-slate-700">
                      {stages[activeStageIndex]?.title} Output Details
                    </span>
                    <span className="text-xs font-mono text-slate-500">
                      {stages[activeStageIndex]?.subtitle}
                    </span>
                  </div>

                  {stages[activeStageIndex]?.data ? (
                    <pre className="p-4 bg-white rounded-lg border border-slate-200 text-xs font-mono text-slate-800 overflow-x-auto max-h-[300px] leading-relaxed">
                      {JSON.stringify(stages[activeStageIndex]?.data, null, 2)}
                    </pre>
                  ) : (
                    <div className="p-8 text-center text-xs text-slate-400 bg-white rounded-lg border border-slate-200">
                      No stage telemetry captured yet. Run a query above to populate stage data.
                    </div>
                  )}
                </div>
              </div>

              {/* 3. Candidate Results Grid */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Film className="h-4 w-4 text-emerald-600" />
                    <h3 className="text-sm font-extrabold text-slate-900 m-0 uppercase tracking-wide">
                      Retrieved &amp; Grounded Candidates ({traceCandidates.length})
                    </h3>
                  </div>

                  {traceCandidates.length > 0 && (
                    <span className="text-xs text-slate-500 font-medium">
                      Showing top ranked matches
                    </span>
                  )}
                </div>

                {traceCandidates.length === 0 ? (
                  <div className="py-12 text-center text-xs text-slate-400 bg-slate-50 rounded-lg border border-slate-200">
                    No candidates retrieved. Execute a query to inspect video frames.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                    {traceCandidates.map((cand) => (
                      <div
                        key={`${cand.video_id}-${cand.frame_id}-${cand.rank}`}
                        className={`p-4 rounded-xl border transition flex flex-col justify-between gap-3 ${
                          cand.rank === 1
                            ? "bg-white border-emerald-500 shadow-md ring-1 ring-emerald-400/30"
                            : "bg-white border-slate-200 shadow-sm hover:border-slate-300"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span
                              className={`w-6 h-6 rounded-md flex items-center justify-center text-xs font-mono font-bold ${
                                cand.rank === 1
                                  ? "bg-emerald-600 text-white"
                                  : "bg-slate-100 text-slate-700"
                              }`}
                            >
                              {cand.rank}
                            </span>
                            <span className="font-mono text-xs font-bold text-slate-900">
                              {cand.video_id}
                            </span>
                          </div>

                          <span className="text-xs font-mono text-slate-500">
                            Frame: {cand.frame_id}
                          </span>
                        </div>

                        {/* Thumbnail */}
                        <div
                          onClick={() =>
                            setBrowseDialog({
                              open: true,
                              videoName: cand.video_id,
                              initialTimestamp: typeof cand.frame_id === "number" ? cand.frame_id / 25 : 0,
                            })
                          }
                          className="h-36 bg-slate-100 rounded-lg border border-slate-200 flex items-center justify-center overflow-hidden cursor-pointer group relative"
                        >
                          {cand.thumbnail_url ? (
                            <img
                              src={cand.thumbnail_url}
                              alt={cand.video_id}
                              className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                              onError={(e) => {
                                ;(e.target as HTMLElement).style.display = "none"
                              }}
                            />
                          ) : (
                            <Film className="h-6 w-6 text-slate-400" />
                          )}
                          <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center gap-1.5 text-xs text-white font-semibold">
                            <Eye className="h-4 w-4" />
                            <span>Inspect Video</span>
                          </div>
                        </div>

                        {cand.answer && (
                          <div className="p-2 bg-amber-50 rounded-md border border-amber-200 text-xs font-medium text-amber-900">
                            Answer: {cand.answer}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          /* TAB 2: BATCH MANIFEST & PRIORS WORKSPACE */
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Query Manifests (4 cols) */}
            <div className="lg:col-span-4 bg-white rounded-xl border border-slate-200 shadow-sm p-4 space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                <div className="flex items-center gap-2">
                  <Folder className="h-4 w-4 text-slate-500" />
                  <Select value={selectedFolder} onValueChange={(val) => setSelectedFolder(val)}>
                    <SelectTrigger className="h-8 text-xs bg-slate-50 border-slate-200 w-[180px] text-slate-800 font-semibold">
                      <SelectValue placeholder="Select folder" />
                    </SelectTrigger>
                    <SelectContent className="bg-white border-slate-200">
                      {folders.map((f) => (
                        <SelectItem key={f.path} value={f.path} className="text-xs">
                          {f.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <button
                  onClick={() => fetchQueries(selectedFolder)}
                  className="p-1 hover:bg-slate-100 rounded text-slate-400 hover:text-slate-700 transition"
                  title="Refresh Queries"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                </button>
              </div>

              <div className="divide-y divide-slate-100 max-h-[640px] overflow-y-auto space-y-1.5 pr-1">
                {loadingQueries ? (
                  <div className="py-12 text-center text-xs text-slate-400 flex flex-col items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-emerald-600" />
                    <span>Loading queries...</span>
                  </div>
                ) : (
                  queries.map((q) => {
                    const isSelected = q.id === selectedQueryId
                    return (
                      <div
                        key={q.id}
                        onClick={() => setSelectedQueryId(q.id)}
                        className={`p-3.5 rounded-lg cursor-pointer transition border text-left mt-1 ${
                          isSelected
                            ? "bg-emerald-50/60 border-emerald-400 shadow-sm"
                            : "bg-white border-slate-200/80 hover:bg-slate-50"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-1 mb-1">
                          <span className="font-mono text-xs font-bold text-slate-900 truncate">
                            {q.id}
                          </span>
                          <span className="px-2 py-0.5 bg-slate-100 text-slate-700 rounded text-[10px] font-bold">
                            {q.type_name}
                          </span>
                        </div>
                        <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">
                          {q.query}
                        </p>
                      </div>
                    )
                  })
                )}
              </div>
            </div>

            {/* Right Column: Query Detail & Candidates (8 cols) */}
            <div className="lg:col-span-8 space-y-6">
              {queryDetail ? (
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-5">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <h2 className="text-base font-extrabold text-slate-900 font-mono">
                          {queryDetail.query_id}
                        </h2>
                        <span className="px-2 py-0.5 bg-slate-100 border border-slate-200 text-slate-800 rounded text-xs font-bold">
                          {queryDetail.type_name}
                        </span>
                        {queryDetail.prior_info && (
                          <span className="px-2 py-0.5 bg-indigo-50 border border-indigo-200 text-indigo-700 rounded text-xs font-semibold flex items-center gap-1">
                            <Sparkles className="h-3 w-3" />
                            {queryDetail.prior_info.prior_count} Evidence Priors Active
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-slate-700 leading-relaxed font-medium">
                        {queryDetail.query_text}
                      </p>
                    </div>

                    <button
                      onClick={() => {
                        window.location.href = `${BACKEND_URL}/api/vbs-audit/download-zip?folder=${encodeURIComponent(selectedFolder)}`
                      }}
                      className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-bold flex items-center gap-1.5 transition shadow-sm shrink-0"
                    >
                      <Download className="h-3.5 w-3.5" />
                      <span>Export Zip</span>
                    </button>
                  </div>

                  {/* Candidate List Grid */}
                  <div className="pt-4 border-t border-slate-100">
                    <h3 className="text-xs font-extrabold uppercase tracking-wide text-slate-700 mb-3">
                      Ranked Candidates ({queryDetail.candidates?.length || 0})
                    </h3>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 max-h-[500px] overflow-y-auto pr-1">
                      {(queryDetail.candidates || []).map((cand: any, idx: number) => (
                        <div
                          key={`${cand.video_id}-${cand.frame_id}-${idx}`}
                          className="p-3 bg-slate-50 rounded-lg border border-slate-200 flex items-center justify-between gap-3"
                        >
                          <div className="flex items-center gap-3">
                            <span className="w-6 h-6 rounded bg-slate-200 text-slate-800 text-xs font-mono font-bold flex items-center justify-center">
                              {cand.rank}
                            </span>
                            <div>
                              <div className="font-mono text-xs font-bold text-slate-900">
                                {cand.video_id}
                              </div>
                              <div className="text-[11px] text-slate-500 font-mono">
                                Frame: {cand.frame_id}
                              </div>
                            </div>
                          </div>

                          {cand.is_prior && (
                            <span className="px-2 py-0.5 bg-indigo-50 border border-indigo-200 text-indigo-700 rounded text-[10px] font-bold">
                              Prior
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-12 text-center text-xs text-slate-400 bg-white rounded-xl border border-slate-200">
                  Select a query on the left to inspect candidates.
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Video Browser Dialog */}
      <BrowseVideoDialog
        videoName={browseDialog.open ? browseDialog.videoName : null}
        onClose={() => setBrowseDialog({ open: false, videoName: "" })}
        onPlayFrame={() => {}}
      />
    </div>
  )
}
