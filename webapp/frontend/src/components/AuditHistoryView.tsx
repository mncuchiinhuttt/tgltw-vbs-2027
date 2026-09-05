import React, { useEffect, useState, useMemo } from "react"
import { useNavigate } from "react-router-dom"
import {
  Activity,
  History,
  RefreshCw,
  Search,
  CheckCircle2,
  Eye,
  Film,
  Play,
  Download,
  Code,
} from "lucide-react"
import { VBS_TASK_LABELS, type VBSTaskType } from "./VBSAuditWorkspace"
import { BrowseVideoDialog } from "./BrowseVideoDialog"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

interface HistoryItem {
  trace_id: string
  query: string
  type: number
  type_name: string
  timestamp: string
  total_latency_ms: number
  results_count: number
  summary?: {
    likely_failure_stage?: string
    confidence?: string
    observed_facts?: string[]
    inferred_cause?: string
    recommendations?: string[]
  }
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
  stageNumber: string
  title: string
  subtitle: string
  status: "completed" | "error" | "idle"
  duration_ms?: number
  data?: Record<string, unknown>
}

export const AuditHistoryView: React.FC = () => {
  const navigate = useNavigate()

  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [searchFilter, setSearchFilter] = useState<string>("")
  const [typeFilter, setTypeFilter] = useState<string>("ALL")
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null)
  const [selectedTraceData, setSelectedTraceData] = useState<Record<string, unknown> | null>(null)
  const [loadingTrace, setLoadingTrace] = useState<boolean>(false)
  const [activeStageIndex, setActiveStageIndex] = useState<number>(0)
  const [showRawJson, setShowRawJson] = useState<boolean>(false)

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

  const fetchHistory = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${BACKEND_URL}/api/diagnostics/history`)
      if (res.ok) {
        const data = await res.json()
        const items: HistoryItem[] = data.history || []
        setHistory(items)
        if (items.length > 0 && !selectedTraceId) {
          handleSelectTrace(items[0].trace_id)
        }
      }
    } catch (err) {
      console.error("Failed to load audit history:", err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [])

  const handleSelectTrace = async (traceId: string) => {
    setSelectedTraceId(traceId)
    setLoadingTrace(true)
    setShowRawJson(false)
    try {
      const res = await fetch(`${BACKEND_URL}/api/diagnostics/trace/${encodeURIComponent(traceId)}`)
      if (res.ok) {
        const data = await res.json()
        setSelectedTraceData(data)
      }
    } catch (err) {
      console.error("Failed to load trace detail:", err)
    } finally {
      setLoadingTrace(false)
    }
  }

  // Filtered History
  const filteredHistory = useMemo(() => {
    return history.filter((item) => {
      const matchesSearch =
        item.query.toLowerCase().includes(searchFilter.toLowerCase()) ||
        item.trace_id.toLowerCase().includes(searchFilter.toLowerCase())
      const matchesType = typeFilter === "ALL" || item.type.toString() === typeFilter
      return matchesSearch && matchesType
    })
  }, [history, searchFilter, typeFilter])

  // Aggregate Metrics
  const stats = useMemo(() => {
    if (history.length === 0) return { total: 0, avgLatency: 0, totalHits: 0 }
    const total = history.length
    const totalLat = history.reduce((acc, curr) => acc + (curr.total_latency_ms || 0), 0)
    const totalHits = history.reduce((acc, curr) => acc + (curr.results_count || 0), 0)
    return {
      total,
      avgLatency: Math.round(totalLat / total),
      totalHits,
    }
  }, [history])

  // Candidates parsed from selected trace
  const candidates: CandidateItem[] = useMemo(() => {
    if (!selectedTraceData) return []
    const finalResult = (selectedTraceData.final_result || {}) as Record<string, unknown>
    const rawResults = (finalResult.results || selectedTraceData.result_preview || selectedTraceData.results || []) as Array<Record<string, unknown>>
    return rawResults.map((r, idx) => {
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
  }, [selectedTraceData])

  // 5 Stages Definition for Stepped Pipeline
  const stages: TraceStageData[] = useMemo(() => {
    if (!selectedTraceData) return []
    const timing = (selectedTraceData.timing || {}) as Record<string, unknown>
    const stageLatencies = (timing.stage_latencies_ms || {}) as Record<string, number>

    return [
      {
        stage: "01_query",
        stageNumber: "Stage 01",
        title: "Query & HyDE Expansion",
        subtitle: "CQR expansion, anchor extraction and hypothetical visual prose generation",
        status: "completed",
        duration_ms: stageLatencies.query_processing,
        data: selectedTraceData.query as Record<string, unknown> | undefined,
      },
      {
        stage: "02_retrieval",
        stageNumber: "Stage 02",
        title: "Multimodal Retrieval & Fusion",
        subtitle: "Dense Qwen3-VL + Sparse BM25 + SigLIP fused via 4-Way Weighted RRF",
        status: "completed",
        duration_ms: stageLatencies.retrieval,
        data: selectedTraceData.retrieval as Record<string, unknown> | undefined,
      },
      {
        stage: "03_diversity",
        stageNumber: "Stage 03",
        title: "Scene Diversity & Ambiguity",
        subtitle: "Temporal coherence boost, KIS-C ambiguity gating and AVS diversity filter",
        status: "completed",
        duration_ms: stageLatencies.context_construction,
        data: selectedTraceData.context as Record<string, unknown> | undefined,
      },
      {
        stage: "04_vlm",
        stageNumber: "Stage 04",
        title: "VLM Grounding & VQA",
        subtitle: "Keyframe resolution, visual verification and grounded factual answer extraction",
        status: "completed",
        duration_ms: stageLatencies.reranking,
        data: selectedTraceData.reranking as Record<string, unknown> | undefined,
      },
      {
        stage: "05_output",
        stageNumber: "Stage 05",
        title: "Priors & Submission Ordering",
        subtitle: "Evidence-backed prior fusion, candidate rank ordering and export packaging",
        status: "completed",
        duration_ms: 12,
        data: selectedTraceData.final_result as Record<string, unknown> | undefined,
      },
    ]
  }, [selectedTraceData])

  const selectedItemMeta = useMemo(() => {
    return history.find((h) => h.trace_id === selectedTraceId)
  }, [history, selectedTraceId])

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans pb-24">
      {/* Top Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-20 shadow-xs">
        <div className="w-full px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-50 border border-indigo-200 flex items-center justify-center text-indigo-700 shadow-xs">
              <History className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-extrabold text-slate-900 m-0 tracking-tight">
                  Audit History &amp; Diagnostics Archive
                </h1>
                <span className="px-2 py-0.5 bg-indigo-50 text-indigo-800 border border-indigo-200 rounded text-[10px] font-bold uppercase tracking-wider">
                  VBS 2027
                </span>
              </div>
              <p className="text-xs text-slate-500 font-medium mt-0.5">
                Inspect historical telemetry traces, intermediate retrieval stages, and candidate pools
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={fetchHistory}
              className="p-2 hover:bg-slate-100 rounded-lg text-slate-600 hover:text-slate-900 transition border border-slate-200 shadow-xs"
              title="Refresh History"
            >
              <RefreshCw className="h-4 w-4" />
            </button>

            <button
              onClick={() => navigate("/audit")}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-bold flex items-center gap-2 transition shadow-sm"
            >
              <Activity className="h-4 w-4" />
              <span>Back to Audit Lab</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="w-full px-4 sm:px-6 lg:px-8 py-5 space-y-6">
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-2">
            <span className="text-xs font-extrabold uppercase tracking-wider text-slate-500">
              Total Audits Executed
            </span>
            <div className="text-3xl font-extrabold text-slate-900 font-mono">
              {stats.total}
            </div>
            <p className="text-xs text-slate-500">
              Recorded across live sessions and offline benchmarks
            </p>
          </div>

          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-2">
            <span className="text-xs font-extrabold uppercase tracking-wider text-slate-500">
              Average Pipeline Latency
            </span>
            <div className="text-3xl font-extrabold text-emerald-700 font-mono">
              {stats.avgLatency} <span className="text-base font-normal text-slate-500">ms</span>
            </div>
            <p className="text-xs text-slate-500">
              Measured from query ingestion to candidate ordering
            </p>
          </div>

          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-2">
            <span className="text-xs font-extrabold uppercase tracking-wider text-slate-500">
              Total Grounded Matches
            </span>
            <div className="text-3xl font-extrabold text-indigo-700 font-mono">
              {stats.totalHits}
            </div>
            <p className="text-xs text-slate-500">
              Retrieved keyframe and video candidates evaluated
            </p>
          </div>
        </section>

        {/* Master-Detail Interactive Inspection View */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Trace List Selector (4 cols) */}
          <div className="lg:col-span-4 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-5 space-y-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-xs font-extrabold uppercase tracking-wider text-slate-800 m-0">
                  Select Audited Run ({filteredHistory.length})
                </h2>
                <button
                  onClick={fetchHistory}
                  className="text-xs text-indigo-600 font-bold hover:underline"
                >
                  Sync
                </button>
              </div>

              {/* Search & Filter */}
              <div className="relative">
                <Search className="h-3.5 w-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <input
                  type="text"
                  placeholder="Filter by query or trace ID..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl pl-9 pr-3 py-2 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:bg-white focus:border-slate-400 font-medium"
                />
              </div>

              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-bold text-slate-700 focus:outline-none focus:bg-white"
              >
                <option value="ALL">All 5 Task Types</option>
                <option value="1">Type 1 (KIS-T)</option>
                <option value="2">Type 2 (VQA)</option>
                <option value="3">Type 3 (KIS-C)</option>
                <option value="4">Type 4 (AVS)</option>
                <option value="5">Type 5 (KIS-V)</option>
              </select>
            </div>

            {/* List */}
            <div className="divide-y divide-slate-100 max-h-[720px] overflow-y-auto space-y-2 pr-1 pt-2">
              {loading ? (
                <div className="py-16 text-center text-xs text-slate-400 flex flex-col items-center gap-2">
                  <RefreshCw className="h-5 w-5 animate-spin text-indigo-600" />
                  <span>Loading audit runs...</span>
                </div>
              ) : filteredHistory.length === 0 ? (
                <div className="py-16 text-center text-xs text-slate-400">
                  No audit traces found.
                </div>
              ) : (
                filteredHistory.map((item) => {
                  const isSelected = item.trace_id === selectedTraceId
                  const taskInfo = VBS_TASK_LABELS[item.type as VBSTaskType] || {
                    label: `Type ${item.type}`,
                    badgeClass: "bg-slate-100 text-slate-800 border-slate-200",
                  }

                  return (
                    <div
                      key={item.trace_id}
                      onClick={() => handleSelectTrace(item.trace_id)}
                      className={`p-4 rounded-xl cursor-pointer transition border text-left mt-2 ${
                        isSelected
                          ? "bg-indigo-50/70 border-indigo-400 shadow-sm ring-1 ring-indigo-300/30"
                          : "bg-white border-slate-200/80 hover:bg-slate-50"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-1 mb-1.5">
                        <span className={`px-2 py-0.5 rounded-md text-[10px] font-extrabold uppercase border ${taskInfo.badgeClass}`}>
                          {item.type_name}
                        </span>
                        <span className="text-[10px] font-mono text-slate-400">
                          {item.timestamp.split(" ")[1] || item.timestamp}
                        </span>
                      </div>

                      <p className="text-xs font-bold text-slate-900 line-clamp-2 leading-relaxed">
                        {item.query}
                      </p>

                      <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono mt-3 pt-2 border-t border-slate-100/80">
                        <span className="font-bold text-slate-700">{item.total_latency_ms} ms</span>
                        <span>{item.results_count} hits</span>
                      </div>
                    </div>
                  )
                })
              )}
            </div>
          </div>

          {/* Right Column: Full Interactive Visual Trace View (8 cols) */}
          <div className="lg:col-span-8 space-y-6">
            {loadingTrace ? (
              <div className="py-32 text-center text-xs text-slate-400 bg-white rounded-2xl border border-slate-200 flex flex-col items-center justify-center gap-3">
                <RefreshCw className="h-7 w-7 animate-spin text-indigo-600" />
                <span>Loading complete visual trace records...</span>
              </div>
            ) : selectedTraceData && selectedItemMeta ? (
              <div className="space-y-6">
                {/* 1. Trace Header Card */}
                <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 md:p-8 space-y-5">
                  <div className="flex items-start justify-between gap-4 flex-wrap">
                    <div className="space-y-1.5 flex-1 min-w-[280px]">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="px-2.5 py-0.5 bg-indigo-50 border border-indigo-200 text-indigo-800 rounded-lg text-xs font-bold uppercase">
                          {selectedItemMeta.type_name}
                        </span>
                        <span className="font-mono text-xs text-slate-400">
                          Trace ID: {selectedItemMeta.trace_id}
                        </span>
                        <span className="text-xs font-mono text-slate-500">
                          {selectedItemMeta.timestamp}
                        </span>
                      </div>

                      <h2 className="text-base font-extrabold text-slate-900 leading-relaxed pt-1">
                        {selectedItemMeta.query}
                      </h2>
                    </div>

                    <div className="flex items-center gap-3 shrink-0">
                      <button
                        onClick={() => setShowRawJson(!showRawJson)}
                        className="px-3.5 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-xl text-xs font-bold flex items-center gap-1.5 transition border border-slate-200"
                      >
                        <Code className="h-3.5 w-3.5" />
                        <span>{showRawJson ? "Visual Dashboard" : "Raw JSON"}</span>
                      </button>

                      <button
                        onClick={() => {
                          const blob = new Blob([JSON.stringify(selectedTraceData, null, 2)], { type: "application/json" })
                          const url = URL.createObjectURL(blob)
                          const a = document.createElement("a")
                          a.href = url
                          a.download = `${selectedItemMeta.trace_id}.json`
                          a.click()
                        }}
                        className="px-3.5 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-xl text-xs font-bold flex items-center gap-1.5 transition border border-slate-200"
                        title="Download JSON"
                      >
                        <Download className="h-3.5 w-3.5" />
                      </button>

                      <button
                        onClick={() => navigate("/audit")}
                        className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold flex items-center gap-1.5 transition shadow-sm"
                      >
                        <Play className="h-3.5 w-3.5 fill-white" />
                        <span>Re-Run in Lab</span>
                      </button>
                    </div>
                  </div>
                </div>

                {showRawJson ? (
                  /* Raw JSON Mode */
                  <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-3">
                    <span className="text-xs font-extrabold uppercase tracking-wider text-slate-800">
                      Complete Diagnostic Telemetry
                    </span>
                    <pre className="p-5 bg-slate-50 rounded-xl border border-slate-200 text-xs font-mono text-slate-800 overflow-x-auto max-h-[520px] leading-relaxed shadow-inner">
                      {JSON.stringify(selectedTraceData, null, 2)}
                    </pre>
                  </div>
                ) : (
                  /* Full Visual Interactive Mode */
                  <>
                    {/* 2. 5-Stage Stepper Cards */}
                    <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 md:p-8 space-y-6">
                      <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                        <div className="space-y-0.5">
                          <h3 className="text-sm font-extrabold text-slate-900 m-0 uppercase tracking-wide">
                            5-Stage Pipeline Breakdown
                          </h3>
                          <p className="text-xs text-slate-500 font-medium">
                            Select a stage to inspect intermediate vectors, query rewriting, and scores
                          </p>
                        </div>

                        <span className="px-3 py-1 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-lg text-xs font-bold flex items-center gap-2">
                          <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                          Total: {selectedItemMeta.total_latency_ms} ms
                        </span>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-5 gap-3.5">
                        {stages.map((st, idx) => {
                          const isActive = activeStageIndex === idx
                          return (
                            <div
                              key={st.stage}
                              onClick={() => setActiveStageIndex(idx)}
                              className={`p-4 rounded-xl border cursor-pointer transition text-left space-y-2.5 relative ${
                                isActive
                                  ? "bg-slate-900 text-white border-slate-900 shadow-md ring-2 ring-indigo-500/20"
                                  : "bg-slate-50 border-slate-200 hover:bg-slate-100/80 text-slate-800 shadow-xs"
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <span className={`text-[10px] font-mono font-extrabold uppercase px-2 py-0.5 rounded ${
                                  isActive ? "bg-slate-800 text-indigo-300" : "bg-slate-200 text-slate-700"
                                }`}>
                                  {st.stageNumber}
                                </span>

                                {st.duration_ms !== undefined && (
                                  <span className={`text-xs font-mono font-bold ${isActive ? "text-indigo-200" : "text-indigo-700"}`}>
                                    {Math.round(st.duration_ms)} ms
                                  </span>
                                )}
                              </div>

                              <div className="space-y-0.5">
                                <h4 className="text-xs font-bold leading-snug">
                                  {st.title}
                                </h4>
                                <p className={`text-[10px] leading-relaxed line-clamp-2 ${isActive ? "text-slate-300" : "text-slate-500"}`}>
                                  {st.subtitle}
                                </p>
                              </div>
                            </div>
                          )
                        })}
                      </div>

                      {/* Stage Inspector Detail Box */}
                      <div className="p-5 bg-slate-50 rounded-xl border border-slate-200 space-y-3">
                        <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                          <span className="text-xs font-extrabold uppercase tracking-wider text-slate-800">
                            {stages[activeStageIndex]?.title} Output Details
                          </span>
                          <span className="text-xs font-mono text-slate-500">
                            {stages[activeStageIndex]?.subtitle}
                          </span>
                        </div>

                        {stages[activeStageIndex]?.data ? (
                          <pre className="p-4 bg-white rounded-lg border border-slate-200 text-xs font-mono text-slate-800 overflow-x-auto max-h-[280px] leading-relaxed shadow-inner">
                            {JSON.stringify(stages[activeStageIndex]?.data, null, 2)}
                          </pre>
                        ) : (
                          <div className="py-8 text-center text-xs text-slate-400 bg-white rounded-lg border border-slate-200">
                            No telemetry captured for this stage.
                          </div>
                        )}
                      </div>
                    </div>

                    {/* 3. Retrieved & Grounded Candidates Grid */}
                    <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 md:p-8 space-y-6">
                      <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                        <div className="space-y-0.5">
                          <h3 className="text-sm font-extrabold text-slate-900 m-0 uppercase tracking-wide">
                            Grounded Candidate Frames ({candidates.length})
                          </h3>
                          <p className="text-xs text-slate-500 font-medium">
                            Ranked keyframe matches recorded during this audit run
                          </p>
                        </div>
                      </div>

                      {candidates.length === 0 ? (
                        <div className="py-12 text-center text-xs text-slate-400 bg-slate-50 rounded-xl border border-slate-200">
                          No candidate frames recorded in this trace.
                        </div>
                      ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                          {candidates.map((cand) => (
                            <div
                              key={`${cand.video_id}-${cand.frame_id}-${cand.rank}`}
                              className={`p-4 rounded-xl border transition flex flex-col justify-between gap-3 ${
                                cand.rank === 1
                                  ? "bg-white border-emerald-500 shadow-md ring-2 ring-emerald-400/20"
                                  : "bg-white border-slate-200 shadow-xs hover:border-slate-300"
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <span
                                    className={`w-6 h-6 rounded-lg flex items-center justify-center text-xs font-mono font-extrabold ${
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

                                <span className="text-xs font-mono text-slate-500 bg-slate-50 px-2 py-0.5 rounded border border-slate-100">
                                  Frame: {cand.frame_id}
                                </span>
                              </div>

                              {/* 16:9 Thumbnail Image */}
                              <div
                                onClick={() =>
                                  setBrowseDialog({
                                    open: true,
                                    videoName: cand.video_id,
                                    initialTimestamp: typeof cand.frame_id === "number" ? cand.frame_id / 25 : 0,
                                  })
                                }
                                className="h-40 bg-slate-100 rounded-lg border border-slate-200 flex items-center justify-center overflow-hidden cursor-pointer group relative"
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
                                  <Film className="h-8 w-8 text-slate-400" />
                                )}
                                <div className="absolute inset-0 bg-slate-900/50 opacity-0 group-hover:opacity-100 transition flex items-center justify-center gap-1.5 text-xs text-white font-bold backdrop-blur-xs">
                                  <Eye className="h-4 w-4" />
                                  <span>Inspect Video at Timestamp</span>
                                </div>
                              </div>

                              {cand.answer && (
                                <div className="p-2.5 bg-amber-50 rounded-lg border border-amber-200 text-xs font-semibold text-amber-950">
                                  Grounded Answer: {cand.answer}
                                </div>
                              )}

                              {cand.score !== undefined && (
                                <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 pt-1.5 border-t border-slate-100">
                                  <span>Confidence:</span>
                                  <span className="font-bold text-slate-800">{cand.score.toFixed(3)}</span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            ) : (
              <div className="py-32 text-center text-xs text-slate-400 bg-white rounded-2xl border border-slate-200">
                Select an audited run from the left list to inspect its pipeline trace.
              </div>
            )}
          </div>
        </section>
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
