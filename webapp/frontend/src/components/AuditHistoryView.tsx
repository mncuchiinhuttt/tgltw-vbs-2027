import React, { useEffect, useState, useMemo } from "react"
import { useNavigate } from "react-router-dom"
import {
  Activity,
  History,
  RefreshCw,
  Search,
  ChevronDown,
  ChevronUp,
} from "lucide-react"
import { VBS_TASK_LABELS, type VBSTaskType } from "./VBSAuditWorkspace"

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

export const AuditHistoryView: React.FC = () => {
  const navigate = useNavigate()

  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [searchFilter, setSearchFilter] = useState<string>("")
  const [typeFilter, setTypeFilter] = useState<string>("ALL")
  const [expandedTraceId, setExpandedTraceId] = useState<string | null>(null)
  const [selectedTraceData, setSelectedTraceData] = useState<Record<string, unknown> | null>(null)
  const [loadingTrace, setLoadingTrace] = useState<boolean>(false)

  const fetchHistory = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${BACKEND_URL}/api/diagnostics/history`)
      if (res.ok) {
        const data = await res.json()
        setHistory(data.history || [])
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

  const handleInspectTrace = async (traceId: string) => {
    if (expandedTraceId === traceId) {
      setExpandedTraceId(null)
      setSelectedTraceData(null)
      return
    }

    setExpandedTraceId(traceId)
    setLoadingTrace(true)
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

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans pb-24">
      {/* Top Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-20 shadow-xs">
        <div className="max-w-[1600px] mx-auto px-6 md:px-10 h-16 flex items-center justify-between gap-6">
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
      <main className="max-w-[1600px] mx-auto px-6 md:px-10 pt-8 space-y-8">
        {/* Metric Cards Row */}
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

        {/* Filter and Search Bar */}
        <section className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-4">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="relative flex-1 min-w-[280px]">
              <Search className="h-4 w-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="Search audit history by query or trace ID..."
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl pl-10 pr-4 py-2.5 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:bg-white focus:border-slate-400 focus:ring-2 focus:ring-emerald-500/20 font-medium transition"
              />
            </div>

            <div className="flex items-center gap-2">
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-xs font-bold text-slate-700 focus:outline-none focus:bg-white"
              >
                <option value="ALL">All 5 Task Types</option>
                <option value="1">Type 1 (KIS-T)</option>
                <option value="2">Type 2 (VQA)</option>
                <option value="3">Type 3 (KIS-C)</option>
                <option value="4">Type 4 (AVS)</option>
                <option value="5">Type 5 (KIS-V)</option>
              </select>
            </div>
          </div>
        </section>

        {/* Trace List */}
        <section className="space-y-4">
          {loading ? (
            <div className="py-20 text-center text-xs text-slate-400 bg-white rounded-2xl border border-slate-200 flex flex-col items-center gap-3">
              <RefreshCw className="h-6 w-6 animate-spin text-indigo-600" />
              <span>Loading audit telemetry archives...</span>
            </div>
          ) : filteredHistory.length === 0 ? (
            <div className="py-20 text-center text-xs text-slate-400 bg-white rounded-2xl border border-slate-200">
              No audit traces found matching the current search filters.
            </div>
          ) : (
            filteredHistory.map((item) => {
              const isExpanded = expandedTraceId === item.trace_id
              const taskInfo = VBS_TASK_LABELS[item.type as VBSTaskType] || {
                label: `Type ${item.type}`,
                badgeClass: "bg-slate-100 text-slate-800 border-slate-200",
              }

              return (
                <div
                  key={item.trace_id}
                  className="bg-white rounded-2xl border border-slate-200/90 shadow-sm overflow-hidden transition"
                >
                  {/* Trace Item Summary Row */}
                  <div className="p-6 flex items-start justify-between gap-6 flex-wrap">
                    <div className="space-y-2 flex-1 min-w-[320px]">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className={`px-2.5 py-0.5 rounded-lg text-xs font-extrabold uppercase border ${taskInfo.badgeClass}`}>
                          {item.type_name}
                        </span>
                        <span className="font-mono text-xs text-slate-400">
                          {item.trace_id}
                        </span>
                        <span className="text-xs font-mono text-slate-500">
                          {item.timestamp}
                        </span>
                      </div>

                      <p className="text-sm font-bold text-slate-900 leading-relaxed">
                        {item.query}
                      </p>
                    </div>

                    <div className="flex items-center gap-4 shrink-0">
                      <div className="text-right space-y-0.5 font-mono">
                        <div className="text-sm font-extrabold text-slate-900">
                          {item.total_latency_ms} ms
                        </div>
                        <div className="text-xs text-slate-400">
                          {item.results_count} candidates
                        </div>
                      </div>

                      <button
                        onClick={() => handleInspectTrace(item.trace_id)}
                        className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-xl text-xs font-bold flex items-center gap-1.5 transition"
                      >
                        <span>{isExpanded ? "Hide Trace" : "Inspect Trace"}</span>
                        {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                      </button>
                    </div>
                  </div>

                  {/* Expanded Stage Telemetry Inspector */}
                  {isExpanded && (
                    <div className="p-6 bg-slate-50 border-t border-slate-200 space-y-4">
                      {loadingTrace ? (
                        <div className="py-8 text-center text-xs text-slate-400 flex items-center justify-center gap-2">
                          <RefreshCw className="h-4 w-4 animate-spin text-indigo-600" />
                          <span>Fetching full trace records...</span>
                        </div>
                      ) : selectedTraceData ? (
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <h3 className="text-xs font-extrabold uppercase tracking-wide text-slate-700 m-0">
                              Full Diagnostic Stage Output
                            </h3>
                            <button
                              onClick={() => {
                                const blob = new Blob([JSON.stringify(selectedTraceData, null, 2)], { type: "application/json" })
                                const url = URL.createObjectURL(blob)
                                const a = document.createElement("a")
                                a.href = url
                                a.download = `${item.trace_id}.json`
                                a.click()
                              }}
                              className="text-xs font-bold text-indigo-700 hover:text-indigo-800 transition"
                            >
                              Download Trace JSON
                            </button>
                          </div>

                          <pre className="p-5 bg-white rounded-xl border border-slate-200 text-xs font-mono text-slate-800 overflow-x-auto max-h-[380px] leading-relaxed shadow-inner">
                            {JSON.stringify(selectedTraceData, null, 2)}
                          </pre>
                        </div>
                      ) : (
                        <div className="py-6 text-center text-xs text-slate-400">
                          Trace record unavailable.
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })
          )}
        </section>
      </main>
    </div>
  )
}
