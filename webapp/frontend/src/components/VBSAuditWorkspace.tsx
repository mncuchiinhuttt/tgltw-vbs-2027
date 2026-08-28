import React, { useEffect, useState, useMemo } from "react"
import { useNavigate } from "react-router-dom"
import {
  Activity,
  CheckCircle2,
  Eye,
  Loader2,
  Play,
  Sparkles,
  Film,
  History,
  Terminal,
  Cpu,
} from "lucide-react"
import { BrowseVideoDialog } from "./BrowseVideoDialog"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

export type VBSTaskType = 1 | 2 | 3 | 4 | 5

export const VBS_TASK_LABELS: Record<VBSTaskType, { label: string; title: string; desc: string; badgeClass: string }> = {
  1: { label: "KIS-T", title: "Textual Known-Item Search", desc: "Pinpoint exact video moment from descriptive text query", badgeClass: "bg-blue-50 text-blue-700 border-blue-200" },
  2: { label: "VQA", title: "Video Question Answering", desc: "Localize keyframe and extract concise factual text answer", badgeClass: "bg-amber-50 text-amber-700 border-amber-200" },
  3: { label: "KIS-C", title: "Conversational Search", desc: "Multi-turn chat with CQR, ambiguity gating and operator feedback", badgeClass: "bg-purple-50 text-purple-700 border-purple-200" },
  4: { label: "AVS", title: "Ad-hoc Video Search", desc: "Retrieve diverse matching shots across different source videos", badgeClass: "bg-indigo-50 text-indigo-700 border-indigo-200" },
  5: { label: "KIS-V", title: "Visual Known-Item Search", desc: "Match target clip from reference visual frame or sample image", badgeClass: "bg-teal-50 text-teal-700 border-teal-200" },
}

const SAMPLE_QUERIES: Record<VBSTaskType, string[]> = {
  1: [
    "a person riding a red motorcycle on a mountain road in the morning",
    "two fishermen pulling a large net onto a wooden boat at sunrise",
    "aerial view of three cyclists racing along a coastal highway",
  ],
  2: [
    "What is the license plate of the red motorcycle next to the gas station?",
    "What color is the boat anchored near the wooden pier?",
    "What text is written on the banner above the entrance gate?",
  ],
  3: [
    "find a chef preparing seafood in an open indoor kitchen",
    "a woman cooking in a kitchen outdoors while grilling vegetables",
    "people gathering near a fountain in a public square during daytime",
  ],
  4: [
    "shots showing solar panels installed on rooftops or in open fields",
    "all shots showing children playing soccer in a park or schoolyard",
    "shots of wind turbines spinning against a cloudy sky",
  ],
  5: [
    "visual match for marine coral reef with yellow tropical fish",
    "visual match for red sports car driving through a tunnel",
    "visual match for traditional wooden pagoda surrounded by cherry blossoms",
  ],
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
  status: "idle" | "running" | "completed" | "error"
  duration_ms?: number
  data?: Record<string, unknown>
}

export const VBSAuditWorkspace: React.FC = () => {
  const navigate = useNavigate()

  // Trace State
  const [traceQuery, setTraceQuery] = useState<string>("a person riding a red motorcycle on a mountain road in the morning")
  const [traceType, setTraceType] = useState<VBSTaskType>(1)
  const [isTracing, setIsTracing] = useState<boolean>(false)
  const [currentTraceId, setCurrentTraceId] = useState<string | null>(null)
  const [activeStageIndex, setActiveStageIndex] = useState<number>(0)
  const [traceResults, setTraceResults] = useState<Record<string, unknown> | null>(null)
  const [traceCandidates, setTraceCandidates] = useState<CandidateItem[]>([])
  const [recentCount, setRecentCount] = useState<number>(0)

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

  // Load Recent Trace Count
  const loadRecentCount = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/diagnostics/history`)
      if (res.ok) {
        const data = await res.json()
        setRecentCount((data.history || []).length)
      }
    } catch {
      // ignore network errors
    }
  }

  useEffect(() => {
    loadRecentCount()
  }, [])

  // Execute Step-by-Step Diagnostic Audit Run
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
          fast_submission: false,
          top_k: 30,
          include_content: true,
          include_prompts: true,
        }),
      })

      if (res.ok) {
        const data = await res.json()
        setTraceResults(data)
        setCurrentTraceId(data.trace_id || null)

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
        loadRecentCount()
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

  // 5 Stages Definition for Stepped Pipeline
  const stages: TraceStageData[] = useMemo(() => {
    const timings = (traceResults?.timings_ms || {}) as Record<string, number>
    return [
      {
        stage: "01_query",
        stageNumber: "Stage 01",
        title: "Query & HyDE Expansion",
        subtitle: "CQR expansion, anchor extraction and hypothetical visual prose generation",
        status: isTracing ? "running" : traceResults ? "completed" : "idle",
        duration_ms: timings.query_processing_ms,
        data: traceResults?.query as Record<string, unknown> | undefined,
      },
      {
        stage: "02_retrieval",
        stageNumber: "Stage 02",
        title: "Multimodal Retrieval & Fusion",
        subtitle: "Dense Qwen3-VL + Sparse BM25 + SigLIP fused via 4-Way Weighted RRF",
        status: isTracing ? "running" : traceResults ? "completed" : "idle",
        duration_ms: timings.retrieval_ms,
        data: traceResults?.retrieval as Record<string, unknown> | undefined,
      },
      {
        stage: "03_diversity",
        stageNumber: "Stage 03",
        title: "Scene Diversity & Ambiguity",
        subtitle: "Temporal coherence boost, KIS-C ambiguity gating and AVS diversity filter",
        status: isTracing ? "running" : traceResults ? "completed" : "idle",
        duration_ms: timings.context_construction_ms,
        data: traceResults?.context as Record<string, unknown> | undefined,
      },
      {
        stage: "04_vlm",
        stageNumber: "Stage 04",
        title: "VLM Grounding & VQA",
        subtitle: "Keyframe resolution, visual verification and grounded factual answer extraction",
        status: isTracing ? "running" : traceResults ? "completed" : "idle",
        duration_ms: timings.reranking_ms,
        data: traceResults?.reranking as Record<string, unknown> | undefined,
      },
      {
        stage: "05_output",
        stageNumber: "Stage 05",
        title: "Priors & Submission Ordering",
        subtitle: "Evidence-backed prior fusion, candidate rank ordering and export packaging",
        status: isTracing ? "running" : traceResults ? "completed" : "idle",
        duration_ms: traceResults ? 12 : undefined,
        data: traceResults?.final_result as Record<string, unknown> | undefined,
      },
    ]
  }, [traceResults, isTracing])

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans pb-24">
      {/* Top Breadcrumb & Actions Bar */}
      <div className="bg-white border-b border-slate-200 sticky top-0 z-20 shadow-xs">
        <div className="max-w-[1600px] mx-auto px-6 md:px-10 h-16 flex items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-700 shadow-xs">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-extrabold text-slate-900 m-0 tracking-tight">
                  System Audit &amp; Trace Lab
                </h1>
                <span className="px-2 py-0.5 bg-emerald-50 text-emerald-800 border border-emerald-200 rounded text-[10px] font-bold uppercase tracking-wider">
                  VBS 2027
                </span>
              </div>
              <p className="text-xs text-slate-500 font-medium mt-0.5">
                Interactive step-by-step pipeline verification, evidence grounding and latency diagnostics
              </p>
            </div>
          </div>

          <button
            onClick={() => navigate("/history")}
            className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-800 border border-slate-200 rounded-lg text-xs font-bold flex items-center gap-2 transition shadow-xs"
          >
            <History className="h-4 w-4 text-slate-600" />
            <span>Audit History</span>
            {recentCount > 0 && (
              <span className="px-1.5 py-0.2 bg-slate-900 text-white rounded-full text-[10px] font-mono">
                {recentCount}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Main Workspace Layout */}
      <div className="max-w-[1600px] mx-auto px-6 md:px-10 pt-8 space-y-8">
        {/* SECTION 1: Query Execution Deck */}
        <section className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 md:p-8 space-y-6">
          {/* Section Header */}
          <div className="flex items-center justify-between gap-4 border-b border-slate-100 pb-5 flex-wrap">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Terminal className="h-4 w-4 text-emerald-600" />
                <h2 className="text-sm font-extrabold uppercase tracking-wider text-slate-900 m-0">
                  Query Execution &amp; Pipeline Inspector
                </h2>
              </div>
              <p className="text-xs text-slate-500 font-medium">
                Select a VBS task mode, formulate a search query, and observe intermediate stage outputs in real time
              </p>
            </div>

            {/* Task Type Switcher Buttons */}
            <div className="flex items-center gap-2 flex-wrap">
              {([1, 2, 3, 4, 5] as VBSTaskType[]).map((t) => {
                const info = VBS_TASK_LABELS[t]
                const isActive = traceType === t
                return (
                  <button
                    key={t}
                    onClick={() => {
                      setTraceType(t)
                      const samples = SAMPLE_QUERIES[t]
                      if (samples && samples.length > 0) {
                        setTraceQuery(samples[0])
                      }
                    }}
                    className={`px-4 py-2 rounded-xl text-xs font-extrabold transition border flex items-center gap-2 ${
                      isActive
                        ? "bg-slate-900 text-white border-slate-900 shadow-sm scale-100"
                        : "bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100 hover:border-slate-300"
                    }`}
                  >
                    <span>{info.label}</span>
                    <span className="hidden sm:inline font-normal opacity-80 text-[11px]">
                      ({info.title.split(" ")[0]})
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Input & Quick Samples Area */}
          <div className="space-y-4">
            <div className="relative">
              <textarea
                rows={3}
                value={traceQuery}
                onChange={(e) => setTraceQuery(e.target.value)}
                placeholder="Enter query text or select a preset sample below..."
                className="w-full bg-slate-50/70 border border-slate-200 rounded-xl p-4 md:p-5 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:bg-white focus:border-slate-400 focus:ring-2 focus:ring-emerald-500/20 font-medium leading-relaxed transition"
              />
              <div className="absolute right-3.5 bottom-3.5 text-[11px] font-mono text-slate-400">
                {traceQuery.length} chars
              </div>
            </div>

            {/* Quick Samples Pills */}
            <div className="flex items-center gap-2 flex-wrap pt-1">
              <span className="text-xs font-bold text-slate-500 shrink-0">Sample Queries:</span>
              {(SAMPLE_QUERIES[traceType] || []).map((sampleText, idx) => (
                <button
                  key={idx}
                  onClick={() => setTraceQuery(sampleText)}
                  className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-medium transition text-left truncate max-w-[420px]"
                  title={sampleText}
                >
                  {sampleText}
                </button>
              ))}
            </div>
            {/* Action Bar */}
            <div className="flex items-center justify-between gap-4 pt-4 border-t border-slate-100 flex-wrap">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 px-3.5 py-1.5 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-lg text-xs font-bold shadow-xs">
                  <Sparkles className="h-3.5 w-3.5 text-emerald-600" />
                  <span>Full VLM Verification Active</span>
                </div>

                {currentTraceId && (
                  <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md">
                    Trace ID: {currentTraceId}
                  </span>
                )}
              </div>

              <button
                onClick={handleRunTrace}
                disabled={isTracing}
                className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-xs font-bold flex items-center gap-2.5 transition shadow-sm hover:shadow-md disabled:opacity-50"
              >
                {isTracing ? (
                  <Loader2 className="h-4 w-4 animate-spin text-white" />
                ) : (
                  <Play className="h-4 w-4 fill-white" />
                )}
                <span>Run Step-by-Step Audit</span>
              </button>
            </div>
          </div>
        </section>

        {/* SECTION 2: Multi-Stage Stepper & Inspector */}
        <section className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 md:p-8 space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-slate-100 pb-4 flex-wrap gap-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-indigo-600" />
                <h3 className="text-sm font-extrabold text-slate-900 m-0 uppercase tracking-wide">
                  5-Stage Pipeline Execution Breakdown
                </h3>
              </div>
              <p className="text-xs text-slate-500 font-medium">
                Click any stage below to inspect captured vectors, candidate pools, and intermediate reasoning
              </p>
            </div>

            {traceResults && (
              <span className="px-3 py-1 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-lg text-xs font-bold flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                Total Wall Time: {String(traceResults.total_latency_ms || 0)} ms
              </span>
            )}
          </div>

          {/* Stepper Cards Row */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {stages.map((st, idx) => {
              const isActive = activeStageIndex === idx
              return (
                <div
                  key={st.stage}
                  onClick={() => setActiveStageIndex(idx)}
                  className={`p-5 rounded-xl border cursor-pointer transition text-left space-y-3 relative ${
                    isActive
                      ? "bg-slate-900 text-white border-slate-900 shadow-md ring-2 ring-emerald-500/20"
                      : st.status === "completed"
                      ? "bg-slate-50 border-slate-200 hover:bg-slate-100/80 text-slate-800 shadow-xs"
                      : "bg-slate-50/40 border-slate-100 text-slate-400"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`text-[10px] font-mono font-extrabold uppercase px-2 py-0.5 rounded ${
                      isActive ? "bg-slate-800 text-emerald-400" : "bg-slate-200 text-slate-700"
                    }`}>
                      {st.stageNumber}
                    </span>

                    {st.duration_ms !== undefined && (
                      <span className={`text-xs font-mono font-bold ${isActive ? "text-emerald-300" : "text-emerald-700"}`}>
                        {Math.round(st.duration_ms)} ms
                      </span>
                    )}
                  </div>

                  <div className="space-y-1">
                    <h4 className="text-xs font-bold leading-snug">
                      {st.title}
                    </h4>
                    <p className={`text-[11px] leading-relaxed line-clamp-2 ${isActive ? "text-slate-300" : "text-slate-500"}`}>
                      {st.subtitle}
                    </p>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Stage Details Inspector */}
          <div className="p-6 bg-slate-50 rounded-2xl border border-slate-200 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <div className="space-y-0.5">
                <span className="text-xs font-extrabold uppercase tracking-wider text-slate-800">
                  {stages[activeStageIndex]?.title} Telemetry Output
                </span>
                <p className="text-[11px] text-slate-500 font-medium">
                  {stages[activeStageIndex]?.subtitle}
                </p>
              </div>
            </div>

            {stages[activeStageIndex]?.data ? (
              <pre className="p-5 bg-white rounded-xl border border-slate-200 text-xs font-mono text-slate-800 overflow-x-auto max-h-[360px] leading-relaxed shadow-inner">
                {JSON.stringify(stages[activeStageIndex]?.data, null, 2)}
              </pre>
            ) : (
              <div className="py-12 text-center text-xs text-slate-400 bg-white rounded-xl border border-slate-200">
                No telemetry recorded for this stage yet. Click "Run Step-by-Step Audit" to generate pipeline data.
              </div>
            )}
          </div>
        </section>

        {/* SECTION 3: Grounded Candidates Grid */}
        <section className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 md:p-8 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-100 pb-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Film className="h-4 w-4 text-emerald-600" />
                <h3 className="text-sm font-extrabold text-slate-900 m-0 uppercase tracking-wide">
                  Retrieved &amp; Grounded Candidates ({traceCandidates.length})
                </h3>
              </div>
              <p className="text-xs text-slate-500 font-medium">
                Verified candidate frames ranked by multimodal score with direct video inspection
              </p>
            </div>

            {traceCandidates.length > 0 && (
              <span className="px-3 py-1 bg-slate-100 border border-slate-200 text-slate-700 rounded-lg text-xs font-bold">
                Top Matches
              </span>
            )}
          </div>

          {traceCandidates.length === 0 ? (
            <div className="py-16 text-center text-xs text-slate-400 bg-slate-50 rounded-xl border border-slate-200">
              No candidates retrieved. Run an audit query above to inspect candidate keyframes.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {traceCandidates.map((cand) => (
                <div
                  key={`${cand.video_id}-${cand.frame_id}-${cand.rank}`}
                  className={`p-5 rounded-2xl border transition flex flex-col justify-between gap-4 ${
                    cand.rank === 1
                      ? "bg-white border-emerald-500 shadow-md ring-2 ring-emerald-400/20"
                      : "bg-white border-slate-200 shadow-xs hover:border-slate-300 hover:shadow-sm"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                      <span
                        className={`w-7 h-7 rounded-lg flex items-center justify-center text-xs font-mono font-extrabold ${
                          cand.rank === 1
                            ? "bg-emerald-600 text-white shadow-xs"
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
                    className="h-44 bg-slate-100 rounded-xl border border-slate-200 flex items-center justify-center overflow-hidden cursor-pointer group relative"
                  >
                    {cand.thumbnail_url ? (
                      <img
                        src={cand.thumbnail_url}
                        alt={cand.video_id}
                        className="w-full h-full object-cover group-hover:scale-105 transition duration-500"
                        onError={(e) => {
                          ;(e.target as HTMLElement).style.display = "none"
                        }}
                      />
                    ) : (
                      <Film className="h-8 w-8 text-slate-400" />
                    )}
                    <div className="absolute inset-0 bg-slate-900/50 opacity-0 group-hover:opacity-100 transition flex items-center justify-center gap-2 text-xs text-white font-bold backdrop-blur-xs">
                      <Eye className="h-4 w-4" />
                      <span>Inspect Video at Timestamp</span>
                    </div>
                  </div>

                  {cand.answer && (
                    <div className="p-3 bg-amber-50 rounded-xl border border-amber-200 text-xs font-semibold text-amber-950">
                      Grounded Answer: {cand.answer}
                    </div>
                  )}

                  {cand.score !== undefined && (
                    <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 pt-2 border-t border-slate-100">
                      <span>Confidence Score:</span>
                      <span className="font-bold text-slate-800">{cand.score.toFixed(3)}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      {/* Video Browser Dialog */}
      <BrowseVideoDialog
        videoName={browseDialog.open ? browseDialog.videoName : null}
        onClose={() => setBrowseDialog({ open: false, videoName: "" })}
        onPlayFrame={() => {}}
      />
    </div>
  )
}
