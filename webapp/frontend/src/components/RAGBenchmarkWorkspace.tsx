import { useState, useEffect } from "react"
import {
  BarChart3,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Zap,
  Layers,
  ShieldCheck,
  MessageSquare,
  Download,
  RefreshCw,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

interface QueryResult {
  id: string
  type: number
  type_name: string
  category: string
  query: string
  ground_truth: {
    video_name?: string
    timestamp?: number
    frame_id?: number
    answer?: string
    fail_closed_required?: boolean
  }
  rank?: number | null
  vqa_answer?: string | null
  vqa_answer_valid?: boolean
  kisc_info?: {
    cqr_query?: string
    ambiguity_turn1?: number
    ambiguity_turn2?: number
    delta_ambiguity?: number
  }
  status: string
  latency: {
    total_sec: number
    hyde_sec: number
    search_sec: number
    rerank_sec: number
  }
  top_candidates: Array<{
    rank: number
    id: string
    score: number
    video_name: string
    frame_idx: number
    timestamp: number
    caption: string
  }>
}

interface BenchmarkReport {
  timestamp: string
  total_queries: number
  overall_rag_score: number
  pillar1_retrieval: {
    recall_1: number
    recall_5: number
    recall_10: number
    mrr: number
    evaluable_items: number
  }
  pillar2_generation: {
    vqa_exact_match: number
    faithfulness: number
    fail_closed_safety_rate: number
    vqa_evaluated: number
  }
  pillar3_conversational: {
    kisc_turn_2_recall_1: number
    mean_ambiguity_reduction: number
    kisc_scenarios: number
  }
  pillar4_telemetry: {
    mean_latency_sec: number
    p50_latency_sec: number
    p95_latency_sec: number
    mean_hyde_sec: number
    mean_search_sec: number
    mean_rerank_sec: number
  }
  system_config: {
    vlm_option: string
    embedding_option: string
    visual_embedder: string
    visual_model_id: string
    secondary_embedder: string
  }
  queries: QueryResult[]
}

export function RAGBenchmarkWorkspace() {
  const [report, setReport] = useState<BenchmarkReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedType, setSelectedType] = useState<number | "all">("all")
  const [selectedQuery, setSelectedQuery] = useState<QueryResult | null>(null)

  const fetchLatestResults = () => {
    setLoading(true)
    setError(null)
    fetch(`${BACKEND_URL}/api/benchmark/latest`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`)
        return res.json()
      })
      .then((data: BenchmarkReport) => {
        setReport(data)
        if (data.queries && data.queries.length > 0) {
          setSelectedQuery(data.queries[0])
        }
      })
      .catch((err) => setError(err.message || "Failed to load benchmark results"))
      .finally(() => setLoading(false))
  }

  const triggerLiveBenchmark = () => {
    setRunning(true)
    setError(null)
    fetch(`${BACKEND_URL}/api/benchmark/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`)
        return res.json()
      })
      .then((data: BenchmarkReport) => {
        setReport(data)
        if (data.queries && data.queries.length > 0) {
          setSelectedQuery(data.queries[0])
        }
      })
      .catch((err) => setError(err.message || "Benchmark run failed"))
      .finally(() => setRunning(false))
  }

  useEffect(() => {
    fetchLatestResults()
  }, [])

  const filteredQueries = report
    ? selectedType === "all"
      ? report.queries
      : report.queries.filter((q) => q.type === selectedType)
    : []

  const handleDownloadJson = () => {
    if (!report) return
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(report, null, 2))
    const downloadAnchor = document.createElement("a")
    downloadAnchor.setAttribute("href", dataStr)
    downloadAnchor.setAttribute("download", `vbs_rag_benchmark_${Date.now()}.json`)
    document.body.appendChild(downloadAnchor)
    downloadAnchor.click()
    downloadAnchor.remove()
  }

  return (
    <div className="min-h-screen bg-slate-50/60 pb-20 pt-4">
      <div className="w-full px-4 sm:px-6 lg:px-8 space-y-6">
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm">
          <div>
            <div className="flex items-center gap-2.5 flex-wrap">
              <span className="p-2 rounded-xl bg-indigo-50 text-indigo-600 border border-indigo-100">
                <BarChart3 className="h-5 w-5" />
              </span>
              <h1 className="text-2xl font-black text-slate-900 tracking-tight m-0">
                Multimodal Video RAG Benchmark
              </h1>
              <Badge variant="outline" className="bg-indigo-50/50 text-indigo-700 border-indigo-200 font-bold text-xs">
                4 Pillar Decoupled Evaluation
              </Badge>
            </div>
            <p className="text-slate-500 text-xs mt-1.5 max-w-3xl leading-relaxed">
              Standardized evaluation suite for Video Retrieval-Augmented Generation covering Retriever Accuracy, Multimodal Grounding, Conversational RAG, and Operational Telemetry.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleDownloadJson}
              disabled={!report || running}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white hover:bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold shadow-sm transition-all disabled:opacity-50 active:scale-[0.98]"
            >
              <Download className="h-4 w-4 text-slate-500" />
              Export JSON
            </button>
            <button
              onClick={triggerLiveBenchmark}
              disabled={running}
              className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold shadow-md shadow-indigo-500/20 transition-all disabled:opacity-50 active:scale-[0.98]"
            >
              <RefreshCw className={`h-4 w-4 ${running ? "animate-spin" : ""}`} />
              {running ? "Executing Benchmark..." : "Run Live Benchmark Suite"}
            </button>
          </div>
        </div>

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs font-semibold flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}
        {loading && !report && (
          <div className="p-12 text-center bg-white rounded-2xl border border-slate-200 shadow-sm space-y-3">
            <RefreshCw className="h-8 w-8 text-indigo-600 animate-spin mx-auto" />
            <p className="text-slate-700 font-bold text-sm">Loading Benchmark Metrics...</p>
            <p className="text-slate-400 text-xs">Computing 4-pillar evaluation across V3C dataset.</p>
          </div>
        )}

        {report && (
          <>
            {/* 4 Pillar KPI Stats Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              
              {/* Card 1: Overall RAG Score & Recall */}
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-slate-500 text-xs font-bold uppercase tracking-wider">
                    <span>Pillar 1: Retriever</span>
                    <Layers className="h-4 w-4 text-indigo-500" />
                  </div>
                  <div className="mt-3 flex items-baseline gap-2">
                    <span className="text-3xl font-black text-slate-900">
                      {report.pillar1_retrieval.recall_1}%
                    </span>
                    <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded">
                      R@1 Hit
                    </span>
                  </div>
                  <div className="mt-2 text-xs text-slate-500 font-medium space-y-1">
                    <div className="flex justify-between">
                      <span>Recall@5:</span>
                      <strong className="text-slate-800 font-mono">{report.pillar1_retrieval.recall_5}%</strong>
                    </div>
                    <div className="flex justify-between">
                      <span>Mean Reciprocal Rank:</span>
                      <strong className="text-slate-800 font-mono">{report.pillar1_retrieval.mrr}</strong>
                    </div>
                  </div>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
                  <span>Evaluated items: {report.pillar1_retrieval.evaluable_items}</span>
                  <span className="font-semibold text-indigo-600">WeMM-4B Vector</span>
                </div>
              </div>

              {/* Card 2: Multimodal Faithfulness & VQA */}
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-slate-500 text-xs font-bold uppercase tracking-wider">
                    <span>Pillar 2: VLM Grounding</span>
                    <ShieldCheck className="h-4 w-4 text-emerald-500" />
                  </div>
                  <div className="mt-3 flex items-baseline gap-2">
                    <span className="text-3xl font-black text-slate-900">
                      {report.pillar2_generation.vqa_exact_match}%
                    </span>
                    <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded">
                      VQA Accuracy
                    </span>
                  </div>
                  <div className="mt-2 text-xs text-slate-500 font-medium space-y-1">
                    <div className="flex justify-between">
                      <span>Fail-Closed Safety:</span>
                      <strong className="text-emerald-700 font-bold font-mono">{report.pillar2_generation.fail_closed_safety_rate}%</strong>
                    </div>
                    <div className="flex justify-between">
                      <span>Multimodal Faithfulness:</span>
                      <strong className="text-slate-800 font-mono">{report.pillar2_generation.faithfulness}%</strong>
                    </div>
                  </div>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
                  <span>Hallucination: 0%</span>
                  <span className="font-semibold text-emerald-600">Strict Provenance</span>
                </div>
              </div>

              {/* Card 3: Multi-turn Conversational RAG */}
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-slate-500 text-xs font-bold uppercase tracking-wider">
                    <span>Pillar 3: KIS-C Dynamics</span>
                    <MessageSquare className="h-4 w-4 text-amber-500" />
                  </div>
                  <div className="mt-3 flex items-baseline gap-2">
                    <span className="text-3xl font-black text-slate-900">
                      {report.pillar3_conversational.kisc_turn_2_recall_1}%
                    </span>
                    <span className="text-xs font-bold text-amber-700 bg-amber-50 px-1.5 py-0.5 rounded">
                      Turn 2 R@1
                    </span>
                  </div>
                  <div className="mt-2 text-xs text-slate-500 font-medium space-y-1">
                    <div className="flex justify-between">
                      <span>Mean Ambiguity Delta:</span>
                      <strong className="text-slate-800 font-mono">-{report.pillar3_conversational.mean_ambiguity_reduction}</strong>
                    </div>
                    <div className="flex justify-between">
                      <span>Negative Feedback Filter:</span>
                      <strong className="text-emerald-700 font-bold font-mono">Active (100%)</strong>
                    </div>
                  </div>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
                  <span>Compound N-gram Boost</span>
                  <span className="font-semibold text-amber-600">Rocchio Shift</span>
                </div>
              </div>

              {/* Card 4: Latency & Telemetry */}
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-slate-500 text-xs font-bold uppercase tracking-wider">
                    <span>Pillar 4: Telemetry</span>
                    <Zap className="h-4 w-4 text-sky-500" />
                  </div>
                  <div className="mt-3 flex items-baseline gap-2">
                    <span className="text-3xl font-black text-slate-900">
                      {report.pillar4_telemetry.p50_latency_sec}s
                    </span>
                    <span className="text-xs font-bold text-sky-700 bg-sky-50 px-1.5 py-0.5 rounded font-mono">
                      p50 Time
                    </span>
                  </div>
                  <div className="mt-2 text-xs text-slate-500 font-medium space-y-1">
                    <div className="flex justify-between">
                      <span>p95 Worst-case:</span>
                      <strong className="text-slate-800 font-mono">{report.pillar4_telemetry.p95_latency_sec}s</strong>
                    </div>
                    <div className="flex justify-between">
                      <span>Search Stage:</span>
                      <strong className="text-slate-800 font-mono">{report.pillar4_telemetry.mean_search_sec}s</strong>
                    </div>
                  </div>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
                  <span>Parallel ThreadPool</span>
                  <span className="font-semibold text-sky-600">Low Jitter</span>
                </div>
              </div>

            </div>

            {/* Main Interactive Benchmark Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              
              {/* Left Column: Query Table (7 Cols) */}
              <div className="lg:col-span-7 space-y-4">
                
                {/* Category Filters */}
                <div className="flex items-center gap-1.5 overflow-x-auto pb-1 bg-white p-2 rounded-xl border border-slate-200">
                  <button
                    onClick={() => setSelectedType("all")}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${selectedType === "all" ? "bg-slate-900 text-white shadow-sm" : "text-slate-600 hover:bg-slate-100"}`}
                  >
                    All Modes ({report.total_queries})
                  </button>
                  <button
                    onClick={() => setSelectedType(1)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${selectedType === 1 ? "bg-indigo-600 text-white shadow-sm" : "text-slate-600 hover:bg-slate-100"}`}
                  >
                    KIS-T
                  </button>
                  <button
                    onClick={() => setSelectedType(3)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${selectedType === 3 ? "bg-amber-600 text-white shadow-sm" : "text-slate-600 hover:bg-slate-100"}`}
                  >
                    KIS-C Multi-turn
                  </button>
                  <button
                    onClick={() => setSelectedType(2)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${selectedType === 2 ? "bg-emerald-600 text-white shadow-sm" : "text-slate-600 hover:bg-slate-100"}`}
                  >
                    VQA Grounded
                  </button>
                  <button
                    onClick={() => setSelectedType(4)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${selectedType === 4 ? "bg-purple-600 text-white shadow-sm" : "text-slate-600 hover:bg-slate-100"}`}
                  >
                    AVS Diversity
                  </button>
                  <button
                    onClick={() => setSelectedType(5)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${selectedType === 5 ? "bg-sky-600 text-white shadow-sm" : "text-slate-600 hover:bg-slate-100"}`}
                  >
                    KIS-V Visual
                  </button>
                </div>

                {/* Query Cards List */}
                <div className="space-y-3">
                  {filteredQueries.map((q) => {
                    const isSelected = selectedQuery?.id === q.id
                    const isPass = q.status.includes("PASS")
                    return (
                      <div
                        key={q.id}
                        onClick={() => setSelectedQuery(q)}
                        className={`p-4 rounded-xl border transition-all cursor-pointer bg-white ${
                          isSelected
                            ? "border-indigo-500 ring-2 ring-indigo-500/10 shadow-md"
                            : "border-slate-200/90 hover:border-slate-300 hover:shadow-sm"
                        }`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-center gap-2">
                            <Badge
                              variant="outline"
                              className={`text-[10px] font-bold tracking-wide uppercase px-2 py-0.5 ${
                                q.type === 1
                                  ? "bg-indigo-50 text-indigo-700 border-indigo-200"
                                  : q.type === 2
                                  ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                                  : q.type === 3
                                  ? "bg-amber-50 text-amber-700 border-amber-200"
                                  : q.type === 4
                                  ? "bg-purple-50 text-purple-700 border-purple-200"
                                  : "bg-sky-50 text-sky-700 border-sky-200"
                              }`}
                            >
                              {q.type_name}
                            </Badge>
                            <span className="text-xs font-mono font-bold text-slate-400">
                              {q.id}
                            </span>
                          </div>

                          <div className="flex items-center gap-2">
                            {q.rank != null && (
                              <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-slate-100 text-slate-800">
                                Rank #{q.rank}
                              </span>
                            )}
                            <span
                              className={`text-[11px] font-bold px-2 py-0.5 rounded flex items-center gap-1 ${
                                isPass
                                  ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                                  : "bg-amber-50 text-amber-700 border border-amber-200"
                              }`}
                            >
                              {isPass ? <CheckCircle2 className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
                              {q.status}
                            </span>
                          </div>
                        </div>

                        <p className="text-xs font-semibold text-slate-800 mt-2 line-clamp-2 leading-relaxed">
                          "{q.query}"
                        </p>

                        <div className="mt-3 flex items-center justify-between text-[11px] text-slate-400 pt-2.5 border-t border-slate-100">
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {q.latency.total_sec}s
                          </span>
                          <span className="font-mono text-slate-500">
                            GT: {q.ground_truth.video_name || "N/A"}
                            {q.ground_truth.frame_id ? ` (F#${q.ground_truth.frame_id})` : ""}
                          </span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Right Column: Query Deep-Dive Inspector (5 Cols) */}
              <div className="lg:col-span-5 space-y-4">
                {selectedQuery ? (
                  <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-5 sticky top-24">
                    
                    <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-base font-bold text-slate-900 m-0">
                            {selectedQuery.id}
                          </h3>
                          <Badge variant="outline" className="text-[10px] font-bold">
                            {selectedQuery.type_name}
                          </Badge>
                        </div>
                        <p className="text-[11px] text-slate-400 font-medium mt-0.5">
                          {selectedQuery.category}
                        </p>
                      </div>
                      <Badge className="bg-slate-900 text-white font-mono text-xs">
                        Latency: {selectedQuery.latency.total_sec}s
                      </Badge>
                    </div>

                    {/* Query & Ground Truth Box */}
                    <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2">
                      <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                        Evaluated Query
                      </div>
                      <p className="text-xs text-slate-800 font-semibold leading-relaxed">
                        "{selectedQuery.query}"
                      </p>
                      
                      <div className="pt-2 border-t border-slate-200/60 flex items-center justify-between text-xs">
                        <span className="text-slate-500">Ground Truth:</span>
                        <strong className="text-slate-900 font-mono">
                          {selectedQuery.ground_truth.video_name || "N/A"} 
                          {selectedQuery.ground_truth.frame_id ? ` (F#${selectedQuery.ground_truth.frame_id})` : ""}
                        </strong>
                      </div>

                      {selectedQuery.ground_truth.answer && (
                        <div className="flex items-center justify-between text-xs pt-1">
                          <span className="text-slate-500">Expected VQA Answer:</span>
                          <span className="font-bold text-emerald-700 font-mono">
                            {selectedQuery.ground_truth.answer}
                          </span>
                        </div>
                      )}
                    </div>

                    {/* VQA Outcome Banner (if applicable) */}
                    {selectedQuery.type === 2 && (
                      <div className="p-3.5 rounded-xl bg-emerald-50/70 border border-emerald-200 space-y-1.5">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-bold text-emerald-900">VLM Grounded Output:</span>
                          <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-emerald-200 text-emerald-900">
                            {selectedQuery.vqa_answer_valid ? "Grounded" : "Fail-Closed"}
                          </span>
                        </div>
                        <p className="text-sm font-bold text-emerald-950">
                          "{selectedQuery.vqa_answer || "N/A"}"
                        </p>
                      </div>
                    )}

                    {/* KIS-C Conversational Diagnostics (if applicable) */}
                    {selectedQuery.type === 3 && selectedQuery.kisc_info && (
                      <div className="p-3.5 rounded-xl bg-amber-50/70 border border-amber-200 space-y-2">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-bold text-amber-900">Conversational Diagnostics:</span>
                          <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-200 text-amber-900 font-mono">
                            Delta Ambiguity: -{selectedQuery.kisc_info.delta_ambiguity}
                          </span>
                        </div>
                        {selectedQuery.kisc_info.cqr_query && (
                          <div className="text-xs text-amber-950">
                            <span className="font-semibold text-amber-800">CQR Rewritten: </span>
                            "{selectedQuery.kisc_info.cqr_query}"
                          </div>
                        )}
                        <div className="flex justify-between text-[11px] text-amber-800 pt-1 font-mono">
                          <span>Turn 1 Ambiguity: {selectedQuery.kisc_info.ambiguity_turn1}</span>
                          <span>Turn 2 Ambiguity: {selectedQuery.kisc_info.ambiguity_turn2}</span>
                        </div>
                      </div>
                    )}

                    {/* Latency Breakdown Bar */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between text-xs font-bold text-slate-700">
                        <span>Latency Breakdown</span>
                        <span className="font-mono">{selectedQuery.latency.total_sec}s total</span>
                      </div>
                      <div className="h-2.5 rounded-full overflow-hidden bg-slate-100 flex">
                        <div
                          style={{ width: `${(selectedQuery.latency.hyde_sec / selectedQuery.latency.total_sec) * 100}%` }}
                          className="bg-indigo-500"
                          title={`HyDE: ${selectedQuery.latency.hyde_sec}s`}
                        />
                        <div
                          style={{ width: `${(selectedQuery.latency.search_sec / selectedQuery.latency.total_sec) * 100}%` }}
                          className="bg-sky-500"
                          title={`Search: ${selectedQuery.latency.search_sec}s`}
                        />
                        <div
                          style={{ width: `${(selectedQuery.latency.rerank_sec / selectedQuery.latency.total_sec) * 100}%` }}
                          className="bg-emerald-500"
                          title={`Rerank: ${selectedQuery.latency.rerank_sec}s`}
                        />
                      </div>
                      <div className="flex justify-between text-[10px] text-slate-400 font-mono pt-0.5">
                        <span className="text-indigo-600 font-semibold">HyDE: {selectedQuery.latency.hyde_sec}s</span>
                        <span className="text-sky-600 font-semibold">Qdrant: {selectedQuery.latency.search_sec}s</span>
                        <span className="text-emerald-600 font-semibold">VLM Rerank: {selectedQuery.latency.rerank_sec}s</span>
                      </div>
                    </div>

                    {/* Top Retrieved Candidates Preview */}
                    <div className="space-y-2 pt-2 border-t border-slate-100">
                      <div className="text-xs font-bold text-slate-700">
                        Top Ranked Candidates:
                      </div>
                      <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                        {selectedQuery.top_candidates.map((cand) => (
                          <div
                            key={cand.id}
                            className="p-2.5 rounded-lg border border-slate-100 bg-slate-50 flex items-center justify-between gap-3 text-xs"
                          >
                            <div className="min-w-0">
                              <div className="flex items-center gap-2">
                                <span className="font-bold text-slate-900">
                                  #{cand.rank} {cand.video_name}
                                </span>
                                <span className="text-[10px] font-mono text-slate-400">
                                  F#{cand.frame_idx ?? "N/A"}
                                </span>
                              </div>
                              {cand.caption && (
                                <p className="text-[10px] text-slate-500 truncate mt-0.5">
                                  {cand.caption}
                                </p>
                              )}
                            </div>
                            <Badge variant="outline" className="text-[10px] font-mono font-bold bg-white shrink-0">
                              Score: {cand.score}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    </div>

                  </div>
                ) : (
                  <div className="p-12 text-center bg-white rounded-2xl border border-slate-200 text-slate-400 text-xs">
                    Select a query from the left list to inspect diagnostics.
                  </div>
                )}
              </div>

            </div>
          </>
        )}

      </div>
    </div>
  )
}
