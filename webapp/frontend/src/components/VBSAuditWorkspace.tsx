import React, { useEffect, useState, useMemo } from "react"
import {
  ArrowUp,
  ArrowDown,
  CheckCircle2,
  Clock,
  Download,
  Eye,
  Folder,
  Layers,
  Loader2,
  Play,
  RefreshCw,
  Save,
  Search,
  Sparkles,
  Trash2,
  Zap,
  AlertCircle,
  Film,
  Check,
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

interface QueryItem {
  id: string
  filename: string
  type: number
  type_name: string
  query: string
  status: "pending" | "running" | "completed" | "failed"
  row_count: number
  has_prior: boolean
  prior_count: number
}

interface CandidateItem {
  rank: number
  video_id: string
  frame_id: string | number
  answer?: string | null
  src_file?: string | null
  thumbnail_url?: string | null
  is_prior?: boolean
  row: string[]
}

interface QueryDetailData {
  query_id: string
  type: number
  type_name: string
  query_text: string
  vqa_answer?: string | null
  timings?: Record<string, unknown>
  candidate_count: number
  candidates: CandidateItem[]
  prior_info?: {
    query_stem: string
    prior_count: number
    top_prior: string[]
    all_priors: string[][]
  } | null
}

export const VBSAuditWorkspace: React.FC = () => {
  // Navigation & Folder state
  const [folders, setFolders] = useState<{ path: string; name: string; has_queries: boolean }[]>([])
  const [selectedFolder, setSelectedFolder] = useState<string>("queries")
  const [queries, setQueries] = useState<QueryItem[]>([])
  const [loadingQueries, setLoadingQueries] = useState<boolean>(false)
  const [searchFilter, setSearchFilter] = useState<string>("")
  const [typeFilter, setTypeFilter] = useState<string>("ALL")
  const [statusFilter, setStatusFilter] = useState<string>("ALL")

  // Selected query detail state
  const [selectedQueryId, setSelectedQueryId] = useState<string | null>(null)
  const [queryDetail, setQueryDetail] = useState<QueryDetailData | null>(null)
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false)
  const [isSaving, setIsSaving] = useState<boolean>(false)
  const [saveSuccessNotice, setSaveSuccessNotice] = useState<string | null>(null)
  const [customQaAnswer, setCustomQaAnswer] = useState<string>("")
  const [isRunningQuery, setIsRunningQuery] = useState<boolean>(false)
  const [isBatchRunning, setIsBatchRunning] = useState<boolean>(false)
  const [fastMode, setFastMode] = useState<boolean>(false)

  // Video browser modal
  const [browseDialog, setBrowseDialog] = useState<{
    open: boolean
    videoName: string
    initialTimestamp?: number
  }>({
    open: false,
    videoName: "",
    initialTimestamp: 0,
  })

  // 1. Fetch available query folders
  const fetchFolders = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/vbs-audit/folders`)
      if (res.ok) {
        const data = await res.json()
        setFolders(data.folders || [])
      }
    } catch (err) {
      console.error("Failed to fetch folders:", err)
    }
  }

  // 2. Fetch queries list for current folder
  const fetchQueries = async (folder: string) => {
    setLoadingQueries(true)
    try {
      const res = await fetch(`${BACKEND_URL}/api/vbs-audit/queries?folder=${encodeURIComponent(folder)}`)
      if (res.ok) {
        const data = await res.json()
        const fetched: QueryItem[] = data.queries || []
        setQueries(fetched)
        if (fetched.length > 0 && (!selectedQueryId || !fetched.some((q) => q.id === selectedQueryId))) {
          setSelectedQueryId(fetched[0].id)
        }
      }
    } catch (err) {
      console.error("Failed to fetch queries:", err)
    } finally {
      setLoadingQueries(false)
    }
  }

  // 3. Fetch query detail
  const fetchQueryDetail = async (folder: string, queryId: string) => {
    setLoadingDetail(true)
    try {
      const res = await fetch(
        `${BACKEND_URL}/api/vbs-audit/query-detail?folder=${encodeURIComponent(folder)}&query_id=${encodeURIComponent(queryId)}`
      )
      if (res.ok) {
        const data: QueryDetailData = await res.json()
        setQueryDetail(data)
        setCustomQaAnswer(data.vqa_answer || "")
      }
    } catch (err) {
      console.error("Failed to fetch query detail:", err)
    } finally {
      setLoadingDetail(false)
    }
  }

  useEffect(() => {
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

  // Periodic poll for running jobs
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/vbs-audit/jobs`)
        if (res.ok) {
          const data = await res.json()
          const running = (data.jobs || []).some((j: { status?: string }) => j.status === "running")
          if (running) {
            if (selectedFolder) fetchQueries(selectedFolder)
            if (selectedFolder && selectedQueryId) fetchQueryDetail(selectedFolder, selectedQueryId)
          }
        }
      } catch {
        // ignore polling errors
      }
    }, 3000)
    return () => clearInterval(interval)
  }, [selectedFolder, selectedQueryId])

  // Filtered queries list
  const filteredQueries = useMemo(() => {
    return queries.filter((q) => {
      const matchesSearch =
        q.id.toLowerCase().includes(searchFilter.toLowerCase()) ||
        q.query.toLowerCase().includes(searchFilter.toLowerCase())
      const matchesType = typeFilter === "ALL" || q.type.toString() === typeFilter
      const matchesStatus =
        statusFilter === "ALL" ||
        (statusFilter === "COMPLETED" && q.status === "completed") ||
        (statusFilter === "PENDING" && q.status === "pending") ||
        (statusFilter === "PRIOR" && q.has_prior)
      return matchesSearch && matchesType && matchesStatus
    })
  }, [queries, searchFilter, typeFilter, statusFilter])

  // Handlers
  const handleRunSingle = async () => {
    if (!selectedFolder || !selectedQueryId) return
    setIsRunningQuery(true)
    try {
      const res = await fetch(`${BACKEND_URL}/api/vbs-audit/run-query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folder: selectedFolder,
          query_id: selectedQueryId,
          fast_mode: fastMode,
          top_k: 100,
        }),
      })
      if (res.ok) {
        setSaveSuccessNotice("Query dispatched in background.")
        setTimeout(() => setSaveSuccessNotice(null), 3000)
        fetchQueries(selectedFolder)
      }
    } catch (err) {
      console.error("Failed to run query:", err)
    } finally {
      setIsRunningQuery(false)
    }
  }

  const handleRunAll = async () => {
    if (!selectedFolder) return
    setIsBatchRunning(true)
    try {
      const res = await fetch(`${BACKEND_URL}/api/vbs-audit/run-all`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folder: selectedFolder,
          fast_mode: fastMode,
          top_k: 100,
        }),
      })
      if (res.ok) {
        setSaveSuccessNotice("Batch execution dispatched.")
        setTimeout(() => setSaveSuccessNotice(null), 3000)
        fetchQueries(selectedFolder)
      }
    } catch (err) {
      console.error("Failed to run all queries:", err)
    } finally {
      setIsBatchRunning(false)
    }
  }

  const handleMakeRank1 = (index: number) => {
    if (!queryDetail || index === 0) return
    const list = [...queryDetail.candidates]
    const item = list.splice(index, 1)[0]
    list.unshift(item)
    const updated = list.map((c, idx) => ({ ...c, rank: idx + 1 }))
    setQueryDetail({ ...queryDetail, candidates: updated })
  }

  const handleMoveUp = (index: number) => {
    if (!queryDetail || index === 0) return
    const list = [...queryDetail.candidates]
    const temp = list[index - 1]
    list[index - 1] = list[index]
    list[index] = temp
    const updated = list.map((c, idx) => ({ ...c, rank: idx + 1 }))
    setQueryDetail({ ...queryDetail, candidates: updated })
  }

  const handleMoveDown = (index: number) => {
    if (!queryDetail || index >= queryDetail.candidates.length - 1) return
    const list = [...queryDetail.candidates]
    const temp = list[index + 1]
    list[index + 1] = list[index]
    list[index] = temp
    const updated = list.map((c, idx) => ({ ...c, rank: idx + 1 }))
    setQueryDetail({ ...queryDetail, candidates: updated })
  }

  const handleExclude = (index: number) => {
    if (!queryDetail) return
    const list = queryDetail.candidates.filter((_, idx) => idx !== index)
    const updated = list.map((c, idx) => ({ ...c, rank: idx + 1 }))
    setQueryDetail({ ...queryDetail, candidates: updated, candidate_count: updated.length })
  }

  const handleSaveRanks = async () => {
    if (!selectedFolder || !selectedQueryId || !queryDetail) return
    setIsSaving(true)
    try {
      const rows = queryDetail.candidates.map((c) => c.row)
      const res = await fetch(`${BACKEND_URL}/api/vbs-audit/update-ranks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folder: selectedFolder,
          query_id: selectedQueryId,
          ranks: rows,
          vqa_answer: customQaAnswer || null,
        }),
      })
      if (res.ok) {
        setSaveSuccessNotice("Saved updated ranks successfully.")
        setTimeout(() => setSaveSuccessNotice(null), 3000)
        fetchQueries(selectedFolder)
      }
    } catch (err) {
      console.error("Failed to save ranks:", err)
    } finally {
      setIsSaving(false)
    }
  }

  const handleDownloadZip = () => {
    if (!selectedFolder) return
    window.location.href = `${BACKEND_URL}/api/vbs-audit/download-zip?folder=${encodeURIComponent(selectedFolder)}`
  }

  return (
    <div className="flex flex-col h-[calc(100vh-64px)] bg-slate-50 text-slate-900 overflow-hidden font-sans">
      {/* Top Toolbar */}
      <header className="h-14 border-b border-slate-200 bg-white px-5 flex items-center justify-between gap-4 shrink-0 shadow-sm">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-emerald-600 shrink-0" />
            <span className="text-xs uppercase tracking-wider font-extrabold text-slate-800">
              VBS System Audit &amp; Benchmark
            </span>
          </div>

          <div className="h-4 w-[1px] bg-slate-200 hidden sm:block" />

          {/* Folder Select */}
          <div className="flex items-center gap-1.5">
            <Folder className="h-3.5 w-3.5 text-slate-400 shrink-0" />
            <Select value={selectedFolder} onValueChange={(val) => setSelectedFolder(val)}>
              <SelectTrigger className="h-8 text-xs bg-slate-50 border-slate-300 w-[180px] text-slate-800">
                <SelectValue placeholder="Select folder" />
              </SelectTrigger>
              <SelectContent className="bg-white border-slate-200 text-slate-800 shadow-md">
                {folders.map((f) => (
                  <SelectItem key={f.path} value={f.path} className="text-xs">
                    {f.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Global Action Buttons */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => setFastMode(!fastMode)}
            className={`px-2.5 py-1 text-xs rounded border transition flex items-center gap-1.5 font-medium ${
              fastMode
                ? "bg-amber-50 border-amber-300 text-amber-800"
                : "bg-slate-100 border-slate-200 text-slate-600 hover:text-slate-900"
            }`}
            title="Toggle Fast Mode (skips heavy VLM calls)"
          >
            <Zap className="h-3 w-3 text-amber-500" />
            <span className="hidden md:inline">{fastMode ? "Fast Mode On" : "Full VLM"}</span>
          </button>

          <button
            onClick={handleRunAll}
            disabled={isBatchRunning}
            className="px-3 py-1 bg-slate-900 hover:bg-slate-800 text-white rounded text-xs font-semibold flex items-center gap-1.5 transition shadow-sm disabled:opacity-50"
          >
            {isBatchRunning ? <Loader2 className="h-3 w-3 animate-spin text-emerald-400" /> : <Play className="h-3 w-3 text-emerald-400" />}
            <span>Process All</span>
          </button>

          <button
            onClick={handleDownloadZip}
            className="px-3 py-1 bg-emerald-600 hover:bg-emerald-700 text-white rounded text-xs font-semibold flex items-center gap-1.5 transition shadow-sm"
          >
            <Download className="h-3 w-3" />
            <span className="hidden sm:inline">Export Zip</span>
          </button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left Column: Query Manifest Sidebar */}
        <aside className="w-[340px] border-r border-slate-200 bg-white flex flex-col shrink-0 shadow-sm">
          {/* Sidebar Filter Controls */}
          <div className="p-3 border-b border-slate-200 space-y-2 shrink-0 bg-slate-50/50">
            <div className="relative">
              <Search className="h-3.5 w-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="Search queries..."
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                className="w-full bg-white border border-slate-200 rounded pl-8 pr-3 py-1.5 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:border-slate-400 focus:ring-1 focus:ring-slate-300"
              />
            </div>

            <div className="flex items-center gap-2">
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="flex-1 bg-white border border-slate-200 rounded px-2 py-1 text-xs text-slate-700 focus:outline-none"
              >
                <option value="ALL">All 5 Types</option>
                <option value="1">Type 1 (KIS-T)</option>
                <option value="2">Type 2 (VQA)</option>
                <option value="3">Type 3 (KIS-C)</option>
                <option value="4">Type 4 (AVS)</option>
                <option value="5">Type 5 (KIS-V)</option>
              </select>

              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="flex-1 bg-white border border-slate-200 rounded px-2 py-1 text-xs text-slate-700 focus:outline-none"
              >
                <option value="ALL">All Status</option>
                <option value="COMPLETED">Completed</option>
                <option value="PENDING">Pending</option>
                <option value="PRIOR">With Prior</option>
              </select>
            </div>

            <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1">
              <span>{filteredQueries.length} of {queries.length} queries</span>
              <button
                onClick={() => fetchQueries(selectedFolder)}
                className="hover:text-slate-800 transition flex items-center gap-1 font-medium"
              >
                <RefreshCw className="h-3 w-3" />
                Refresh
              </button>
            </div>
          </div>

          {/* Query List Scrollable */}
          <div className="flex-1 overflow-y-auto divide-y divide-slate-100 p-2 space-y-1.5">
            {loadingQueries ? (
              <div className="p-8 text-center text-xs text-slate-400 flex flex-col items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin text-emerald-600" />
                <span>Loading queries...</span>
              </div>
            ) : filteredQueries.length === 0 ? (
              <div className="p-8 text-center text-xs text-slate-400">
                No matching queries found.
              </div>
            ) : (
              filteredQueries.map((q) => {
                const isSelected = q.id === selectedQueryId
                return (
                  <div
                    key={q.id}
                    onClick={() => setSelectedQueryId(q.id)}
                    className={`p-3 rounded-lg cursor-pointer transition border text-left ${
                      isSelected
                        ? "bg-emerald-50/50 border-emerald-400 shadow-sm"
                        : "bg-white border-slate-200/80 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1 mb-1">
                      <span className="font-mono text-[11px] font-bold text-slate-900 truncate">
                        {q.id}
                      </span>
                      <div className="flex items-center gap-1 shrink-0">
                        {q.has_prior && (
                          <span className="px-1.5 py-0.2 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded text-[9px] font-bold">
                            Prior
                          </span>
                        )}
                        <span
                          className={`px-1.5 py-0.2 rounded text-[9px] font-bold uppercase ${
                            q.type === 1
                              ? "bg-blue-50 text-blue-700 border border-blue-200"
                              : q.type === 2
                              ? "bg-amber-50 text-amber-700 border border-amber-200"
                              : q.type === 3
                              ? "bg-purple-50 text-purple-700 border border-purple-200"
                              : q.type === 4
                              ? "bg-indigo-50 text-indigo-700 border border-indigo-200"
                              : "bg-teal-50 text-teal-700 border border-teal-200"
                          }`}
                        >
                          {q.type_name}
                        </span>
                      </div>
                    </div>

                    <p className="text-[11px] text-slate-600 line-clamp-2 leading-relaxed">
                      {q.query}
                    </p>

                    <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2">
                      <span className="flex items-center gap-1">
                        {q.status === "completed" ? (
                          <CheckCircle2 className="h-3 w-3 text-emerald-600" />
                        ) : q.status === "running" ? (
                          <Loader2 className="h-3 w-3 animate-spin text-amber-500" />
                        ) : (
                          <Clock className="h-3 w-3 text-slate-400" />
                        )}
                        <span className="capitalize text-slate-600 font-medium">{q.status}</span>
                      </span>
                      <span className="font-mono">{q.row_count} rows</span>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </aside>

        {/* Center / Right Column: Candidate Workspace */}
        <main className="flex-1 flex flex-col min-w-0 bg-slate-50 overflow-hidden">
          {/* Query Header & Controls */}
          {queryDetail ? (
            <div className="p-4 border-b border-slate-200 bg-white space-y-3 shrink-0 shadow-sm">
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h2 className="text-sm font-bold text-slate-900 font-mono">
                      {queryDetail.query_id}
                    </h2>
                    <span className="px-2 py-0.5 bg-slate-100 border border-slate-200 rounded text-[10px] font-bold text-slate-700">
                      {queryDetail.type_name}
                    </span>
                    {queryDetail.prior_info && (
                      <span className="px-2 py-0.5 bg-indigo-50 border border-indigo-200 rounded text-[10px] font-semibold text-indigo-700 flex items-center gap-1">
                        <Sparkles className="h-3 w-3 text-indigo-600" />
                        {queryDetail.prior_info.prior_count} Evidence Priors Active
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-700 leading-relaxed font-normal">
                    {queryDetail.query_text}
                  </p>
                </div>

                {/* Query Level Run & Save Actions */}
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={handleRunSingle}
                    disabled={isRunningQuery}
                    className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-800 border border-slate-300 rounded text-xs font-semibold flex items-center gap-1.5 transition disabled:opacity-50"
                  >
                    {isRunningQuery ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-600" />
                    ) : (
                      <Play className="h-3.5 w-3.5 text-emerald-600" />
                    )}
                    <span>Run Query</span>
                  </button>

                  <button
                    onClick={handleSaveRanks}
                    disabled={isSaving}
                    className="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded text-xs font-bold flex items-center gap-1.5 transition shadow-sm disabled:opacity-50"
                  >
                    {isSaving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                    <span>Save Changes</span>
                  </button>
                </div>
              </div>

              {/* VQA Answer Bar if applicable */}
              {queryDetail.type === 2 && (
                <div className="p-2.5 bg-amber-50/80 border border-amber-200 rounded-md flex items-center gap-3">
                  <span className="text-xs font-bold text-amber-800 shrink-0">Grounded VQA Answer:</span>
                  <input
                    type="text"
                    value={customQaAnswer}
                    onChange={(e) => setCustomQaAnswer(e.target.value)}
                    placeholder="Enter factual VQA answer..."
                    className="flex-1 bg-white border border-amber-300 rounded px-2.5 py-1 text-xs text-slate-900 focus:outline-none focus:border-amber-500"
                  />
                </div>
              )}

              {/* Success Notification Banner */}
              {saveSuccessNotice && (
                <div className="p-2 bg-emerald-50 border border-emerald-300 rounded text-xs text-emerald-800 flex items-center gap-2">
                  <Check className="h-3.5 w-3.5 text-emerald-600" />
                  <span>{saveSuccessNotice}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="p-4 border-b border-slate-200 text-xs text-slate-500 bg-white">
              Select a query on the left to inspect candidates.
            </div>
          )}

          {/* Candidate Grid / Table */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {loadingDetail ? (
              <div className="h-full flex flex-col items-center justify-center gap-3 text-xs text-slate-400">
                <Loader2 className="h-6 w-6 animate-spin text-emerald-600" />
                <span>Loading candidates and media provenance...</span>
              </div>
            ) : !queryDetail || queryDetail.candidates.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center gap-2 text-xs text-slate-400">
                <AlertCircle className="h-6 w-6 text-slate-300" />
                <span>No candidate rows available yet. Click "Run Query" to generate results.</span>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3.5">
                {queryDetail.candidates.map((cand, idx) => (
                  <div
                    key={`${cand.video_id}-${cand.frame_id}-${idx}`}
                    className={`p-3.5 rounded-lg border transition flex flex-col justify-between gap-3 relative ${
                      cand.rank === 1
                        ? "bg-white border-emerald-500 shadow-md ring-1 ring-emerald-400/30"
                        : cand.is_prior
                        ? "bg-white border-indigo-300 shadow-sm"
                        : "bg-white border-slate-200 shadow-sm hover:border-slate-300"
                    }`}
                  >
                    {/* Rank Badge & Meta */}
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`w-6 h-6 rounded flex items-center justify-center text-xs font-mono font-bold ${
                            cand.rank === 1
                              ? "bg-emerald-600 text-white"
                              : cand.rank <= 5
                              ? "bg-slate-800 text-white"
                              : "bg-slate-200 text-slate-700"
                          }`}
                        >
                          {cand.rank}
                        </span>
                        <span className="font-mono text-xs font-bold text-slate-900">
                          {cand.video_id}
                        </span>
                      </div>

                      <div className="flex items-center gap-1">
                        {cand.is_prior && (
                          <span className="px-1.5 py-0.2 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded text-[9px] font-bold">
                            Verified Prior
                          </span>
                        )}
                        <span className="text-[11px] font-mono text-slate-500">
                          Frame: {cand.frame_id}
                        </span>
                      </div>
                    </div>

                    {/* Thumbnail / Media Preview Area */}
                    <div
                      onClick={() =>
                        setBrowseDialog({
                          open: true,
                          videoName: cand.video_id,
                          initialTimestamp: typeof cand.frame_id === "number" ? cand.frame_id / 25 : 0,
                        })
                      }
                      className="h-36 bg-slate-100 rounded-md border border-slate-200 flex items-center justify-center overflow-hidden cursor-pointer group relative"
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
                        <div className="flex flex-col items-center gap-1 text-slate-400 group-hover:text-slate-600 transition">
                          <Film className="h-6 w-6" />
                          <span className="text-[10px] font-mono">{cand.video_id}</span>
                        </div>
                      )}
                      <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center gap-1.5 text-xs text-white font-semibold">
                        <Eye className="h-4 w-4" />
                        <span>Inspect Video</span>
                      </div>
                    </div>

                    {/* VQA Answer if present */}
                    {cand.answer && (
                      <div className="text-[11px] text-amber-900 bg-amber-50 px-2.5 py-1.5 rounded border border-amber-200 font-medium">
                        Answer: {cand.answer}
                      </div>
                    )}

                    {/* Action Controls for Rank Tuning */}
                    <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                      <div className="flex items-center gap-1">
                        {cand.rank !== 1 && (
                          <button
                            onClick={() => handleMakeRank1(idx)}
                            className="px-2 py-0.5 bg-emerald-50 hover:bg-emerald-100 border border-emerald-300 text-emerald-800 rounded text-[10px] font-bold transition"
                            title="Make Top Rank 1"
                          >
                            Set #1
                          </button>
                        )}
                        <button
                          onClick={() => handleMoveUp(idx)}
                          disabled={idx === 0}
                          className="p-1 hover:bg-slate-100 rounded text-slate-500 hover:text-slate-900 transition disabled:opacity-30"
                          title="Move Up"
                        >
                          <ArrowUp className="h-3.5 w-3.5" />
                        </button>
                        <button
                          onClick={() => handleMoveDown(idx)}
                          disabled={idx >= queryDetail.candidates.length - 1}
                          className="p-1 hover:bg-slate-100 rounded text-slate-500 hover:text-slate-900 transition disabled:opacity-30"
                          title="Move Down"
                        >
                          <ArrowDown className="h-3.5 w-3.5" />
                        </button>
                      </div>

                      <button
                        onClick={() => handleExclude(idx)}
                        className="p-1 hover:bg-red-50 rounded text-slate-400 hover:text-red-600 transition"
                        title="Exclude Frame"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </main>
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
