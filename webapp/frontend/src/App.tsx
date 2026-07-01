import React, { useState, useEffect, useRef } from "react"
import { HashRouter as Router, Routes, Route, Link, useLocation } from "react-router-dom"
import { 
  Search as SearchIcon, 
  Play, 
  Database, 
  Cpu, 
  RefreshCw, 
  Sliders, 
  Tag, 
  FileText, 
  Sparkles, 
  Clock, 
  AlertCircle, 
  CheckCircle2, 
  Volume2, 
  Terminal,
  ChevronDown,
  ChevronUp,
  FileCode,
  FolderOpen,
  Eye,
  Settings,
  Layers,
  PlayCircle
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

const BACKEND_URL = "http://localhost:8000"

// -------------------------------------------------------------
// NAVIGATION COMPONENT
// -------------------------------------------------------------
function Navbar() {
  const location = useLocation()
  
  const navItems = [
    { path: "/", label: "Query Console", icon: SearchIcon },
    { path: "/preprocess", label: "Pipeline & Indexing", icon: Cpu },
    { path: "/database", label: "DB Statistics", icon: Database }
  ]

  return (
    <nav className="bg-white/80 backdrop-blur-md border-b border-indigo-100/60 sticky top-0 z-40 px-6 py-4 flex items-center justify-between shadow-sm">
      <div className="flex items-center gap-3">
        <div className="bg-indigo-600/10 p-2 rounded-lg border border-indigo-500/20">
          <Sparkles className="h-6 w-6 text-indigo-600 animate-pulse" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-slate-800 tracking-tight m-0 p-0 text-left">
            The Gays Lead The World
            <span className="text-indigo-600 font-semibold text-[10px] border border-indigo-500/20 bg-indigo-50 px-1.5 py-0.5 rounded ml-1">RMIT UNIVERSITY VIETNAM</span>
          </h1>
        </div>
      </div>
      
      <div className="flex gap-1.5">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path
          const Icon = item.icon
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 ${
                isActive 
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/20" 
                  : "text-slate-500 hover:text-slate-900 hover:bg-slate-100"
              }`}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          )
        })}
      </div>
    </nav>
  )
}

// -------------------------------------------------------------
// VIEW 1: SEARCH PAGE & BATCH QUERIES
// -------------------------------------------------------------
function SearchView() {
  const [activeTab, setActiveTab] = useState<"single" | "batch">("single")
  
  // Single Query states
  const [query, setQuery] = useState("")
  const [queryType, setQueryType] = useState<number>(1)
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null)
  
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

  // -----------------------------------------------------------
  // SINGLE QUERY INFERENCE
  // -----------------------------------------------------------
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setResults([])
    setExpandedIndex(null)

    try {
      const response = await fetch(`${BACKEND_URL}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: queryType, query: query })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Search request failed")
      }

      const data = await response.json()
      setResults(data.results || [])
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.")
    } finally {
      setLoading(false)
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
    <div className="max-w-7xl mx-auto px-6 py-8 relative">
      
      {/* Subnavigation Tabs */}
      <div className="flex justify-start gap-2 mb-6 border-b border-slate-200 pb-2">
        <button
          onClick={() => setActiveTab("single")}
          className={`px-4 py-2 text-sm font-bold border-b-2 transition-all duration-200 flex items-center gap-2 ${
            activeTab === "single"
              ? "border-indigo-600 text-indigo-650"
              : "border-transparent text-slate-400 hover:text-slate-600"
          }`}
        >
          <SearchIcon className="h-4 w-4" />
          Single Query Search
        </button>
        <button
          onClick={() => setActiveTab("batch")}
          className={`px-4 py-2 text-sm font-bold border-b-2 transition-all duration-200 flex items-center gap-2 ${
            activeTab === "batch"
              ? "border-indigo-600 text-indigo-650"
              : "border-transparent text-slate-400 hover:text-slate-600"
          }`}
        >
          <Layers className="h-4 w-4" />
          Process Multiple Queries
        </button>
      </div>

      {/* ------------------------------------------------------- */}
      {/* TAB A: SINGLE QUERY CONSOLE */}
      {/* ------------------------------------------------------- */}
      {activeTab === "single" && (
        <>
          <Card className="tech-card border-indigo-100/60 mb-8 overflow-hidden relative bg-white">
            {loading && <div className="scan-line" />}
            <CardContent className="pt-6">
              <form onSubmit={handleSearch} className="flex flex-col md:flex-row gap-4">
                
                {/* Shadcn Select Component */}
                <div className="w-full md:w-56 text-left">
                  <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
                    Retrieval Method
                  </label>
                  <Select
                    value={String(queryType)}
                    onValueChange={(val) => setQueryType(Number(val))}
                  >
                    <SelectTrigger className="w-full bg-white border border-slate-200 rounded-lg text-slate-800">
                      <SelectValue placeholder="Select method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">Type 1: Textual-KIS</SelectItem>
                      <SelectItem value="2">Type 2: VQA Reasoning</SelectItem>
                      <SelectItem value="3">Type 3: Temporal Alignment</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex-1 text-left">
                  <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
                    Natural Text Description
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder={
                        queryType === 1 
                          ? "Search details (e.g. một người đang lái xe máy dưới mưa...)" 
                          : queryType === 2 
                          ? "Visual Question (e.g. người mặc áo đỏ đang dắt xe đạp ở giây thứ mấy?)" 
                          : "Chronological actions (e.g. đầu tiên xe máy đi qua, sau đó ô tô đi qua...)"
                      }
                      className="w-full bg-white border border-slate-200 rounded-lg pl-10 pr-4 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                    />
                    <SearchIcon className="absolute left-3.5 top-3.5 h-4 w-4 text-slate-400" />
                  </div>
                </div>

                <div className="flex items-end">
                  <button
                    type="submit"
                    disabled={loading}
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
                        Query DB
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
                Retrieved <span className="text-indigo-600 font-bold">{results.length}</span> matches for: "{query}"
              </p>
              <Badge variant="outline" className="border-indigo-100 text-indigo-600 bg-indigo-50/50">
                FusedDenseSparse
              </Badge>
            </div>
          )}

          {!loading && results.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {results.map((hit, idx) => {
                const payload = hit.payload || {}
                const videoName = payload.source_file || "Unknown File"
                const timestamp = payload.timestamp !== undefined ? payload.timestamp : 0.0
                const frameUrl = `${BACKEND_URL}/api/media/frame?video_name=${encodeURIComponent(videoName)}&timestamp=${timestamp}`
                const isExpanded = expandedIndex === idx

                return (
                  <Card key={hit.id || idx} className="tech-card overflow-hidden flex flex-col justify-between">
                    <div>
                      <div className="relative group aspect-video bg-slate-100 flex items-center justify-center overflow-hidden border-b border-slate-100">
                        <img
                          src={frameUrl}
                          alt={`Frame ${videoName}`}
                          className="object-cover w-full h-full group-hover:scale-102 transition-transform duration-500"
                          loading="lazy"
                        />
                        <div className="absolute top-3 left-3 flex gap-1.5">
                          <Badge variant="default" className="bg-slate-900/90 text-white font-bold text-xs py-0.5 border border-slate-800">
                            #{idx + 1}
                          </Badge>
                          <Badge variant="outline" className="bg-white/95 text-slate-700 font-bold text-xs border-slate-200/80 shadow-sm">
                            Score: {hit.score ? hit.score.toFixed(3) : (hit.rrf_score ? hit.rrf_score.toFixed(4) : "N/A")}
                          </Badge>
                        </div>
                        <div className="absolute top-3 right-3">
                          <Badge variant="outline" className="bg-white/95 text-xs font-semibold border-slate-200 text-indigo-600 shadow-sm flex gap-1 items-center capitalize">
                            {payload.modality === "ambient_audio" ? <Volume2 className="h-3 w-3" /> : <Tag className="h-3 w-3" />}
                            {payload.modality || "visual"}
                          </Badge>
                        </div>
                        <div className="absolute inset-0 bg-indigo-900/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                          <button
                            onClick={() => setSelectedVideo({ name: videoName, time: timestamp })}
                            className="bg-indigo-600 hover:bg-indigo-500 text-white rounded-full p-4"
                          >
                            <Play className="h-6 w-6 fill-white" />
                          </button>
                        </div>
                      </div>

                      <div className="p-5 text-left">
                        <h4 className="font-bold text-slate-800 truncate text-base mb-1" title={videoName}>
                          {videoName}
                        </h4>
                        <div className="flex items-center gap-1.5 text-slate-500 text-xs font-semibold mb-3">
                          <Clock className="h-4 w-4 text-indigo-500" />
                          <span>Timestamp:</span>
                          <span className="text-indigo-600 bg-indigo-50 border border-indigo-100/50 px-1.5 py-0.5 rounded font-mono">
                            {timestamp.toFixed(2)}s
                          </span>
                        </div>

                        {hit.answer && (
                          <div className="bg-emerald-50 border border-emerald-100 p-3 rounded-lg mb-3">
                            <p className="text-[10px] font-bold text-emerald-700 uppercase tracking-wider mb-1 flex items-center gap-1">
                              <CheckCircle2 className="h-3 w-3" />
                              VQA Generated Answer
                            </p>
                            <p className="text-sm text-slate-700 font-semibold">
                              {hit.answer}
                            </p>
                          </div>
                        )}

                        <p className="text-slate-600 text-sm leading-relaxed line-clamp-2">
                          {payload.caption || "No visual caption metadata indexed."}
                        </p>

                        {isExpanded && (
                          <div className="space-y-4 pt-4 border-t border-slate-100 mt-3 animate-in fade-in duration-200">
                            {payload.scene_narrative && (
                              <div className="space-y-1">
                                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Scene Narrative</span>
                                <p className="text-xs text-slate-600 bg-slate-50 p-2.5 rounded-lg border border-slate-100 leading-relaxed font-medium">
                                  {payload.scene_narrative}
                                </p>
                              </div>
                            )}
                            {payload.ocr_text && (
                              <div className="space-y-1">
                                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">OCR Normalized Text</span>
                                <Badge variant="outline" className="bg-slate-50 border-slate-200 text-slate-700 font-mono text-xs py-1 px-2.5">
                                  {payload.ocr_text}
                                </Badge>
                              </div>
                            )}
                            {payload.detected_objects && payload.detected_objects.length > 0 && (
                              <div className="space-y-1">
                                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Detected Objects</span>
                                <div className="flex flex-wrap gap-1">
                                  {payload.detected_objects.map((obj: any, oIdx: number) => (
                                    <Badge key={oIdx} variant="secondary" className="bg-slate-100 hover:bg-slate-200 text-slate-600 text-xs py-0.5 px-2">
                                      {obj.label} ({obj.conf.toFixed(2)})
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>

                    <CardFooter className="px-5 pb-5 pt-0 flex justify-between gap-3 border-t border-slate-100 mt-4 pt-3">
                      <button
                        onClick={() => setExpandedIndex(isExpanded ? null : idx)}
                        className="flex items-center gap-1 text-xs font-bold text-slate-400 hover:text-indigo-600 transition-colors"
                      >
                        {isExpanded ? <>Hide Metadata<ChevronUp className="h-4 w-4" /></> : <>Inspect Metadata<ChevronDown className="h-4 w-4" /></>}
                      </button>
                      <button
                        onClick={() => setSelectedVideo({ name: videoName, time: timestamp })}
                        className="bg-indigo-50 hover:bg-indigo-100 border border-indigo-100 text-indigo-750 text-xs font-bold px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-colors"
                      >
                        <Play className="h-3.5 w-3.5 fill-indigo-600 text-indigo-600" />
                        Play Clip
                      </button>
                    </CardFooter>
                  </Card>
                )
              })}
            </div>
          )}

          {/* Empty States */}
          {!loading && results.length === 0 && query && (
            <Card className="tech-card py-16 text-center max-w-lg mx-auto border-indigo-100/60 bg-white/75">
              <CardContent className="space-y-4">
                <div className="bg-slate-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto border border-slate-200">
                  <SearchIcon className="h-6 w-6 text-slate-400" />
                </div>
                <h3 className="text-lg font-bold text-slate-800">No matches found</h3>
                <p className="text-slate-500 text-sm">
                  We couldn't retrieve any candidate frames. Check descriptors or try another query.
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
                Perform Multimodal Search
              </h2>
              <p className="text-slate-500 text-sm leading-relaxed font-semibold">
                Search visual frames, spoken speech, and sound environments. Switch between Textual-KIS, VQA (Visual Question Answering), or Temporal alignment.
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
                    Query Registry
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
                      To run a custom batch search: place your list of queries in <code className="text-indigo-600 bg-indigo-100/50 px-1 rounded">queries/queries.json</code> following the query dictionary structure, then click "Process Multiple Queries".
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
                      Process Multiple Queries
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
    </div>
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
            <CardDescription className="text-slate-500 font-semibold">Scanned datasets</CardDescription>
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
              Detailed breakdown of ingested multimedia vectors in Qdrant DB.
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
                  The <code className="text-indigo-600 bg-indigo-50 border border-indigo-100/50 px-1 rounded font-mono">visual_index</code> maps visual keyframes using dense representations. Indexed metadata payloads contain raw transcripts, normalised Vietnamese NFC Unicode strings, detected object tags (bounding box coords resolved via Rex-Omni), and detailed keyframe caption narratives.
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
          <p className="font-semibold">HCMC AI Challenge 2026 &bull; The Gays Lead The World &bull; RMIT University Vietnam</p>
        </footer>
      </div>
    </Router>
  )
}

export default App
