import { useState, useEffect } from "react"
import {
  CheckCircle2,
  Play,
  Send,
  Film,
  Maximize2,
  Clock,
  Layers,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Sparkles,
  ArrowRight,
  RotateCcw,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { type ResultHit, ResultCard } from "@/components/ResultCard"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

interface TimelineFrame {
  id?: string
  frame_idx: number
  timestamp: number
}

interface KISVWorkspaceProps {
  queryImageFile: File | null
  results: ResultHit[]
  onPlay: (videoName: string, time: number, frameIdx?: number | null) => void
  onInspect: (hit: ResultHit, rank: number) => void
  onSubmitToDres: (hit: ResultHit) => void
  onFeedback?: (id: string, positive: boolean) => void
  onUseAsQuery?: (id: string) => void
  onInVideoSearch?: (videoName: string) => void
  onBrowseVideo: (videoName: string) => void
  onReset: () => void
}

export function KISVWorkspace({
  queryImageFile,
  results,
  onPlay,
  onInspect,
  onSubmitToDres,
  onFeedback,
  onUseAsQuery,
  onInVideoSearch,
  onBrowseVideo,
  onReset,
}: KISVWorkspaceProps) {
  const primaryHit = results[0]
  const payload = primaryHit?.payload || {}
  const videoName = payload.source_file || "Unknown Video"

  // Primary matched frame index and timestamp
  const initialFrameIdx = Number.isInteger(primaryHit?.evidence_frame_idx)
    ? (primaryHit.evidence_frame_idx as number)
    : Number.isInteger(payload.frame_idx)
    ? (payload.frame_idx as number)
    : null

  const initialTimestamp =
    typeof primaryHit?.evidence_timestamp === "number" && Number.isFinite(primaryHit.evidence_timestamp)
      ? primaryHit.evidence_timestamp
      : typeof payload.timestamp === "number" && Number.isFinite(payload.timestamp)
      ? payload.timestamp
      : 0.0

  const [selectedFrameIdx, setSelectedFrameIdx] = useState<number | null>(initialFrameIdx)
  const [selectedTimestamp, setSelectedTimestamp] = useState<number>(initialTimestamp)
  const [timelineFrames, setTimelineFrames] = useState<TimelineFrame[]>([])
  const [loadingTimeline, setLoadingTimeline] = useState(false)
  const [showSecondaryCandidates, setShowSecondaryCandidates] = useState(false)
  const [queryImagePreview, setQueryImagePreview] = useState<string | null>(null)

  // Generate object URL for the query image
  useEffect(() => {
    if (!queryImageFile) {
      setQueryImagePreview(null)
      return
    }
    const url = URL.createObjectURL(queryImageFile)
    setQueryImagePreview(url)
    return () => URL.revokeObjectURL(url)
  }, [queryImageFile])

  // Reset selected frame when primary hit changes
  useEffect(() => {
    setSelectedFrameIdx(initialFrameIdx)
    setSelectedTimestamp(initialTimestamp)
  }, [primaryHit?.id, initialFrameIdx, initialTimestamp])

  // Fetch adjacent contextual frames from the matched video
  useEffect(() => {
    if (!videoName || videoName === "Unknown Video") return
    let isCancelled = false
    setLoadingTimeline(true)

    const centerTime = initialTimestamp
    const url = `${BACKEND_URL}/api/video/${encodeURIComponent(videoName)}/timeline?center_timestamp=${centerTime}&window_sec=30`

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load video timeline")
        return res.json()
      })
      .then((data) => {
        if (isCancelled) return
        if (Array.isArray(data.frames)) {
          const sorted = [...data.frames].sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0))
          setTimelineFrames(sorted)
        }
      })
      .catch((err) => {
        console.warn("Timeline lookup error:", err)
      })
      .finally(() => {
        if (!isCancelled) setLoadingTimeline(false)
      })

    return () => {
      isCancelled = true
    }
  }, [videoName, initialTimestamp])

  if (!primaryHit) return null

  const matchScore = primaryHit.score ?? primaryHit.rrf_score ?? 0.0
  const matchPercentage = (matchScore * 100).toFixed(1)

  // URL for the matched keyframe in the dataset
  const matchedParams = new URLSearchParams({ video_name: videoName })
  if (selectedFrameIdx != null) {
    matchedParams.set("frame_idx", String(selectedFrameIdx))
  } else if (selectedTimestamp != null) {
    matchedParams.set("timestamp", String(selectedTimestamp))
  }
  const matchedFrameUrl = `${BACKEND_URL}/api/media/frame?${matchedParams.toString()}`

  return (
    <div className="w-full space-y-4 text-left">
      {/* 1. Target Video Identification Banner */}
      <div className="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-white rounded-xl p-5 sm:p-6 shadow-xl border border-indigo-500/30 relative overflow-hidden">
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-5">
          <div className="space-y-2.5 flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-400/30 text-[11px] font-extrabold tracking-wider uppercase flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                Target Video Located
              </span>
              <Badge variant="success" className="text-[10px] font-bold px-2 py-0.5 uppercase tracking-wider">
                Visual Similarity: {matchPercentage}%
              </Badge>
            </div>

            {/* Video & Frame Identification */}
            <div className="space-y-1">
              <div className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-baseline gap-3">
                <span className="text-slate-400 text-base sm:text-lg font-normal">Video:</span>
                <span className="text-indigo-300 font-mono drop-shadow-sm">{videoName}</span>
              </div>
              <div className="flex items-center gap-3 text-xs sm:text-sm text-slate-300 font-mono">
                <span>Location: <strong className="text-emerald-400">Frame #{selectedFrameIdx ?? "N/A"}</strong></span>
                <span>·</span>
                <span>Timestamp: <strong className="text-emerald-400">{selectedTimestamp.toFixed(2)}s</strong></span>
              </div>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex flex-row md:flex-col gap-2 shrink-0 w-full md:w-auto">
            <button
              onClick={() => onPlay(videoName, selectedTimestamp, selectedFrameIdx)}
              className="flex-1 md:flex-none inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold shadow-md transition-all active:scale-95"
            >
              <Play className="w-4 h-4 fill-white" />
              Play Video ({selectedTimestamp.toFixed(1)}s)
            </button>
            <button
              onClick={() => onSubmitToDres(primaryHit)}
              className="flex-1 md:flex-none inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold shadow-md transition-all active:scale-95"
            >
              <Send className="w-4 h-4" />
              Submit to DRES
            </button>
            <button
              onClick={onReset}
              className="flex-1 md:flex-none inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-800 text-slate-300 text-xs font-semibold border border-slate-700 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              New Frame Search
            </button>
          </div>
        </div>
      </div>

      {/* 2. Side-by-Side Visual Verification Workspace */}
      <div className="bg-white rounded-xl border border-slate-200/90 shadow-sm p-4 sm:p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-sm font-extrabold text-slate-800 flex items-center gap-2 m-0">
              <Sparkles className="w-4 h-4 text-indigo-600" />
              Side-by-Side Visual Verification
            </h3>
            <p className="text-xs text-slate-500 m-0">
              Compare your query frame against the exact video frame found in the dataset archive:
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 hidden sm:inline">
            Match Confidence: <strong className="text-emerald-600">{matchPercentage}%</strong>
          </span>
        </div>

        {/* Side-by-Side Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Left: Input Query Frame */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-extrabold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-indigo-600" />
                Query Target Frame (Input)
              </span>
              <span className="text-[11px] text-slate-400 truncate max-w-[200px]">
                {queryImageFile?.name || "Target Frame"}
              </span>
            </div>
            <div className="relative aspect-video bg-slate-900 rounded-lg overflow-hidden border-2 border-indigo-500/40 shadow-xs">
              {queryImagePreview ? (
                <img
                  src={queryImagePreview}
                  alt="Query frame"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-slate-500 text-xs">
                  Query frame preview unavailable
                </div>
              )}
              <div className="absolute top-2 left-2">
                <span className="px-2 py-0.5 bg-slate-950/85 backdrop-blur-xs text-white font-mono text-[10px] font-bold rounded">
                  Query Input
                </span>
              </div>
            </div>
          </div>

          {/* Right: Matched Archive Video Frame */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-extrabold text-emerald-800 uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                Matched Video Frame (Archive)
              </span>
              <span className="text-[11px] text-slate-500 font-mono">
                {videoName} @ {selectedTimestamp.toFixed(2)}s
              </span>
            </div>
            <div className="relative aspect-video bg-slate-900 rounded-lg overflow-hidden border-2 border-emerald-500/60 shadow-xs group">
              <img
                src={matchedFrameUrl}
                alt="Matched archive frame"
                className="w-full h-full object-cover"
              />
              <div className="absolute top-2 left-2 flex items-center gap-1.5">
                <span className="px-2 py-0.5 bg-emerald-600 text-white font-mono text-[10px] font-extrabold rounded flex items-center gap-1 uppercase tracking-wider">
                  <CheckCircle2 className="w-3 h-3" />
                  Exact Match
                </span>
                <span className="px-2 py-0.5 bg-slate-950/85 backdrop-blur-xs text-white font-mono text-[10px] font-bold rounded">
                  Frame #{selectedFrameIdx ?? "N/A"}
                </span>
              </div>
              <div className="absolute top-2 right-2">
                <span className="px-2 py-0.5 bg-white/95 backdrop-blur-xs text-slate-900 font-mono text-[10px] font-bold rounded shadow-xs">
                  {selectedTimestamp.toFixed(2)}s
                </span>
              </div>
              <div className="absolute bottom-2 right-2 flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={() => onPlay(videoName, selectedTimestamp, selectedFrameIdx)}
                  className="px-2.5 py-1 bg-indigo-600/90 hover:bg-indigo-600 text-white text-xs font-bold rounded shadow-sm flex items-center gap-1"
                >
                  <Play className="w-3.5 h-3.5 fill-white" />
                  Play
                </button>
                <button
                  onClick={() => onInspect(primaryHit, 1)}
                  className="px-2.5 py-1 bg-slate-800/90 hover:bg-slate-800 text-white text-xs font-bold rounded shadow-sm flex items-center gap-1"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                  Inspect
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Video metadata caption */}
        {payload.caption && (
          <div className="p-2.5 bg-slate-50 border border-slate-200/80 rounded-lg text-xs text-slate-700 flex items-start gap-2">
            <span className="font-bold text-slate-500 uppercase tracking-wider text-[10px] shrink-0 mt-0.5">
              Archive Caption:
            </span>
            <span className="italic">{payload.caption}</span>
          </div>
        )}

        {/* 3. In-Video Timeline Sequence (Scrub adjacent frames to verify the segment) */}
        <div className="pt-2 border-t border-slate-100 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-extrabold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              In-Video Frame Timeline ({timelineFrames.length} keyframes around this moment)
            </span>
            {loadingTimeline && <RefreshCw className="w-3 h-3 animate-spin text-indigo-500" />}
          </div>

          <div className="flex gap-2.5 overflow-x-auto pb-2 pt-1 custom-scrollbar">
            {timelineFrames.map((frame, idx) => {
              const isCurrent = frame.frame_idx === selectedFrameIdx
              const isInitialTarget = frame.frame_idx === initialFrameIdx
              const frameParams = new URLSearchParams({
                video_name: videoName,
                frame_idx: String(frame.frame_idx),
              })
              const thumbUrl = `${BACKEND_URL}/api/media/frame?${frameParams.toString()}`

              return (
                <div
                  key={frame.frame_idx || idx}
                  onClick={() => {
                    setSelectedFrameIdx(frame.frame_idx)
                    setSelectedTimestamp(frame.timestamp)
                  }}
                  className={`relative w-28 aspect-video rounded-md overflow-hidden shrink-0 cursor-pointer border-2 transition-all ${
                    isCurrent
                      ? "border-emerald-500 ring-2 ring-emerald-500/20 shadow-xs"
                      : "border-slate-200 hover:border-indigo-400 opacity-80 hover:opacity-100"
                  }`}
                  title={`Frame #${frame.frame_idx} (${frame.timestamp.toFixed(2)}s)`}
                >
                  <img
                    src={thumbUrl}
                    alt={`Frame ${frame.frame_idx}`}
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                  <div className="absolute bottom-0 inset-x-0 bg-slate-950/80 backdrop-blur-2xs px-1 py-0.5 text-[9px] text-white font-mono flex justify-between">
                    <span>#{frame.frame_idx}</span>
                    <span>{frame.timestamp.toFixed(1)}s</span>
                  </div>
                  {isInitialTarget && (
                    <div className="absolute top-1 left-1 w-2 h-2 rounded-full bg-emerald-400 shadow-xs" />
                  )}
                </div>
              )
            })}
          </div>

          <div className="pt-1 flex justify-end">
            <button
              onClick={() => onBrowseVideo(videoName)}
              className="inline-flex items-center gap-1.5 px-3 py-1 text-xs text-indigo-600 hover:text-indigo-800 font-bold hover:underline"
            >
              <Film className="w-3.5 h-3.5" />
              Browse Entire Video Keyframes
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* 4. Alternative Candidate Videos (Collapsible) */}
      {results.length > 1 && (
        <div className="bg-white rounded-xl border border-slate-200/90 p-4 space-y-3">
          <button
            onClick={() => setShowSecondaryCandidates(!showSecondaryCandidates)}
            className="w-full flex items-center justify-between text-left font-bold text-xs text-slate-700 hover:text-indigo-600 transition-colors"
          >
            <span className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-slate-400" />
              Alternative Candidate Videos Analyzed ({results.length - 1} other videos)
            </span>
            {showSecondaryCandidates ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>

          {showSecondaryCandidates && (
            <div className="pt-2 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2.5">
              {results.slice(1).map((hit, idx) => (
                <ResultCard
                  key={hit.id || idx + 1}
                  hit={hit}
                  idx={idx + 1}
                  onInspect={(h, i) => onInspect(h, i)}
                  onPlay={onPlay}
                  onFeedback={onFeedback}
                  onUseAsQuery={onUseAsQuery}
                  onInVideoSearch={onInVideoSearch}
                  onBrowseVideo={onBrowseVideo}
                  onSubmitToDres={onSubmitToDres}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
