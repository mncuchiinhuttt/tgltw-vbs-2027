import { useState, useEffect } from "react"
import {
  HelpCircle,
  CheckCircle2,
  Play,
  Send,
  Film,
  Maximize2,
  Sparkles,
  Clock,
  Layers,
  ChevronDown,
  ChevronUp,
  RefreshCw,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { type ResultHit, ResultCard } from "@/components/ResultCard"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

interface TimelineFrame {
  id?: string
  frame_idx: number
  timestamp: number
  is_focus?: boolean
}

interface VQAWorkspaceProps {
  query: string
  results: ResultHit[]
  onPlay: (videoName: string, time: number, frameIdx?: number | null) => void
  onInspect: (hit: ResultHit, rank: number) => void
  onSubmitToDres: (hit: ResultHit) => void
  onFeedback?: (id: string, positive: boolean) => void
  onUseAsQuery?: (id: string) => void
  onInVideoSearch?: (videoName: string) => void
  onBrowseVideo: (videoName: string) => void
}

export function VQAWorkspace({
  query,
  results,
  onPlay,
  onInspect,
  onSubmitToDres,
  onFeedback,
  onUseAsQuery,
  onInVideoSearch,
  onBrowseVideo,
}: VQAWorkspaceProps) {
  const primaryHit = results[0]
  const payload = primaryHit?.payload || {}
  const videoName = payload.source_file || "Unknown Video"

  // Primary evidence frame and timestamp
  const initialFrameIdx: number | null = Number.isInteger(primaryHit?.evidence_frame_idx)
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

  // State for user-selected preview frame in the contextual timeline
  const [selectedFrameIdx, setSelectedFrameIdx] = useState<number | null>(initialFrameIdx)
  const [selectedTimestamp, setSelectedTimestamp] = useState<number>(initialTimestamp)
  const [timelineFrames, setTimelineFrames] = useState<TimelineFrame[]>([])
  const [loadingTimeline, setLoadingTimeline] = useState(false)
  const [showSecondaryCandidates, setShowSecondaryCandidates] = useState(false)

  // Reset selected frame when primary hit changes
  useEffect(() => {
    setSelectedFrameIdx(initialFrameIdx ?? null)
    setSelectedTimestamp(initialTimestamp)
  }, [primaryHit?.id, initialFrameIdx, initialTimestamp])

  // Fetch adjacent contextual frames for the timeline
  useEffect(() => {
    if (!videoName || videoName === "Unknown Video") return
    let isCancelled = false
    setLoadingTimeline(true)

    const centerTime = initialTimestamp
    const url = `${BACKEND_URL}/api/video/${encodeURIComponent(videoName)}/timeline?center_timestamp=${centerTime}&window_sec=35`

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load timeline")
        return res.json()
      })
      .then((data) => {
        if (isCancelled) return
        if (Array.isArray(data.frames)) {
          // Sort chronologically
          const sorted = [...data.frames].sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0))
          setTimelineFrames(sorted)
        }
      })
      .catch((err) => {
        console.warn("Timeline fetch error:", err)
      })
      .finally(() => {
        if (!isCancelled) setLoadingTimeline(false)
      })

    return () => {
      isCancelled = true
    }
  }, [videoName, initialTimestamp])

  if (!primaryHit) return null

  const answerText = primaryHit.answer || primaryHit.vqa_answer || "N/A"
  const isAnswerValid = primaryHit.vqa_answer_valid ?? Boolean(primaryHit.answer && primaryHit.answer !== "N/A")
  const confidenceScore = primaryHit.score ?? primaryHit.rrf_score ?? 0.0

  // URL for the large active evidence preview image
  const previewParams = new URLSearchParams({ video_name: videoName })
  if (selectedFrameIdx != null) {
    previewParams.set("frame_idx", String(selectedFrameIdx))
  } else if (selectedTimestamp != null) {
    previewParams.set("timestamp", String(selectedTimestamp))
  }
  const activeFrameUrl = `${BACKEND_URL}/api/media/frame?${previewParams.toString()}`

  return (
    <div className="w-full space-y-4 text-left">
      {/* 1. Executive VQA Answer Banner */}
      <div className="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-white rounded-xl p-5 sm:p-6 shadow-xl border border-indigo-500/30 relative overflow-hidden">
        {/* Subtle background glow */}
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-5">
          <div className="space-y-3 flex-1 min-w-0">
            {/* Task mode chip & Question */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-400/30 text-[11px] font-extrabold tracking-wider uppercase flex items-center gap-1.5">
                <HelpCircle className="w-3.5 h-3.5 text-indigo-400" />
                VQA Grounded Inquiry
              </span>
              <span className="text-xs text-slate-400">
                Grounded Video: <strong className="text-slate-200 font-mono">{videoName}</strong>
              </span>
            </div>

            <div className="text-slate-300 text-sm sm:text-base font-medium italic border-l-2 border-indigo-400/60 pl-3">
              "{query}"
            </div>

            {/* Answer Display */}
            <div className="pt-1 flex flex-wrap items-baseline gap-3">
              <span className="text-xs uppercase tracking-wider font-extrabold text-slate-400">
                Verified Answer:
              </span>
              <span className="text-2xl sm:text-3xl font-black text-emerald-400 tracking-tight flex items-center gap-2 drop-shadow-sm">
                <CheckCircle2 className="w-7 h-7 text-emerald-400 shrink-0" />
                {answerText}
              </span>
              {isAnswerValid && (
                <Badge variant="success" className="text-[10px] font-bold px-2 py-0.5 uppercase tracking-wider">
                  Grounded Match (Score: {(confidenceScore * 100).toFixed(1)}%)
                </Badge>
              )}
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
          </div>
        </div>
      </div>

      {/* 2. Visual Evidence & Context Timeline Section */}
      <div className="bg-white rounded-xl border border-slate-200/90 shadow-sm p-4 sm:p-5 space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-2 border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-sm font-extrabold text-slate-800 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-600" />
              Visual Evidence & Temporal Context
            </h3>
            <p className="text-xs text-slate-500">
              The model verified the answer <strong className="text-slate-700">"{answerText}"</strong> from the visual frame sequence below:
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
            <Clock className="w-3.5 h-3.5 text-slate-400" />
            <span>Active Frame: #{selectedFrameIdx ?? "N/A"} ({selectedTimestamp.toFixed(2)}s)</span>
          </div>
        </div>

        {/* Evidence Grid: Large Active Frame (Left) + Context Sequence Filmstrip (Right) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
          {/* Main Selected Frame Preview */}
          <div className="lg:col-span-7 space-y-2">
            <div className="relative aspect-video bg-slate-900 rounded-lg overflow-hidden border-2 border-indigo-500/40 shadow-md group">
              <img
                src={activeFrameUrl}
                alt={`Evidence frame ${selectedFrameIdx}`}
                className="w-full h-full object-cover"
              />
              <div className="absolute top-2 left-2 flex items-center gap-1.5">
                <span className="px-2 py-0.5 bg-slate-950/85 backdrop-blur-xs text-white font-mono text-[11px] font-bold rounded">
                  Frame #{selectedFrameIdx ?? "N/A"}
                </span>
                {selectedFrameIdx === initialFrameIdx && (
                  <span className="px-2 py-0.5 bg-emerald-600 text-white text-[10px] font-extrabold rounded shadow-xs flex items-center gap-1 uppercase tracking-wider">
                    <CheckCircle2 className="w-3 h-3" />
                    Key Grounded Evidence
                  </span>
                )}
              </div>
              <div className="absolute top-2 right-2">
                <span className="px-2 py-0.5 bg-white/95 backdrop-blur-xs text-slate-900 font-mono text-[11px] font-bold rounded shadow-xs">
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

            {/* Video description or caption snippet */}
            {payload.caption && (
              <p className="text-xs text-slate-600 bg-slate-50 border border-slate-100 rounded-md p-2 italic">
                "{payload.caption}"
              </p>
            )}
          </div>

          {/* Context Sequence Filmstrip */}
          <div className="lg:col-span-5 space-y-2">
            <div className="flex items-center justify-between text-xs text-slate-500 font-bold uppercase tracking-wider">
              <span>Adjacent Frame Filmstrip ({timelineFrames.length} frames)</span>
              {loadingTimeline && <RefreshCw className="w-3 h-3 animate-spin text-indigo-500" />}
            </div>

            <div className="max-h-[380px] overflow-y-auto space-y-2 pr-1 custom-scrollbar">
              {timelineFrames.length > 0 ? (
                timelineFrames.map((frame, idx) => {
                  const isCurrentActive = frame.frame_idx === selectedFrameIdx
                  const isKeyEvidence = frame.frame_idx === initialFrameIdx
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
                      className={`flex items-center gap-3 p-2 rounded-lg border transition-all cursor-pointer select-none text-left ${
                        isCurrentActive
                          ? "bg-indigo-50/80 border-indigo-500 shadow-xs ring-1 ring-indigo-500/20"
                          : "bg-white hover:bg-slate-50 border-slate-200/80"
                      }`}
                    >
                      <div className="relative w-20 aspect-video bg-slate-900 rounded overflow-hidden shrink-0">
                        <img
                          src={thumbUrl}
                          alt={`Frame ${frame.frame_idx}`}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                        {isKeyEvidence && (
                          <div className="absolute inset-0 border-2 border-emerald-500 pointer-events-none" />
                        )}
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span className="text-xs font-bold text-slate-800 font-mono">
                            Frame #{frame.frame_idx}
                          </span>
                          {isKeyEvidence && (
                            <span className="text-[9px] px-1.5 py-0.2 bg-emerald-100 text-emerald-800 font-extrabold rounded">
                              Evidence
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] text-slate-500 font-mono">
                          {frame.timestamp.toFixed(2)}s
                        </div>
                      </div>

                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          onPlay(videoName, frame.timestamp, frame.frame_idx)
                        }}
                        className="p-1.5 text-slate-400 hover:text-indigo-600 hover:bg-white rounded transition-colors shrink-0"
                        title="Play from this frame"
                      >
                        <Play className="w-3.5 h-3.5 fill-current" />
                      </button>
                    </div>
                  )
                })
              ) : (
                <div className="text-center py-8 text-xs text-slate-400">
                  {loadingTimeline ? "Loading timeline sequence..." : "No additional frames in sequence"}
                </div>
              )}
            </div>

            <div className="pt-1">
              <button
                onClick={() => onBrowseVideo(videoName)}
                className="w-full py-1.5 px-3 rounded-lg border border-slate-200 bg-slate-50 hover:bg-slate-100 text-slate-600 hover:text-slate-800 text-xs font-bold flex items-center justify-center gap-1.5 transition-colors"
              >
                <Film className="w-3.5 h-3.5 text-slate-500" />
                Browse Entire Video ({videoName})
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Secondary Candidate Shots (Collapsible) */}
      {results.length > 1 && (
        <div className="bg-white rounded-xl border border-slate-200/90 p-4 space-y-3">
          <button
            onClick={() => setShowSecondaryCandidates(!showSecondaryCandidates)}
            className="w-full flex items-center justify-between text-left font-bold text-xs text-slate-700 hover:text-indigo-600 transition-colors"
          >
            <span className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-slate-400" />
              Alternative Candidate Shots Analyzed ({results.length - 1} other shots)
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
