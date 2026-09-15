import { useState, useEffect, useCallback } from "react"
import {
  Play,
  Film,
  Send,
  Maximize2,
  ChevronLeft,
  ChevronRight,
  Clock,
  CheckCircle2,
  LayoutGrid,
  Tv,
  Volume2,
  Tag,
  FileText,
  Sparkles,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { type ResultHit, ResultCard } from "@/components/ResultCard"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

interface KISTWorkspaceProps {
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

export function KISTWorkspace({
  query,
  results,
  onPlay,
  onInspect,
  onSubmitToDres,
  onFeedback,
  onUseAsQuery,
  onInVideoSearch,
  onBrowseVideo,
}: KISTWorkspaceProps) {
  // Default selected candidate is Rank 1 (index 0)
  const [selectedIndex, setSelectedIndex] = useState<number>(0)
  const [viewMode, setViewMode] = useState<"gallery" | "grid">("gallery")

  // Reset to Rank 1 whenever search results change
  useEffect(() => {
    setSelectedIndex(0)
  }, [results[0]?.id, results.length])

  // Active selected hit
  const selectedHit = results[selectedIndex] || results[0]

  // Keyboard navigation for fast competition workflow (Left/Right arrow, Numbers 1-9)
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const activeElement = document.activeElement
      const tagName = activeElement?.tagName?.toLowerCase()
      if (tagName === "input" || tagName === "textarea" || activeElement?.getAttribute("contenteditable")) {
        return
      }

      if (e.key === "ArrowLeft") {
        e.preventDefault()
        setSelectedIndex((prev) => Math.max(0, prev - 1))
      } else if (e.key === "ArrowRight") {
        e.preventDefault()
        setSelectedIndex((prev) => Math.min(results.length - 1, prev + 1))
      } else if (e.key >= "1" && e.key <= "9") {
        const targetIdx = parseInt(e.key, 10) - 1
        if (targetIdx < results.length) {
          e.preventDefault()
          setSelectedIndex(targetIdx)
        }
      } else if (e.key === "Enter" && selectedHit) {
        // Optional hotkey: Enter to submit selected candidate
        e.preventDefault()
        onSubmitToDres(selectedHit)
      }
    },
    [results.length, selectedHit, onSubmitToDres]
  )

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [handleKeyDown])

  if (!selectedHit) return null

  // Metadata extraction for selected hit
  const payload = selectedHit.payload || {}
  const videoName = payload.source_file || "Unknown Video"
  const cleanVideoName = videoName.replace(/\.[^/.]+$/, "")

  const evidenceFrameIdx = Number.isInteger(selectedHit.evidence_frame_idx)
    ? selectedHit.evidence_frame_idx
    : Number.isInteger(payload.frame_idx)
    ? payload.frame_idx
    : null

  const evidenceTimestamp =
    typeof selectedHit.evidence_timestamp === "number" && Number.isFinite(selectedHit.evidence_timestamp)
      ? selectedHit.evidence_timestamp
      : typeof payload.timestamp === "number" && Number.isFinite(payload.timestamp)
      ? payload.timestamp
      : 0.0

  const displayMediaName = selectedHit.evidence_media_name || videoName

  const frameParams = new URLSearchParams({ video_name: displayMediaName })
  if (evidenceFrameIdx != null) frameParams.set("frame_idx", String(evidenceFrameIdx))
  else if (evidenceTimestamp != null) frameParams.set("timestamp", String(evidenceTimestamp))
  const frameUrl = `${BACKEND_URL}/api/media/frame?${frameParams.toString()}`

  const scoreDisplay =
    selectedHit.score != null
      ? selectedHit.score.toFixed(3)
      : selectedHit.rrf_score != null
      ? selectedHit.rrf_score.toFixed(4)
      : "N/A"

  const captionText = payload.caption || payload.scene_narrative || "No caption text recorded."
  const ocrText = payload.ocr_text
  const detectedObjects = payload.detected_objects || []

  return (
    <div className="space-y-4 text-left">
      {/* Top Controls Bar: Switch View & Quick Stepper */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-white border border-slate-200/90 rounded-xl px-4 py-2.5 shadow-2xs">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200 font-bold px-2 py-0.5 text-xs">
            KIS-T
          </Badge>
          <span className="text-xs font-semibold text-slate-800 flex items-center gap-1.5">
            <span>Query:</span>
            <strong className="text-slate-950 font-bold max-w-[280px] sm:max-w-md truncate inline-block">
              "{query}"
            </strong>
          </span>
        </div>

        <div className="flex items-center gap-2.5">
          {/* Active Rank Counter & Stepper */}
          <div className="flex items-center bg-slate-100/90 rounded-lg p-0.5 border border-slate-200/80">
            <button
              type="button"
              onClick={() => setSelectedIndex((prev) => Math.max(0, prev - 1))}
              disabled={selectedIndex === 0}
              className="p-1 rounded hover:bg-white text-slate-700 disabled:opacity-30 disabled:pointer-events-none transition-colors"
              title="Previous Candidate (Left Arrow)"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="px-2 text-xs font-mono font-bold text-slate-800">
              Rank #{selectedIndex + 1} <span className="text-slate-400 font-normal">/ {results.length}</span>
            </span>
            <button
              type="button"
              onClick={() => setSelectedIndex((prev) => Math.min(results.length - 1, prev + 1))}
              disabled={selectedIndex === results.length - 1}
              className="p-1 rounded hover:bg-white text-slate-700 disabled:opacity-30 disabled:pointer-events-none transition-colors"
              title="Next Candidate (Right Arrow)"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center bg-slate-100/90 rounded-lg p-0.5 border border-slate-200/80">
            <button
              type="button"
              onClick={() => setViewMode("gallery")}
              className={`px-2.5 py-1 text-xs font-bold rounded flex items-center gap-1.5 transition-colors ${
                viewMode === "gallery"
                  ? "bg-white text-indigo-700 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
              title="Featured Gallery View (Large Preview + Thumbnails)"
            >
              <Tv className="h-3.5 w-3.5" />
              <span>Gallery</span>
            </button>
            <button
              type="button"
              onClick={() => setViewMode("grid")}
              className={`px-2.5 py-1 text-xs font-bold rounded flex items-center gap-1.5 transition-colors ${
                viewMode === "grid"
                  ? "bg-white text-indigo-700 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
              title="Grid Overview (All Cards Equal)"
            >
              <LayoutGrid className="h-3.5 w-3.5" />
              <span>Grid</span>
            </button>
          </div>
        </div>
      </div>

      {/* Mode 1: Featured Gallery View (Hero Selected Frame + Interactive Candidate Strip) */}
      {viewMode === "gallery" && (
        <>
          {/* Main Selected Hero Stage */}
          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm overflow-hidden grid grid-cols-1 lg:grid-cols-12 gap-0">
            {/* Left: Large 16:9 Image Display */}
            <div className="lg:col-span-7 xl:col-span-8 bg-slate-950 flex flex-col justify-center relative group min-h-[280px] sm:min-h-[380px] md:min-h-[440px]">
              <div
                onClick={() => onInspect(selectedHit, selectedIndex + 1)}
                className="relative aspect-video w-full h-full flex items-center justify-center cursor-pointer overflow-hidden"
                title="Click to open full inspection dialog"
              >
                <img
                  src={frameUrl}
                  alt={`Rank ${selectedIndex + 1}: ${videoName}`}
                  className="w-full h-full object-contain group-hover:scale-[1.01] transition-transform duration-200"
                />

                {/* Top-left: Prominent Selected Rank Badge */}
                <div className="absolute top-3 left-3 flex items-center gap-2 z-10">
                  <span className="px-3 py-1 bg-indigo-600/95 backdrop-blur-xs text-white text-xs sm:text-sm font-black rounded-lg shadow-md font-mono flex items-center gap-1.5 border border-indigo-400/30">
                    <Sparkles className="h-3.5 w-3.5 text-yellow-300" />
                    Rank #{selectedIndex + 1}
                  </span>
                  {selectedIndex === 0 && (
                    <span className="px-2 py-0.5 bg-emerald-500/95 backdrop-blur-xs text-white text-[11px] font-bold rounded shadow-sm">
                      Top Match
                    </span>
                  )}
                </div>

                {/* Top-right: Score Badge */}
                <div className="absolute top-3 right-3 flex items-center gap-2 z-10">
                  <span className="px-2.5 py-1 bg-slate-900/90 backdrop-blur-xs text-white text-xs font-mono font-bold rounded-lg shadow-md border border-slate-700">
                    Score: {scoreDisplay}
                  </span>
                  {payload.modality === "ambient_audio" && (
                    <span className="p-1 bg-indigo-600/90 text-white rounded-lg shadow-md" title="Audio Modality">
                      <Volume2 className="h-4 w-4" />
                    </span>
                  )}
                </div>

                {/* Bottom Overlay: Video & Timestamp Scrim */}
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent pt-8 pb-2.5 px-4 flex items-center justify-between text-white text-xs sm:text-sm font-mono">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="font-bold text-white truncate max-w-[200px] sm:max-w-xs md:max-w-md">
                      {cleanVideoName}
                    </span>
                    <span className="text-slate-400 text-xs hidden sm:inline">({videoName})</span>
                  </div>
                  <div className="flex items-center gap-2 font-semibold text-slate-200 shrink-0">
                    <Clock className="h-3.5 w-3.5 text-indigo-400" />
                    <span>{evidenceTimestamp.toFixed(1)}s</span>
                    {evidenceFrameIdx != null && (
                      <span className="text-slate-400 text-xs">· Frame #{evidenceFrameIdx}</span>
                    )}
                  </div>
                </div>

                {/* Center Hover Action Pill */}
                <div className="absolute inset-0 bg-slate-950/20 opacity-0 group-hover:opacity-100 transition-opacity duration-150 flex items-center justify-center pointer-events-none">
                  <span className="px-3 py-1.5 bg-slate-900/85 backdrop-blur-xs text-white text-xs font-semibold rounded-full shadow-lg border border-white/20 flex items-center gap-1.5">
                    <Maximize2 className="h-3.5 w-3.5 text-indigo-400" />
                    Click image to inspect high-res & evidence
                  </span>
                </div>
              </div>
            </div>

            {/* Right: Selected Candidate Inspector & Quick Action Panel */}
            <div className="lg:col-span-5 xl:col-span-4 p-4 sm:p-5 flex flex-col justify-between bg-slate-50/50 border-t lg:border-t-0 lg:border-l border-slate-200/90">
              <div className="space-y-4">
                {/* Header Info */}
                <div className="flex items-start justify-between gap-2 border-b border-slate-200/80 pb-3">
                  <div>
                    <h3 className="text-sm font-extrabold text-slate-900 flex items-center gap-1.5">
                      <span>Candidate Inspector</span>
                      <span className="text-xs font-mono font-normal text-slate-500">
                        ({selectedIndex + 1}/{results.length})
                      </span>
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Direct submission and provenance verification
                    </p>
                  </div>
                  <Badge variant="outline" className="font-mono text-xs font-bold bg-white text-indigo-700 border-indigo-200">
                    Rank #{selectedIndex + 1}
                  </Badge>
                </div>

                {/* Primary CTA: Big DRES Submit Button */}
                <div>
                  <button
                    type="button"
                    onClick={() => onSubmitToDres(selectedHit)}
                    className="w-full bg-emerald-600 hover:bg-emerald-700 active:scale-[0.98] text-white font-black py-3 px-4 rounded-xl shadow-sm hover:shadow-md transition-all flex items-center justify-center gap-2.5 text-sm"
                    title="Submit this exact video moment to DRES (Press Enter)"
                  >
                    <Send className="h-4 w-4" />
                    <span>Submit Candidate #{selectedIndex + 1} to DRES</span>
                    <span className="ml-1 px-1.5 py-0.5 bg-emerald-700/80 rounded text-[10px] font-mono font-normal opacity-85">
                      ↵ Enter
                    </span>
                  </button>
                </div>

                {/* Secondary Action Buttons Grid */}
                <div className="grid grid-cols-3 gap-2">
                  <button
                    type="button"
                    onClick={() => onPlay(videoName, evidenceTimestamp, evidenceFrameIdx)}
                    className="bg-white hover:bg-slate-100 active:scale-[0.97] text-slate-800 border border-slate-200 font-semibold py-2 px-2.5 rounded-lg flex items-center justify-center gap-1.5 text-xs shadow-2xs transition-all"
                    title="Play video snippet around this moment"
                  >
                    <Play className="h-3.5 w-3.5 fill-indigo-600 text-indigo-600" />
                    <span>Play Video</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onBrowseVideo(videoName)}
                    className="bg-white hover:bg-indigo-50 active:scale-[0.97] text-indigo-700 border border-indigo-200/80 font-semibold py-2 px-2.5 rounded-lg flex items-center justify-center gap-1.5 text-xs shadow-2xs transition-all"
                    title="Open Timeline to view surrounding keyframes (±30s)"
                  >
                    <Film className="h-3.5 w-3.5 text-indigo-600" />
                    <span>Timeline</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onInspect(selectedHit, selectedIndex + 1)}
                    className="bg-white hover:bg-slate-100 active:scale-[0.97] text-slate-700 border border-slate-200 font-semibold py-2 px-2.5 rounded-lg flex items-center justify-center gap-1.5 text-xs shadow-2xs transition-all"
                    title="View detailed JSON evidence and multimodal vectors"
                  >
                    <Maximize2 className="h-3.5 w-3.5 text-slate-600" />
                    <span>Details</span>
                  </button>
                </div>

                {/* Evidence Caption Block */}
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs font-bold text-slate-700">
                    <span className="flex items-center gap-1">
                      <FileText className="h-3.5 w-3.5 text-indigo-600" />
                      Visual Caption / Context
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono font-normal">WeMM-4B</span>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-slate-200 text-xs text-slate-700 leading-relaxed max-h-32 overflow-y-auto font-sans shadow-2xs">
                    {captionText}
                  </div>
                </div>

                {/* OCR Text (if present) */}
                {ocrText && (
                  <div className="space-y-1">
                    <span className="text-[11px] font-bold text-slate-700 flex items-center gap-1">
                      <Tag className="h-3 w-3 text-emerald-600" />
                      Detected OCR Text
                    </span>
                    <div className="bg-emerald-50/60 border border-emerald-200/60 rounded px-2.5 py-1 text-xs font-mono text-emerald-900 truncate">
                      "{ocrText}"
                    </div>
                  </div>
                )}

                {/* Detected Objects Tags */}
                {detectedObjects.length > 0 && (
                  <div className="space-y-1.5">
                    <span className="text-[11px] font-bold text-slate-600 flex items-center gap-1">
                      <Tag className="h-3 w-3 text-slate-400" />
                      YOLOE-26 Grounded Objects
                    </span>
                    <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto">
                      {detectedObjects.slice(0, 8).map((obj, i) => (
                        <span
                          key={i}
                          className="px-1.5 py-0.5 bg-slate-100 text-slate-700 rounded text-[10px] font-mono border border-slate-200"
                        >
                          {obj.label} {(obj.conf * 100).toFixed(0)}%
                        </span>
                      ))}
                      {detectedObjects.length > 8 && (
                        <span className="px-1.5 py-0.5 bg-slate-100 text-slate-500 rounded text-[10px] font-mono">
                          +{detectedObjects.length - 8} more
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Bottom Keyboard Quick Nav Guide */}
              <div className="pt-3 mt-3 border-t border-slate-200/80 flex items-center justify-between text-[11px] text-slate-500 font-mono">
                <span className="flex items-center gap-1">
                  <kbd className="px-1 py-0.5 bg-white border border-slate-200 rounded text-[10px] font-bold text-slate-700">←</kbd>
                  <kbd className="px-1 py-0.5 bg-white border border-slate-200 rounded text-[10px] font-bold text-slate-700">→</kbd>
                  <span className="text-slate-400 ml-1">Step</span>
                </span>
                <span className="flex items-center gap-1">
                  <kbd className="px-1 py-0.5 bg-white border border-slate-200 rounded text-[10px] font-bold text-slate-700">1-9</kbd>
                  <span className="text-slate-400 ml-1">Jump</span>
                </span>
                <span className="text-indigo-600 font-semibold">
                  Click below to switch
                </span>
              </div>
            </div>
          </div>

          {/* Interactive Candidate Thumbnails Gallery Strip */}
          <div className="space-y-2">
            <div className="flex items-center justify-between px-1">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                <span>All Candidate Shots</span>
                <span className="px-1.5 py-0.2 bg-slate-200 text-slate-700 rounded text-[11px] font-mono font-semibold">
                  {results.length}
                </span>
              </h4>
              <span className="text-[11px] text-slate-500 font-mono">
                Click any thumbnail to preview in large hero view
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8 gap-2.5">
              {results.map((hit, idx) => {
                const isSelected = idx === selectedIndex
                const p = hit.payload || {}
                const vName = p.source_file || "Unknown Video"
                const cleanVName = vName.replace(/\.[^/.]+$/, "")
                const fIdx = Number.isInteger(hit.evidence_frame_idx)
                  ? hit.evidence_frame_idx
                  : Number.isInteger(p.frame_idx)
                  ? p.frame_idx
                  : null
                const ts =
                  typeof hit.evidence_timestamp === "number" && Number.isFinite(hit.evidence_timestamp)
                    ? hit.evidence_timestamp
                    : typeof p.timestamp === "number" && Number.isFinite(p.timestamp)
                    ? p.timestamp
                    : 0.0

                const fParams = new URLSearchParams({ video_name: hit.evidence_media_name || vName })
                if (fIdx != null) fParams.set("frame_idx", String(fIdx))
                else if (ts != null) fParams.set("timestamp", String(ts))
                const thumbUrl = `${BACKEND_URL}/api/media/frame?${fParams.toString()}`

                const scoreTxt =
                  hit.score != null
                    ? hit.score.toFixed(3)
                    : hit.rrf_score != null
                    ? hit.rrf_score.toFixed(4)
                    : "N/A"

                return (
                  <div
                    key={hit.id || idx}
                    onClick={() => setSelectedIndex(idx)}
                    className={`group relative bg-white rounded-xl border transition-all duration-150 flex flex-col overflow-hidden cursor-pointer select-none text-left ${
                      isSelected
                        ? "ring-2 ring-indigo-600 ring-offset-2 border-indigo-600 shadow-md bg-indigo-50/20 scale-[1.02] z-10"
                        : "border-slate-200/90 hover:border-indigo-400 hover:shadow-xs"
                    }`}
                  >
                    {/* Thumbnail Image */}
                    <div className="relative aspect-video bg-slate-950 overflow-hidden">
                      <img
                        src={thumbUrl}
                        alt={`Rank ${idx + 1}`}
                        className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-200"
                        loading="lazy"
                      />

                      {/* Top-left: Rank Badge */}
                      <div className="absolute top-1.5 left-1.5 flex items-center gap-1 z-10">
                        <span
                          className={`px-1.5 py-0.5 text-[10px] font-black rounded shadow-xs font-mono ${
                            isSelected
                              ? "bg-indigo-600 text-white ring-1 ring-white/50"
                              : "bg-slate-900/90 backdrop-blur-xs text-white"
                          }`}
                        >
                          #{idx + 1}
                        </span>
                        {isSelected && (
                          <span className="px-1 py-0.5 bg-emerald-500 text-white text-[9px] font-bold rounded shadow-xs">
                            Active
                          </span>
                        )}
                      </div>

                      {/* Top-right: Score Badge */}
                      <div className="absolute top-1.5 right-1.5 flex items-center gap-1 z-10">
                        <span className="px-1.5 py-0.5 bg-white/95 backdrop-blur-xs text-slate-800 text-[10px] font-bold rounded shadow-xs border border-slate-200/80 font-mono">
                          {scoreTxt}
                        </span>
                      </div>

                      {/* Bottom Scrim: Video Name & Timestamp */}
                      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 via-black/40 to-transparent pt-4 pb-1 px-2 flex items-center justify-between text-white text-[10px] font-mono">
                        <span className="truncate max-w-[65%] font-medium" title={vName}>
                          {cleanVName}
                        </span>
                        <span className="font-semibold shrink-0 text-slate-200">
                          {ts.toFixed(1)}s
                        </span>
                      </div>

                      {/* Fast Hover Action Overlay */}
                      <div className="absolute inset-0 bg-slate-950/40 opacity-0 group-hover:opacity-100 transition-opacity duration-150 flex items-center justify-center gap-1.5 backdrop-blur-2xs z-20">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            onPlay(vName, ts, fIdx)
                          }}
                          className="p-1.5 rounded-full bg-white/95 hover:bg-white text-indigo-600 shadow-md transition-transform hover:scale-110"
                          title="Play video"
                        >
                          <Play className="h-3 w-3 fill-indigo-600" />
                        </button>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            onBrowseVideo(vName)
                          }}
                          className="p-1.5 rounded-full bg-white/95 hover:bg-white text-indigo-600 shadow-md transition-transform hover:scale-110"
                          title="Browse Timeline"
                        >
                          <Film className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            onInspect(hit, idx + 1)
                          }}
                          className="p-1.5 rounded-full bg-white/95 hover:bg-white text-indigo-600 shadow-md transition-transform hover:scale-110"
                          title="Inspect Details"
                        >
                          <Maximize2 className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            onSubmitToDres(hit)
                          }}
                          className="p-1.5 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white shadow-md transition-transform hover:scale-110"
                          title="Submit to DRES"
                        >
                          <Send className="h-3 w-3" />
                        </button>
                      </div>
                    </div>

                    {/* Sub-bar: Status / Caption preview */}
                    <div
                      className={`px-2 py-1.5 border-t flex items-center justify-between gap-1 text-[11px] ${
                        isSelected
                          ? "bg-indigo-50/70 border-indigo-200 text-indigo-950 font-semibold"
                          : "bg-slate-50/80 border-slate-100 text-slate-500"
                      }`}
                    >
                      <span className="truncate text-[10px]" title={p.caption || ""}>
                        {p.caption || cleanVName}
                      </span>
                      {isSelected ? (
                        <span className="text-[10px] text-indigo-700 font-bold shrink-0 flex items-center gap-0.5">
                          <CheckCircle2 className="h-2.5 w-2.5 text-indigo-600" />
                          Viewing
                        </span>
                      ) : (
                        <span className="text-[10px] text-slate-400 group-hover:text-indigo-600 font-bold shrink-0">
                          Select
                        </span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </>
      )}

      {/* Mode 2: Traditional Grid Overview (For operators who prefer equal-sized cards) */}
      {viewMode === "grid" && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8 gap-2.5">
          {results.map((hit, idx) => (
            <ResultCard
              key={hit.id || idx}
              hit={hit}
              idx={idx}
              onInspect={(h, i) => onInspect(h, i + 1)}
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
  )
}
