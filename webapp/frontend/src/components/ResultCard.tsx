import {
  Play,
  Film,
  Send,
  Maximize2,
  Volume2,
  CheckCircle2,
} from "lucide-react"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

export interface ResultHit {
  id: string
  score?: number
  rrf_score?: number
  answer?: string | null
  vqa_answer?: string | null
  vqa_score?: number
  vqa_answer_valid?: boolean
  vqa_evidence_available?: boolean
  vqa_evidence_reason?: string
  answer_candidate_id?: string | null
  answer_video_id?: string | null
  answer_frame_idx?: number | null
  evidence_media_name?: string | null
  evidence_frame_idx?: number | null
  evidence_timestamp?: number | null
  matched_via?: string[]
  payload: {
    source_file?: string
    timestamp?: number
    frame_idx?: number
    caption?: string
    scene_narrative?: string
    ocr_text?: string
    modality?: string
    detected_objects?: Array<{ label: string; conf: number }>
  }
}

interface ResultCardProps {
  hit: ResultHit
  idx: number
  isExpanded?: boolean
  onToggleExpand?: () => void
  onInspect?: (hit: ResultHit, idx: number) => void
  onPlay: (videoName: string, time: number, frameIdx?: number | null) => void
  onFeedback?: (id: string, positive: boolean) => void
  onUseAsQuery?: (id: string) => void
  onInVideoSearch?: (videoName: string) => void
  onBrowseVideo: (videoName: string) => void
  onSubmitToDres: (hit: ResultHit) => void
}

export function ResultCard({
  hit,
  idx,
  onInspect,
  onPlay,
  onBrowseVideo,
  onSubmitToDres,
}: ResultCardProps) {
  const payload = hit.payload || {}
  const videoName = payload.source_file || "Unknown Video"
  const evidenceFrameIdx = Number.isInteger(hit.evidence_frame_idx)
    ? hit.evidence_frame_idx
    : Number.isInteger(payload.frame_idx)
    ? payload.frame_idx
    : null
  const evidenceTimestamp =
    typeof hit.evidence_timestamp === "number" && Number.isFinite(hit.evidence_timestamp)
      ? hit.evidence_timestamp
      : typeof payload.timestamp === "number" && Number.isFinite(payload.timestamp)
      ? payload.timestamp
      : null
  const displayMediaName = hit.evidence_media_name || videoName
  const playbackTimestamp = evidenceTimestamp ?? 0.0

  const frameParams = new URLSearchParams({ video_name: displayMediaName })
  if (evidenceFrameIdx != null) frameParams.set("frame_idx", String(evidenceFrameIdx))
  else if (evidenceTimestamp != null) frameParams.set("timestamp", String(evidenceTimestamp))
  const frameUrl = `${BACKEND_URL}/api/media/frame?${frameParams.toString()}`

  const scoreDisplay =
    hit.score != null
      ? hit.score.toFixed(3)
      : hit.rrf_score != null
      ? hit.rrf_score.toFixed(4)
      : "N/A"

  const hasVqa = Boolean(hit.answer)

  return (
    <div
      onClick={() => onInspect?.(hit, idx)}
      className="group relative bg-white rounded-lg border border-slate-200/90 hover:border-indigo-500 hover:shadow-lg transition-all duration-150 flex flex-col overflow-hidden cursor-pointer select-none text-left"
    >
      {/* 16:9 Thumbnail Image */}
      <div className="relative aspect-video bg-slate-900 overflow-hidden">
        <img
          src={frameUrl}
          alt={`Frame ${displayMediaName}`}
          className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-200"
          loading="lazy"
        />

        {/* Top-left: Rank Badge */}
        <div className="absolute top-1.5 left-1.5 flex items-center gap-1 z-10">
          <span className="px-1.5 py-0.5 bg-slate-900/90 backdrop-blur-xs text-white text-[10px] font-black rounded shadow-xs font-mono">
            #{idx + 1}
          </span>
        </div>

        {/* Top-right: Score Badge */}
        <div className="absolute top-1.5 right-1.5 flex items-center gap-1 z-10">
          <span className="px-1.5 py-0.5 bg-white/95 backdrop-blur-xs text-slate-800 text-[10px] font-bold rounded shadow-xs border border-slate-200/80 font-mono">
            {scoreDisplay}
          </span>
          {payload.modality === "ambient_audio" && (
            <span className="p-0.5 bg-white/95 text-indigo-600 rounded shadow-xs" title="Audio modality">
              <Volume2 className="h-3 w-3" />
            </span>
          )}
        </div>

        {/* Bottom Scrim: Video Name & Timestamp */}
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 via-black/40 to-transparent pt-4 pb-1 px-2 flex items-center justify-between text-white text-[10px] font-mono">
          <span className="truncate max-w-[65%] font-medium" title={videoName}>
            {videoName.replace(/\.[^/.]+$/, "")}
          </span>
          <span className="font-semibold shrink-0 text-slate-200">
            {playbackTimestamp.toFixed(1)}s
          </span>
        </div>

        {/* Hover Action Overlay */}
        <div className="absolute inset-0 bg-slate-950/40 opacity-0 group-hover:opacity-100 transition-opacity duration-150 flex items-center justify-center gap-1.5 backdrop-blur-2xs z-20">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              onInspect?.(hit, idx)
            }}
            className="p-2 rounded-full bg-white/95 hover:bg-white text-indigo-600 shadow-md transition-transform hover:scale-110"
            title="Inspect Details (Mở chi tiết)"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              onPlay(videoName, playbackTimestamp, evidenceFrameIdx)
            }}
            className="p-2 rounded-full bg-white/95 hover:bg-white text-indigo-600 shadow-md transition-transform hover:scale-110"
            title="Play video"
          >
            <Play className="h-3.5 w-3.5 fill-indigo-600" />
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              onBrowseVideo(videoName)
            }}
            className="p-2 rounded-full bg-white/95 hover:bg-white text-indigo-600 shadow-md transition-transform hover:scale-110"
            title="Inspect Timeline (±30s)"
          >
            <Film className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              onSubmitToDres(hit)
            }}
            className="p-2 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white shadow-md transition-transform hover:scale-110"
            title="Submit to DRES"
          >
            <Send className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* VQA Answer Banner (if present) */}
      {hasVqa && (
        <div className="px-2 py-1 bg-emerald-50 border-t border-emerald-100 flex items-center justify-between text-[11px] font-bold text-emerald-800 truncate">
          <span className="truncate flex items-center gap-1">
            <CheckCircle2 className="h-3 w-3 shrink-0 text-emerald-600" />
            <span className="truncate">{hit.answer}</span>
          </span>
        </div>
      )}

      {/* Sub-bar: Compact metadata & quick action trigger */}
      <div className="px-2 py-1.5 bg-slate-50/80 border-t border-slate-100 flex items-center justify-between gap-1 text-[11px]">
        <span className="text-slate-500 truncate text-[10px] font-medium max-w-[70%]" title={payload.caption || ""}>
          {payload.caption || videoName}
        </span>
        <span className="text-[10px] text-indigo-600 font-bold hover:underline shrink-0 flex items-center gap-0.5">
          Chi tiết
        </span>
      </div>
    </div>
  )
}
