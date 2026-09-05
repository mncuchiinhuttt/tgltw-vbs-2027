import {
  Play,
  Film,
  Send,
  Search,
  Sparkles,
  ThumbsUp,
  ThumbsDown,
  Tag,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Eye,
} from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { type ResultHit } from "@/components/ResultCard"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

interface CandidateInspectionDialogProps {
  hit: ResultHit | null
  rank: number | null
  open: boolean
  onClose: () => void
  onPlay: (videoName: string, time: number, frameIdx?: number | null) => void
  onFeedback: (id: string, positive: boolean) => void
  onUseAsQuery: (id: string) => void
  onInVideoSearch: (videoName: string) => void
  onBrowseVideo: (videoName: string) => void
  onSubmitToDres: (hit: ResultHit) => void
}

export function CandidateInspectionDialog({
  hit,
  rank,
  open,
  onClose,
  onPlay,
  onFeedback,
  onUseAsQuery,
  onInVideoSearch,
  onBrowseVideo,
  onSubmitToDres,
}: CandidateInspectionDialogProps) {
  if (!hit) return null

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
      ? hit.score.toFixed(4)
      : hit.rrf_score != null
      ? hit.rrf_score.toFixed(4)
      : "N/A"

  const hasVqa = Boolean(hit.answer) || hit.vqa_answer_valid !== undefined
  const detectedObjects = payload.detected_objects || []

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <DialogContent className="border-slate-200 bg-white p-0 overflow-hidden shadow-2xl rounded-2xl max-w-5xl w-[95vw] max-h-[92vh] flex flex-col">
        {/* Header Bar */}
        <DialogHeader className="p-4 sm:p-5 border-b border-slate-100 bg-slate-50/70 flex-shrink-0">
          <div className="flex flex-wrap items-center justify-between gap-3 pr-8">
            <div className="flex items-center gap-2.5 min-w-0">
              {rank != null && (
                <span className="px-2.5 py-1 bg-slate-900 text-white text-xs font-black rounded-md shadow-xs">
                  #{rank}
                </span>
              )}
              <div className="min-w-0">
                <DialogTitle className="text-slate-900 font-bold text-base sm:text-lg truncate">
                  {videoName}
                </DialogTitle>
                <DialogDescription className="text-slate-500 text-xs flex items-center gap-2 flex-wrap mt-0.5">
                  <span className="font-mono font-medium">
                    Time: {playbackTimestamp.toFixed(2)}s
                  </span>
                  <span>·</span>
                  <span className="font-mono font-medium">
                    Frame: {evidenceFrameIdx ?? "N/A"}
                  </span>
                  <span>·</span>
                  <span className="font-mono font-semibold text-indigo-600">
                    Score: {scoreDisplay}
                  </span>
                </DialogDescription>
              </div>
            </div>

            {/* Header Quick Actions */}
            <div className="flex items-center gap-2 flex-wrap">
              <button
                type="button"
                onClick={() => {
                  onSubmitToDres(hit)
                  onClose()
                }}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg shadow-sm transition-all"
              >
                <Send className="h-3.5 w-3.5" />
                Nộp DRES
              </button>
              <button
                type="button"
                onClick={() => {
                  onBrowseVideo(videoName)
                  onClose()
                }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-semibold rounded-lg shadow-2xs transition-all"
              >
                <Film className="h-3.5 w-3.5 text-indigo-600" />
                Timeline ±30s
              </button>
              <button
                type="button"
                onClick={() => {
                  onPlay(videoName, playbackTimestamp, evidenceFrameIdx)
                  onClose()
                }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-semibold rounded-lg shadow-2xs transition-all"
              >
                <Play className="h-3.5 w-3.5 fill-slate-700" />
                Phát video
              </button>
            </div>
          </div>
        </DialogHeader>

        {/* Scrollable Body Content */}
        <div className="p-4 sm:p-6 overflow-y-auto flex-1 grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Left: Keyframe Visual Inspection (7 cols) */}
          <div className="lg:col-span-7 flex flex-col gap-3">
            <div className="relative aspect-video bg-slate-950 rounded-xl overflow-hidden border border-slate-200 shadow-md group flex items-center justify-center">
              <img
                src={frameUrl}
                alt={`Keyframe ${videoName}`}
                className="object-contain w-full h-full max-h-[55vh]"
              />
              <div className="absolute top-2.5 left-2.5 flex items-center gap-1.5">
                <span className="px-2 py-0.5 bg-black/75 backdrop-blur-sm text-white text-[11px] font-mono font-bold rounded">
                  {playbackTimestamp.toFixed(2)}s
                </span>
                {evidenceFrameIdx != null && (
                  <span className="px-2 py-0.5 bg-black/75 backdrop-blur-sm text-slate-300 text-[11px] font-mono rounded">
                    Frame {evidenceFrameIdx}
                  </span>
                )}
              </div>
              <div className="absolute top-2.5 right-2.5">
                <Badge variant="outline" className="bg-black/75 backdrop-blur-sm text-white border-white/20 text-[10px] uppercase font-bold tracking-wider">
                  {payload.modality || "Visual Evidence"}
                </Badge>
              </div>
            </div>

            {/* Image Action Row */}
            <div className="flex items-center justify-between gap-2 p-2.5 bg-slate-50 border border-slate-200/80 rounded-xl text-xs">
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => onFeedback(hit.id, true)}
                  className="p-1.5 text-slate-600 hover:text-emerald-600 hover:bg-emerald-50 rounded-md transition"
                  title="Mark relevant (Positive feedback)"
                >
                  <ThumbsUp className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  onClick={() => onFeedback(hit.id, false)}
                  className="p-1.5 text-slate-600 hover:text-rose-600 hover:bg-rose-50 rounded-md transition"
                  title="Mark irrelevant (Negative feedback)"
                >
                  <ThumbsDown className="h-4 w-4" />
                </button>
                <span className="text-slate-300">|</span>
                <span className="text-slate-500 text-[11px]">Relevance feedback</span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => {
                    onUseAsQuery(hit.id)
                    onClose()
                  }}
                  className="inline-flex items-center gap-1 text-[11px] font-semibold text-slate-700 hover:text-indigo-600 hover:bg-white px-2 py-1 rounded transition"
                >
                  <Search className="h-3 w-3" />
                  Tìm tương tự
                </button>
                <button
                  type="button"
                  onClick={() => {
                    onInVideoSearch(videoName)
                    onClose()
                  }}
                  className="inline-flex items-center gap-1 text-[11px] font-semibold text-slate-700 hover:text-indigo-600 hover:bg-white px-2 py-1 rounded transition"
                >
                  <Sparkles className="h-3 w-3 text-indigo-600" />
                  Tìm trong video
                </button>
              </div>
            </div>
          </div>

          {/* Right: Detailed Multimodal Evidence & Metadata (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-3.5 text-left">
            {/* Grounded VQA Card (if available) */}
            {hasVqa && (
              <div className="p-3.5 rounded-xl border border-indigo-200 bg-indigo-50/50 shadow-2xs space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-700 flex items-center gap-1">
                    <Sparkles className="h-3 w-3" /> Grounded VQA Answer
                  </span>
                  {hit.vqa_evidence_available !== false ? (
                    <span className="inline-flex items-center gap-1 text-[10px] font-bold text-emerald-700 bg-emerald-100/70 px-2 py-0.5 rounded-full">
                      <CheckCircle2 className="h-3 w-3" /> Grounding Verified
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-[10px] font-bold text-amber-700 bg-amber-100/70 px-2 py-0.5 rounded-full">
                      <AlertTriangle className="h-3 w-3" /> Fail-Closed Refusal
                    </span>
                  )}
                </div>
                <p className="text-sm font-bold text-slate-900 m-0">
                  {hit.answer || "UNKNOWN / N/A"}
                </p>
                {hit.vqa_evidence_reason && (
                  <p className="text-[11px] text-slate-600 italic m-0">
                    {hit.vqa_evidence_reason}
                  </p>
                )}
              </div>
            )}

            {/* Detected Objects (YOLOE-26) */}
            <div className="p-3.5 rounded-xl border border-slate-200/80 bg-white shadow-2xs space-y-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                <Tag className="h-3 w-3 text-slate-400" /> Detected Objects ({detectedObjects.length})
              </span>
              {detectedObjects.length > 0 ? (
                <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
                  {detectedObjects.map((obj, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 bg-slate-100 border border-slate-200 text-slate-700 text-[11px] font-semibold rounded-md"
                    >
                      {obj.label}{" "}
                      <span className="text-slate-400 font-normal">
                        {Math.round(obj.conf * 100)}%
                      </span>
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-400 italic m-0">No object boxes in this shot</p>
              )}
            </div>

            {/* OCR Text */}
            {payload.ocr_text && (
              <div className="p-3.5 rounded-xl border border-slate-200/80 bg-white shadow-2xs space-y-1.5">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                  <FileText className="h-3 w-3 text-slate-400" /> On-Screen Text (PP-OCRv6)
                </span>
                <p className="text-xs text-slate-800 font-mono bg-slate-50 p-2 rounded-lg border border-slate-200/60 m-0 max-h-20 overflow-y-auto break-words">
                  {payload.ocr_text}
                </p>
              </div>
            )}

            {/* Caption & Scene Narrative */}
            <div className="p-3.5 rounded-xl border border-slate-200/80 bg-white shadow-2xs space-y-1.5 flex-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                <Eye className="h-3 w-3 text-slate-400" /> Scene Caption & Narrative
              </span>
              <p className="text-xs text-slate-700 leading-relaxed m-0 max-h-28 overflow-y-auto">
                {payload.caption || payload.scene_narrative || "No narrative caption available for this keyframe."}
              </p>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
