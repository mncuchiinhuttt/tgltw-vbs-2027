import {
  Play,
  Tag,
  Clock,
  CheckCircle2,
  Volume2,
  ChevronDown,
  ChevronUp,
  ThumbsUp,
  ThumbsDown,
  ImageIcon,
  Video,
  Send,
} from "lucide-react"
import { Card, CardFooter } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

const BACKEND_URL = "http://localhost:8000"

export interface ResultHit {
  id: string
  score?: number
  rrf_score?: number
  answer?: string | null
  payload: {
    source_file?: string
    timestamp?: number
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
  isExpanded: boolean
  onToggleExpand: () => void
  onPlay: (videoName: string, time: number) => void
  onFeedback: (id: string, positive: boolean) => void
  onUseAsQuery: (id: string) => void
  onInVideoSearch: (videoName: string) => void
  onBrowseVideo: (videoName: string) => void
  onSubmitToDres: (hit: ResultHit) => void
}

/**
 * Single search-result card, extracted out of SearchView (was inline JSX
 * spanning ~120 lines) both to keep files under the project's 200-line
 * guideline and to give the VBS interactive-session actions (feedback,
 * query-by-example, in-video search, DRES submit) a natural home per-card
 * without growing App.tsx further.
 */
export function ResultCard({
  hit,
  idx,
  isExpanded,
  onToggleExpand,
  onPlay,
  onFeedback,
  onUseAsQuery,
  onInVideoSearch,
  onBrowseVideo,
  onSubmitToDres,
}: ResultCardProps) {
  const payload = hit.payload || {}
  const videoName = payload.source_file || "Unknown File"
  const timestamp = payload.timestamp !== undefined ? payload.timestamp : 0.0
  const frameUrl = `${BACKEND_URL}/api/media/frame?video_name=${encodeURIComponent(videoName)}&timestamp=${timestamp}`

  return (
    <Card className="tech-card overflow-hidden flex flex-col justify-between">
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
              Score: {hit.score != null ? hit.score.toFixed(3) : (hit.rrf_score != null ? hit.rrf_score.toFixed(4) : "N/A")}
            </Badge>
          </div>
          <div className="absolute top-3 right-3">
            <Badge variant="outline" className="bg-white/95 text-xs font-semibold border-slate-200 text-indigo-600 shadow-sm flex gap-1 items-center capitalize">
              {payload.modality === "ambient_audio" ? <Volume2 className="h-3 w-3" /> : <Tag className="h-3 w-3" />}
              {payload.modality || "visual"}
            </Badge>
          </div>
          <div className="absolute inset-0 bg-indigo-900/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
            <button onClick={() => onPlay(videoName, timestamp)} className="bg-indigo-600 hover:bg-indigo-500 text-white rounded-full p-4">
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
              <p className="text-sm text-slate-700 font-semibold">{hit.answer}</p>
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
                    {payload.detected_objects.map((obj, oIdx) => (
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

      <CardFooter className="px-5 pb-5 pt-0 flex flex-col gap-2 border-t border-slate-100 mt-4 pt-3">
        <div className="flex justify-between w-full">
          <button
            onClick={onToggleExpand}
            className="flex items-center gap-1 text-xs font-bold text-slate-400 hover:text-indigo-600 transition-colors"
          >
            {isExpanded ? <>Hide Metadata<ChevronUp className="h-4 w-4" /></> : <>Inspect Metadata<ChevronDown className="h-4 w-4" /></>}
          </button>
          <button
            onClick={() => onPlay(videoName, timestamp)}
            className="bg-indigo-50 hover:bg-indigo-100 border border-indigo-100 text-indigo-750 text-xs font-bold px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-colors"
          >
            <Play className="h-3.5 w-3.5 fill-indigo-600 text-indigo-600" />
            Play Clip
          </button>
        </div>

        {/* VBS interactive-session actions (relevance feedback,
            query-by-example, in-video search, DRES submit) - see Phase C
            (backend) / Phase D (this UI) of the batch-to-interactive plan. */}
        <div className="flex flex-wrap gap-1.5 w-full pt-1">
          <button
            onClick={() => onFeedback(hit.id, true)}
            title="Mark relevant (Rocchio feedback)"
            className="bg-emerald-50 hover:bg-emerald-100 border border-emerald-100 text-emerald-700 text-xs font-bold px-2 py-1 rounded-lg flex items-center gap-1 transition-colors"
          >
            <ThumbsUp className="h-3 w-3" />
          </button>
          <button
            onClick={() => onFeedback(hit.id, false)}
            title="Mark not relevant (Rocchio feedback)"
            className="bg-red-50 hover:bg-red-100 border border-red-100 text-red-700 text-xs font-bold px-2 py-1 rounded-lg flex items-center gap-1 transition-colors"
          >
            <ThumbsDown className="h-3 w-3" />
          </button>
          <button
            onClick={() => onUseAsQuery(hit.id)}
            title="Search for more like this frame"
            className="bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 text-xs font-bold px-2 py-1 rounded-lg flex items-center gap-1 transition-colors"
          >
            <ImageIcon className="h-3 w-3" />
            Dùng làm query
          </button>
          <button
            onClick={() => onInVideoSearch(videoName)}
            title="Search this video's full timeline"
            className="bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 text-xs font-bold px-2 py-1 rounded-lg flex items-center gap-1 transition-colors"
          >
            <Video className="h-3 w-3" />
            Tìm sâu trong video
          </button>
          <button
            onClick={() => onBrowseVideo(videoName)}
            title="Browse all indexed keyframes for this video"
            className="bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 text-xs font-bold px-2 py-1 rounded-lg flex items-center gap-1 transition-colors"
          >
            Browse video
          </button>
          <button
            onClick={() => onSubmitToDres(hit)}
            title="Submit this answer to DRES"
            className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold px-2 py-1 rounded-lg flex items-center gap-1 transition-colors ml-auto"
          >
            <Send className="h-3 w-3" />
            Nộp câu trả lời
          </button>
        </div>
      </CardFooter>
    </Card>
  )
}
