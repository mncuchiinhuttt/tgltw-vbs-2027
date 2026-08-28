import { useEffect, useState } from "react"
import { Search, RotateCcw, Play, Sparkles } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

interface BrowseFrame {
  id: string
  score?: number
  dense_score?: number
  text_score?: number
  payload: {
    frame_idx?: number
    timestamp?: number
    caption?: string
    ocr_text?: string
    scene_narrative?: string
  }
}

interface BrowseVideoDialogProps {
  videoName: string | null
  onClose: () => void
  onPlayFrame: (videoName: string, time: number, frameIdx?: number | null) => void
}

/**
 * Full keyframe browse for one video (GET /api/browse-video/{video_name},
 * which reuses HybridSearcher.get_all_points_for_video verbatim) - lets
 * the operator page through everything indexed for a video they've
 * spotted as promising, separate from the single-frame video-modal
 * player already used elsewhere in SearchView.
 */
export function BrowseVideoDialog({ videoName, onClose, onPlayFrame }: BrowseVideoDialogProps) {
  const [frames, setFrames] = useState<BrowseFrame[]>([])
  const [loading, setLoading] = useState(false)
  const [reranking, setReranking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [subQuery, setSubQuery] = useState("")
  const [isReranked, setIsReranked] = useState(false)

  const fetchFrames = (video: string) => {
    setLoading(true)
    setError(null)
    setIsReranked(false)
    setSubQuery("")
    fetch(`${BACKEND_URL}/api/browse-video/${encodeURIComponent(video)}`)
      .then((res) => res.json())
      .then((data) => setFrames(data.frames || []))
      .catch((err) => setError(err.message || "Failed to load video frames"))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    if (!videoName) return
    fetchFrames(videoName)
  }, [videoName])

  const handleInVideoRerank = (e: React.FormEvent) => {
    e.preventDefault()
    if (!videoName || !subQuery.trim()) return
    setReranking(true)
    setError(null)
    fetch(`${BACKEND_URL}/api/video/${encodeURIComponent(videoName)}/rerank`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: subQuery.trim(), top_k: 30 }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.results) {
          setFrames(data.results)
          setIsReranked(true)
        }
      })
      .catch((err) => setError(err.message || "In-video search failed"))
      .finally(() => setReranking(false))
  }

  return (
    <Dialog open={!!videoName} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="border-slate-200 bg-white p-0 overflow-hidden shadow-2xl rounded-2xl max-w-4xl">
        <DialogHeader className="p-6 pb-3 border-b border-slate-100 bg-slate-50/50">
          <div className="flex items-center justify-between">
            <div>
              <DialogTitle className="text-slate-900 font-bold text-lg truncate pr-6">{videoName}</DialogTitle>
              <DialogDescription className="text-slate-500 text-xs mt-0.5">
                {loading
                  ? "Loading indexed frames..."
                  : isReranked
                  ? `Ranked top ${frames.length} frames for "${subQuery}"`
                  : `${frames.length} indexed keyframes in chronological timeline`}
              </DialogDescription>
            </div>
          </div>

          <form onSubmit={handleInVideoRerank} className="mt-3 flex items-center gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search scene or object within this video (e.g. red umbrella, person enters car)..."
                value={subQuery}
                onChange={(e) => setSubQuery(e.target.value)}
                className="w-full pl-9 pr-3 py-1.5 text-xs bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all placeholder:text-slate-400"
              />
            </div>
            <button
              type="submit"
              disabled={reranking || !subQuery.trim()}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-xs font-semibold rounded-lg shadow-sm transition-all"
            >
              <Sparkles className="h-3.5 w-3.5" />
              {reranking ? "Ranking..." : "Rerank Video"}
            </button>
            {isReranked && (
              <button
                type="button"
                onClick={() => videoName && fetchFrames(videoName)}
                className="inline-flex items-center gap-1 px-2.5 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium rounded-lg transition-all"
                title="Reset to chronological order"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset
              </button>
            )}
          </form>
        </DialogHeader>

        <div className="p-6 pt-4 max-h-[62vh] overflow-y-auto">
          {error && <div className="p-3 mb-4 rounded-lg bg-red-50 text-red-600 text-xs font-medium border border-red-100">{error}</div>}
          
          <div className="grid grid-cols-4 gap-3">
            {frames.map((frame, index) => {
              const timestamp = typeof frame.payload.timestamp === "number" && Number.isFinite(frame.payload.timestamp)
                ? frame.payload.timestamp : null
              const frameIdx = Number.isInteger(frame.payload.frame_idx) ? frame.payload.frame_idx : null
              const frameParams = new URLSearchParams({ video_name: videoName || "" })
              if (frameIdx != null) frameParams.set("frame_idx", String(frameIdx))
              else if (timestamp != null) frameParams.set("timestamp", String(timestamp))
              const frameUrl = `${BACKEND_URL}/api/media/frame?${frameParams.toString()}`
              
              return (
                <div
                  key={frame.id}
                  className="group relative bg-white rounded-xl overflow-hidden border border-slate-200 hover:border-indigo-400 hover:shadow-md transition-all flex flex-col"
                >
                  <div className="relative aspect-video bg-slate-100 overflow-hidden">
                    <img src={frameUrl} alt="" className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-300" loading="lazy" />
                    
                    <button
                      onClick={() => videoName && onPlayFrame(videoName, timestamp ?? 0.0, frameIdx)}
                      className="absolute inset-0 bg-slate-900/30 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity"
                    >
                      <div className="p-2 rounded-full bg-white/90 text-indigo-600 shadow">
                        <Play className="h-4 w-4 fill-indigo-600" />
                      </div>
                    </button>

                    <div className="absolute top-1.5 left-1.5 flex gap-1">
                      {isReranked && (
                        <span className="bg-indigo-600 text-white text-[9px] font-bold px-1.5 py-0.5 rounded shadow">
                          #{index + 1}
                        </span>
                      )}
                      {frame.score != null && (
                        <span className="bg-slate-900/80 text-white text-[9px] font-mono px-1.5 py-0.5 rounded shadow">
                          {(frame.score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>

                    <span className="absolute bottom-1.5 right-1.5 bg-slate-900/80 text-white text-[9px] font-mono px-1.5 py-0.5 rounded shadow">
                      {frameIdx != null ? `F#${frameIdx}` : (timestamp != null ? `${timestamp.toFixed(1)}s` : "?")}
                    </span>
                  </div>

                  {frame.payload.caption && (
                    <div className="p-2 bg-slate-50/80 border-t border-slate-100">
                      <p className="text-[10px] text-slate-600 line-clamp-2 leading-tight">
                        {frame.payload.caption}
                      </p>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
