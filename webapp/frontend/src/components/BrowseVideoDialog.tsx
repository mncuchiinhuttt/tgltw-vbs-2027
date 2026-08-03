import { useEffect, useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ""

interface BrowseFrame {
  id: string
  payload: {
    frame_idx?: number
    timestamp?: number
    caption?: string
  }
}

interface BrowseVideoDialogProps {
  videoName: string | null
  onClose: () => void
  onPlayFrame: (videoName: string, time: number) => void
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
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!videoName) return
    setLoading(true)
    setError(null)
    fetch(`${BACKEND_URL}/api/browse-video/${encodeURIComponent(videoName)}`)
      .then((res) => res.json())
      .then((data) => setFrames(data.frames || []))
      .catch((err) => setError(err.message || "Failed to load video frames"))
      .finally(() => setLoading(false))
  }, [videoName])

  return (
    <Dialog open={!!videoName} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="border-slate-200 bg-white p-0 overflow-hidden shadow-2xl rounded-xl max-w-3xl">
        <DialogHeader className="p-6 pb-2">
          <DialogTitle className="text-slate-800 truncate pr-6">{videoName}</DialogTitle>
          <DialogDescription className="text-slate-500">
            {loading ? "Loading indexed frames..." : `${frames.length} indexed keyframes`}
          </DialogDescription>
        </DialogHeader>

        <div className="p-6 pt-2 max-h-[60vh] overflow-y-auto">
          {error && <div className="text-red-600 text-sm font-semibold">{error}</div>}
          <div className="grid grid-cols-4 gap-2">
            {frames.map((frame) => {
              const timestamp = frame.payload.timestamp ?? 0.0
              const frameUrl = `${BACKEND_URL}/api/media/frame?video_name=${encodeURIComponent(videoName || "")}&timestamp=${timestamp}`
              return (
                <button
                  key={frame.id}
                  onClick={() => videoName && onPlayFrame(videoName, timestamp)}
                  className="relative aspect-video bg-slate-100 rounded-lg overflow-hidden border border-slate-200 hover:border-indigo-400 transition-colors"
                  title={`frame_idx ${frame.payload.frame_idx ?? "N/A"}`}
                >
                  <img src={frameUrl} alt="" className="object-cover w-full h-full" loading="lazy" />
                  <span className="absolute bottom-0.5 right-0.5 bg-slate-900/80 text-white text-[9px] font-mono px-1 rounded">
                    {frame.payload.frame_idx ?? "?"}
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
