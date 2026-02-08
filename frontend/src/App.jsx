import { useState, useRef, useCallback, useEffect } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  CartesianGrid,
} from 'recharts'
import './App.css'

const VIDEO_SIZE_THRESHOLD = 1024 * 1024 // 1MB - show processing for larger videos
const REVERSE_SPEED = 1 // normal speed when reversing
const POST_PROCESSING_DELAY_MS = 1000 // wait 1 sec after processing before showing playback view (for small videos)
const LARGE_VIDEO_PROCESSING_MS = 2500 // fixed delay for large videos - don't wait for metadata (can hang)
const TRANSITION_VIDEO_PATH = '/video2.mp4'
const TRANSITION_VIDEO_TO_PLAYBACK_MS = 600 // fade duration from transition video end to playback view
const TRANSITION_VIDEO_PLAYBACK_RATE = 1 // 1x for smoother frame pacing (avoid fractional-rate stutter)
// Backend proxy: Vite rewrites /api → http://127.0.0.1:8000
const API_BASE = '/api'
// Endpoints: POST /jobs, GET /jobs/:id, GET /jobs/:id/input-video, GET /jobs/:id/annotated-video
const STATUS_POLL_MS = 2000 // poll job status every 2s during Vision analysis

const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.webm', '.mkv', '.m4v', '.avi']

function isVideoFile(file) {
  if (file.type && file.type.startsWith('video/')) return true
  const name = (file.name || '').toLowerCase()
  return VIDEO_EXTENSIONS.some((ext) => name.endsWith(ext))
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

function abbreviateFileName(name, maxLen = 24) {
  if (name.length <= maxLen) return name
  const ext = name.includes('.') ? name.slice(name.lastIndexOf('.')) : ''
  const base = name.slice(0, name.length - ext.length)
  const keep = maxLen - ext.length - 3
  return keep > 0 ? base.slice(0, keep) + '...' + ext : name.slice(0, maxLen - 3) + '...'
}

export default function App() {
  const [isDragging, setIsDragging] = useState(false)
  const [droppedFiles, setDroppedFiles] = useState([])
  const [playbackIndex, setPlaybackIndex] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [showPlaybackView, setShowPlaybackView] = useState(false)
  const [showTransitionVideo, setShowTransitionVideo] = useState(false)
  const [transitionFadingOut, setTransitionFadingOut] = useState(false)
  const [playbackViewZoomingIn, setPlaybackViewZoomingIn] = useState(false)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [inputVideoUrl, setInputVideoUrl] = useState(null)
  const [playbackBlobUrl, setPlaybackBlobUrl] = useState(null)
  const [showResultView, setShowResultView] = useState(false)
  const [jobError, setJobError] = useState(null)
  const [annotatedVideoBlobUrl, setAnnotatedVideoBlobUrl] = useState(null)
  const [resultActionsOpen, setResultActionsOpen] = useState(false)
  const [resultActionPending, setResultActionPending] = useState(null)
  const [showApprovePage, setShowApprovePage] = useState(false)
  const [approveJobId, setApproveJobId] = useState(null)
  const [reportData, setReportData] = useState({ metrics: null, confidence: null, timeline: null, worldGt: null })
  const [reportMetricsRun, setReportMetricsRun] = useState('baseline')
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState(null)
  const [jsonTab, setJsonTab] = useState('metrics')
  const [resultVideoPlaying, setResultVideoPlaying] = useState(true)
  const videoRef = useRef(null)
  const playbackVideoRef = useRef(null)
  const playbackBlobUrlRef = useRef(null)
  const annotatedBlobUrlRef = useRef(null)
  const transitionVideoRef = useRef(null)
  const dropZoneRef = useRef(null)
  const hasSeenTransitionVideoRef = useRef(false)
  const processingVideosRef = useRef(0)
  const progressIntervalRef = useRef(null)
  const processingTimeoutRef = useRef(null)
  const postProcessingTimerRef = useRef(null)
  const reverseRafRef = useRef(null)
  const statusPollIntervalRef = useRef(null)
  const resultVideoRef = useRef(null)
  const resultViewRef = useRef(null)

  // PiP layout for result view: position (left, top) and width; height = width * 9/16. Draggable, resizable, snaps to edges.
  const PIP_MIN_WIDTH = 160
  const PIP_MAX_WIDTH_RATIO = 0.4
  const PIP_MARGIN = 16
  const [pipLayout, setPipLayout] = useState(null)
  const pipDragRef = useRef({ dragging: false, startX: 0, startY: 0, startLeft: 0, startTop: 0 })
  const pipResizeRef = useRef({ resizing: false, startX: 0, startY: 0, startLeft: 0, startTop: 0, startWidth: 0, startHeight: 0, corner: 'br' })
  const pipLayoutRef = useRef(null)
  pipLayoutRef.current = pipLayout

  const hadFilesRef = useRef(0)
  hadFilesRef.current = droppedFiles.length

  // Open existing job from URL (?job=id): show annotated video in Google Meet layout if job is completed
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const jobFromUrl = params.get('job')
    if (!jobFromUrl || jobId) return
    let cancelled = false
    fetch(`${API_BASE}/jobs/${jobFromUrl}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (cancelled || !data) return
        if (data.status === 'completed') {
          setJobId(data.job_id || jobFromUrl)
          setJobStatus(data)
          setShowPlaybackView(true)
          setShowResultView(true)
        }
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  const playReverseToStart = useCallback(() => {
    const video = videoRef.current
    if (!video || !Number.isFinite(video.duration) || video.duration <= 0) return

    if (reverseRafRef.current) cancelAnimationFrame(reverseRafRef.current)

    video.pause()

    const runReverse = () => {
      const v = videoRef.current
      if (!v) return
      const step = (1 / 60) * REVERSE_SPEED
      const next = Math.max(0, v.currentTime - step)
      v.currentTime = next
      if (next <= 0) {
        v.currentTime = 0
        reverseRafRef.current = null
        return
      }
      reverseRafRef.current = requestAnimationFrame(runReverse)
    }
    reverseRafRef.current = requestAnimationFrame(runReverse)
  }, [])

  const resetOnCancelDrag = useCallback(() => {
    if (reverseRafRef.current) cancelAnimationFrame(reverseRafRef.current)
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.currentTime = 0
    }
  }, [])

  const handleDragEnter = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.dataTransfer?.types?.includes('Files')) {
      if (reverseRafRef.current) {
        cancelAnimationFrame(reverseRafRef.current)
        reverseRafRef.current = null
      }
      setIsDragging(true)
      if (hadFilesRef.current === 0) {
        if (videoRef.current) {
          videoRef.current.currentTime = 0
          videoRef.current.play()
        }
      }
    }
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragging(false)
      if (hadFilesRef.current === 0) playReverseToStart()
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const finishProcessing = useCallback(() => {
    setProcessingProgress(100)
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current)
      progressIntervalRef.current = null
    }
    if (processingTimeoutRef.current) {
      clearTimeout(processingTimeoutRef.current)
      processingTimeoutRef.current = null
    }
    setTimeout(() => {
      setIsProcessing(false)
      setProcessingProgress(0)
      if (postProcessingTimerRef.current) clearTimeout(postProcessingTimerRef.current)
      postProcessingTimerRef.current = setTimeout(() => {
        if (hasSeenTransitionVideoRef.current) {
          setShowPlaybackView(true)
        } else {
          setShowTransitionVideo(true)
        }
        postProcessingTimerRef.current = null
      }, 0) // immediately show transition video
    }, 300)
  }, [])

  const handleVideoLoaded = useCallback(() => {
    processingVideosRef.current = Math.max(0, processingVideosRef.current - 1)
    if (processingVideosRef.current === 0) finishProcessing()
  }, [finishProcessing])

  // No longer mark as unsupported: backend converts any video to MP4 for Vision Pro playback
  const handleThumbnailError = useCallback((_index) => {
    // Browser may not decode the video for thumbnail, but backend will convert it
  }, [])

  const handlePlaybackVideoError = useCallback((_index) => {
    // Local blob may fail; backend returns converted MP4 via inputVideoUrl
  }, [])

  const handleTransitionVideoEnded = useCallback(() => {
    setTransitionFadingOut(true)
    hasSeenTransitionVideoRef.current = true
    setTimeout(() => {
      setShowTransitionVideo(false)
      setTransitionFadingOut(false)
      setPlaybackViewZoomingIn(true)
      setShowPlaybackView(true)
    }, TRANSITION_VIDEO_TO_PLAYBACK_MS)
  }, [])

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    const allFiles = [...(e.dataTransfer?.files || [])]
    const files = allFiles.filter(isVideoFile)
    if (files.length) {
      const newEntries = files.map((file) => ({
        file,
        url: URL.createObjectURL(file),
        error: null,
      }))
      const videoCount = newEntries.filter((e) => e.url).length
      const needsProcessing = newEntries.some(
        (e) => e.url && e.file.size > VIDEO_SIZE_THRESHOLD
      )
      if (needsProcessing && videoCount > 0) {
        setIsProcessing(true)
        setProcessingProgress(0)
        processingVideosRef.current = videoCount
        progressIntervalRef.current = setInterval(() => {
          setProcessingProgress((p) => (p >= 90 ? p : p + Math.random() * 6 + 2))
        }, 350)
        processingTimeoutRef.current = setTimeout(() => {
          if (processingVideosRef.current > 0) {
            processingVideosRef.current = 0
            finishProcessing()
          }
        }, LARGE_VIDEO_PROCESSING_MS)
      }
      setDroppedFiles((prev) => {
        const isFirstDrop = prev.length === 0
        if (isFirstDrop && videoRef.current) videoRef.current.currentTime = videoRef.current.duration
        const next = [...prev, ...newEntries]
        if (!needsProcessing && next.length > 0) {
          if (postProcessingTimerRef.current) clearTimeout(postProcessingTimerRef.current)
          postProcessingTimerRef.current = setTimeout(() => {
            if (hasSeenTransitionVideoRef.current) {
              setShowPlaybackView(true)
            } else {
              setShowTransitionVideo(true)
            }
            postProcessingTimerRef.current = null
          }, POST_PROCESSING_DELAY_MS)
        }
        return next
      })
    } else {
      resetOnCancelDrag()
    }
  }

  const handleRemoveFile = useCallback((index) => {
    setDroppedFiles((prev) => {
      const entry = prev[index]
      if (entry?.url) URL.revokeObjectURL(entry.url)
      const next = prev.filter((_, i) => i !== index)
      if (next.length === 0) {
        hasSeenTransitionVideoRef.current = false
        if (videoRef.current) videoRef.current.currentTime = 0
      }
      return next
    })
    setPlaybackIndex((prev) => {
      if (index < prev) return prev - 1
      if (index === prev) return Math.max(0, prev - 1)
      return prev
    })
  }, [])

  const effectivePlaybackIndex =
    droppedFiles.length > 0 ? Math.min(playbackIndex, droppedFiles.length - 1) : 0
  const playbackEntry = droppedFiles[effectivePlaybackIndex]
  // Video in Vision Pro mask: job's input-video (converted MP4)
  const playbackUrl = inputVideoUrl || (jobId ? `${API_BASE}/jobs/${jobId}/input-video` : null)

  // Fetch backend video as blob so it plays in Vision Pro (streaming URL can fail there)
  useEffect(() => {
    if (!playbackUrl || !playbackUrl.startsWith('/api')) {
      if (playbackBlobUrlRef.current) {
        URL.revokeObjectURL(playbackBlobUrlRef.current)
        playbackBlobUrlRef.current = null
      }
      setPlaybackBlobUrl(null)
      return
    }
    let cancelled = false
    fetch(playbackUrl)
      .then((r) => {
        if (!r.ok) throw new Error(r.statusText)
        return r.blob()
      })
      .then((blob) => {
        if (cancelled) return
        if (playbackBlobUrlRef.current) URL.revokeObjectURL(playbackBlobUrlRef.current)
        const url = URL.createObjectURL(blob)
        playbackBlobUrlRef.current = url
        setPlaybackBlobUrl(url)
      })
      .catch(() => {
        if (!cancelled) setPlaybackBlobUrl(null)
      })
    return () => {
      cancelled = true
      if (playbackBlobUrlRef.current) {
        URL.revokeObjectURL(playbackBlobUrlRef.current)
        playbackBlobUrlRef.current = null
      }
      setPlaybackBlobUrl(null)
    }
  }, [playbackUrl])

  const effectivePlaybackSrc = playbackUrl?.startsWith('/api') ? playbackBlobUrl : playbackUrl

  // Preload video2 when app mounts for smooth playback (avoid buffering stutter)
  useEffect(() => {
    const link = document.createElement('link')
    link.rel = 'preload'
    link.as = 'video'
    link.href = TRANSITION_VIDEO_PATH
    document.head.appendChild(link)
    return () => link.remove()
  }, [])

  useEffect(() => {
    if (showTransitionVideo && transitionVideoRef.current) {
      const v = transitionVideoRef.current
      v.currentTime = 0
      v.playbackRate = TRANSITION_VIDEO_PLAYBACK_RATE
      // Wait for enough data to avoid mid-play stutter; then play
      const playWhenReady = () => {
        v.playbackRate = TRANSITION_VIDEO_PLAYBACK_RATE
        v.play().catch(() => {})
      }
      if (v.readyState >= 3) playWhenReady()
      else v.addEventListener('canplaythrough', playWhenReady, { once: true })
    }
  }, [showTransitionVideo])

  useEffect(() => {
    if (showPlaybackView && playbackVideoRef.current) {
      playbackVideoRef.current.play().catch(() => {})
    }
  }, [showPlaybackView])

  // When converted input-video URL arrives, ensure video loads and plays in Vision Pro mask
  useEffect(() => {
    if (!effectivePlaybackSrc || !showPlaybackView || !playbackVideoRef.current) return
    const el = playbackVideoRef.current
    const play = () => el.play().catch(() => {})
    if (el.readyState >= 2) play()
    else {
      el.addEventListener('loadeddata', play, { once: true })
      el.addEventListener('canplay', play, { once: true })
      return () => {
        el.removeEventListener('loadeddata', play)
        el.removeEventListener('canplay', play)
      }
    }
  }, [effectivePlaybackSrc, showPlaybackView])

  useEffect(() => {
    if (playbackViewZoomingIn) {
      const id = setTimeout(() => setPlaybackViewZoomingIn(false), 700)
      return () => clearTimeout(id)
    }
  }, [playbackViewZoomingIn])

  // Flow: drag & drop → process → enter playback view → POST /jobs (file) → remember job_id → use /jobs/{job_id}/input-video in Vision Pro mask
  useEffect(() => {
    if (!showPlaybackView || showResultView || !playbackEntry?.file || jobError) return

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/jobs/${jobId}`)
        if (!res.ok) return
        const data = await res.json()
        setJobStatus(data)
        if (data.status === 'completed') {
          if (statusPollIntervalRef.current) {
            clearInterval(statusPollIntervalRef.current)
            statusPollIntervalRef.current = null
          }
          setShowResultView(true)
        } else if (data.status === 'failed') {
          if (statusPollIntervalRef.current) {
            clearInterval(statusPollIntervalRef.current)
            statusPollIntervalRef.current = null
          }
          setJobError(data.error || 'Job failed')
        }
      } catch {
        // ignore network errors, will retry next poll
      }
    }

    if (jobId) {
      poll()
      statusPollIntervalRef.current = setInterval(poll, STATUS_POLL_MS)
      return () => {
        if (statusPollIntervalRef.current) {
          clearInterval(statusPollIntervalRef.current)
          statusPollIntervalRef.current = null
        }
      }
    } else {
      // POST /jobs with dropped file → backend creates job, returns job_id and input_video_url
      const form = new FormData()
      form.append('file', playbackEntry.file)
      fetch(`${API_BASE}/jobs`, { method: 'POST', body: form })
        .then((r) => r.json())
        .then((data) => {
          if (data.job_id) {
            setJobId(data.job_id)
            // job_id/input-video = converted mp4; used for <video> inside Vision Pro mask
            const path = (data.input_video_url || '').replace(/^\//, '')
            setInputVideoUrl(path ? `${API_BASE}/${path}` : null)
            if (!path) setJobError('No input video URL from server')
          } else setJobError('Failed to create job')
        })
        .catch(() => setJobError('Failed to upload video'))
    }
  }, [showPlaybackView, showResultView, jobId, playbackEntry?.file, jobError])

  const handleBackFromResult = useCallback(() => {
    setShowResultView(false)
    setShowPlaybackView(false)
    setJobId(null)
    setJobStatus(null)
    setInputVideoUrl(null)
    setJobError(null)
    setAnnotatedVideoBlobUrl(null)
    setResultActionsOpen(false)
    setResultActionPending(null)
    if (playbackBlobUrlRef.current) {
      URL.revokeObjectURL(playbackBlobUrlRef.current)
      playbackBlobUrlRef.current = null
    }
    if (annotatedBlobUrlRef.current) {
      URL.revokeObjectURL(annotatedBlobUrlRef.current)
      annotatedBlobUrlRef.current = null
    }
    setPlaybackBlobUrl(null)
    setDroppedFiles((prev) => {
      prev.forEach((e) => e?.url && URL.revokeObjectURL(e.url))
      return []
    })
    hasSeenTransitionVideoRef.current = false
  }, [])

  const resultVideoUrl = jobId ? `${API_BASE}/jobs/${jobId}/annotated-video` : null

  // As soon as result view is shown, fetch annotated video as blob so we can show it (and for Vision Pro compatibility)
  useEffect(() => {
    if (!showResultView || !resultVideoUrl) {
      if (annotatedBlobUrlRef.current) {
        URL.revokeObjectURL(annotatedBlobUrlRef.current)
        annotatedBlobUrlRef.current = null
      }
      setAnnotatedVideoBlobUrl(null)
      return
    }
    let cancelled = false
    fetch(resultVideoUrl)
      .then((r) => {
        if (!r.ok) throw new Error(r.statusText)
        return r.blob()
      })
      .then((blob) => {
        if (cancelled) return
        if (annotatedBlobUrlRef.current) URL.revokeObjectURL(annotatedBlobUrlRef.current)
        const url = URL.createObjectURL(blob)
        annotatedBlobUrlRef.current = url
        setAnnotatedVideoBlobUrl(url)
      })
      .catch(() => {
        if (!cancelled) setAnnotatedVideoBlobUrl(null)
      })
    return () => {
      cancelled = true
      if (annotatedBlobUrlRef.current) {
        URL.revokeObjectURL(annotatedBlobUrlRef.current)
        annotatedBlobUrlRef.current = null
      }
      setAnnotatedVideoBlobUrl(null)
    }
  }, [showResultView, resultVideoUrl])

  // Default PiP position (bottom-right) when entering result view
  useEffect(() => {
    if (!showResultView) return
    const w = window.innerWidth
    const h = window.innerHeight
    const width = 240
    const height = Math.round((width * 9) / 16)
    setPipLayout({
      left: w - width - PIP_MARGIN,
      top: h - height - PIP_MARGIN,
      width,
    })
  }, [showResultView])

  // PiP drag: move and snap to edges on release
  const getContainerRect = useCallback(() => {
    const el = resultViewRef.current
    if (el) return el.getBoundingClientRect()
    return { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight }
  }, [])

  const snapPipToEdges = useCallback(() => {
    const layout = pipLayoutRef.current
    if (!layout) return
    const rect = getContainerRect()
    const height = (layout.width * 9) / 16
    const halfW = rect.width / 2
    const halfH = rect.height / 2
    const centerX = layout.left + layout.width / 2
    const centerY = layout.top + height / 2
    let left = layout.left
    let top = layout.top
    if (centerX < halfW) left = PIP_MARGIN
    else left = rect.width - layout.width - PIP_MARGIN
    if (centerY < halfH) top = PIP_MARGIN
    else top = rect.height - height - PIP_MARGIN
    setPipLayout((prev) => (prev ? { ...prev, left, top } : null))
  }, [getContainerRect])

  const handlePipMouseDown = useCallback(
    (e) => {
      if (e.target.closest('.result-view__pip-resize')) return
      e.preventDefault()
      pipDragRef.current = {
        dragging: true,
        startX: e.clientX,
        startY: e.clientY,
        startLeft: pipLayout.left,
        startTop: pipLayout.top,
      }
    },
    [pipLayout?.left, pipLayout?.top]
  )

  const handlePipResizeDown = useCallback(
    (e) => {
      e.preventDefault()
      e.stopPropagation()
      const layout = pipLayoutRef.current
      if (!layout) return
      const rect = getContainerRect()
      const h = (layout.width * 9) / 16
      const rightGap = rect.width - (layout.left + layout.width)
      const leftGap = layout.left
      const bottomGap = rect.height - (layout.top + h)
      const topGap = layout.top
      const handleOnLeft = rightGap < leftGap
      const handleOnTop = bottomGap < topGap
      const corner = handleOnLeft ? (handleOnTop ? 'tl' : 'bl') : (handleOnTop ? 'tr' : 'br')
      pipResizeRef.current = {
        resizing: true,
        startX: e.clientX,
        startY: e.clientY,
        startLeft: layout.left,
        startTop: layout.top,
        startWidth: layout.width,
        startHeight: h,
        corner,
      }
    },
    [getContainerRect]
  )

  useEffect(() => {
    if (!showResultView || !pipLayout) return
    const onMove = (e) => {
      const rect = getContainerRect()
      const height = (pipLayout.width * 9) / 16
      if (pipDragRef.current.dragging) {
        const dx = e.clientX - pipDragRef.current.startX
        const dy = e.clientY - pipDragRef.current.startY
        let left = pipDragRef.current.startLeft + dx
        let top = pipDragRef.current.startTop + dy
        left = Math.max(PIP_MARGIN, Math.min(rect.width - pipLayout.width - PIP_MARGIN, left))
        top = Math.max(PIP_MARGIN, Math.min(rect.height - height - PIP_MARGIN, top))
        setPipLayout((prev) => ({ ...prev, left, top }))
      }
      if (pipResizeRef.current.resizing) {
        const { startX, startLeft, startTop, startWidth, startHeight, corner } = pipResizeRef.current
        const dx = e.clientX - startX
        const minW = PIP_MIN_WIDTH
        const maxW = Math.floor(rect.width * PIP_MAX_WIDTH_RATIO)
        let width = (corner === 'br' || corner === 'tr') ? startWidth + dx : startWidth - dx
        width = Math.max(minW, Math.min(maxW, width))
        const height2 = (width * 9) / 16
        let left = startLeft
        let top = startTop
        if (corner === 'br') {
          left = startLeft
          top = startTop
          if (left + width > rect.width - PIP_MARGIN) left = rect.width - width - PIP_MARGIN
          if (top + height2 > rect.height - PIP_MARGIN) top = rect.height - height2 - PIP_MARGIN
        } else if (corner === 'bl') {
          left = startLeft + startWidth - width
          top = startTop
          if (left < PIP_MARGIN) left = PIP_MARGIN
          if (top + height2 > rect.height - PIP_MARGIN) top = rect.height - height2 - PIP_MARGIN
        } else if (corner === 'tr') {
          left = startLeft
          top = startTop + startHeight - height2
          if (left + width > rect.width - PIP_MARGIN) left = rect.width - width - PIP_MARGIN
          if (top < PIP_MARGIN) top = PIP_MARGIN
        } else {
          left = startLeft + startWidth - width
          top = startTop + startHeight - height2
          if (left < PIP_MARGIN) left = PIP_MARGIN
          if (top < PIP_MARGIN) top = PIP_MARGIN
        }
        setPipLayout((prev) => (prev ? { ...prev, left, top, width } : null))
      }
    }
    const onUp = () => {
      if (pipDragRef.current.dragging) {
        pipDragRef.current.dragging = false
        snapPipToEdges()
      }
      if (pipResizeRef.current.resizing) pipResizeRef.current.resizing = false
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [showResultView, pipLayout, getContainerRect, snapPipToEdges])

  // Resize handle corner: opposite to the screen edge the PiP is closest to
  const pipResizeCorner = (() => {
    if (!pipLayout || !resultViewRef.current) return 'br'
    const rect = getContainerRect()
    const w = pipLayout.width
    const h = (w * 9) / 16
    const leftGap = pipLayout.left
    const rightGap = rect.width - (pipLayout.left + w)
    const topGap = pipLayout.top
    const bottomGap = rect.height - (pipLayout.top + h)
    const handleOnLeft = rightGap < leftGap
    const handleOnTop = bottomGap < topGap
    return handleOnLeft ? (handleOnTop ? 'tl' : 'bl') : (handleOnTop ? 'tr' : 'br')
  })()

  // Main video must always use annotated-video URL so it never shows input (no blob mix-up)
  const mainVideoSrc = resultVideoUrl || null

  const handleResultAction = useCallback(
    async (action) => {
      setResultActionsOpen(false)
      if (!jobId) return
      if (action === 'approve') {
        setApproveJobId(jobId)
        setShowApprovePage(true)
        setResultActionPending(null)
        return
      }
      if (action === 'reject') {
        setResultActionPending('reject')
        setTimeout(() => setResultActionPending(null), 1500)
        return
      }
      if (action === 'correction') {
        setResultActionPending('correction')
        try {
          await fetch(`${API_BASE}/jobs/${jobId}/corrections`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ corrections: [] }),
          })
          const r = await fetch(`${API_BASE}/jobs/${jobId}/rerun`, { method: 'POST' })
          if (r.ok) {
            setResultActionPending('rerunning')
            setShowResultView(false)
          } else {
            setResultActionPending(null)
          }
        } catch (e) {
          console.warn('[RoboSight] Correction/rerun failed', e)
          setResultActionPending(null)
        }
      }
    },
    [jobId]
  )

  // Poll job when rerunning; go back to playback view, then show result view with new annotated video when completed
  useEffect(() => {
    if (resultActionPending !== 'rerunning' || !jobId) return
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/jobs/${jobId}`)
        const data = await res.json()
        setJobStatus(data)
        if (data.status === 'completed') {
          setResultActionPending(null)
          setAnnotatedVideoBlobUrl(null)
          if (annotatedBlobUrlRef.current) {
            URL.revokeObjectURL(annotatedBlobUrlRef.current)
            annotatedBlobUrlRef.current = null
          }
          const b = await fetch(`${API_BASE}/jobs/${jobId}/annotated-video`)
          if (b.ok) {
            const blob = await b.blob()
            const url = URL.createObjectURL(blob)
            annotatedBlobUrlRef.current = url
            setAnnotatedVideoBlobUrl(url)
          }
          setShowResultView(true)
        } else if (data.status === 'failed') {
          setResultActionPending(null)
        }
      } catch (_) {}
    }, 2000)
    return () => clearInterval(poll)
  }, [resultActionPending, jobId])

  // Load report data when Approve page is shown; prefer calibrated metrics (from reanalysis) when available
  useEffect(() => {
    const id = showApprovePage ? approveJobId || jobId : null
    if (!id) {
      setReportData({ metrics: null, confidence: null, timeline: null, worldGt: null })
      setReportMetricsRun('baseline')
      return
    }
    setReportLoading(true)
    setReportError(null)
    const base = `${API_BASE}/jobs/${id}`
    const metricsPromise = fetch(`${base}/metrics?run=calibrated`)
      .then((r) => (r.ok ? r.json() : null))
      .then((cal) => cal != null ? { metrics: cal, run: 'calibrated' } : fetch(`${base}/metrics?run=baseline`).then((r) => (r.ok ? r.json() : null)).then((m) => ({ metrics: m, run: 'baseline' })))
    Promise.all([
      metricsPromise,
      fetch(`${base}/confidence-report`).then((r) => (r.ok ? r.json() : null)),
      fetch(`${base}/timeline`).then((r) => (r.ok ? r.json() : null)),
      fetch(`${base}/results`).then((r) => (r.ok ? r.json() : null)),
    ])
      .then(([metricsResult, confidence, timeline, worldGt]) => {
        setReportMetricsRun(metricsResult.run)
        setReportData({ metrics: metricsResult.metrics, confidence, timeline, worldGt })
        setReportError(null)
      })
      .catch((e) => {
        setReportError(e.message || 'Failed to load report data')
        setReportData({ metrics: null, confidence: null, timeline: null, worldGt: null })
        setReportMetricsRun('baseline')
      })
      .finally(() => setReportLoading(false))
  }, [showApprovePage, approveJobId, jobId])

  // URL: ?view=approve&job=xxx opens reports page for that job
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    if (params.get('view') === 'approve') {
      const id = params.get('job')
      if (id) {
        setApproveJobId(id)
        setShowApprovePage(true)
        setShowPlaybackView(true)
        setShowResultView(false)
      }
    }
  }, [])

  const reportJobId = showApprovePage ? (approveJobId || jobId) : null
  const reportVideoUrl = reportJobId ? `${API_BASE}/jobs/${reportJobId}/annotated-video` : null

  return (
    <div
      className={`page ${showPlaybackView || showResultView || showApprovePage || transitionFadingOut ? 'page--playback' : ''}`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {/* Approve / Reports page: annotated video top-left, graphs from job JSON */}
      {showApprovePage && reportJobId && (
        <div className="reports-page reports-page--vision-pro">
          <button
            type="button"
            className="result-view__back result-view__back--vision-pro reports-page__back"
            onClick={() => { setShowApprovePage(false); setApproveJobId(null) }}
          >
            Back
          </button>
          <div className="reports-page__layout">
            <aside className="reports-page__left">
              <div className="reports-page__video-wrap">
                <div className="reports-page__video-label">Annotated view</div>
                {reportVideoUrl && (
                  <video
                    key={`report-video-${reportJobId}`}
                    className="reports-page__video"
                    src={reportVideoUrl}
                    muted
                    playsInline
                    loop
                    preload="auto"
                    onLoadedData={(e) => e.target.play().catch(() => {})}
                  />
                )}
              </div>
              <div className="reports-page__json-wrap">
                <h3 className="reports-page__card-title">Raw JSON</h3>
                <div className="reports-page__json-tabs">
                  {['metrics', 'confidence', 'timeline', 'world_gt'].map((key) => (
                    <button
                      key={key}
                      type="button"
                      className={`reports-page__json-tab ${jsonTab === key ? 'reports-page__json-tab--active' : ''}`}
                      onClick={() => setJsonTab(key)}
                    >
                      {key.replace('_', ' ')}
                    </button>
                  ))}
                </div>
                <pre className="reports-page__json-pre">
                  {(jsonTab === 'world_gt' ? reportData.worldGt : reportData[jsonTab]) != null
                    ? JSON.stringify(jsonTab === 'world_gt' ? reportData.worldGt : reportData[jsonTab], null, 2)
                    : '—'}
                </pre>
              </div>
            </aside>
            <div className="reports-page__right">
              <div className="reports-page__graphs">
                {reportLoading && (
                  <div className="reports-page__loading" aria-live="polite">
                    <div className="reports-page__loading-bar" />
                    <div className="reports-page__loading-shimmer">
                      <div className="reports-page__loading-shimmer-inner" />
                    </div>
                    <span className="reports-page__loading-text">Loading report data…</span>
                  </div>
                )}
                {reportError && (
                  <div className="reports-page__error">{reportError}</div>
                )}
                {!reportLoading && !reportError && reportData.metrics && (
                  <>
                    <div className="reports-page__card reports-page__card--metrics">
                      <h3 className="reports-page__card-title">Metrics</h3>
                      <div className="reports-page__metrics-grid">
                        <div className="reports-page__metric">
                          <span className="reports-page__metric-value">{((reportData.metrics.event_precision ?? 0) * 100).toFixed(1)}%</span>
                          <span className="reports-page__metric-label">Precision</span>
                        </div>
                        <div className="reports-page__metric">
                          <span className="reports-page__metric-value">{((reportData.metrics.event_recall ?? 0) * 100).toFixed(1)}%</span>
                          <span className="reports-page__metric-label">Recall</span>
                        </div>
                        <div className="reports-page__metric">
                          <span className="reports-page__metric-value">{(reportData.metrics.avg_boundary_error ?? 0).toFixed(2)}s</span>
                          <span className="reports-page__metric-label">Boundary error</span>
                        </div>
                        <div className="reports-page__metric">
                          <span className="reports-page__metric-value">{((reportData.metrics.review_percentage ?? 0) * 100).toFixed(1)}%</span>
                          <span className="reports-page__metric-label">Review %</span>
                        </div>
                        <div className="reports-page__metric">
                          <span className="reports-page__metric-value">{Math.round(reportData.metrics.labeling_time_saved_estimate ?? 0)}</span>
                          <span className="reports-page__metric-label">Time saved (est.)</span>
                        </div>
                      </div>
                    </div>
                    <div className="reports-page__card">
                      <h3 className="reports-page__card-title">Precision & Recall</h3>
                      <div className="reports-page__chart-wrap">
                        <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={[
                          { name: 'Precision', value: (reportData.metrics.event_precision ?? 0) * 100, fill: '#007AFF' },
                          { name: 'Recall', value: (reportData.metrics.event_recall ?? 0) * 100, fill: '#5856D6' },
                        ]}
                        margin={{ top: 8, right: 8, left: 8, bottom: 8 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                        <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                        <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} />
                        <Bar dataKey="value" fill="#007AFF" radius={[6, 6, 0, 0]} />
                      </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </>
                )}
                {!reportLoading && !reportError && reportData.confidence && (
                  <div className="reports-page__card">
                    <h3 className="reports-page__card-title">Confidence</h3>
                    <div className="reports-page__confidence-row">
                      <div className="reports-page__chart-wrap">
                        <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={(() => {
                            const high = reportData.confidence.high_confidence_count ?? 0
                            const low = reportData.confidence.low_confidence_count ?? 0
                            if (high === 0 && low === 0) return [{ name: 'No events', value: 1, color: '#E5E5EA' }]
                            return [
                              { name: 'High confidence', value: high, color: '#34C759' },
                              { name: 'Low confidence', value: low, color: '#FF3B30' },
                            ]
                          })()}
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={80}
                          paddingAngle={2}
                          dataKey="value"
                          label={({ name, value }) => (value > 0 ? `${name}: ${value}` : null)}
                        >
                          {(() => {
                            const high = reportData.confidence.high_confidence_count ?? 0
                            const low = reportData.confidence.low_confidence_count ?? 0
                            const arr = high === 0 && low === 0
                              ? [{ color: '#E5E5EA' }]
                              : [{ color: '#34C759' }, { color: '#FF3B30' }]
                            return arr.map((entry, i) => <Cell key={i} fill={entry.color} />)
                          })()}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="reports-page__confidence-summary">
                        <p>Overall: {((reportData.confidence.overall_confidence ?? 0) * 100).toFixed(0)}%</p>
                        <p>Total events: {reportData.confidence.total_events ?? 0}</p>
                      </div>
                    </div>
                  </div>
                )}
                {!reportLoading && !reportError && reportData.worldGt?.objects?.length > 0 && (
                  <div className="reports-page__card">
                    <h3 className="reports-page__card-title">Object confidence</h3>
                    <div className="reports-page__chart-wrap">
                      <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={(reportData.worldGt.objects || []).map((o) => ({ name: o.label || o.id, value: Number(((o.confidence ?? 0) * 100).toFixed(1)) }))}
                      layout="vertical"
                      margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                      <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} />
                      <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 11 }} />
                      <Tooltip formatter={(v) => `${v}%`} />
                      <Bar dataKey="value" fill="#007AFF" radius={[0, 6, 6, 0]} />
                    </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
                {!reportLoading && !reportError && reportData.timeline?.length > 0 && (
                  <div className="reports-page__card">
                    <h3 className="reports-page__card-title">Events over time</h3>
                    <div className="reports-page__chart-wrap">
                      <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={reportData.timeline.map((e) => ({ time: e.start_time ?? 0, count: 1 }))}
                      margin={{ top: 8, right: 8, left: 8, bottom: 8 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                      <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#5856D6" radius={[6, 6, 0, 0]} />
                    </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
                {!reportLoading && !reportError && reportData.worldGt?.video && (
                  <div className="reports-page__card">
                    <h3 className="reports-page__card-title">Video info</h3>
                    <div className="reports-page__video-info">
                      <p>Duration: {(reportData.worldGt.video.duration_seconds ?? 0).toFixed(1)}s</p>
                      <p>FPS: {(reportData.worldGt.video.fps ?? 0).toFixed(1)}</p>
                      <p>Resolution: {reportData.worldGt.video.resolution?.join(' × ') || '—'}</p>
                    </div>
                  </div>
                )}
                {!reportLoading && !reportError && reportMetricsRun === 'calibrated' && reportData.metrics && (
                  <div className="reports-page__card reports-page__card--reanalysis" style={{ gridColumn: '1 / -1' }}>
                    <h3 className="reports-page__card-title">Precision & Recall from reanalysis</h3>
                    <p className="reports-page__reanalysis-desc">Metrics after correction and video reanalysis.</p>
                    <div className="reports-page__reanalysis-metrics">
                      <div className="reports-page__metric">
                        <span className="reports-page__metric-value">{((reportData.metrics.event_precision ?? 0) * 100).toFixed(1)}%</span>
                        <span className="reports-page__metric-label">Precision</span>
                      </div>
                      <div className="reports-page__metric">
                        <span className="reports-page__metric-value">{((reportData.metrics.event_recall ?? 0) * 100).toFixed(1)}%</span>
                        <span className="reports-page__metric-label">Recall</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Black underlay during video2 fade-out: prevents side-faced Vision Pro / white bg from showing through */}
      {transitionFadingOut && (
        <div className="transition-black-underlay" aria-hidden="true" />
      )}
      {showTransitionVideo && (
        <div className={`transition-video-layer ${transitionFadingOut ? 'transition-video-layer--fade-out' : ''}`}>
          <video
            ref={transitionVideoRef}
            src={TRANSITION_VIDEO_PATH}
            className="transition-video-layer__video"
            muted
            playsInline
            autoPlay
            preload="auto"
            disablePictureInPicture
            disableRemotePlayback
            onEnded={handleTransitionVideoEnded}
          />
        </div>
      )}
      {/* Result view: Google Meet layout — annotated video from backend job as main stage, original as PiP */}
      {showResultView && !showApprovePage && resultVideoUrl && (
        <div ref={resultViewRef} className="result-view result-view--vision-pro">
          {/* Main stage: annotated video from /jobs/:id/annotated-video (Meet “main speaker”) */}
          <div className="result-view__main">
            {mainVideoSrc ? (
              <video
                ref={resultVideoRef}
                key={`annotated-${jobId}`}
                className="result-view__main-video"
                src={mainVideoSrc}
                muted
                playsInline
                autoPlay
                loop
                preload="auto"
                disablePictureInPicture
                disableRemotePlayback
                onLoadedData={(e) => { e.target.play().catch(() => {}); setResultVideoPlaying(true) }}
                onCanPlay={(e) => { e.target.play().catch(() => {}); setResultVideoPlaying(true) }}
                onPlay={() => setResultVideoPlaying(true)}
                onPause={() => setResultVideoPlaying(false)}
                onError={(e) => console.warn('[RoboSight] Annotated video failed to load', e.target?.error)}
              />
            ) : (
              <div className="result-view__loading">
                <span className="playback-view__error-text">Loading annotated video…</span>
              </div>
            )}
            <div className="result-view__main-label">Annotated view</div>
          </div>
          <button
            type="button"
            className="result-view__play-pause result-view__actions-btn"
            aria-label={resultVideoPlaying ? 'Pause' : 'Play'}
            onClick={() => {
              const v = resultVideoRef.current
              if (!v) return
              if (resultVideoPlaying) {
                v.pause()
                setResultVideoPlaying(false)
              } else {
                v.play().catch(() => {})
                setResultVideoPlaying(true)
              }
            }}
          >
            {resultVideoPlaying ? (
              <span className="result-view__play-pause-icon result-view__play-pause-icon--pause" aria-hidden>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg>
              </span>
            ) : (
              <span className="result-view__play-pause-icon result-view__play-pause-icon--play" aria-hidden>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
              </span>
            )}
          </button>
          {effectivePlaybackSrc && pipLayout && (
            <div
              className="result-view__pip"
              aria-label="Original video"
              style={{
                left: pipLayout.left,
                top: pipLayout.top,
                width: pipLayout.width,
                height: (pipLayout.width * 9) / 16,
              }}
              onMouseDown={handlePipMouseDown}
            >
              <video
                className="result-view__pip-video"
                src={effectivePlaybackSrc}
                data-video-role="original"
                muted
                playsInline
                autoPlay
                loop
                preload="auto"
                disablePictureInPicture
                disableRemotePlayback
                onLoadedData={(e) => e.target.play().catch(() => {})}
              />
              <span className="result-view__pip-label">Original</span>
              <span
                className={`result-view__pip-resize result-view__pip-resize--${pipResizeCorner}`}
                aria-label="Resize"
                onMouseDown={handlePipResizeDown}
              />
            </div>
          )}
          <button type="button" className="result-view__back result-view__back--vision-pro" onClick={handleBackFromResult}>
            Leave
          </button>
          <div className="result-view__actions-wrap">
            {resultActionPending && (
              <span className="result-view__action-status" aria-live="polite">
                {resultActionPending === 'rerunning' ? 'Rerunning…' : resultActionPending === 'correction' ? 'Starting…' : resultActionPending === 'approve' ? 'Approved' : 'Rejected'}
              </span>
            )}
            <div className="result-view__actions">
              <button
                type="button"
                className="result-view__actions-btn"
                aria-expanded={resultActionsOpen}
                aria-haspopup="true"
                aria-label="Result actions"
                onClick={() => setResultActionsOpen((o) => !o)}
              >
                <span className="result-view__actions-btn-dot" />
              </button>
              {resultActionsOpen && (
                <>
                  <div className="result-view__actions-backdrop" onClick={() => setResultActionsOpen(false)} aria-hidden="true" />
                  <div className="result-view__actions-dropdown" role="menu">
                    <button type="button" role="menuitem" className="result-view__actions-item" onClick={() => handleResultAction('approve')}>Approve</button>
                    <button type="button" role="menuitem" className="result-view__actions-item" onClick={() => handleResultAction('correction')}>Correction</button>
                    <button type="button" role="menuitem" className="result-view__actions-item" onClick={() => handleResultAction('reject')}>Reject</button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
      {/* Vision analysis: back-vision pro mask + user video playing while we poll job status */}
      {showPlaybackView && !showResultView && droppedFiles.length > 0 && (
        <div className={`playback-view ${playbackViewZoomingIn ? 'playback-view--zoom-in' : ''}`}>
          {jobError && (
            <div className="playback-view__job-error">
              <span className="playback-view__error-text">{jobError}</span>
              <button type="button" className="result-view__back" onClick={() => { setJobError(null); setJobId(null); setInputVideoUrl(null) }}>Retry</button>
            </div>
          )}
          <div className="playback-view__headset">
            <div className="playback-view__screen">
              {/* During vision analysis: wave animation (Apple Intelligence colors) bottom to top. Otherwise: scanner. */}
              {effectivePlaybackSrc && !playbackEntry?.error && jobId && jobStatus?.status !== 'completed' && jobStatus?.status !== 'failed' && (
                <div className="playback-view__apple-intel" aria-hidden="true">
                  <div className="playback-view__apple-intel__wave" />
                </div>
              )}
              {effectivePlaybackSrc && !playbackEntry?.error && (!jobId || jobStatus?.status === 'completed' || jobStatus?.status === 'failed') && (
                <div className="playback-view__scanner" aria-hidden="true">
                  <div className="playback-view__scanner__grid" />
                  <div className="playback-view__scanner__line playback-view__scanner__line--1" />
                  <div className="playback-view__scanner__line playback-view__scanner__line--2" />
                  <div className="playback-view__scanner__line playback-view__scanner__line--3" />
                  <div className="playback-view__scanner__corner playback-view__scanner__corner--tl" />
                  <div className="playback-view__scanner__corner playback-view__scanner__corner--tr" />
                  <div className="playback-view__scanner__corner playback-view__scanner__corner--bl" />
                  <div className="playback-view__scanner__corner playback-view__scanner__corner--br" />
                  <div className="playback-view__scanner__particles">
                    {[...Array(8)].map((_, i) => (
                      <div key={i} className="playback-view__scanner__dot-wrap" style={{ '--i': i }}>
                        <span className="playback-view__scanner__dot" />
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {playbackEntry?.error ? (
                <div className="playback-view__error">
                  <span className="playback-view__error-text">{playbackEntry.error}</span>
                  <span className="playback-view__error-hint">Try another video or convert to H.264/MP4</span>
                </div>
              ) : effectivePlaybackSrc ? (
                <video
                  key={jobId ? `job-${jobId}` : `blob-${effectivePlaybackIndex}`}
                  ref={playbackVideoRef}
                  src={effectivePlaybackSrc}
                  className="playback-view__video"
                  muted
                  playsInline
                  autoPlay
                  loop
                  preload="auto"
                  disablePictureInPicture
                  disableRemotePlayback
                  onLoadedData={(e) => e.target.play().catch(() => {})}
                  onCanPlay={(e) => e.target.play().catch(() => {})}
                  onError={() => handlePlaybackVideoError(effectivePlaybackIndex)}
                />
              ) : playbackUrl ? (
                <div className="playback-view__error">
                  <span className="playback-view__error-text">Loading video…</span>
                  <span className="playback-view__error-hint">Preparing playback</span>
                </div>
              ) : (
                <div className="playback-view__error">
                  <span className="playback-view__error-text">Converting video…</span>
                  <span className="playback-view__error-hint">Preparing MP4 for playback</span>
                </div>
              )}
              <div className="playback-view__gradient" aria-hidden="true" />
            </div>
            <img src="/back-vision%20pro.png" alt="" className="playback-view__headset-img" />
          </div>
          {jobId && jobStatus?.status !== 'completed' && jobStatus?.status !== 'failed' && (
            <div className="vision-analysis-label" aria-live="polite">
              {resultActionPending === 'rerunning' || jobStatus?.status === 'rerunning'
                ? `Video analysis (reanalyzing)… ${jobStatus != null ? `${Math.round((jobStatus.progress ?? 0) * 100)}%` : 'starting'}`
                : `Vision analysis… ${jobStatus != null ? `${Math.round((jobStatus.progress ?? 0) * 100)}%` : 'starting'}`}
            </div>
          )}
        </div>
      )}
      {!showPlaybackView && !showResultView && (
      <>
      <div className={`vision-pro-wrapper ${isProcessing ? 'vision-pro-wrapper--processing' : ''}`}>
        {isProcessing && (
          <div className="apple-intelligence-orb" aria-hidden="true">
            <div className="apple-intelligence-orb__ring apple-intelligence-orb__ring--1" />
            <div className="apple-intelligence-orb__ring apple-intelligence-orb__ring--2" />
            <div className="apple-intelligence-orb__ring apple-intelligence-orb__ring--3" />
            <div className="apple-intelligence-orb__particles">
              {[...Array(12)].map((_, i) => (
                <div key={i} className="apple-intelligence-orb__dot-wrap" style={{ '--i': i }}>
                  <span className="apple-intelligence-orb__dot" />
                </div>
              ))}
            </div>
          </div>
        )}
        <video
          ref={videoRef}
          src="/vision-pro.mp4"
          className="vision-pro-image vision-pro-video"
          muted
          playsInline
          preload="auto"
          draggable={false}
          onDragStart={(e) => e.preventDefault()}
        />
      </div>
      <div
        ref={dropZoneRef}
        className={`drop-zone ${isDragging ? 'drop-zone--active' : ''} ${droppedFiles.length > 0 ? 'drop-zone--has-files' : ''}`}
      >
        {droppedFiles.length > 0 && isProcessing && (
          <div className="processing-bar">
            <div className="processing-bar__track">
              <div
                className="processing-bar__fill"
                style={{ width: `${processingProgress}%` }}
              />
            </div>
            <span className="processing-bar__label">Processing...</span>
          </div>
        )}
        {droppedFiles.length > 0 && (
          <span className="drop-zone__label">Drag & drop videos to add more</span>
        )}
        <div className="drop-zone__glass">
          {droppedFiles.length === 0 && (
            <span className="drop-zone__text">
              {isDragging ? 'Drop here' : 'Drag & drop video files here'}
            </span>
          )}
          {droppedFiles.length > 0 && (
            <div className="uploaded-files">
              {droppedFiles.map((entry, index) => (
                <div key={`${entry.file.name}-${index}`} className={`uploaded-file ${entry.error ? 'uploaded-file--error' : ''}`}>
                  <div className="uploaded-file__preview">
                    {entry.error ? (
                      <span className="uploaded-file__error" title={entry.error}>!</span>
                    ) : entry.url ? (
                      <video
                        src={entry.url}
                        muted
                        playsInline
                        preload="metadata"
                        onLoadedMetadata={handleVideoLoaded}
                        onError={() => handleThumbnailError(index)}
                      />
                    ) : null}
                  </div>
                  <div className="uploaded-file__info">
                    <span className="uploaded-file__name" title={entry.file.name}>{abbreviateFileName(entry.file.name)}</span>
                    <span className="uploaded-file__size">{entry.error || formatFileSize(entry.file.size)}</span>
                  </div>
                  <button
                    type="button"
                    className="uploaded-file__delete"
                    onClick={() => handleRemoveFile(index)}
                    title="Remove file"
                    aria-label="Remove file"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      </>
      )}
    </div>
  )
}
