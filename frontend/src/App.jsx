import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

const VIDEO_SIZE_THRESHOLD = 1024 * 1024 // 1MB - show processing for larger videos
const REVERSE_SPEED = 1 // normal speed when reversing
const POST_PROCESSING_DELAY_MS = 1000 // wait 1 sec after processing before showing playback view (for small videos)
const LARGE_VIDEO_PROCESSING_MS = 2500 // fixed delay for large videos - don't wait for metadata (can hang)
const TRANSITION_VIDEO_PATH = '/video2.mp4'
const TRANSITION_VIDEO_TO_PLAYBACK_MS = 600 // fade duration from transition video end to playback view
const TRANSITION_VIDEO_PLAYBACK_RATE = 0.97 // slightly slower for smoother scroll recording playback
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
  const pipVideoRef = useRef(null)
  const [isResultPlaying, setIsResultPlaying] = useState(true)

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
          // Don't setShowResultView here — the blob-fetch useEffect will do it once the annotated video is fully downloaded
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

  useEffect(() => {
    if (showTransitionVideo && transitionVideoRef.current) {
      const v = transitionVideoRef.current
      v.currentTime = 0
      v.playbackRate = TRANSITION_VIDEO_PLAYBACK_RATE
      v.play().catch(() => {})
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
          // Don't setShowResultView here — the blob-fetch useEffect will do it once the annotated video is fully downloaded
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
    setIsResultPlaying(true)
  }, [])

  const toggleResultPlayback = useCallback(() => {
    const main = resultVideoRef.current
    const pip = pipVideoRef.current
    setIsResultPlaying((playing) => {
      if (playing) {
        main?.pause()
        pip?.pause()
      } else {
        main?.play().catch(() => {})
        pip?.play().catch(() => {})
      }
      return !playing
    })
  }, [])

  const resultVideoUrl = jobId ? `${API_BASE}/jobs/${jobId}/annotated-video` : null

  // When job completes, fetch annotated video as a full blob BEFORE showing result view
  const isJobCompleted = jobStatus?.status === 'completed'
  useEffect(() => {
    if (!isJobCompleted || !resultVideoUrl) {
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
        setShowResultView(true) // Only show result view once blob is fully downloaded
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
  }, [isJobCompleted, resultVideoUrl])

  // Main video uses only the fully-downloaded blob URL (never the raw streaming URL)
  const mainVideoSrc = annotatedVideoBlobUrl || null

  return (
    <div
      className={`page ${showPlaybackView || showResultView || transitionFadingOut ? 'page--playback' : ''}`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
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
      {showResultView && resultVideoUrl && (
        <div className="result-view result-view--meet">
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
                onClick={toggleResultPlayback}
                onLoadedData={(e) => e.target.play().catch(() => {})}
                onCanPlay={(e) => e.target.play().catch(() => {})}
                onError={(e) => console.warn('[RoboSight] Annotated video failed to load', e.target?.error)}
              />
            ) : (
              <div className="result-view__loading">
                <span className="playback-view__error-text">Loading annotated video…</span>
              </div>
            )}
            <div className="result-view__main-label">Annotated view</div>
          </div>
          {/* PiP: original input video (bottom-right like Google Meet) */}
          {effectivePlaybackSrc && (
            <div className="result-view__pip" aria-label="Original video">
              <video
                ref={pipVideoRef}
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
            </div>
          )}
          {/* Top-left: Leave call (Meet style) */}
          <button type="button" className="result-view__back" onClick={handleBackFromResult}>
            Leave
          </button>
          {/* Bottom bar: Meet-style control strip */}
          <div className="result-view__bar">
            <span className="result-view__bar-title">RoboSight — Annotated result</span>
            <button type="button" className="result-view__play-btn" onClick={toggleResultPlayback} aria-label={isResultPlaying ? 'Pause' : 'Play'}>
              {isResultPlaying ? '⏸' : '▶'}
            </button>
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
              {/* AI / game scanner style overlay when input video is playing (Satisfactory-like) */}
              {effectivePlaybackSrc && !playbackEntry?.error && (
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
              Vision analysis… {jobStatus != null ? `${Math.round((jobStatus.progress ?? 0) * 100)}%` : 'starting'}
            </div>
          )}
        </div>
      )}
      {!showPlaybackView && !showResultView && (
      <>
      <h1 className="hero-title" aria-hidden="true">ROBO SIGHT</h1>
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
