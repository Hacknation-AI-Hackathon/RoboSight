import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

const VIDEO_SIZE_THRESHOLD = 1024 * 1024 // 1MB - show processing for larger videos
const REVERSE_SPEED = 1 // normal speed when reversing
const POST_PROCESSING_DELAY_MS = 1000 // wait 1 sec after processing before showing playback view (for small videos)
const LARGE_VIDEO_PROCESSING_MS = 2500 // fixed delay for large videos - don't wait for metadata (can hang)
const TRANSITION_VIDEO_PATH = '/video2.mp4'
const TRANSITION_VIDEO_TO_PLAYBACK_MS = 600 // fade duration from transition video end to playback view
const TRANSITION_VIDEO_PLAYBACK_RATE = 0.97 // slightly slower for smoother scroll recording playback
const API_BASE = '/api'
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
  const [showResultView, setShowResultView] = useState(false)
  const [jobError, setJobError] = useState(null)
  const videoRef = useRef(null)
  const playbackVideoRef = useRef(null)
  const transitionVideoRef = useRef(null)
  const dropZoneRef = useRef(null)
  const hasSeenTransitionVideoRef = useRef(false)
  const processingVideosRef = useRef(0)
  const progressIntervalRef = useRef(null)
  const processingTimeoutRef = useRef(null)
  const postProcessingTimerRef = useRef(null)
  const reverseRafRef = useRef(null)
  const statusPollIntervalRef = useRef(null)

  const hadFilesRef = useRef(0)
  hadFilesRef.current = droppedFiles.length

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

  const handleThumbnailError = useCallback((index) => {
    setDroppedFiles((prev) =>
      prev.map((e, i) => (i === index ? { ...e, error: 'Unsupported format' } : e))
    )
  }, [])

  const handlePlaybackVideoError = useCallback((index) => {
    setDroppedFiles((prev) =>
      prev.map((e, i) => (i === index ? { ...e, error: 'Unsupported format' } : e))
    )
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
  const playbackUrl = playbackEntry?.url ?? null

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

  useEffect(() => {
    if (playbackViewZoomingIn) {
      const id = setTimeout(() => setPlaybackViewZoomingIn(false), 700)
      return () => clearTimeout(id)
    }
  }, [playbackViewZoomingIn])

  // Vision analysis: when we enter playback view (mask + video), create job and poll status every 2s until completed/failed
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
      // Create job when entering Vision analysis screen
      const form = new FormData()
      form.append('file', playbackEntry.file)
      fetch(`${API_BASE}/jobs`, { method: 'POST', body: form })
        .then((r) => r.json())
        .then((data) => {
          if (data.job_id) setJobId(data.job_id)
          else setJobError('Failed to create job')
        })
        .catch(() => setJobError('Failed to upload video'))
    }
  }, [showPlaybackView, showResultView, jobId, playbackEntry?.file, jobError])

  const handleBackFromResult = useCallback(() => {
    setShowResultView(false)
    setShowPlaybackView(false)
    setJobId(null)
    setJobStatus(null)
    setJobError(null)
  }, [])

  const resultVideoUrl = jobId ? `${API_BASE}/jobs/${jobId}/annotated-video` : null

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
            onEnded={handleTransitionVideoEnded}
          />
        </div>
      )}
      {/* Result view: full-screen backend video + floating original video on the right */}
      {showResultView && resultVideoUrl && (
        <div className="result-view">
          <video
            className="result-view__main-video"
            src={resultVideoUrl}
            muted
            playsInline
            autoPlay
            loop
            preload="auto"
          />
          <div className="result-view__floating-original">
            <span className="result-view__floating-label">Original</span>
            {playbackUrl && (
              <video
                className="result-view__floating-video"
                src={playbackUrl}
                muted
                playsInline
                loop
                preload="auto"
              />
            )}
          </div>
          <button
            type="button"
            className="result-view__back"
            onClick={handleBackFromResult}
            aria-label="Back to analysis"
          >
            New video
          </button>
        </div>
      )}
      {/* Vision analysis: back-vision pro mask + user video playing while we poll job status */}
      {showPlaybackView && !showResultView && droppedFiles.length > 0 && (
        <div className={`playback-view ${playbackViewZoomingIn ? 'playback-view--zoom-in' : ''}`}>
          {jobError && (
            <div className="playback-view__job-error">
              <span className="playback-view__error-text">{jobError}</span>
              <button type="button" className="result-view__back" onClick={() => { setJobError(null); setJobId(null) }}>Retry</button>
            </div>
          )}
          <div className="playback-view__headset">
            <div className="playback-view__screen">
              {playbackEntry?.error ? (
                <div className="playback-view__error">
                  <span className="playback-view__error-text">{playbackEntry.error}</span>
                  <span className="playback-view__error-hint">Try another video or convert to H.264/MP4</span>
                </div>
              ) : playbackUrl ? (
                <video
                  key={effectivePlaybackIndex}
                  ref={playbackVideoRef}
                  src={playbackUrl}
                  className="playback-view__video"
                  muted
                  playsInline
                  autoPlay
                  loop
                  preload="auto"
                  onLoadedData={(e) => e.target.play().catch(() => {})}
                  onCanPlay={(e) => e.target.play().catch(() => {})}
                  onError={() => handlePlaybackVideoError(effectivePlaybackIndex)}
                />
              ) : null}
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
