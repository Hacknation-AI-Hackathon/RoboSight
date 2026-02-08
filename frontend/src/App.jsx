import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

const VIDEO_SIZE_THRESHOLD = 1024 * 1024 // 1MB - show processing for larger videos
const REVERSE_SPEED = 1 // normal speed when reversing
const POST_PROCESSING_DELAY_MS = 1000 // wait 1 sec after processing before showing playback view
const LARGE_VIDEO_PROCESSING_MS = 2500 // fixed delay for large videos - don't wait for metadata (can hang)

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
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [showPlaybackView, setShowPlaybackView] = useState(false)
  const videoRef = useRef(null)
  const playbackVideoRef = useRef(null)
  const dropZoneRef = useRef(null)
  const processingVideosRef = useRef(0)
  const progressIntervalRef = useRef(null)
  const processingTimeoutRef = useRef(null)
  const postProcessingTimerRef = useRef(null)
  const reverseRafRef = useRef(null)

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
        setShowPlaybackView(true)
        postProcessingTimerRef.current = null
      }, POST_PROCESSING_DELAY_MS)
    }, 300)
  }, [])

  const handleVideoLoaded = useCallback(() => {
    processingVideosRef.current = Math.max(0, processingVideosRef.current - 1)
    if (processingVideosRef.current === 0) finishProcessing()
  }, [finishProcessing])

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    const allFiles = [...(e.dataTransfer?.files || [])]
    const files = allFiles.filter((file) => file.type.startsWith('video/'))
    if (files.length) {
      const newEntries = files.map((file) => ({
        file,
        url: URL.createObjectURL(file),
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
            setShowPlaybackView(true)
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
      if (next.length === 0 && videoRef.current) videoRef.current.currentTime = 0
      return next
    })
  }, [])

  const firstVideoUrl = droppedFiles.length > 0 ? droppedFiles[0]?.url : null

  useEffect(() => {
    if (showPlaybackView && playbackVideoRef.current) {
      playbackVideoRef.current.play().catch(() => {})
    }
  }, [showPlaybackView])

  return (
    <div
      className="page"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {showPlaybackView && firstVideoUrl && (
        <div className="playback-view">
          <div className="playback-view__headset">
            <div className="playback-view__screen">
              <video
                ref={playbackVideoRef}
                src={firstVideoUrl}
                className="playback-view__video"
                muted
                playsInline
                autoPlay
                loop
                preload="auto"
                onLoadedData={(e) => e.target.play().catch(() => {})}
                onCanPlay={(e) => e.target.play().catch(() => {})}
                onError={(e) => console.warn('Playback video error:', e.target.error)}
              />
              <div className="playback-view__gradient" aria-hidden="true" />
            </div>
            <img src="/back-vision%20pro.png" alt="" className="playback-view__headset-img" />
          </div>
        </div>
      )}
      {!showPlaybackView && (
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
                <div key={`${entry.file.name}-${index}`} className="uploaded-file">
                  <div className="uploaded-file__preview">
                    {entry.url ? (
                      <video
                        src={entry.url}
                        muted
                        playsInline
                        preload="metadata"
                        onLoadedMetadata={handleVideoLoaded}
                      />
                    ) : null}
                  </div>
                  <div className="uploaded-file__info">
                    <span className="uploaded-file__name" title={entry.file.name}>{abbreviateFileName(entry.file.name)}</span>
                    <span className="uploaded-file__size">{formatFileSize(entry.file.size)}</span>
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
