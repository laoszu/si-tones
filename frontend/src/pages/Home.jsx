import { useState, useRef, useCallback } from 'react'
import './Home.css'

export default function Home() {

  const [file, setFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const handleFile = (f) => {
    if (!f)
        return
    setFile(f)
    setAudioUrl(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const handleSubmit = async () => {
    if (!file)
        return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const form = new FormData()
      form.append('audio', file)
      const res = await fetch('/api/transcribe', { method: 'POST', body: form })
      if (!res.ok)
        throw new Error(`Server error: ${res.status}`)
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const trans = result?.trans ?? result?.transcription
  const wav = result?.wav  ?? result?.wave
  const mel = result?.mel

  return (
    <div className="main">
      <div
        className={`dropzone ${dragging ? 'dragover' : ''} ${file ? 'has-file' : ''}`}
        onClick={() => inputRef.current.click()}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".wav,.mp3,.flac"
          onChange={(e) => handleFile(e.target.files[0])}
          style={{ display: 'none' }}
        />
        {file ? (
          <div className="file-info">
            <span className="file-name">{file.name}</span>
            <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
          </div>
        ) : (
          <div className="drop-prompt">
            <span className="drop-text">Drop audio file here</span>
            <span className="drop-hint">.wav · .mp3 · .flac</span>
          </div>
        )}
      </div>

      {audioUrl && <audio controls src={audioUrl} className="player" />}

      <button
        className="btn-transcribe"
        onClick={handleSubmit}
        disabled={!file || loading}
      >
        {loading ? <><span className="spinner" /> Processing…</> : '转录  Transcribe'}
      </button>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="result-panel">
          <div className="trans-block">
            <span className="trans-label">Transcription</span>
            <p className="trans-text">
              {trans || <span style={{ color: 'var(--muted)' }}>(empty response)</span>}
            </p>
          </div>
          {wav && (
            <div className="viz-block">
              <span className="viz-label">Waveform + Pitch (F₀)</span>
              <img src={`data:image/png;base64,${wav}`} alt="waveform and pitch" className="viz-img" />
            </div>
          )}
          {mel && (
            <div className="viz-block">
              <span className="viz-label">Mel Spectrogram</span>
              <img src={`data:image/png;base64,${mel}`} alt="mel spectrogram" className="viz-img" />
            </div>
          )}
        </div>
      )}
    </div> )
  }