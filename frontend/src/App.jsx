import { useState } from 'react'
import './App.css'
import Home from './pages/Home'
import About from './pages/About'

export default function App() {
  const [tab, setTab] = useState(
    window.location.pathname === '/about' ? 'about' : 'home'
  )

  const navigate = (t) => {
    window.history.pushState({}, '', t === 'about' ? '/about' : '/')
    setTab(t)
  }

  return (
    <div className="shell">
      <header className="header">
        <div className="logo-mark">四</div>
        <div className="header-text">
          <h1 className="site-title">Si Tone</h1>
          <p className="site-sub">Mandarin Speech Recognition</p>
        </div>
        <nav className="nav">
          <button
            className={`nav-btn ${tab === 'home' ? 'active' : ''}`}
            onClick={() => navigate('home')}
          >Transcribe</button>
          <button
            className={`nav-btn ${tab === 'about' ? 'active' : ''}`}
            onClick={() => navigate('about')}
          >About</button>
        </nav>
      </header>

      {tab === 'home' ? <Home /> : <About />}

      <footer className="footer">© 2026 Oliwia Pawelec (程安蕙) :3</footer>
    </div>
  )
}