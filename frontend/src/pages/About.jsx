import './About.css'

const CARDS = [
    { 
        num: '01', 
        title: 'Conformer Architecture', 
        body: 'Combines CNN layers for local feature extraction with multi-head self-attention for global context - outperforming pure transformer and CNN models on speech tasks.' 
    },
    { 
        num: '02', 
        title: 'CTC Decoding', 
        body: 'Connectionist Temporal Classification allows the model to learn alignments between audio frames and characters without explicit segmentation. Greedy decoding collapses repeated tokens.' 
    },
    { 
        num: '03', 
        title: 'CER Benchmarking', 
        body: 'Evaluates the model using Character Error Rate, calculated as: CER = (S + D + I) / N, where S is substitutions, D is deletions, I is insertions, and N is the total number of reference characters in the Aishell-1 dataset.' 
    },
    { 
        num: '04', 
        title: 'Spectral Feature Extraction', 
        body: 'Transforms raw audio waveforms into visual frequency maps. It extracts 80-channel log-mel filterbanks over time, highlighting the acoustic features.' 
    },
]

const STACK = ['PyTorch', 'NumPy', 'torchaudio', 'FastAPI', 'React', 'Vite']

export default function About() {
  return (
    <main className="about">
      <section className="about-hero">
        <span className="about-tag">Speech Recognition</span>
        <h2 className="about-heading">Mandarin ASR,<br />Engineering project.</h2>
        <div className="about-project">
        <p className="about-lead">
            大家好! This is my engineering project: an Automatic Speech Recognition (ASR) system for Mandarin Chinese. 
            Currently, it only features Speech-to-Text (STT), but I plan to upgrade it to work in real-time 
            (making it both faster and more accurate) and to display mapped-out acoustic features (tones, Pinyin's 
            initials and finals...), turning it into a Chinese learning tool for my fellow sinology 同志 &lt;3
        </p>
        <br/>
        <p className="about-more">
            This entire thingy was built from scratch for my engineering degree. If you want to check out my thesis, 
            take a look <a href="https://www.youtube.com/watch?v=_FlE_HHttPc" target="_blank" rel="noopener noreferrer">here (here, really)</a> :p
        </p>
        </div>
      </section>

      <div className="about-grid">
        {CARDS.map(c => (
          <div key={c.num} className="about-card">
            <span className="card-num">{c.num}</span>
            <h3 className="card-title">{c.title}</h3>
            <p className="card-body">{c.body}</p>
          </div>
        ))}
      </div>
    </main>
  )
}