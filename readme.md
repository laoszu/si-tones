# Si-Tones (四 Tones, or 四声)

ASR (**A**utomatic **S**peech **R**ecognition) project for Mandarin Chinese (普通话), focused on speech recognition and pronunciation assessment.

Part of a phoneme-level pronunciation assessment pipeline using the **GOP** (Goodness of Pronunciation) metric - trained purely on L1 data, requiring no L2 corpus. This is especially practical for Mandarin, where L2 corpora are scarce and Polish-speaker data is virtually nonexistent.

---

## Architecture

- **Model:** Conformer (CTC)
- **Input:** 80-band Mel spectrogram (16 kHz, hop 160)
- **Training data:** [AISHELL-1](https://huggingface.co/datasets/AISHELL/AISHELL-1) - 170h of Mandarin read speech
- **Output:** Temporary **only** Pinyin character sequence

---

## Project structure

```
si-tone/
├── data/               # dataset, collate
├── models/
│   └── confromer/      # Conformer encoder, attention, conv
├── utils/              # tokenizer, vocab, pinyin helpers
├── inference/          # inference scripts, notebooks
│   └── inference.py    # single-file inference
├── examples/           # sample audio files
├── checkpoints/        # saved model checkpoints + config.json
└── train/
```

---

## Setup

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verify:
```bash
python --version
which pip
echo $VIRTUAL_ENV
```

Deactivate:
```bash
deactivate
```

---

## Training

Download AISHELL-1:
```bash
wget http://www.openslr.org/33/data_aishell.tgz
tar -xzf data_aishell.tgz
```

---

## Inference

Single file:
```bash
python inference/inference.py --audio examples/chinese_granny.mp3
python inference/inference.py --audio examples/chinese_granny.mp3 --checkpoint checkpoints/conformer_epoch_10.pt
```

Saves to `results/<filename>_epoch<N>_<timestamp>/`:
```
prediction.txt   # predicted pinyin + diagnostics
mel.png          # mel spectrogram
probs.png        # blank vs. max token probability over time
```

---

## Notes on convergence

The model needs ~30-50 epochs before CER drops meaningfully. Typical trajectory on AISHELL-1:

| Epoch | CER  |
|-------|------|
| 5–10  | ~90% |
| 20–30 | ~50% |
| 50+   | ~20% |
| 100+  | <10% |

A `blank_frac` above 90% in inference output means the model is still in early training.

---

## Acknowledgements

- Native speech (L1): [AISHELL-1](https://huggingface.co/datasets/AISHELL/AISHELL-1)
- L2 learner speech: [Mandarin Learners' Speech Bank](https://sites.google.com/site/tehsinphono/resources/mandarin-learners-speech-bank)