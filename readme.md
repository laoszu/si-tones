# Si-tones (四 Tones)

ASR (**A**utomatic **S**peech **R**ecognition) / STT (**S**peech **T**o **T**ext) project for Mandarin Chinese (普通话), focused purely on speech recognition.

Part of a phoneme-level speech recognition and pronunciation assessment project.

Training based on [AISHELL-1](https://huggingface.co/datasets/AISHELL/AISHELL-1).

# Overview
Using GOP (Goodness of Pronunciation) metric to assess the correctness of users' inputs. It relies solely on an L1 dataset (e.g. AISHELL-1) and doesn't require L2 data — which is especially convenient, as Mandarin L2 corpora are scarce, and ones including Polish speakers practically don't exist.

## Running

### Virtual environment
```
python3.13 -m venv venv
source venv/bin/activate
```

Verify:
```
python --version
which pip
echo $VIRTUAL_ENV
```

To close:
```
deactivate
```

### Training

Download the dataset first:
```
wget http://www.openslr.org/33/data_aishell.tgz
tar -xzf data_aishell.tgz
```

# Acknowledgements
* Native speakers training: https://huggingface.co/datasets/AISHELL/AISHELL-1
* L2 speakers training: https://sites.google.com/site/tehsinphono/resources/mandarin-learners-speech-bank