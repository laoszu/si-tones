# 四 Tones

ASR (**A**utomatic **S**peech **R**ecognition) project for Mandarin Chinese (普通话), focused purely on speech recognition.

Training based on [AISHELL-1](https://huggingface.co/datasets/AISHELL/AISHELL-1).

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