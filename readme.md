# 四 Tones
ASR (**A**utomatic **S**peach **R**ecognition) project done on a mandarin chinese language (普通话).

# Running

## Virtual environment
Librosa only supports python <3.10; 3.14) versions, so it might be required to create virtual enviroment.
```
python3.13 -m venv venv
source venv/bin/activate
```
And then verify:
```
which python
which pip
echo $VIRTUAL_ENV
```

## Training
Download the dataset first.
```
wget http://www.openslr.org/resources/18/data_thchs30.tgz
tar -xzf data_thchs30.tgz
```

```
python train_model.py
```

# Sources