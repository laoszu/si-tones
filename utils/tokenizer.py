import torch
import json
import os
from to_pinyin import to_pinyin

class Tokenizer:

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

        # special tokens resevred
        self.blank_token = "<blank>"
        self.unk_token = "<unk>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"

    def build_vocab(self, path):
        all = []

        with open(path, 'r') as f:
            for line in f:
                pure_parts = "".join(line.strip().split()[1:])
                pinyined = to_pinyin(pure_parts, True).split()

                for pin in pinyined:
                    all.append(pin)

        all = sorted(set(all))

        vocab = [self.blank_token, self.unk_token, self.sos_token, self.eos_token] + all

        self.char_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        self.id_to_char = {idx: tok for idx, tok in enumerate(vocab)}

        #print(f"Vocab size: { len(self.char_to_id) }")

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.char_to_id, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path):
        with open(path, 'r') as f:
            self.char_to_id = json.load(f)
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}

    def encode(self, pinyin_sentence):
        unk_id = self.char_to_id[self.unk_token]
        return [self.char_to_id.get(tok, unk_id) for tok in pinyin_sentence.strip().split()]

    def decode(self, ids, skip_special=True):
        tokens = []
        
        for i in ids:
            tok = self.id_to_char.get(i)
            if tok is None:
                continue
            if skip_special and tok in (self.blank_token, self.unk_token, self.sos_token, self.eos_token):
                continue
            tokens.append(tok)

        return " ".join(tokens)

    def __len__(self):
        return len(self.char_to_id)