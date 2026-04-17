import tokenizer

tok = tokenizer.Tokenizer()
tok.build_vocab(path='../data_aishell/transcript/aishell_transcript_v0.8.txt')
tok.save_vocab("vocab.json")