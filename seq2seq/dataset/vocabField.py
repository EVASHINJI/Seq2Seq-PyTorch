from collections import defaultdict

class VocabField():
    def __init__(self, vocab, vocab_size=None, unk_token="<UNK>", pad_token="<PAD>", sos_token=None, eos_token=None):
        default_tokens = [unk_token, pad_token]
        if sos_token: default_tokens.append(sos_token)
        if eos_token: default_tokens.append(eos_token)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        vocab = default_tokens + vocab
        if vocab_size: vocab = vocab[:vocab_size]
        unk_id = vocab.index(unk_token)
        self.vocab = vocab
        self.word2idx = defaultdict(lambda : unk_id)
        self.idx2word = defaultdict(lambda : unk_token)
        for i, w in enumerate(vocab):
            self.word2idx[w] = i
            self.idx2word[i] = w

    @staticmethod
    def load_vocab(vocab_fp):
        vocab = []
        with open(vocab_fp, 'r') as f:
            for line in f:
                line = line.strip()
                if line: vocab.append(line)
        return vocab
