import torch
from torch.utils.data.dataset import Dataset

from tqdm import tqdm

# def vanilla_dialog_data(subs, obj):
#     src, tgt = subs
#     src = src.split(' ')
#     tgt = tgt.split(' ')
#     if len(src) > obj.max_src_length or len(tgt) > obj.max_tgt_length:
#         return None
#     src_ids = [obj.src_vocab_dic[w] for w in src] + [EOS_token]
#     tgt_ids = [obj.tgt_vocab_dic[w] for w in tgt] + [EOS_token]
#     return {"post": torch.LongTensor(src_ids), 
#             "resp": torch.LongTensor(tgt_ids)}

def translate_data(subs, obj):
    import re
    import unicodedata
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    src, tgt = subs
    src = normalizeString(src).split(' ')
    tgt = normalizeString(tgt).split(' ')
    tgt = [obj.tgt_vocab.sos_token] + tgt + [obj.tgt_vocab.eos_token]
    if len(src) > obj.max_src_length or len(tgt) > obj.max_tgt_length:
        return None
    src_length, tgt_length = len(src), len(tgt)
    src.extend([obj.src_vocab.pad_token] * (obj.max_src_length - src_length))
    tgt.extend([obj.tgt_vocab.pad_token] * (obj.max_tgt_length - tgt_length))
    src_ids = [obj.src_vocab.word2idx[w] for w in src]
    tgt_ids = [obj.tgt_vocab.word2idx[w] for w in tgt]
    return {"src": torch.LongTensor(src_ids), 
            "tgt": torch.LongTensor(tgt_ids), 
            "src_len": torch.LongTensor([src_length]),
            "tgt_len": torch.LongTensor([tgt_length])}


class DialogDataset(Dataset):
    def __init__(self, data_fp, transform_fuc, src_vocab, tgt_vocab, max_src_length, max_tgt_length):
        self.datasets = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        
        loaded = 0
        data_monitor = 0
        with open(data_fp, 'r') as f:
            for line in tqdm(f, desc="Load Data: "):
                subs = line.strip().split('\t')
                loaded += 1
                if not data_monitor: data_monitor = len(subs)
                else: assert data_monitor == len(subs)
                item = transform_fuc(subs, self)
                if item: self.datasets.append(item)

        print(f"{loaded} paris loaded. {len(self.datasets)} are valid. Rate {1.0 * len(self.datasets)/loaded:.4f}")

    
    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]