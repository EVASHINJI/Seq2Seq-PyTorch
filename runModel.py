import os
import logging

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch.utils.data.distributed as dist
import horovod.torch as hvd
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, TopKDecoder,Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import VocabField
from seq2seq.dataset.dialogDatasets import *
from seq2seq.evaluator import Predictor

from configParser import opt

if opt.random_seed is not None: torch.cuda.manual_seed_all(opt.random_seed)

multi_gpu = False
if opt.device == 'cpu' or opt.device.isdigit():
    device = torch.device(f"cuda:{opt.device}" if opt.device.isdigit() else 'cpu')
else:
    multi_gpu = True
    hvd.init()
    device = torch.device(f"cuda" if opt.device else 'cpu')
    torch.cuda.set_device(hvd.local_rank())
    opt.batch_size = opt.batch_size

LOG_FORMAT = '%(asctime)s %(levelname)-8s %(message)s'
if opt.phase == 'train':
    logging.basicConfig(format=LOG_FORMAT, 
                        level=getattr(logging, opt.log_level.upper()),
                        filename=os.path.join(opt.model_dir, opt.log_file),
                        filemode='a' if opt.resume else 'w')
else:
    logging.basicConfig(format=LOG_FORMAT, 
                        level=getattr(logging, opt.log_level.upper()))
logger = logging.getLogger('train')
if not multi_gpu or hvd.rank() == 0: 
    logger.info(f"Train Log")
    logger.info(opt)

def get_last_checkpoint(model_dir):
    checkpoints_fp = os.path.join(model_dir, "checkpoints")
    try:
        with open(checkpoints_fp, 'r') as f:
            checkpoint = f.readline().strip()
    except:
        return None
    return checkpoint

if __name__ == "__main__":
    # Prepare Datasets and Vocab
    src_vocab_list = VocabField.load_vocab(opt.src_vocab_file)
    tgt_vocab_list = VocabField.load_vocab(opt.tgt_vocab_file)
    src_vocab = VocabField(src_vocab_list, vocab_size=opt.src_vocab_size)
    tgt_vocab = VocabField(tgt_vocab_list, vocab_size=opt.tgt_vocab_size, 
                            sos_token="<SOS>", eos_token="<EOS>")
    pad_id = tgt_vocab.word2idx[tgt_vocab.pad_token]

    # Prepare loss
    weight = torch.ones(len(tgt_vocab.vocab))
    loss = Perplexity(weight, pad_id)
    loss.to(device)

    # Initialize model
    encoder = EncoderRNN(len(src_vocab.vocab),
                         opt.max_src_length,
                         embedding_size=opt.embedding_size,
                         rnn_cell=opt.rnn_cell,
                         n_layers=opt.n_hidden_layer,
                         hidden_size=opt.hidden_size,
                         bidirectional=opt.bidirectional, 
                         variable_lengths=False)

    decoder = DecoderRNN(len(tgt_vocab.vocab),
                         opt.max_tgt_length,
                         embedding_size=opt.embedding_size,
                         rnn_cell=opt.rnn_cell,
                         n_layers=opt.n_hidden_layer,
                         hidden_size=opt.hidden_size * 2 if opt.bidirectional else opt.hidden_size,
                         bidirectional=opt.bidirectional,
                         dropout_p=0.2,
                         use_attention=opt.use_attn, 
                         eos_id=tgt_vocab.word2idx[tgt_vocab.eos_token], 
                         sos_id=tgt_vocab.word2idx[tgt_vocab.sos_token])
    seq2seq = Seq2seq(encoder, decoder)
    seq2seq.to(device)

    if opt.resume and not opt.load_checkpoint:
        last_checkpoint = get_last_checkpoint(opt.model_dir)
        if last_checkpoint:
            opt.load_checkpoint = os.path.join(opt.model_dir, last_checkpoint)
            opt.skip_steps = int(last_checkpoint.strip('.pt').split('/')[-1])

    if opt.load_checkpoint:
        seq2seq.load_state_dict(torch.load(opt.load_checkpoint))
        opt.skip_steps = int(opt.load_checkpoint.strip('.pt').split('/')[-1])
        if not multi_gpu or hvd.rank() == 0: logger.info(f"\nLoad from {opt.load_checkpoint}\n")
    else:
        for param in seq2seq.parameters():
            param.data.uniform_(-opt.init_weight, opt.init_weight)
    
    if opt.beam_width > 1 and opt.phase == "infer":
        if not multi_gpu or hvd.rank() == 0: logger.info(f"Beam Width {opt.beam_width}")
        seq2seq.decoder = TopKDecoder(seq2seq.decoder, opt.beam_width)

    if opt.phase == "train":
        # Prepare Train Data
        trans_data = TranslateData(pad_id)
        train_set = DialogDataset(opt.train_path,
                                  trans_data.translate_data,
                                  src_vocab,
                                  tgt_vocab,
                                  max_src_length=opt.max_src_length,
                                  max_tgt_length=opt.max_tgt_length)
        train_sampler = dist.DistributedSampler(train_set, num_replicas=hvd.size(), rank=hvd.rank()) \
                            if multi_gpu else None
        train = DataLoader(train_set, 
                           batch_size=opt.batch_size, 
                           shuffle=False if multi_gpu else True,
                           sampler=train_sampler,
                           drop_last=True,
                           collate_fn=trans_data.collate_fn)

        dev_set = DialogDataset(opt.dev_path,
                                trans_data.translate_data,
                                src_vocab,
                                tgt_vocab,
                                max_src_length=opt.max_src_length,
                                max_tgt_length=opt.max_tgt_length)
        dev_sampler = dist.DistributedSampler(dev_set, num_replicas=hvd.size(), rank=hvd.rank()) \
                            if multi_gpu else None
        dev = DataLoader(dev_set, 
                        batch_size=opt.batch_size, 
                        shuffle=False, 
                        sampler=dev_sampler, 
                        collate_fn=trans_data.collate_fn)

        # Prepare optimizer
        # optimizer = Optimizer(optim.Adam(seq2seq.parameters(), lr=opt.learning_rate), max_grad_norm=opt.clip_grad)
        optimizer = optim.Adam(seq2seq.parameters(), lr=opt.learning_rate)
        if multi_gpu: 
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=seq2seq.named_parameters())
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            hvd.broadcast_parameters(seq2seq.state_dict(), root_rank=0)
        optimizer = Optimizer(optimizer, max_grad_norm=opt.clip_grad)
        if opt.decay_factor:
            optimizer.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, 'min', factor=opt.decay_factor, patience=1))

        # Prepare trainer and train
        t = SupervisedTrainer(loss=loss, 
                              model_dir=opt.model_dir,
                              best_model_dir=opt.best_model_dir,
                              batch_size=opt.batch_size,
                              checkpoint_every=opt.checkpoint_every,
                              print_every=opt.print_every,
                              max_epochs=opt.max_epochs,
                              max_steps=opt.max_steps,
                              max_checkpoints_num=opt.max_checkpoints_num,
                              best_ppl=opt.best_ppl,
                              device=device,
                              multi_gpu=multi_gpu,
                              logger=logger)

        seq2seq = t.train(seq2seq, 
                          data=train,
                          start_step=opt.skip_steps, 
                          dev_data=dev,
                          optimizer=optimizer,
                          teacher_forcing_ratio=opt.teacher_forcing_ratio)

    elif opt.phase == "infer":
        # Predict
        predictor = Predictor(seq2seq, src_vocab.word2idx, tgt_vocab.idx2word, device)

        while True:
            seq_str = input("Type in a source sequence:")
            seq = seq_str.strip().split()
            ans = predictor.predict_n(seq, n=opt.beam_width) \
                if opt.beam_width > 1 else predictor.predict(seq)
            print(ans)
