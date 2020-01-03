import os
import logging

import torch
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
# from seq2seq.util.checkpoint import Checkpoint

from configParser import opt

# path = os.path.join(os.getcwd(), 'chatbot/S2S_pytorch/IBM_s2s_1.3')
# os.chdir(path)

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR




LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

device = torch.device('cuda:%s' % opt.device if opt.device.isdigit() else 'cpu')

    

if __name__ == "__main__":
    # if opt.load_checkpoint is not None:
    #     logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    #     checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    #     checkpoint = Checkpoint.load(checkpoint_path)
    #     seq2seq = checkpoint.model
    #     input_vocab = checkpoint.input_vocab
    #     output_vocab = checkpoint.output_vocab
        
    # else:
    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    max_len = 50
    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab
    print(type(input_vocab))

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    loss.to(device)

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size=128
        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                            bidirectional=bidirectional, variable_lengths=True)
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                            dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                            eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        seq2seq = Seq2seq(encoder, decoder)
        seq2seq.to(device)

        if opt.load_checkpoint is not None:
            seq2seq.load_state_dict(torch.load(opt.load_checkpoint))
            print(opt.load_checkpoint)
        else:
            for param in seq2seq.parameters():
                param.data.uniform_(-0.08, 0.08)

    if opt.phase == "train":
        # train
        t = SupervisedTrainer(loss=loss, batch_size=32,
                            checkpoint_every=500,
                            print_every=500, expt_dir=opt.expt_dir)

        seq2seq = t.train(seq2seq, train,
                        num_epochs=6, dev_data=dev,
                        optimizer=optimizer,
                        teacher_forcing_ratio=0.5,
                        resume=opt.resume)

    elif opt.phase == 'infer':
        # Predict
        
        predictor = Predictor(seq2seq, input_vocab, output_vocab)

        while True:
            seq_str = raw_input("Type in a source sequence:")
            seq = seq_str.strip().split()
            print(predictor.predict(seq))
