import os
import logging

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import VocabField
from seq2seq.dataset.dialogDatasets import *
from seq2seq.evaluator import Predictor

from configParser import opt

# path = os.path.join(os.getcwd(), 'chatbot/S2S_pytorch/Seq2Seq-PyTorch')
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

    # Prepare Datasets and Vocab
    src_vocab_list = VocabField.load_vocab(opt.src_vocab_file)
    tgt_vocab_list = VocabField.load_vocab(opt.tgt_vocab_file)
    src_vocab = VocabField(src_vocab_list, vocab_size=opt.src_vocab_size)
    tgt_vocab = VocabField(tgt_vocab_list, vocab_size=opt.tgt_vocab_size, 
                            sos_token="<SOS>", eos_token="<EOS>")
    
    train_set = DialogDataset(opt.train_path,
                              translate_data,
                              src_vocab,
                              tgt_vocab,
                              max_src_length=opt.max_src_length,
                              max_tgt_length=opt.max_tgt_length)
    train = DataLoader(train_set, batch_size=opt.batch_size, shuffle=False)

    dev_set = DialogDataset(opt.dev_path,
                            translate_data,
                            src_vocab,
                            tgt_vocab,
                            max_src_length=opt.max_src_length,
                            max_tgt_length=opt.max_tgt_length)
    dev = DataLoader(dev_set, batch_size=opt.batch_size, shuffle=False)

    # Prepare loss
    weight = torch.ones(len(tgt_vocab.vocab))
    pad_id = tgt_vocab.word2idx[tgt_vocab.pad_token]
    loss = Perplexity(weight, pad_id)
    loss.to(device)

    seq2seq = None
    if not opt.resume:
        # Initialize model
        encoder = EncoderRNN(len(src_vocab.vocab), 
                             opt.max_src_length, 
                             hidden_size=opt.hidden_size,
                             bidirectional=opt.bidirectional, 
                             variable_lengths=False)

        decoder = DecoderRNN(len(tgt_vocab.vocab), 
                             opt.max_tgt_length, 
                             hidden_size=opt.hidden_size * 2 if opt.bidirectional else opt.hidden_size,
                             dropout_p=0.2, 
                             use_attention=opt.use_attn, 
                             bidirectional=opt.bidirectional,
                             eos_id=tgt_vocab.word2idx[tgt_vocab.eos_token], 
                             sos_id=tgt_vocab.word2idx[tgt_vocab.sos_token])
        
        seq2seq = Seq2seq(encoder, decoder)
        seq2seq.to(device)

        if opt.load_checkpoint:
            seq2seq.load_state_dict(torch.load(opt.load_checkpoint))
            print(opt.load_checkpoint)
        elif opt.resume or opt.phase == "infer":
            all_times = sorted(os.listdir(opt.model_dir), reverse=True)
        else:
            for param in seq2seq.parameters():
                param.data.uniform_(-opt.init_weight, opt.init_weight)

    if opt.phase == "train":
        # train
        optimizer = Optimizer(optim.Adam(seq2seq.parameters(), lr=opt.learning_rate), max_grad_norm=opt.clip_grad)
        t = SupervisedTrainer(loss=loss, 
                              batch_size=opt.batch_size,
                              max_epochs=opt.max_epochs,
                              max_steps=opt.max_steps,
                              checkpoint_every=opt.checkpoint_every,
                              print_every=opt.checkpoint_every, 
                              model_dir=opt.model_dir, 
                              device=device)

        seq2seq = t.train(seq2seq, 
                          data=train,
                          start_step=opt.skip_steps, 
                          dev_data=dev,
                          optimizer=optimizer,
                          teacher_forcing_ratio=opt.teacher_forcing_ratio)

    elif opt.phase == "infer":
        # Predict
        predictor = Predictor(seq2seq, src_vocab.word2idx, tgt_vocab.idx2word)

        while True:
            seq_str = raw_input("Type in a source sequence:")
            seq = seq_str.strip().split()
            ans = predictor.predict(seq)
            print(ans)
            print(len(ans))